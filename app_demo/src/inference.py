from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .models import ModelBundle, compute_motion, create_model, load_state, safe_torch_load


@dataclass(frozen=True)
class Prediction:
    text: str
    task: str
    elapsed_ms: float
    confidence: float | None = None
    topk: list[tuple[str, float]] | None = None
    frames: int | None = None


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(app_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (app_root / path).resolve()


def load_model_specs(app_root: Path, config_path: Path | None = None) -> list[dict[str, Any]]:
    cfg_path = config_path or app_root / "config" / "models.json"
    data = read_json(cfg_path)
    return data.get("models", data)


def normalize_id_map(raw: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for key, value in raw.items():
        out[int(key)] = str(value)
    return out


def load_demo_model(spec: dict[str, Any], app_root: Path, device: torch.device | None = None) -> ModelBundle:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_path(app_root, spec["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path)
    config: dict[str, Any] = {}
    if "config_path" in spec:
        config_path = resolve_path(app_root, spec["config_path"])
        if config_path.exists():
            config.update(read_json(config_path))
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        config.update(checkpoint["config"])

    task = spec["task"]
    architecture = spec["architecture"]
    labels: dict[int, str] | None = None
    vocab: dict[str, int] | None = None

    if task == "islr":
        if "label_map" in spec:
            labels = normalize_id_map(read_json(resolve_path(app_root, spec["label_map"])))
        elif isinstance(checkpoint, dict) and "id_to_label" in checkpoint:
            labels = normalize_id_map(checkpoint["id_to_label"])
        else:
            raise ValueError(f"ISLR model {spec['id']} needs id_to_label mapping.")
        output_size = len(labels)
    elif task == "cslr":
        vocab = {str(k): int(v) for k, v in read_json(resolve_path(app_root, spec["vocab"])).items()}
        output_size = len(vocab)
    else:
        raise ValueError(f"Unsupported task: {task}")

    model = create_model(architecture, output_size, config)
    load_state(model, checkpoint_path)
    model.to(device)
    model.eval()
    return ModelBundle(model=model, task=task, architecture=architecture, labels=labels, vocab=vocab, config=config, device=device)


def pooled_lengths(input_lens: torch.Tensor, output_time: int) -> torch.Tensor:
    return ((input_lens + 1) // 2).clamp(min=1, max=output_time)


def adjust_logits_for_ctc(logits: torch.Tensor, config: dict[str, Any], decode: bool = True) -> torch.Tensor:
    key = "blank_decode_penalty" if decode else "blank_logit_bias"
    penalty = float(config.get(key, 0.0) or 0.0)
    if penalty > 0:
        logits = logits.clone()
        logits[:, :, 0] = logits[:, :, 0] - penalty
    return logits


def greedy_decode(log_probs: torch.Tensor, feat_lens: torch.Tensor, inv_vocab: dict[int, str]) -> list[str]:
    best = log_probs.argmax(dim=-1).cpu()
    decoded: list[str] = []
    for i in range(best.shape[0]):
        seq = best[i, : int(feat_lens[i])].tolist()
        out: list[str] = []
        prev = None
        for idx in seq:
            if idx != prev and idx != 0:
                out.append(inv_vocab.get(int(idx), ""))
            prev = idx
        decoded.append(" ".join(token for token in out if token))
    return decoded


def beam_decode_one(log_probs: torch.Tensor, inv_vocab: dict[int, str], beam_size: int = 8, topk: int = 25, length_reward: float = 0.0) -> str:
    arr = log_probs.detach().cpu().numpy()
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -np.inf)}
    vocab_size = arr.shape[1]
    topk = min(topk, vocab_size)
    for t in range(arr.shape[0]):
        ids = np.argpartition(arr[t], -topk)[-topk:]
        if 0 not in ids:
            ids = np.concatenate([[0], ids])
        next_beams = defaultdict(lambda: [-np.inf, -np.inf])
        for prefix, (pb, pnb) in beams.items():
            total = np.logaddexp(pb, pnb)
            next_beams[prefix][0] = np.logaddexp(next_beams[prefix][0], total + arr[t, 0])
            for c in ids:
                c = int(c)
                if c == 0:
                    continue
                p = arr[t, c]
                new_prefix = prefix + (c,)
                if prefix and c == prefix[-1]:
                    next_beams[new_prefix][1] = np.logaddexp(next_beams[new_prefix][1], pb + p)
                    next_beams[prefix][1] = np.logaddexp(next_beams[prefix][1], pnb + p)
                else:
                    next_beams[new_prefix][1] = np.logaddexp(next_beams[new_prefix][1], total + p)
        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]) + length_reward * len(kv[0]),
                reverse=True,
            )[:beam_size]
        )
    best = max(beams, key=lambda k: np.logaddexp(beams[k][0], beams[k][1]) + length_reward * len(k)) if beams else ()
    return " ".join(inv_vocab.get(int(c), "") for c in best)


@torch.no_grad()
def predict_skeleton(bundle: ModelBundle, skeleton: np.ndarray, decode_method: str = "greedy", topk: int = 5) -> Prediction:
    if skeleton.ndim != 3 or skeleton.shape[1:] != (86, 3):
        raise ValueError(f"Expected skeleton shape (T, 86, 3), got {skeleton.shape}")
    started = time.time()
    device = bundle.device or torch.device("cpu")
    sk = torch.from_numpy(np.nan_to_num(skeleton, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)).unsqueeze(0).to(device)
    lengths = torch.tensor([sk.shape[1]], dtype=torch.long, device=device)

    if bundle.task == "islr":
        logits = bundle.model(sk, lengths)
        probs = torch.softmax(logits, dim=-1)[0]
        values, indices = torch.topk(probs, k=min(topk, probs.numel()))
        top_items = [
            (bundle.labels.get(int(idx), str(int(idx))) if bundle.labels else str(int(idx)), float(val))
            for val, idx in zip(values.cpu(), indices.cpu())
        ]
        elapsed = (time.time() - started) * 1000.0
        return Prediction(
            text=top_items[0][0] if top_items else "",
            task="islr",
            elapsed_ms=elapsed,
            confidence=top_items[0][1] if top_items else None,
            topk=top_items,
            frames=int(skeleton.shape[0]),
        )

    config = bundle.config or {}
    inv_vocab = {idx: token for token, idx in (bundle.vocab or {}).items()}
    if bundle.architecture == "mska_plus_cslr":
        outputs = bundle.model(sk)
        logits = outputs[-1]
    else:
        mo = compute_motion(sk)
        _, logits = bundle.model(sk, mo)
    logits = adjust_logits_for_ctc(logits, config, decode=True)
    log_probs = logits.log_softmax(-1)
    feat_lens = pooled_lengths(lengths.cpu(), logits.shape[1])
    if decode_method == "beam":
        text = beam_decode_one(
            log_probs[0, : int(feat_lens[0])],
            inv_vocab,
            beam_size=int(config.get("beam_width", 8)),
            topk=int(config.get("beam_topk", 25)),
            length_reward=float(config.get("length_reward", 0.0)),
        )
    else:
        text = greedy_decode(log_probs, feat_lens, inv_vocab)[0]
    probs = log_probs.exp()[0, : int(feat_lens[0])]
    nonblank = probs.argmax(dim=-1) != 0
    confidence = float(probs.max(dim=-1).values[nonblank].mean().cpu()) if bool(nonblank.any()) else 0.0
    elapsed = (time.time() - started) * 1000.0
    return Prediction(text=text, task="cslr", elapsed_ms=elapsed, confidence=confidence, frames=int(skeleton.shape[0]))
