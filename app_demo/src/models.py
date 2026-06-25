from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import math
import numpy as np
import torch
from torch import nn


EDGES_BODY = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]
EDGES_HAND = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]
EDGES_FACE = [(0, 2), (1, 3), (2, 4), (3, 4), (4, 5), (4, 6), (0, 1)]
EDGES_MOUTH = [(0, 1), (1, 2), (2, 3), (3, 0)]


def build_adjacency(num_nodes: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    matrix = np.zeros((2, num_nodes, num_nodes), dtype=np.float32)
    matrix[0] = np.eye(num_nodes, dtype=np.float32)
    for i, j in edges:
        if 0 <= i < num_nodes and 0 <= j < num_nodes:
            matrix[1, i, j] = 1.0
            matrix[1, j, i] = 1.0
    for k in range(matrix.shape[0]):
        deg = matrix[k].sum(axis=1)
        deg[deg == 0] = 1.0
        norm = np.diag(1.0 / np.sqrt(deg))
        matrix[k] = norm @ matrix[k] @ norm
    return torch.tensor(matrix, dtype=torch.float32)


def compute_motion(sk: torch.Tensor) -> torch.Tensor:
    xy = sk[:, :, :, :2]
    dx_prev = torch.cat([torch.zeros_like(xy[:, :1]), xy[:, 1:] - xy[:, :-1]], dim=1)
    dx_next = torch.cat([xy[:, 1:] - xy[:, :-1], torch.zeros_like(xy[:, :1])], dim=1)
    return torch.cat([dx_prev, dx_next], dim=-1).clamp(-1.0, 1.0)


def add_motion_features(sk: torch.Tensor) -> torch.Tensor:
    xy = sk[:, :, :, :2]
    dx = torch.cat([torch.zeros_like(xy[:, :1]), xy[:, 1:] - xy[:, :-1]], dim=1)
    return torch.cat([sk, dx.clamp(-1.0, 1.0)], dim=-1)


def center_xy(x: torch.Tensor, root_idx: int) -> torch.Tensor:
    out = x.clone()
    out[:, :, :, :2] = out[:, :, :, :2] - out[:, :, root_idx:root_idx + 1, :2]
    return out


class STGCNBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, num_nodes: int, edges: list[tuple[int, int]], adaptive: bool = False):
        super().__init__()
        adjacency = build_adjacency(num_nodes, edges)
        if adaptive:
            self.A = nn.Parameter(adjacency.clone())
        else:
            self.register_buffer("A", adjacency)
        self.spatial = nn.Conv2d(in_c * 2, out_c, kernel_size=1, bias=False)
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=(3, 1), padding=(1, 0), groups=out_c, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.residual = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()
        parts = [torch.einsum("bctv,vw->bctw", x, self.A[k]) for k in range(2)]
        y = self.spatial(torch.cat(parts, dim=1))
        y = self.temporal(y) + self.residual(x)
        return self.act(y).permute(0, 2, 3, 1).contiguous()


class StreamEncoder(nn.Module):
    def __init__(self, in_c: int, hidden: int, num_nodes: int, edges: list[tuple[int, int]], adaptive: bool = False):
        super().__init__()
        mid = max(64, hidden)
        self.blocks = nn.Sequential(
            STGCNBlock(in_c, 64, num_nodes, edges, adaptive),
            STGCNBlock(64, mid, num_nodes, edges, adaptive),
            STGCNBlock(mid, hidden, num_nodes, edges, adaptive),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x).mean(dim=2)


class PipelineTransferStreamEncoder(nn.Module):
    def __init__(self, in_c: int, hidden: int, num_nodes: int, edges: list[tuple[int, int]], adaptive: bool = False):
        super().__init__()
        self.blocks = nn.Sequential(
            STGCNBlock(in_c, 64, num_nodes, edges, adaptive),
            STGCNBlock(64, hidden, num_nodes, edges, adaptive),
            STGCNBlock(hidden, hidden, num_nodes, edges, adaptive),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x).mean(dim=2)


class ISLRStreamEncoder(nn.Module):
    def __init__(self, in_c: int, hidden: int, num_nodes: int, edges: list[tuple[int, int]]):
        super().__init__()
        self.blocks = nn.Sequential(
            STGCNBlock(in_c, 64, num_nodes, edges, adaptive=False),
            STGCNBlock(64, hidden, num_nodes, edges, adaptive=False),
            STGCNBlock(hidden, hidden, num_nodes, edges, adaptive=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x).mean(dim=2)


class PipelineISLRModel(nn.Module):
    def __init__(self, num_classes: int, config: dict[str, Any]):
        super().__init__()
        hidden = int(config.get("hidden", 256))
        dropout = float(config.get("dropout", 0.35))
        part_hidden = max(32, hidden // 5)
        self.body_sk = ISLRStreamEncoder(3, part_hidden, 33, EDGES_BODY)
        self.lh_sk = ISLRStreamEncoder(3, part_hidden, 21, EDGES_HAND)
        self.rh_sk = ISLRStreamEncoder(3, part_hidden, 21, EDGES_HAND)
        self.face_sk = ISLRStreamEncoder(3, part_hidden, 7, EDGES_FACE)
        self.mouth_sk = ISLRStreamEncoder(3, part_hidden, 4, EDGES_MOUTH)
        self.body_mo = ISLRStreamEncoder(4, part_hidden, 33, EDGES_BODY)
        self.lh_mo = ISLRStreamEncoder(4, part_hidden, 21, EDGES_HAND)
        self.rh_mo = ISLRStreamEncoder(4, part_hidden, 21, EDGES_HAND)
        self.face_mo = ISLRStreamEncoder(4, part_hidden, 7, EDGES_FACE)
        self.mouth_mo = ISLRStreamEncoder(4, part_hidden, 4, EDGES_MOUTH)
        feat_dim = part_hidden * 10
        self.temporal = nn.Sequential(
            nn.Conv1d(feat_dim, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def encode(self, sk: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        mo = compute_motion(sk)
        body = center_xy(sk[:, :, 0:33, :], 0)
        lh = center_xy(sk[:, :, 33:54, :], 0)
        rh = center_xy(sk[:, :, 54:75, :], 0)
        face = center_xy(sk[:, :, 75:82, :], 4)
        mouth = center_xy(sk[:, :, 82:86, :], 0)
        feat = torch.cat([
            self.body_sk(body), self.lh_sk(lh), self.rh_sk(rh), self.face_sk(face), self.mouth_sk(mouth),
            self.body_mo(mo[:, :, 0:33, :]), self.lh_mo(mo[:, :, 33:54, :]), self.rh_mo(mo[:, :, 54:75, :]),
            self.face_mo(mo[:, :, 75:82, :]), self.mouth_mo(mo[:, :, 82:86, :]),
        ], dim=-1)
        z = self.temporal(feat.transpose(1, 2)).transpose(1, 2)
        if lengths is None:
            return z.mean(dim=1)
        pooled_lengths = torch.clamp(lengths.to(z.device), max=z.size(1))
        mask = torch.arange(z.size(1), device=z.device)[None, :] < pooled_lengths[:, None]
        z = z * mask.unsqueeze(-1)
        return z.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)

    def forward(self, sk: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.cls(self.encode(sk, lengths))


class TCNBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, max(32, dim // 2)),
            nn.Tanh(),
            nn.Linear(max(32, dim // 2), 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None].to(x.device)
            scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class LiteTCNBiGRUISLRModel(nn.Module):
    def __init__(self, num_classes: int, config: dict[str, Any]):
        super().__init__()
        hidden = int(config.get("hidden", 256))
        dropout = float(config.get("dropout", 0.25))
        use_motion = bool(config.get("use_motion", True))
        self.use_motion = use_motion
        in_channels = 5 if use_motion else 3
        self.input_dim = 86 * in_channels
        self.frame_proj = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tcn = nn.Sequential(
            TCNBlock(hidden, dropout, kernel_size=5, dilation=1),
            TCNBlock(hidden, dropout, kernel_size=5, dilation=2),
            TCNBlock(hidden, dropout, kernel_size=5, dilation=4),
        )
        self.rnn = nn.GRU(
            input_size=hidden,
            hidden_size=hidden // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.pool = MaskedAttentionPool(hidden)
        self.cls = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def make_frame_features(self, sk: torch.Tensor) -> torch.Tensor:
        if self.use_motion:
            sk = add_motion_features(sk)
        return sk.reshape(sk.size(0), sk.size(1), -1)

    def encode_sequence(self, sk: torch.Tensor) -> torch.Tensor:
        x = self.make_frame_features(sk)
        x = self.frame_proj(x)
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.rnn(x)
        return x

    def encode(self, sk: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.pool(self.encode_sequence(sk), lengths)

    def forward(self, sk: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.cls(self.encode(sk, lengths))


class PipelineCSLRModel(nn.Module):
    def __init__(self, vocab_size: int, config: dict[str, Any]):
        super().__init__()
        hidden = int(config.get("hidden", 256))
        dropout = float(config.get("dropout", 0.28))
        adaptive = bool(config.get("adaptive_gcn", True))
        part_hidden = int(config.get("part_hidden", hidden // 4))
        encoder_cls = PipelineTransferStreamEncoder if "part_hidden" in config else StreamEncoder
        self.body_sk = encoder_cls(3, part_hidden, 33, EDGES_BODY, adaptive)
        self.lh_sk = encoder_cls(3, part_hidden, 21, EDGES_HAND, adaptive)
        self.rh_sk = encoder_cls(3, part_hidden, 21, EDGES_HAND, adaptive)
        self.face_sk = encoder_cls(3, part_hidden, 7, EDGES_FACE, adaptive)
        self.mouth_sk = encoder_cls(3, part_hidden, 4, EDGES_MOUTH, adaptive)
        self.body_mo = encoder_cls(4, part_hidden, 33, EDGES_BODY, adaptive)
        self.lh_mo = encoder_cls(4, part_hidden, 21, EDGES_HAND, adaptive)
        self.rh_mo = encoder_cls(4, part_hidden, 21, EDGES_HAND, adaptive)
        self.face_mo = encoder_cls(4, part_hidden, 7, EDGES_FACE, adaptive)
        self.mouth_mo = encoder_cls(4, part_hidden, 4, EDGES_MOUTH, adaptive)
        feat_dim = part_hidden * 10
        self.temporal = nn.Sequential(
            nn.Conv1d(feat_dim, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),
        )
        self.lstm = nn.LSTM(256, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.cls_aux = nn.Linear(256, vocab_size)
        self.cls_pri = nn.Linear(512, vocab_size)

    def extract_features(self, sk: torch.Tensor, mo: torch.Tensor) -> torch.Tensor:
        body = center_xy(sk[:, :, 0:33, :], 0)
        lh = center_xy(sk[:, :, 33:54, :], 0)
        rh = center_xy(sk[:, :, 54:75, :], 0)
        face = center_xy(sk[:, :, 75:82, :], 4)
        mouth = center_xy(sk[:, :, 82:86, :], 0)
        return torch.cat([
            self.body_sk(body), self.lh_sk(lh), self.rh_sk(rh), self.face_sk(face), self.mouth_sk(mouth),
            self.body_mo(mo[:, :, 0:33, :]), self.lh_mo(mo[:, :, 33:54, :]), self.rh_mo(mo[:, :, 54:75, :]),
            self.face_mo(mo[:, :, 75:82, :]), self.mouth_mo(mo[:, :, 82:86, :]),
        ], dim=-1)

    def forward(self, sk: torch.Tensor, mo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(sk, mo)
        z = self.temporal(feat.transpose(1, 2)).transpose(1, 2)
        aux = self.cls_aux(z)
        z, _ = self.lstm(z)
        pri = self.cls_pri(z)
        return aux, pri


class LSTMAttentionISLRModel(nn.Module):
    def __init__(self, num_classes: int, config: dict[str, Any]):
        super().__init__()
        hidden = int(config.get("hidden", 256))
        dropout = float(config.get("dropout", 0.30))
        self.use_motion = bool(config.get("use_motion", True))
        in_channels = 5 if self.use_motion else 3
        self.input_dim = 86 * in_channels
        self.frame_proj = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rnn = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.pool = MaskedAttentionPool(hidden)
        self.cls = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(128, hidden // 2)),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(max(128, hidden // 2), num_classes),
        )

    def make_frame_features(self, sk: torch.Tensor) -> torch.Tensor:
        if self.use_motion:
            sk = add_motion_features(sk)
        return sk.reshape(sk.size(0), sk.size(1), -1)

    def encode_sequence(self, sk: torch.Tensor) -> torch.Tensor:
        x = self.make_frame_features(sk)
        x = self.frame_proj(x)
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        return x

    def encode(self, sk: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.pool(self.encode_sequence(sk), lengths)

    def forward(self, sk: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.cls(self.encode(sk, lengths))


class LiteTCNBiGRUCTCModel(nn.Module):
    def __init__(self, vocab_size: int, config: dict[str, Any]):
        super().__init__()
        hidden = int(config.get("hidden", 256))
        dropout = float(config.get("dropout", 0.25))
        self.use_motion = bool(config.get("use_motion", True))
        in_channels = 5 if self.use_motion else 3
        self.input_dim = 86 * in_channels
        self.frame_proj = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tcn = nn.Sequential(
            TCNBlock(hidden, dropout, kernel_size=5, dilation=1),
            TCNBlock(hidden, dropout, kernel_size=5, dilation=2),
            TCNBlock(hidden, dropout, kernel_size=5, dilation=4),
        )
        self.downsample = nn.MaxPool1d(kernel_size=2, ceil_mode=True)
        self.aux = nn.Linear(hidden, vocab_size)
        self.rnn = nn.GRU(
            input_size=hidden,
            hidden_size=hidden // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.pri = nn.Linear(hidden, vocab_size)

    def make_frame_features(self, sk: torch.Tensor) -> torch.Tensor:
        if self.use_motion:
            sk = add_motion_features(sk)
        return sk.reshape(sk.size(0), sk.size(1), -1)

    def forward(self, sk: torch.Tensor, mo: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.make_frame_features(sk)
        x = self.frame_proj(x)
        x = self.tcn(x.transpose(1, 2))
        x = self.downsample(x).transpose(1, 2)
        aux = self.aux(x)
        self.rnn.flatten_parameters()
        z, _ = self.rnn(x)
        pri = self.pri(z)
        return aux, pri


class LSTMAttentionCTCModel(nn.Module):
    def __init__(self, vocab_size: int, config: dict[str, Any]):
        super().__init__()
        hidden = int(config.get("hidden", 256))
        dropout = float(config.get("dropout", 0.30))
        self.use_motion = bool(config.get("use_motion", True))
        in_channels = 5 if self.use_motion else 3
        self.input_dim = 86 * in_channels
        self.frame_proj = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.MaxPool1d(kernel_size=2, ceil_mode=True)
        self.aux = nn.Linear(hidden, vocab_size)
        self.rnn = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.pri = nn.Linear(hidden, vocab_size)

    def make_frame_features(self, sk: torch.Tensor) -> torch.Tensor:
        if self.use_motion:
            sk = add_motion_features(sk)
        return sk.reshape(sk.size(0), sk.size(1), -1)

    def forward(self, sk: torch.Tensor, mo: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.make_frame_features(sk)
        x = self.frame_proj(x)
        x = self.downsample(x.transpose(1, 2)).transpose(1, 2)
        aux = self.aux(x)
        self.rnn.flatten_parameters()
        z, _ = self.rnn(x)
        pri = self.pri(z)
        return aux, pri


MSKA_STREAMS = {
    "body": [0, 11, 12, 13, 14, 15, 16, 23, 24],
    "lhand": list(range(33, 54)),
    "rhand": list(range(54, 75)),
    "face": list(range(75, 86)),
}


class MSKASpatialAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_points: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model)
        self.spatial_bias = nn.Parameter(torch.zeros(n_heads, n_points, n_points))
        self.attn_drop = nn.Dropout(dropout * 0.5)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout * 0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, timesteps, points, dim = x.shape

        def heads(z: torch.Tensor) -> torch.Tensor:
            return z.view(bsz, timesteps, points, self.n_heads, self.d_head).permute(0, 1, 3, 2, 4).contiguous()

        q = heads(self.q(x))
        k = heads(self.k(x))
        v = heads(self.v(x))
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores + self.spatial_bias.float().unsqueeze(0).unsqueeze(0)
        attn = self.attn_drop(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v.float())
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(bsz, timesteps, points, dim)
        x = self.norm1(x + self.o(out.to(x.dtype)))
        return self.norm2(x + self.ffn(x))


class MSKAStreamEncoder(nn.Module):
    def __init__(self, n_points: int, config: dict[str, Any]):
        super().__init__()
        hidden = int(config.get("hidden", 320))
        dropout = float(config.get("dropout", 0.30))
        input_channels = 5 if bool(config.get("use_motion", True)) else 3
        self.use_conf_gate = bool(config.get("use_conf_gate", True))
        self.input_norm = nn.LayerNorm(input_channels)
        self.proj = nn.Linear(input_channels, hidden)
        self.point_embed = nn.Parameter(torch.zeros(1, 1, n_points, hidden))
        self.conf_alpha = nn.Parameter(torch.tensor(0.5))
        self.blocks = nn.ModuleList([
            MSKASpatialAttentionBlock(hidden, n_points, int(config.get("num_heads", 8)), dropout)
            for _ in range(int(config.get("num_blocks", 4)))
        ])
        self.pool_score = nn.Linear(hidden, 1)
        self.out_norm = nn.LayerNorm(hidden)
        nn.init.trunc_normal_(self.point_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        confidence = x[..., 2:3].clamp(0.0, 1.0) if x.shape[-1] >= 3 else None
        if self.use_conf_gate and confidence is not None:
            x = x * (1.0 + torch.sigmoid(self.conf_alpha) * confidence)
        h = self.proj(self.input_norm(x)) + self.point_embed
        for block in self.blocks:
            h = block(h)
        scores = self.pool_score(h).squeeze(-1)
        if confidence is not None:
            scores = scores + torch.log(confidence.squeeze(-1).clamp_min(1e-4))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return self.out_norm((h * weights).sum(dim=2))


class MSKAMultiScaleTemporalBlock(nn.Module):
    def __init__(self, hidden: int, config: dict[str, Any]):
        super().__init__()
        dropout = float(config.get("dropout", 0.30))
        kernels = config.get("temporal_kernel_sizes", [3, 5, 7])
        self.norm = nn.LayerNorm(hidden)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel_size=int(k), padding=int(k) // 2, groups=hidden, bias=False),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
            )
            for k in kernels
        ])
        self.mix = nn.Sequential(
            nn.Conv1d(hidden * len(kernels), hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout * 0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x).transpose(1, 2)
        z = torch.cat([branch(z) for branch in self.branches], dim=1)
        z = self.mix(z).transpose(1, 2)
        x = x + z
        return x + self.ffn(x)


class MSKAAttentiveTemporalPool(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None].to(x.device)
            scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class MSKAPlusISLRModel(nn.Module):
    def __init__(self, num_classes: int, config: dict[str, Any]):
        super().__init__()
        hidden = int(config.get("hidden", 320))
        dropout = float(config.get("dropout", 0.30))
        self.config = config
        self.enc = nn.ModuleDict({name: MSKAStreamEncoder(len(indices), config) for name, indices in MSKA_STREAMS.items()})
        self.stream_gate = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 4),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.ModuleList([
            MSKAMultiScaleTemporalBlock(hidden, config)
            for _ in range(int(config.get("temporal_blocks", 3)))
        ])
        self.rnn = nn.GRU(
            input_size=hidden,
            hidden_size=hidden // 2,
            num_layers=int(config.get("temporal_rnn_layers", 2)),
            batch_first=True,
            bidirectional=True,
            dropout=dropout if int(config.get("temporal_rnn_layers", 2)) > 1 else 0.0,
        )
        self.pool = MSKAAttentiveTemporalPool(hidden)
        self.cls = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def encode_sequence(self, sk: torch.Tensor) -> torch.Tensor:
        sk_in = add_motion_features(sk) if bool(self.config.get("use_motion", True)) else sk
        features = [self.enc[name](sk_in[:, :, indices, :]) for name, indices in MSKA_STREAMS.items()]
        concat = torch.cat(features, dim=-1)
        gates = torch.softmax(self.stream_gate(concat), dim=-1)
        gated = torch.cat([feat * gates[:, :, i:i + 1] for i, feat in enumerate(features)], dim=-1)
        z = self.fuse(gated)
        for block in self.temporal:
            z = block(z)
        self.rnn.flatten_parameters()
        z, _ = self.rnn(z)
        return z

    def encode(self, sk: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.pool(self.encode_sequence(sk), lengths)

    def forward(self, sk: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.cls(self.encode(sk, lengths))


class MSKAStrongTemporalEncoder(nn.Module):
    def __init__(self, hidden: int, config: dict[str, Any]):
        super().__init__()
        dropout = float(config.get("dropout", 0.30))
        self.blocks = nn.ModuleList([
            MSKAMultiScaleTemporalBlock(hidden, config)
            for _ in range(int(config.get("temporal_blocks", 4)))
        ])
        self.downsample = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)
        layers = int(config.get("temporal_rnn_layers", 2))
        self.rnn = nn.GRU(
            input_size=hidden,
            hidden_size=hidden // 2,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.downsample(x.transpose(1, 2)).transpose(1, 2)
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        return self.out(x)


class MSKAAuxTemporalHead(nn.Module):
    def __init__(self, in_dim: int, vocab_size: int, config: dict[str, Any]):
        super().__init__()
        dropout = float(config.get("dropout", 0.30))
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=5, padding=2, groups=in_dim, bias=False),
            nn.BatchNorm1d(in_dim),
            nn.GELU(),
            nn.Conv1d(in_dim, in_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_dim),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Dropout(dropout),
        )
        self.cls = nn.Linear(in_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(self.net(x.transpose(1, 2)).transpose(1, 2))


class MSKAFuseCTCHead(nn.Module):
    def __init__(self, in_dim: int, vocab_size: int, config: dict[str, Any]):
        super().__init__()
        self.temporal = MSKAStrongTemporalEncoder(in_dim, config)
        self.cls = nn.Linear(in_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(self.temporal(x))


class MSKAPlusCSLRModel(nn.Module):
    def __init__(self, vocab_size: int, config: dict[str, Any]):
        super().__init__()
        hidden = int(config.get("hidden", 320))
        dropout = float(config.get("dropout", 0.30))
        self.config = config
        self.enc = nn.ModuleDict({name: MSKAStreamEncoder(len(indices), config) for name, indices in MSKA_STREAMS.items()})
        self.stream_gate = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 4),
        )
        self.fuse_proj = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.aux_heads = nn.ModuleDict({name: MSKAAuxTemporalHead(hidden, vocab_size, config) for name in MSKA_STREAMS})
        self.fuse_head = MSKAFuseCTCHead(hidden, vocab_size, config)

    def forward(self, sk: torch.Tensor) -> tuple[torch.Tensor, ...]:
        sk_in = add_motion_features(sk) if bool(self.config.get("use_motion", True)) else sk
        features = {name: self.enc[name](sk_in[:, :, indices, :]) for name, indices in MSKA_STREAMS.items()}
        ordered = [features[name] for name in MSKA_STREAMS]
        concat = torch.cat(ordered, dim=-1)
        gates = torch.softmax(self.stream_gate(concat), dim=-1)
        gated = torch.cat([feat * gates[:, :, i:i + 1] for i, feat in enumerate(ordered)], dim=-1)
        fused = self.fuse_proj(gated)
        aux = [self.aux_heads[name](features[name]) for name in MSKA_STREAMS]
        return tuple(aux + [self.fuse_head(fused)])


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    task: str
    architecture: str
    labels: dict[int, str] | None = None
    vocab: dict[str, int] | None = None
    config: dict[str, Any] | None = None
    device: torch.device | None = None


def safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_state(model: nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = safe_torch_load(checkpoint_path)
    state = checkpoint.get("model_state", checkpoint.get("model", checkpoint))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"Checkpoint load warning for {checkpoint_path.name}: missing={len(missing)}, unexpected={len(unexpected)}")
    return checkpoint


def create_model(architecture: str, output_size: int, config: dict[str, Any]) -> nn.Module:
    if architecture == "pipeline_islr":
        return PipelineISLRModel(output_size, config)
    if architecture == "lite_tcn_bigru_islr":
        return LiteTCNBiGRUISLRModel(output_size, config)
    if architecture == "lstm_attention_islr":
        return LSTMAttentionISLRModel(output_size, config)
    if architecture == "mska_plus_islr":
        return MSKAPlusISLRModel(output_size, config)
    if architecture == "pipeline_cslr":
        return PipelineCSLRModel(output_size, config)
    if architecture == "lite_tcn_bigru_cslr":
        return LiteTCNBiGRUCTCModel(output_size, config)
    if architecture == "lstm_attention_cslr":
        return LSTMAttentionCTCModel(output_size, config)
    if architecture == "mska_plus_cslr":
        return MSKAPlusCSLRModel(output_size, config)
    raise ValueError(f"Unsupported model architecture: {architecture}")
