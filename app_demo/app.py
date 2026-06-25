from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
import torch

from src.inference import Prediction, load_demo_model, load_model_specs, predict_skeleton
from src.skeleton import (
    FpsMeter,
    LiveSegmenter,
    draw_status,
    extract_frame,
    extract_video_skeleton,
    hand_present,
)

try:
    import av
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False


APP_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = APP_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(page_title="VieSL Demo", page_icon="VieSL", layout="wide")


@st.cache_resource(show_spinner=False)
def cached_model(spec_id: str) -> tuple[dict, object]:
    specs = load_model_specs(APP_ROOT)
    spec = next(item for item in specs if item["id"] == spec_id)
    bundle = load_demo_model(spec, APP_ROOT)
    return spec, bundle


def model_options(task: str) -> list[dict]:
    specs = [spec for spec in load_model_specs(APP_ROOT) if spec["task"] == task]
    return sorted(specs, key=lambda spec: (not bool(spec.get("recommended", False)), not spec_available(spec), spec["name"]))


def spec_path_exists(spec: dict, key: str) -> bool:
    value = spec.get(key)
    if not value:
        return False
    path = Path(value)
    if not path.is_absolute():
        path = (APP_ROOT / path).resolve()
    return path.exists()


def spec_available(spec: dict) -> bool:
    required = ["checkpoint", "config_path"]
    required.append("label_map" if spec["task"] == "islr" else "vocab")
    return all(spec_path_exists(spec, key) for key in required)


def render_prediction(pred: Prediction) -> None:
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Prediction", pred.text or "(blank)")
    col_b.metric("Inference", f"{pred.elapsed_ms:.1f} ms")
    if pred.confidence is not None:
        col_c.metric("Confidence", f"{pred.confidence * 100:.1f}%")
    if pred.topk:
        st.dataframe(
            [{"rank": idx + 1, "gloss": label, "confidence": f"{score * 100:.2f}%"} for idx, (label, score) in enumerate(pred.topk)],
            hide_index=True,
            use_container_width=True,
        )


def format_model_label(spec: dict) -> str:
    status = "ready" if spec_available(spec) else "missing"
    badge = "recommended" if spec.get("recommended") else status
    return f"{spec['name']} ({badge})"


def process_uploaded_video(uploaded_file, bundle, decode_method: str) -> None:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=OUTPUT_DIR) as tmp:
        tmp.write(uploaded_file.getbuffer())
        video_path = Path(tmp.name)

    st.video(uploaded_file.getvalue())
    with st.spinner("Extracting skeleton and running inference..."):
        result = extract_video_skeleton(video_path, trim_active=True)
        pred = predict_skeleton(bundle, result.skeleton, decode_method=decode_method)

    stats = st.columns(5)
    stats[0].metric("Frames", result.frames)
    stats[1].metric("Source FPS", f"{result.fps:.2f}")
    stats[2].metric("Hand rate", f"{result.hand_rate:.1f}%")
    stats[3].metric("Pose rate", f"{result.pose_rate:.1f}%")
    stats[4].metric("Extraction", f"{result.elapsed_sec:.2f}s")
    render_prediction(pred)


class DemoVideoProcessor(VideoProcessorBase):
    def __init__(self, bundle, decode_method: str, task_label: str):
        self.bundle = bundle
        self.decode_method = decode_method
        self.task_label = task_label
        self.segmenter = LiveSegmenter(pre_roll=8, min_active_frames=12, idle_frames_to_close=14, max_segment_frames=210)
        self.fps_meter = FpsMeter()
        self.prev_pose = None
        self.prev_lh = None
        self.prev_rh = None
        self.latest_prediction = ""
        self.latest_ms = 0.0
        self.latest_confidence = 0.0
        self.last_error = ""

        import mediapipe as mp

        self.holistic = mp.solutions.holistic.Holistic()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        kp, pose_detected, left_detected, right_detected = extract_frame(
            image,
            self.holistic,
            self.prev_pose,
            self.prev_lh,
            self.prev_rh,
        )
        if pose_detected:
            self.prev_pose = kp[0:33, :2].copy()
        if left_detected:
            self.prev_lh = kp[33:54, :2].copy()
        if right_detected:
            self.prev_rh = kp[54:75, :2].copy()

        has_hand = hand_present(kp)
        segment = self.segmenter.update(kp, has_hand)
        if segment is not None:
            try:
                pred = predict_skeleton(self.bundle, segment, decode_method=self.decode_method)
                self.latest_prediction = pred.text
                self.latest_ms = pred.elapsed_ms
                self.latest_confidence = pred.confidence or 0.0
                self.last_error = ""
            except Exception as exc:  # pragma: no cover - UI path
                self.last_error = str(exc)

        fps = self.fps_meter.tick()
        state = "recording" if self.segmenter.recording else "waiting for hands"
        lines = [
            f"{self.task_label} | Display FPS {fps:05.2f} | {state}",
            f"Prediction: {self.latest_prediction or '(none yet)'}",
            f"Inference: {self.latest_ms:.1f} ms | Confidence {self.latest_confidence * 100:.1f}%",
        ]
        if self.last_error:
            lines.append(f"Error: {self.last_error[:70]}")
        image = draw_status(image, lines, has_hand)
        return av.VideoFrame.from_ndarray(image, format="bgr24")


def render_camera(bundle, decode_method: str, task_label: str, stream_key: str) -> None:
    if not WEBRTC_AVAILABLE:
        st.error("Camera mode needs streamlit-webrtc and av. Install requirements.txt, then rerun the app.")
        return
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_streamer(
        key=f"viesl-{task_label}-{decode_method}-{stream_key}",
        video_processor_factory=lambda: DemoVideoProcessor(bundle, decode_method, task_label),
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        desired_playing_state=True,
        async_processing=True,
    )


def main() -> None:
    st.title("VieSL Sign Language Demo")

    with st.sidebar:
        mode_label = st.radio("Mode", ["ISLR", "CSLR"], horizontal=True)
        task = "islr" if mode_label == "ISLR" else "cslr"
        specs = model_options(task)
        labels = [format_model_label(spec) for spec in specs]
        selected_label = st.selectbox("Model", labels)
        spec = specs[labels.index(selected_label)]
        bundle = None
        if spec_available(spec):
            spec, bundle = cached_model(spec["id"])
        st.caption(spec.get("metric_note", ""))
        source = st.radio("Input", ["Upload video", "Camera"], horizontal=False)
        decode_method = "greedy"
        if task == "cslr":
            decode_method = st.radio("Decode", ["greedy", "beam"], horizontal=True)

    if bundle is None:
        st.warning("Model artifact is not available locally yet. Export the Kaggle result folder into codex/model, then refresh this app.")
        st.json({key: spec.get(key) for key in ["checkpoint", "config_path", "label_map", "vocab"] if spec.get(key)})
        return

    device = bundle.device or torch.device("cpu")
    status_cols = st.columns(4)
    status_cols[0].metric("Task", mode_label)
    status_cols[1].metric("Device", str(device))
    status_cols[2].metric("Model", spec["name"])
    status_cols[3].metric("Decode", decode_method if task == "cslr" else "softmax")

    if source == "Upload video":
        uploaded = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv", "webm", "m4v"])
        if uploaded is not None:
            process_uploaded_video(uploaded, bundle, decode_method)
    else:
        render_camera(bundle, decode_method, mode_label, spec["id"])


if __name__ == "__main__":
    main()
