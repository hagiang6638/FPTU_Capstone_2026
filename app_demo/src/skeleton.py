from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover
    savgol_filter = None


NUM_KEYPOINTS = 86
FACE_INDICES = [70, 105, 336, 334, 1, 13, 14, 61, 291, 152, 10]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass
class ExtractionResult:
    skeleton: np.ndarray
    hand_flags: np.ndarray
    fps: float
    frames: int
    width: int
    height: int
    elapsed_sec: float
    hand_rate: float
    pose_rate: float


def smooth_savgol(sk: np.ndarray, window: int = 5, poly: int = 2) -> np.ndarray:
    if savgol_filter is None:
        return sk.astype(np.float32)
    total_frames = sk.shape[0]
    if total_frames < window:
        if total_frames > poly:
            window = total_frames if total_frames % 2 else total_frames - 1
        else:
            return sk.astype(np.float32)
    return savgol_filter(sk, window_length=window, polyorder=poly, axis=0, mode="nearest").astype(np.float32)


def normalize_zen(kp: np.ndarray) -> np.ndarray:
    out = kp.copy()
    if out.shape[0] == 0:
        return out.astype(np.float32)
    left_shoulder = out[:, 11, :2]
    right_shoulder = out[:, 12, :2]
    centers = (left_shoulder + right_shoulder) / 2
    dists = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
    thresh = max(1e-3, float(np.percentile(dists, 10)))
    valid = dists > thresh
    if valid.sum() > 0:
        anchor = np.median(centers[valid], axis=0)
        scale = float(np.median(dists[valid]))
    else:
        anchor = np.zeros(2, dtype=np.float32)
        scale = 1.0
    if scale < 1e-4:
        scale = 1.0
    out[:, :, :2] = (out[:, :, :2] - anchor) / scale
    out[:, :, :2] = smooth_savgol(out[:, :, :2])
    return out.astype(np.float32)


def extract_frame(
    frame: np.ndarray,
    holistic: Any,
    prev_pose: np.ndarray | None = None,
    prev_lh: np.ndarray | None = None,
    prev_rh: np.ndarray | None = None,
) -> tuple[np.ndarray, bool, bool, bool]:
    kp = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    height, width, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    pose_detected = False
    if results.pose_landmarks:
        pose_detected = True
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            kp[idx] = [lm.x * width, lm.y * height, lm.visibility]
    elif prev_pose is not None:
        kp[0:33, :2] = prev_pose

    left_detected = False
    if results.left_hand_landmarks:
        left_detected = True
        for idx, lm in enumerate(results.left_hand_landmarks.landmark):
            kp[33 + idx] = [lm.x * width, lm.y * height, 1.0]
    elif prev_lh is not None:
        kp[33:54, :2] = prev_lh
    elif pose_detected:
        kp[33:54, :2] = kp[15, :2]

    right_detected = False
    if results.right_hand_landmarks:
        right_detected = True
        for idx, lm in enumerate(results.right_hand_landmarks.landmark):
            kp[54 + idx] = [lm.x * width, lm.y * height, 1.0]
    elif prev_rh is not None:
        kp[54:75, :2] = prev_rh
    elif pose_detected:
        kp[54:75, :2] = kp[16, :2]

    if results.face_landmarks:
        for idx, face_idx in enumerate(FACE_INDICES):
            lm = results.face_landmarks.landmark[face_idx]
            kp[75 + idx] = [lm.x * width, lm.y * height, 1.0]

    return kp, pose_detected, left_detected, right_detected


def hand_present(kp: np.ndarray) -> bool:
    left = np.mean(kp[33:54, 2] > 0.1) > 0.25
    right = np.mean(kp[54:75, 2] > 0.1) > 0.25
    return bool(left or right)


def trim_to_active_segment(kp: np.ndarray, hand_flags: np.ndarray, pre_roll: int = 4, post_roll: int = 8) -> np.ndarray:
    if kp.shape[0] == 0 or not hand_flags.any():
        return kp
    idx = np.where(hand_flags)[0]
    start = max(0, int(idx[0]) - pre_roll)
    end = min(kp.shape[0], int(idx[-1]) + post_roll + 1)
    return kp[start:end]


def extract_video_skeleton(video_path: Path, trim_active: bool = True, max_frames: int | None = None) -> ExtractionResult:
    started = time.time()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    keypoints: list[np.ndarray] = []
    hand_flags: list[bool] = []
    pose_flags: list[bool] = []
    prev_pose = None
    prev_lh = None
    prev_rh = None

    with mp.solutions.holistic.Holistic() as holistic:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            kp, pose_detected, left_detected, right_detected = extract_frame(frame, holistic, prev_pose, prev_lh, prev_rh)
            if pose_detected:
                prev_pose = kp[0:33, :2].copy()
            if left_detected:
                prev_lh = kp[33:54, :2].copy()
            if right_detected:
                prev_rh = kp[54:75, :2].copy()
            keypoints.append(kp)
            hand_flags.append(left_detected or right_detected)
            pose_flags.append(pose_detected)
            if max_frames is not None and len(keypoints) >= max_frames:
                break

    capture.release()
    if not keypoints:
        raise RuntimeError(f"No frames were read from video: {video_path}")

    raw = np.stack(keypoints, axis=0).astype(np.float32)
    flags = np.array(hand_flags, dtype=bool)
    pose = np.array(pose_flags, dtype=bool)
    if trim_active:
        raw = trim_to_active_segment(raw, flags)
    normalized = normalize_zen(raw)
    return ExtractionResult(
        skeleton=normalized,
        hand_flags=flags,
        fps=fps,
        frames=int(normalized.shape[0]),
        width=width,
        height=height,
        elapsed_sec=round(time.time() - started, 3),
        hand_rate=float(flags.mean() * 100.0),
        pose_rate=float(pose.mean() * 100.0),
    )


class LiveSegmenter:
    def __init__(self, pre_roll: int = 8, min_active_frames: int = 10, idle_frames_to_close: int = 14, max_segment_frames: int = 180):
        self.pre_roll = pre_roll
        self.min_active_frames = min_active_frames
        self.idle_frames_to_close = idle_frames_to_close
        self.max_segment_frames = max_segment_frames
        self.preroll = deque(maxlen=pre_roll)
        self.active: list[np.ndarray] = []
        self.idle_count = 0
        self.recording = False

    def reset(self) -> None:
        self.preroll.clear()
        self.active = []
        self.idle_count = 0
        self.recording = False

    def update(self, kp: np.ndarray, has_hand: bool) -> np.ndarray | None:
        if not self.recording:
            self.preroll.append(kp.copy())
            if has_hand:
                self.recording = True
                self.active = list(self.preroll)
                self.idle_count = 0
            return None

        self.active.append(kp.copy())
        if has_hand:
            self.idle_count = 0
        else:
            self.idle_count += 1

        too_long = len(self.active) >= self.max_segment_frames
        closed = self.idle_count >= self.idle_frames_to_close
        if not (closed or too_long):
            return None

        segment = np.stack(self.active, axis=0).astype(np.float32)
        self.reset()
        if segment.shape[0] < self.min_active_frames:
            return None
        return normalize_zen(segment)


class FpsMeter:
    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.last_time = time.time()
        self.fps = 0.0

    def tick(self) -> float:
        now = time.time()
        dt = max(1e-6, now - self.last_time)
        instant = 1.0 / dt
        self.fps = instant if self.fps == 0 else self.smoothing * self.fps + (1.0 - self.smoothing) * instant
        self.last_time = now
        return self.fps


def load_overlay_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


_FONT_MAIN = load_overlay_font(24)
_FONT_SMALL = load_overlay_font(21)


def draw_status(frame: np.ndarray, lines: list[str], has_hand: bool) -> np.ndarray:
    color = (70, 220, 90) if has_hand else (80, 180, 255)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 90 + 28 * max(0, len(lines) - 2)), (15, 20, 28), -1)
    frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for idx, text in enumerate(lines):
        draw.text(
            (16, 12 + idx * 28),
            text,
            font=_FONT_MAIN if idx == 0 else _FONT_SMALL,
            fill=color if idx == 0 else (245, 245, 245),
        )
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
