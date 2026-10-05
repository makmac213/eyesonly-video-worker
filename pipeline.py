"""
EyesOnly video pipeline: turn a <=10 s selfie video into an eyes-only clip.

1. Find the eyes in every frame (OpenCV YuNet face detector, landmarks for both eyes).
2. Smooth the eye track so the frame doesn't jitter; fill short gaps.
3. Move a fixed-size crop (Band / Square / Portrait / One eye / Candid) with the eyes,
   always clamped to a safe zone: never below the eyes (no nose tip / mouth).
4. FAIL SAFE: any frame where the eyes can't be found is pixelated + darkened,
   and a video where the eyes are lost too often is rejected.
5. Encode H.264 MP4 (muted unless the member turned sound on) + a poster JPEG.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_detection_yunet_2023mar.onnx")

MAX_SECONDS = 10.5          # 10 s clips, small tolerance
DETECT_WIDTH = 480          # detection runs on a downscaled frame
MIN_SCORE = 0.7
MIN_VALID_FRACTION = 0.6    # reject if eyes are found in fewer frames than this
MAX_GAP_FILL = 6            # frames; longer gaps get pixelated
SMOOTH = 7                  # moving-average window (frames)

# Same geometry as the app (multiples of the distance between the eyes).
STYLES = {
    "band":     {"aspect": 2.6,   "width": 2.3},
    "square":   {"aspect": 1.0,   "width": 1.8},
    "portrait": {"aspect": 0.8,   "width": 1.7},
    "single":   {"aspect": 1.0,   "width": 0.95},
    "candid":   {"aspect": 4 / 3, "width": 2.4},
}
ABOVE, BELOW, SIDE = 2.4, 0.38, 1.7


class VideoRejected(Exception):
    """A friendly reason shown to the member."""


@dataclass
class Eyes:
    lx: float
    ly: float
    rx: float
    ry: float

    @property
    def d(self) -> float:
        return float(np.hypot(self.rx - self.lx, self.ry - self.ly))


def _detect(det, frame, scale) -> Eyes | None:
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    det.setInputSize((small.shape[1], small.shape[0]))
    _, faces = det.detect(small)
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda f: f[2] * f[3])
    if best[14] < MIN_SCORE:
        return None
    # YuNet landmarks: [4:6] right eye, [6:8] left eye (subject's), in small-frame coords.
    ax, ay, bx, by = best[4] / scale, best[5] / scale, best[6] / scale, best[7] / scale
    if ax > bx:
        ax, ay, bx, by = bx, by, ax, ay
    eyes = Eyes(ax, ay, bx, by)
    return eyes if eyes.d > 8 else None


def _fill_and_smooth(track: list[Eyes | None]) -> list[Eyes | None]:
    n = len(track)
    arr = np.full((n, 4), np.nan)
    for i, e in enumerate(track):
        if e is not None:
            arr[i] = (e.lx, e.ly, e.rx, e.ry)
    valid = ~np.isnan(arr[:, 0])
    # Fill short gaps by linear interpolation.
    idx = np.where(valid)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= MAX_GAP_FILL:
            for k in range(1, gap + 1):
                t = k / (gap + 1)
                arr[a + k] = arr[a] * (1 - t) + arr[b] * t
    valid = ~np.isnan(arr[:, 0])
    # Centered moving average over valid frames only.
    out: list[Eyes | None] = []
    half = SMOOTH // 2
    for i in range(n):
        if not valid[i]:
            out.append(None)
            continue
        lo, hi = max(0, i - half), min(n, i + half + 1)
        win = arr[lo:hi][valid[lo:hi]]
        m = win.mean(axis=0)
        out.append(Eyes(*m))
    return out


def _rect(style: str, s: Eyes, raw: Eyes, w: float, h: float, W: int, H: int):
    """Crop rect for one frame. `s` = smoothed eyes (position), `raw` = this frame's eyes (safety)."""
    d = raw.d
    mx, my = (s.lx + s.rx) / 2, (s.ly + s.ry) / 2
    safe_bottom = max(raw.ly, raw.ry) + BELOW * d        # never below the eyes
    if style == "band":
        x, y = mx - w / 2, my - 0.58 * h
    elif style == "single":
        x, y = s.lx - w / 2, s.ly - 0.52 * h
    else:
        x, y = mx - w / 2, safe_bottom - h
        if style == "candid":
            x += 0.12 * w
    # Safe zone (sides / above / below), then the frame edges.
    rmx = (raw.lx + raw.rx) / 2
    x = min(max(x, rmx - SIDE * d), rmx + SIDE * d - w)
    y = max(y, my - ABOVE * d)
    y = min(y, safe_bottom - h)
    x = min(max(x, 0), W - w)
    y = max(y, 0)
    if y + h > safe_bottom + 0.5 or y + h > H:
        return None  # can't fit without showing too much -> treat as lost
    return int(round(x)), int(round(y))


def _pixelate(img):
    h, w = img.shape[:2]
    tiny = cv2.resize(img, (8, max(1, round(8 * h / w))), interpolation=cv2.INTER_AREA)
    return (cv2.resize(tiny, (w, h), interpolation=cv2.INTER_NEAREST) * 0.45).astype(np.uint8)


def _has_audio(path: str) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return bool(r.stdout.strip())


def process(src: str, out_mp4: str, out_jpg: str, style: str = "band", sound: bool = False) -> dict:
    if style not in STYLES:
        style = "band"
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise VideoRejected("We couldn't read that video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps > 120 or fps < 1:
        fps = 30.0
    det = cv2.FaceDetectorYN.create(MODEL, "", (320, 320), MIN_SCORE, 0.3, 5)

    # Pass 1: detect eyes in every frame.
    track: list[Eyes | None] = []
    W = H = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        if len(track) / fps > MAX_SECONDS:
            cap.release()
            raise VideoRejected("Videos can be up to 10 seconds.")
        scale = min(1.0, DETECT_WIDTH / W)
        track.append(_detect(det, frame, scale))
    cap.release()
    n = len(track)
    if n == 0:
        raise VideoRejected("We couldn't read that video.")
    found = sum(e is not None for e in track)
    if found / n < MIN_VALID_FRACTION:
        raise VideoRejected("We lost track of your eyes too often. Keep your face in view and try again.")

    filled = _fill_and_smooth(track)
    ds = sorted(e.d for e in filled if e is not None)
    d_med = ds[len(ds) // 2]
    st = STYLES[style]
    w = min(st["width"] * d_med, W)
    h = w / st["aspect"]
    if h > (ABOVE + BELOW) * d_med or h > H:
        h = min((ABOVE + BELOW) * d_med, H)
        w = h * st["aspect"]
    w, h = int(round(w)), int(round(h))

    out_w = 720 if st["aspect"] >= 1 else 576
    out_h = int(round(out_w / st["aspect"] / 2) * 2)

    # Pass 2: render.
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", f"{fps:.3f}", "-i", "-"]
    keep_audio = sound and _has_audio(src)
    if keep_audio:
        cmd += ["-i", src, "-map", "0:v", "-map", "1:a:0", "-c:a", "aac", "-b:a", "96k", "-shortest"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-t", "10", out_mp4]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    cap = cv2.VideoCapture(src)
    last_xy = None
    poster_saved = False
    pixelated = 0
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        e, raw = filled[i], track[i] or filled[i]
        xy = _rect(style, e, raw, w, h, W, H) if e is not None else None
        if xy is not None:
            x, y = xy
            crop = cv2.resize(frame[y:y + h, x:x + w], (out_w, out_h), interpolation=cv2.INTER_AREA)
            last_xy = xy
            if not poster_saved:
                cv2.imwrite(out_jpg, crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                poster_saved = True
        else:
            pixelated += 1
            x, y = last_xy or (max(0, (W - w) // 2), max(0, (H - h) // 2))
            crop = _pixelate(cv2.resize(frame[y:y + h, x:x + w], (out_w, out_h), interpolation=cv2.INTER_AREA))
        enc.stdin.write(np.ascontiguousarray(crop).tobytes())
    cap.release()
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("encoding failed")
    if not poster_saved:
        raise VideoRejected("We couldn't find your eyes in this video.")
    return {
        "aspect": round(out_w / out_h, 3),
        "frames": n,
        "pixelated_frames": pixelated,
        "duration": round(n / fps, 2),
        "sound": keep_audio,
    }
