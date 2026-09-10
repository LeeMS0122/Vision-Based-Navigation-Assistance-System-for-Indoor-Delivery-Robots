from pathlib import Path
from typing import Dict, List, Tuple
import shutil
import subprocess

import cv2


def extract_sampled_frames(video_path: Path, frames_dir: Path, sample_fps: float = 5.0) -> Tuple[List[Dict], Dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        native_fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / native_fps if native_fps > 0 else 0.0

    sample_fps = max(0.5, float(sample_fps))
    sample_every_n = max(1, int(round(native_fps / sample_fps)))
    sampled_actual_fps = native_fps / sample_every_n

    frames_dir.mkdir(parents=True, exist_ok=True)

    sampled = []
    idx = 0
    sampled_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % sample_every_n == 0:
            frame_name = f"frame_{sampled_idx:06d}.jpg"
            frame_path = frames_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            timestamp_sec = idx / native_fps
            sampled.append(
                {
                    "sampled_index": sampled_idx,
                    "source_frame_index": idx,
                    "timestamp_sec": round(float(timestamp_sec), 3),
                    "frame_path": frame_path,
                }
            )
            sampled_idx += 1

        idx += 1

    cap.release()

    video_info = {
        "native_fps": float(native_fps),
        "sample_fps_requested": float(sample_fps),
        "sample_fps_actual": float(sampled_actual_fps),
        "sample_every_n": int(sample_every_n),
        "frame_count": int(frame_count),
        "sampled_frame_count": int(len(sampled)),
        "width": int(width),
        "height": int(height),
        "duration_sec": round(float(duration_sec), 3),
    }
    return sampled, video_info


def _transcode_to_h264_if_available(src_path: Path, dst_path: Path, fps: float) -> bool:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return False

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(src_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-movflags",
        "+faststart",
        "-r",
        str(max(0.5, float(fps))),
        str(dst_path),
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0 and dst_path.exists() and dst_path.stat().st_size > 0


def compose_video_from_frames(frame_paths: List[Path], output_path: Path, fps: float) -> None:
    if not frame_paths:
        raise RuntimeError("No frames to compose")

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")

    h, w = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_output = output_path.with_name(f"{output_path.stem}_tmp_mp4v.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_output), fourcc, max(0.5, float(fps)), (w, h))

    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        writer.write(img)

    writer.release()

    # Prefer H.264 for browser playback compatibility.
    if _transcode_to_h264_if_available(tmp_output, output_path, fps):
        try:
            tmp_output.unlink(missing_ok=True)
        except Exception:
            pass
        return

    # Fallback: keep OpenCV mp4v output if ffmpeg is unavailable.
    tmp_output.replace(output_path)
