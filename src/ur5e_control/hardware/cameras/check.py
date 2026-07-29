"""Camera stream check utilities."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable, Mapping

VideoWriterFactory = Callable[[Path, float, tuple[int, int]], Any]
SleepFn = Callable[[float], None]


def record_camera_check(
    cameras: Mapping[str, Any],
    camera_name_id: Mapping[str, str],
    output_dir: str | Path,
    duration_s: float = 5.0,
    fps: float = 30.0,
    video_writer_factory: VideoWriterFactory | None = None,
    sleep: SleepFn = time.sleep,
) -> dict[str, Path]:
    """Record a short MP4 for each configured logical camera view."""

    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")

    missing_names = set(cameras).difference(camera_name_id)
    if missing_names:
        missing = ", ".join(sorted(missing_names))
        raise ValueError(f"camera_name_id is missing camera IDs: {missing}")

    writer_factory = video_writer_factory or _create_mp4_writer
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, int(round(duration_s * fps)))
    outputs: dict[str, Path] = {}
    writers: dict[str, Any] = {}
    frame_period_s = 1.0 / fps

    try:
        for _ in range(frame_count):
            frame_start = time.monotonic()
            for camera_id, camera in cameras.items():
                logical_name = camera_name_id[camera_id]
                if logical_name not in outputs:
                    outputs[logical_name] = output_path / f"{logical_name}.mp4"
                image, _depth = camera.read()
                if logical_name not in writers:
                    writers[logical_name] = writer_factory(
                        outputs[logical_name],
                        fps,
                        _frame_size(image),
                    )
                writers[logical_name].write(_to_bgr(image))
            sleep(max(0.0, frame_period_s - (time.monotonic() - frame_start)))
    finally:
        for writer in writers.values():
            writer.release()

    return outputs


def _create_mp4_writer(path: Path, fps: float, frame_size: tuple[int, int]):
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open MP4 writer: {path}")
    return writer


def _frame_size(image: Any) -> tuple[int, int]:
    try:
        height = int(image.shape[0])
        width = int(image.shape[1])
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "camera image must expose shape=(height, width, channels)"
        ) from exc
    return width, height


def _to_bgr(image: Any) -> Any:
    try:
        if len(image.shape) == 3 and image.shape[2] == 3:
            return image[:, :, ::-1]
    except (AttributeError, IndexError, TypeError):
        return image
    return image
