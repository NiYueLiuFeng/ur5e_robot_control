"""Record one policy episode as a side-by-side third-person + wrist video."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np


class EpisodeVideoRecorder:
    """Stream merged camera frames to `save_video/<model>/<time>/video_<n>.mp4`.

    Frames are written as they arrive, so a long episode does not grow in
    memory. `save()` keeps the file; `discard()` deletes the unfinished one.
    """

    def __init__(
        self,
        root: str | Path = "save_video",
        model: str = "json",
        fps: float = 15.0,
    ) -> None:
        self.root = Path(root)
        self.model = model
        self.fps = fps
        self._writer = None
        self._path: Path | None = None
        self._last_t = 0.0
        self._frames = 0
        self._failed = False

    @property
    def has_frames(self) -> bool:
        return self._frames > 0

    def add(self, obs: dict[str, Any]) -> None:
        """Append one merged frame, throttled to `fps`."""

        if self._failed:
            return
        now = time.monotonic()
        if self._frames and now - self._last_t < 1.0 / self.fps:
            return
        frame = merge_views(obs["third_left_top_rgb"], obs["wrist_rgb"])
        if self._writer is None:
            self._path = self._next_path()
            try:
                self._writer = _open_writer(self._path, frame, self.fps)
            except RuntimeError as exc:
                # 编码器不可用时只关掉录像，不能把机器人控制回路带崩。
                self._path.unlink(missing_ok=True)
                self._path = None
                self._failed = True
                print(f"WARNING: video recording disabled: {exc}")
                return
        # cv2 的编码器按 BGR 处理，相机给的是 RGB。
        self._writer.write(np.ascontiguousarray(frame[:, :, ::-1]))
        self._frames += 1
        self._last_t = now

    def save(self) -> Path | None:
        """Finish the current file and return its path (None if empty)."""

        path = self._path
        frames = self._frames
        self._close()
        if path is None or not frames:
            return None
        print(f"Saved episode video: {path}")
        return path

    def discard(self) -> None:
        """Drop the unfinished episode and delete its partial file."""

        path = self._path
        self._close()
        if path is not None:
            path.unlink(missing_ok=True)

    def _close(self) -> None:
        if self._writer is not None:
            self._writer.release()
        self._writer = None
        self._path = None
        self._frames = 0
        self._last_t = 0.0
        # 下一段 episode 重新尝试一次，临时故障（如磁盘满）可以恢复。
        self._failed = False

    def _next_path(self) -> Path:
        out_dir = self.root / self.model / time.strftime("%Y%m%d-%H:%M")
        out_dir.mkdir(parents=True, exist_ok=True)
        index = 1
        while (out_dir / f"video_{index}.mp4").exists():
            index += 1
        return out_dir / f"video_{index}.mp4"


def merge_views(third_person: Any, wrist: Any) -> np.ndarray:
    """Put the two RGB views side by side; the result stays RGB."""

    import cv2

    left = np.asarray(third_person, dtype=np.uint8)
    right = np.asarray(wrist, dtype=np.uint8)
    height = min(left.shape[0], right.shape[0])
    return cv2.hconcat(
        [_scale_to_height(left, height), _scale_to_height(right, height)]
    )


def _scale_to_height(image: np.ndarray, height: int) -> np.ndarray:
    import cv2

    if image.shape[0] == height:
        return image
    width = max(1, round(image.shape[1] * height / image.shape[0]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _open_writer(path: Path, frame: np.ndarray, fps: float) -> Any:
    import cv2

    # MPEG-4 的 timebase 分母上限是 65535，过大的帧率会让 ffmpeg 打不开编码器。
    fps = min(max(float(fps), 1.0), 240.0)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    return writer
