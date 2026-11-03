"""Record one policy episode as a side-by-side third-person + wrist video."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

# 单次补帧最多覆盖的时长，避免服务端长时间无响应时写出海量重复帧。
MAX_GAP_S = 120.0
# 缓冲区占用达到这个值时提醒一次：视频是整段编码的，长 episode 很吃内存。
MEMORY_WARNING_BYTES = 1024**3


class EpisodeVideoRecorder:
    """Buffer one episode in memory and write `save_video/<model>/<time>/video_<n>.mp4`.

    Every `add()` merges the two camera views and appends the frame with its
    capture time. Nothing is encoded until `save()` (called when the operator
    answers `y` at the N/T prompt, or when the loop exits), so the control loop
    never waits on the encoder between actions. The trade-off is memory: a
    1280x480 BGR frame is
    ~1.8 MB, i.e. about 1.7 GB per minute at 15 fps.
    """

    def __init__(
        self,
        root: str | Path = "save_video",
        model: str = "json",
        fps: float = 15.0,
    ) -> None:
        self.root = Path(root)
        self.model = model
        # 与 _open_writer 的时基限制一致，补帧间隔才能和实际编码帧率对上。
        self.fps = min(max(float(fps), 1.0), 240.0)
        self._frames: list[tuple[np.ndarray, float]] = []
        self._bytes = 0
        self._stamp: str | None = None
        self._failed = False
        self._warned_memory = False

    @property
    def has_frames(self) -> bool:
        return bool(self._frames)

    def add(self, obs: dict[str, Any]) -> None:
        """Append one merged frame, throttled to `fps`."""

        if self._failed:
            return
        now = time.monotonic()
        if self._frames and now - self._frames[-1][1] < 1.0 / self.fps:
            return
        try:
            frame = merge_views(obs["third_left_top_rgb"], obs["wrist_rgb"])
        except ImportError as exc:
            # 没装 opencv 时只关掉录像，不打断控制回路。
            self._failed = True
            print(f"WARNING: video recording disabled: {exc}")
            return
        # 每一步的观测都留在内存里，按 Y 时再整段编码成视频。
        # 存 BGR 是编码器需要的格式，也省掉保存时再转换一次。
        bgr = np.ascontiguousarray(frame[:, :, ::-1])
        if self._stamp is None:
            # 目录用第一帧的时间命名，而不是保存时的时间。
            self._stamp = time.strftime("%Y%m%d-%H:%M")
        self._frames.append((bgr, now))
        self._bytes += bgr.nbytes
        self._warn_if_large()

    def save(self, end: float | None = None) -> Path | None:
        """Encode the buffered episode and return its path (None if empty)."""

        frames = self._frames
        if not frames:
            self._reset()
            return None
        end_t = time.monotonic() if end is None else end
        span = max(0.0, end_t - frames[0][1])
        print(f"Encoding episode video ({len(frames)} steps, {span:.1f} s)...")
        path = self._next_path()
        try:
            writer = _open_writer(path, frames[0][0], self.fps)
        except (ImportError, RuntimeError) as exc:
            # 编码器不可用时只关掉录像，不能把机器人控制回路带崩。
            path.unlink(missing_ok=True)
            self._reset()
            print(f"WARNING: video recording disabled: {exc}")
            return None
        try:
            _write_timeline(writer, frames, end_t, self.fps)
        finally:
            writer.release()
        self._reset()
        print(f"Saved episode video: {path}")
        return path

    def discard(self) -> None:
        """Drop the buffered episode without writing anything."""

        self._reset()

    def _warn_if_large(self) -> None:
        if self._warned_memory or self._bytes < MEMORY_WARNING_BYTES:
            return
        self._warned_memory = True
        print(
            "WARNING: episode video buffer is "
            f"{self._bytes / 1024**3:.1f} GiB; a long episode can exhaust memory. "
            "Press T to save, or Ctrl+C to stop."
        )

    def _reset(self) -> None:
        self._frames = []
        self._bytes = 0
        self._stamp = None
        # 下一段 episode 重新尝试一次，临时故障（如磁盘满）可以恢复。
        self._failed = False

    def _next_path(self) -> Path:
        stamp = self._stamp or time.strftime("%Y%m%d-%H:%M")
        out_dir = self.root / self.model / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        index = 1
        while (out_dir / f"video_{index}.mp4").exists():
            index += 1
        return out_dir / f"video_{index}.mp4"


def _write_timeline(
    writer: Any,
    frames: list[tuple[np.ndarray, float]],
    end: float,
    fps: float,
) -> None:
    """Write the buffered frames, repeating the previous one to fill real gaps."""

    previous, last_t = frames[0]
    writer.write(previous)
    for frame, now in frames[1:]:
        # 两次 action 之间可能隔着推理等待，只写一帧会让视频比现场快很多。
        gap = max(1, int(round(min(now - last_t, MAX_GAP_S) * fps)))
        for _ in range(gap - 1):
            writer.write(previous)
        writer.write(frame)
        previous, last_t = frame, now
    # 最后一条 action 到 episode 结束（按 Y / Ctrl+C）之间的等待也要补上。
    tail = int(round(min(max(end - last_t, 0.0), MAX_GAP_S) * fps)) - 1
    for _ in range(max(0, tail)):
        writer.write(previous)


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
