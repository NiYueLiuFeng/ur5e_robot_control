"""Record one policy episode as a side-by-side third-person + wrist video."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

# 单次补帧最多覆盖的时长，避免服务端长时间无响应时写出海量重复帧。
MAX_GAP_S = 120.0


class EpisodeVideoRecorder:
    """Stream episodes to `save_video/<model>/<session start>/video_<n>.mp4`.

    The directory is named once when the recorder is built (the ws-control
    session start) and every episode of that process writes into it, numbered
    `video_1.mp4`, `video_2.mp4`, ... The first `add()` of an episode creates
    the directory, opens the encoder and writes that frame, so the file exists
    from the beginning of the episode. Every later `add()` writes the merged
    frame straight to disk (repeating the previous one to fill the wall-clock
    gap). `save()` only closes the file and keeps it; `discard()` closes and
    deletes it. A crash or Ctrl+C therefore loses at most the tail instead of
    the whole episode.

    Only the previous frame is held in memory so gaps can be filled, which
    keeps a long episode at constant memory instead of ~1.8 MB per frame
    (~1.7 GB per minute at 15 fps). The encoder runs between actions; writing
    one 1280x480 frame at 15 fps is negligible compared with the policy
    request that precedes it.
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
        # 会话目录在进程启动时定下来：多个 episode 都写进同一个目录，
        # 而不是按各自开始的时间散落到不同分钟的目录里。
        self._stamp = time.strftime("%Y%m%d-%H:%M")
        self._writer: Any | None = None
        self._path: Path | None = None
        self._frames = 0
        self._start_t: float | None = None
        self._last_t: float | None = None
        # 只保留上一帧：两次 action 之间隔着的推理等待用它补帧。
        self._previous: np.ndarray | None = None
        self._size: tuple[int, int] | None = None
        self._failed = False
        self._warned_size = False

    @property
    def has_frames(self) -> bool:
        return self._frames > 0

    def add(self, obs: dict[str, Any]) -> None:
        """Append one merged frame, throttled to `fps`, written immediately."""

        if self._failed:
            return
        now = time.monotonic()
        if self._last_t is not None and now - self._last_t < 1.0 / self.fps:
            return
        try:
            frame = merge_views(obs["third_left_top_rgb"], obs["wrist_rgb"])
        except ImportError as exc:
            # 没装 opencv 时只关掉录像，不打断控制回路。
            self._fail(str(exc))
            return
        # cv2 的编码器按 BGR 处理，相机给的是 RGB。
        bgr = np.ascontiguousarray(frame[:, :, ::-1])
        if self._writer is None:
            self._open(bgr, now)
            return
        bgr = self._fit(bgr)
        # 两次 action 之间可能隔着推理等待，只写一帧会让视频比现场快很多。
        gap = max(1, int(round(min(now - self._last_t, MAX_GAP_S) * self.fps)))
        for _ in range(gap - 1):
            self._writer.write(self._previous)
        self._writer.write(bgr)
        self._previous = bgr
        self._last_t = now
        self._frames += gap

    def save(self, end: float | None = None, *, partial: bool = False) -> Path | None:
        """Close the episode's file and return its path (None if empty).

        ``partial`` marks a recording that was not explicitly confirmed by the
        operator (Ctrl+C / ``--max-steps`` / error at exit, or an unanswered /
        unrecognized N/T prompt): the file is renamed to
        ``video_<n>_partial.mp4`` so it is not mistaken for an episode the
        operator chose to keep.
        """

        writer = self._writer
        path = self._path
        if writer is None or path is None:
            self._reset()
            return None
        # 先摘下 writer：编码尾帧时按 Ctrl+C 不会让退出逻辑重复写同一个文件。
        self._writer = None
        end_t = time.monotonic() if end is None else end
        try:
            self._write_tail(writer, end_t)
        finally:
            writer.release()
        frames = self._frames
        span = max(0.0, end_t - (self._start_t or end_t))
        self._reset()
        if partial:
            path = _mark_partial(path)
            print(
                f"Saved partial episode video (not explicitly confirmed): {path} "
                f"({frames} frames, {span:.1f} s)"
            )
        else:
            print(f"Saved episode video: {path} ({frames} frames, {span:.1f} s)")
        return path

    def discard(self) -> None:
        """Close and delete the episode's partially written file."""

        writer = self._writer
        path = self._path
        self._writer = None
        if writer is not None:
            writer.release()
        self._reset()
        if path is not None:
            path.unlink(missing_ok=True)
            print(f"Discarded episode video: {path}")

    def _open(self, frame: np.ndarray, now: float) -> None:
        """Create the file and write the episode's first frame."""

        self._size = (frame.shape[1], frame.shape[0])
        path = self._next_path()
        try:
            writer = _open_writer(path, frame, self.fps)
        except (ImportError, RuntimeError) as exc:
            # 编码器不可用时只关掉录像，不能把机器人控制回路带崩。
            path.unlink(missing_ok=True)
            self._fail(str(exc))
            return
        # 目录和文件在 episode 开始时就存在，Ctrl+C 最多丢掉最后几帧。
        writer.write(frame)
        self._path = path
        self._writer = writer
        self._previous = frame
        self._start_t = now
        self._last_t = now
        self._frames = 1
        print(f"Recording episode video: {path}")

    def _write_tail(self, writer: Any, end: float) -> None:
        """Hold the last frame until the episode actually ends."""

        if self._previous is None or self._last_t is None:
            return
        # 最后一条 action 到 episode 结束（按 Y / Ctrl+C）之间的等待也要补上。
        tail = int(round(min(max(end - self._last_t, 0.0), MAX_GAP_S) * self.fps)) - 1
        for _ in range(max(0, tail)):
            writer.write(self._previous)

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        """Keep every frame at the size the encoder was opened with.

        OpenCV silently drops frames whose size differs from the writer's, so
        a mid-episode resolution change would otherwise truncate the video.
        """

        if self._size is None or frame.shape[1::-1] == self._size:
            return frame
        import cv2

        if not self._warned_size:
            self._warned_size = True
            height, width = frame.shape[:2]
            print(
                "WARNING: camera frame size changed "
                f"({width}x{height} -> {self._size[0]}x{self._size[1]}); "
                "resizing to keep the video playable."
            )
        return cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)

    def _fail(self, message: str) -> None:
        """Disable recording until the next episode, without raising."""

        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:  # 编码器已经坏了，释放失败也不能影响控制回路
                pass
        path = self._path
        self._reset()
        if path is not None:
            path.unlink(missing_ok=True)
        self._failed = True
        print(f"WARNING: video recording disabled: {message}")

    def _reset(self) -> None:
        """Forget the current episode; the caller must close the writer first."""

        self._writer = None
        self._path = None
        self._frames = 0
        self._start_t = None
        self._last_t = None
        self._previous = None
        self._size = None
        # 注意：不清 _stamp —— 会话目录在整个 ws-control 进程内保持不变。
        self._warned_size = False
        # 下一段 episode 重新尝试一次，临时故障（如磁盘满）可以恢复。
        self._failed = False

    def _next_path(self) -> Path:
        out_dir = self.root / self.model / self._stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        index = 1
        # 退出时保存的文件已经改名成 video_<n>_partial.mp4，同样算占用这个序号，
        # 否则下一段 episode 会重新用 video_<n>.mp4，看起来像两个 episode 重号。
        while _episode_exists(out_dir, index):
            index += 1
        return out_dir / f"video_{index}.mp4"


def _episode_exists(out_dir: Path, index: int) -> bool:
    return (out_dir / f"video_{index}.mp4").exists() or (
        out_dir / f"video_{index}_partial.mp4"
    ).exists()


def _mark_partial(path: Path) -> Path:
    """Rename `video_<n>.mp4` to `video_<n>_partial.mp4`; fall back on failure."""

    target = path.with_name(f"{path.stem}_partial{path.suffix}")
    try:
        path.rename(target)
    except OSError as exc:
        print(f"WARNING: could not rename {path} to {target}: {exc}")
        return path
    return target


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
