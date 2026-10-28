"""Repair LeRobot episode video metadata from the encoded frame PTS values."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class EpisodeTimestampSpan:
    """The frame count and existing timestamps for one encoded episode."""

    episode_index: int
    length: int
    from_timestamp: float
    to_timestamp: float


@dataclass(frozen=True)
class RebuiltTimestampSpan:
    """Timestamp metadata reconstructed from a contiguous frame PTS segment."""

    episode_index: int
    from_timestamp: float
    to_timestamp: float
    max_frame_error_s: float


@dataclass(frozen=True)
class VideoTimestampRepairReport:
    """Summary returned by :func:`repair_lerobot_video_timestamps`."""

    video_keys: tuple[str, ...]
    video_files: int
    episodes: int
    frames: int
    changed_episodes: int
    max_metadata_correction_s: float
    applied: bool


FramePtsReader = Callable[[Path], tuple[list[int], Fraction]]


def rebuild_episode_timestamps(
    frame_pts: Sequence[int],
    time_base: Fraction,
    episodes: Sequence[EpisodeTimestampSpan],
    fps: int,
    tolerance_s: float = 1e-4,
) -> list[RebuiltTimestampSpan]:
    """Map consecutive encoded frames back to episode timestamp ranges.

    Metadata alone can fix a constant offset introduced at a concatenation
    boundary. It cannot fix variable-rate or missing frames inside an episode,
    so every relative PTS is checked against the declared fixed frame rate.
    """

    if fps <= 0:
        raise ValueError("fps must be positive")
    if tolerance_s < 0:
        raise ValueError("tolerance_s must be non-negative")
    if time_base <= 0:
        raise ValueError("time_base must be positive")

    expected_frames = sum(episode.length for episode in episodes)
    if len(frame_pts) != expected_frames:
        raise ValueError(
            "video frame count does not match episode metadata: "
            f"decoded={len(frame_pts)}, expected={expected_frames}"
        )

    frame_period = Fraction(1, fps)
    cursor = 0
    rebuilt: list[RebuiltTimestampSpan] = []
    for episode in episodes:
        if episode.length <= 0:
            raise ValueError(
                f"episode {episode.episode_index} has invalid length {episode.length}"
            )
        segment = frame_pts[cursor : cursor + episode.length]
        if any(right <= left for left, right in zip(segment, segment[1:])):
            raise ValueError(
                f"episode {episode.episode_index} has non-increasing frame PTS values"
            )

        first_pts = segment[0]
        errors = [
            abs(float((pts - first_pts) * time_base - index * frame_period))
            for index, pts in enumerate(segment)
        ]
        max_error_s = max(errors, default=0.0)
        if max_error_s > tolerance_s:
            raise ValueError(
                f"episode {episode.episode_index} is not fixed {fps} FPS: "
                f"maximum internal PTS error is {max_error_s * 1000:.3f} ms "
                f"(tolerance {tolerance_s * 1000:.3f} ms); re-encode this video file"
            )

        from_timestamp = float(first_pts * time_base)
        to_timestamp = float(segment[-1] * time_base + frame_period)
        rebuilt.append(
            RebuiltTimestampSpan(
                episode_index=episode.episode_index,
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp,
                max_frame_error_s=max_error_s,
            )
        )
        cursor += episode.length

    return rebuilt


def read_video_frame_pts(video_path: Path) -> tuple[list[int], Fraction]:
    """Decode a video in display order and return its frame PTS and time base."""

    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is required to inspect encoded video timestamps"
        ) from exc

    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream found in {video_path}")
        stream = container.streams.video[0]
        time_base = Fraction(stream.time_base)
        frame_pts = []
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"frame without PTS found in {video_path}")
            frame_pts.append(int(frame.pts))
    return frame_pts, time_base


def repair_lerobot_video_timestamps(
    dataset_root: str | Path,
    *,
    video_keys: Iterable[str] | None = None,
    apply: bool = False,
    backup: bool = True,
    tolerance_s: float = 1e-4,
    frame_pts_reader: FramePtsReader = read_video_frame_pts,
) -> VideoTimestampRepairReport:
    """Audit or repair episode ``from_timestamp``/``to_timestamp`` columns.

    By default only features marked ``is_depth_map`` are processed. The repair
    is applied atomically per episode parquet file. When ``backup`` is enabled,
    an untouched ``.pre-timestamp-repair.bak`` copy is created first.
    """

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to repair LeRobot metadata") from exc

    root = Path(dataset_root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = int(info["fps"])
    selected_keys = tuple(video_keys or _depth_video_keys(info))
    if not selected_keys:
        raise ValueError("no depth video features found; pass video_keys explicitly")

    episode_paths = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(
            f"no episode parquet files under {root / 'meta/episodes'}"
        )

    tables = {path: pq.read_table(path) for path in episode_paths}
    updates: dict[Path, dict[str, list[Any]]] = {}
    for path, table in tables.items():
        updates[path] = {
            key: table[key].to_pylist()
            for key in table.column_names
            if any(
                key == f"videos/{video_key}/from_timestamp"
                or key == f"videos/{video_key}/to_timestamp"
                for video_key in selected_keys
            )
        }

    video_file_count = 0
    episode_count = 0
    frame_count = 0
    changed_episode_count = 0
    max_correction_s = 0.0

    for video_key in selected_keys:
        columns = _video_columns(video_key)
        records_by_file: dict[
            tuple[int, int], list[tuple[Path, int, EpisodeTimestampSpan]]
        ] = {}
        for path, table in tables.items():
            missing = set(columns.values()).difference(table.column_names)
            if missing:
                missing_text = ", ".join(sorted(missing))
                raise ValueError(
                    f"{path} is missing video metadata columns: {missing_text}"
                )
            for row_index in range(table.num_rows):
                episode = EpisodeTimestampSpan(
                    episode_index=int(table["episode_index"][row_index].as_py()),
                    length=int(table["length"][row_index].as_py()),
                    from_timestamp=float(table[columns["from"]][row_index].as_py()),
                    to_timestamp=float(table[columns["to"]][row_index].as_py()),
                )
                file_key = (
                    int(table[columns["chunk"]][row_index].as_py()),
                    int(table[columns["file"]][row_index].as_py()),
                )
                records_by_file.setdefault(file_key, []).append(
                    (path, row_index, episode)
                )

        for (chunk_index, file_index), records in sorted(records_by_file.items()):
            records.sort(key=lambda item: item[2].episode_index)
            video_path = root / info["video_path"].format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            if not video_path.exists():
                raise FileNotFoundError(video_path)
            pts, time_base = frame_pts_reader(video_path)
            spans = rebuild_episode_timestamps(
                pts,
                time_base,
                [record[2] for record in records],
                fps,
                tolerance_s=tolerance_s,
            )
            video_file_count += 1
            frame_count += len(pts)

            for (path, row_index, old), new in zip(records, spans, strict=True):
                correction = max(
                    abs(new.from_timestamp - old.from_timestamp),
                    abs(new.to_timestamp - old.to_timestamp),
                )
                max_correction_s = max(max_correction_s, correction)
                if correction > 1e-12:
                    changed_episode_count += 1
                updates[path][columns["from"]][row_index] = new.from_timestamp
                updates[path][columns["to"]][row_index] = new.to_timestamp
                episode_count += 1

    if apply:
        if backup:
            for path in tables:
                backup_path = _backup_path(path)
                if backup_path.exists():
                    raise FileExistsError(
                        f"backup already exists: {backup_path}; "
                        "move it or use backup=False"
                    )
            for path in tables:
                shutil.copy2(path, _backup_path(path))
        for path, table in tables.items():
            updated_table = table
            for column_name, values in updates[path].items():
                field_index = updated_table.schema.get_field_index(column_name)
                field = updated_table.schema.field(field_index)
                updated_table = updated_table.set_column(
                    field_index,
                    field,
                    pa.array(values, type=field.type),
                )
            _write_parquet_atomically(
                path,
                updated_table,
                pq=pq,
            )

    return VideoTimestampRepairReport(
        video_keys=selected_keys,
        video_files=video_file_count,
        episodes=episode_count,
        frames=frame_count,
        changed_episodes=changed_episode_count,
        max_metadata_correction_s=max_correction_s,
        applied=apply,
    )


def _depth_video_keys(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
        and (feature.get("info") or {}).get("is_depth_map") is True
    ]


def _video_columns(video_key: str) -> dict[str, str]:
    prefix = f"videos/{video_key}"
    return {
        "chunk": f"{prefix}/chunk_index",
        "file": f"{prefix}/file_index",
        "from": f"{prefix}/from_timestamp",
        "to": f"{prefix}/to_timestamp",
    }


def _write_parquet_atomically(path: Path, table: Any, *, pq: Any) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        pq.write_table(table, temporary_path)
        shutil.copymode(path, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _backup_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".pre-timestamp-repair.bak")
