"""Audit and repair LeRobot video timestamps using encoded frame PTS values."""

from __future__ import annotations

import argparse

from ur5e_control.data_collection.video_timestamps import (
    repair_lerobot_video_timestamps,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild LeRobot episode video timestamp metadata from actual frame PTS. "
            "The default mode is read-only."
        )
    )
    parser.add_argument("--root", required=True, help="LeRobot dataset root")
    parser.add_argument(
        "--video-key",
        action="append",
        dest="video_keys",
        help=(
            "Video feature to process; repeat for multiple keys. "
            "Defaults to depth videos."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write corrected metadata. Without this flag only an audit is performed.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .pre-timestamp-repair.bak files when applying.",
    )
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=0.1,
        help="Maximum allowed internal fixed-FPS error in milliseconds.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.tolerance_ms < 0:
        raise ValueError("--tolerance-ms must be non-negative")
    report = repair_lerobot_video_timestamps(
        args.root,
        video_keys=args.video_keys,
        apply=args.apply,
        backup=not args.no_backup,
        tolerance_s=args.tolerance_ms / 1000.0,
    )
    mode = "repaired" if report.applied else "audit only"
    print(f"Video timestamp result ({mode}):")
    print(f"- keys: {', '.join(report.video_keys)}")
    print(f"- video files: {report.video_files}")
    print(f"- episode/video entries checked: {report.episodes}")
    print(f"- decoded frame entries checked: {report.frames}")
    print(f"- episode/video entries needing correction: {report.changed_episodes}")
    print(
        "- maximum metadata correction: "
        f"{report.max_metadata_correction_s * 1000:.3f} ms"
    )
    if not args.apply and report.changed_episodes:
        print("Re-run with --apply after reviewing this audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
