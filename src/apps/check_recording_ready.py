"""Check whether the current shell is ready for UR5e recording."""

import argparse

from ur5e_control.data_collection.preflight import (
    build_lerobot_record_command,
    collect_recording_preflight,
    format_preflight_report,
    preflight_has_blockers,
)
from ur5e_control.runtime.config import load_runtime_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check UR5e recording readiness")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Hardware/runtime YAML config. Defaults to UR5E_HARDWARE_CONFIG "
            "or configs/hardware/ur5e_d455.yaml."
        ),
    )
    parser.add_argument("--gello-port", default=None)
    parser.add_argument("--pgc-port", default=None)
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--check-robot-tcp", action="store_true")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="LeRobot dataset repo id for the generated command",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="LeRobot local dataset root for the generated command",
    )
    parser.add_argument("--fps", type=int, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_runtime_config(args.config)
    gello_port = args.gello_port or config.agent.gello_port
    pgc_port = args.pgc_port or config.gripper.port
    robot_ip = args.robot_ip or config.robot.ip
    repo_id = args.repo_id or config.dataset.lerobot_repo_id
    root = args.root or config.dataset.lerobot_root
    fps = args.fps or config.dataset.lerobot_fps

    results = collect_recording_preflight(
        gello_port=gello_port,
        robot_ip=robot_ip,
        agent_name=config.agent.name,
        check_robot_tcp=args.check_robot_tcp,
        gripper_port=pgc_port,
        spacemouse_device_path=config.agent.spacemouse.device_path,
        realsense_ids=config.camera.realsense_ids,
    )
    print(format_preflight_report(results))

    print()
    print("LeRobot recording command:")
    print(
        build_lerobot_record_command(
            repo_id,
            root,
            fps,
            config_path=config.source_path,
        )
    )

    return 1 if preflight_has_blockers(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
