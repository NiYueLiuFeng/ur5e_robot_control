"""Entry point for UR5e teleoperation data collection."""

import argparse

from ur5e_control.data_collection.teleop.runner import main as run_teleop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record UR5e teleoperation data")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Hardware/runtime YAML config. Defaults to UR5E_HARDWARE_CONFIG "
            "or configs/hardware/ur5e_d455.yaml."
        ),
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    run_teleop(config_path=args.config)


if __name__ == "__main__":
    main()
