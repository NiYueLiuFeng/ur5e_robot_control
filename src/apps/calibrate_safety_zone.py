"""Calibrate UR5e joint safety limits through teleoperation."""

import argparse

from ur5e_control.runtime.safety_calibration import main as run_calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate UR5e joint safety limits")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Hardware/runtime YAML config. Defaults to UR5E_HARDWARE_CONFIG "
            "or configs/hardware/ur5e_d455.yaml."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output YAML for calibrated limits. Defaults to safety.joint_limits_file "
            "from the runtime config."
        ),
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Safety margin in radians applied inside the calibrated min/max range.",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    run_calibration(
        config_path=args.config,
        output_path=args.output,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()
