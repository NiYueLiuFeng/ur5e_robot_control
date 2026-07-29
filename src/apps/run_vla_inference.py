"""Run UR5e control from an already-running CloudRobo-VLA server."""

from __future__ import annotations

import argparse
from dataclasses import replace

from ur5e_control.inference.controller import build_vla_inference_controller
from ur5e_control.runtime.config import RuntimeConfig, load_runtime_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Runtime YAML config. Defaults to configs/hardware/ur5e_d455.yaml.",
    )
    parser.add_argument(
        "--server-url",
        "--server",
        default=None,
        help="CloudRobo-VLA WebSocket URL, for example ws://127.0.0.1:8000.",
    )
    parser.add_argument("--api-key", default=None, help="Optional VLA server API key.")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Dataset/schema repo id sent in each CloudRobo-VLA request.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Optional language task override sent to the VLA server.",
    )
    parser.add_argument(
        "--sub-task",
        default=None,
        help="Optional sub_task field sent to the VLA server.",
    )
    parser.add_argument(
        "--action-space",
        choices=("abs", "absolute", "delta", "relative"),
        default=None,
        help="Interpret server actions as absolute joint targets or joint deltas.",
    )
    parser.add_argument(
        "--execute-horizon",
        type=int,
        default=None,
        help="Number of actions to execute from each returned action chunk.",
    )
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=None,
        help="Maximum arm joint movement per command in radians.",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=None,
        help="VLA WebSocket connection timeout in seconds.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after this many robot commands. Omit for continuous control.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not send the VLA server reset command before the first step.",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    config = _apply_cli_overrides(load_runtime_config(args.config), args)
    controller = build_vla_inference_controller(
        runtime_config=config,
        task_name=args.task,
        sub_task=args.sub_task,
    )
    try:
        print("Starting UR5e VLA inference control. Press Ctrl+C to stop.")
        steps = controller.run(init_pos=config.robot.reset_joints, max_steps=args.max_steps)
        print(f"Stopped after executing {steps} robot command(s).")
    except KeyboardInterrupt:
        print("Stopping UR5e VLA inference control.")
    finally:
        controller.close()


def _apply_cli_overrides(
    config: RuntimeConfig,
    args: argparse.Namespace,
) -> RuntimeConfig:
    vla = config.vla
    if args.server_url is not None:
        vla = replace(vla, server_url=args.server_url)
    if args.api_key is not None:
        vla = replace(vla, api_key=args.api_key)
    if args.repo_id is not None:
        vla = replace(vla, repo_id=args.repo_id)
    if args.action_space is not None:
        vla = replace(vla, action_space=args.action_space)
    if args.execute_horizon is not None:
        vla = replace(vla, execute_horizon=args.execute_horizon)
    if args.max_joint_delta is not None:
        vla = replace(vla, max_joint_delta=args.max_joint_delta)
    if args.request_timeout_s is not None:
        vla = replace(vla, request_timeout_s=args.request_timeout_s)
    if args.no_reset:
        vla = replace(vla, reset_on_start=False)
    return replace(config, vla=vla)


if __name__ == "__main__":
    main()
