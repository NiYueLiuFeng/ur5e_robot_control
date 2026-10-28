"""Run UR5e inference through the FlexPi HTTP server.

This is a separate entrypoint; the existing CloudRobo WebSocket inference code
is left unchanged.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ur5e_control.data_collection.tasks import load_task_file
from ur5e_control.inference.flexpi_controller import (
    build_flexpi_inference_controller,
)
from ur5e_control.inference.flexpi_http import FLEXPI_MODES, FlexPiHTTPClient
from ur5e_control.runtime.config import load_runtime_config, resolve_repo_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Runtime hardware YAML.")
    parser.add_argument(
        "--server-url",
        default=None,
        help=(
            "FlexPi HTTP base URL. Defaults to vla.server_url from the runtime "
            "YAML, with ws(s) converted to http(s)."
        ),
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help="Override vla.request_timeout_s from the runtime YAML.",
    )
    parser.add_argument("--mode", choices=FLEXPI_MODES, default=FLEXPI_MODES[-1])
    parser.add_argument("--inference-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--task",
        default=None,
        help="Language instruction override; defaults to the configured task file.",
    )
    parser.add_argument(
        "--expected-profile",
        default=None,
        help="Fail unless /health reports this profile, e.g. ur5e/block.",
    )
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Check the FlexPi service without connecting cameras or robot.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually command the robot; default is one inference preview only.",
    )
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument(
        "--execute-horizon",
        type=int,
        default=None,
        help="Override vla.execute_horizon from the runtime YAML.",
    )
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=None,
        help="Override vla.max_joint_delta from the runtime YAML.",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=None,
        help="Override vla.threadhold from the runtime YAML.",
    )
    parser.add_argument(
        "--action-rate-hz",
        type=float,
        default=None,
        help="Override dataset.lerobot.fps from the runtime YAML.",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    config = load_runtime_config(args.config)
    if config.vla.action_space != "abs":
        raise ValueError(
            "FlexPi UR5e returns absolute joint targets; set "
            "vla.action_space: abs in the runtime YAML"
        )
    server_url = _flexpi_http_url(args.server_url or config.vla.server_url)
    timeout_s = (
        config.vla.request_timeout_s if args.timeout_s is None else args.timeout_s
    )
    execute_horizon = (
        config.vla.execute_horizon
        if args.execute_horizon is None
        else args.execute_horizon
    )
    max_joint_delta = (
        config.vla.max_joint_delta
        if args.max_joint_delta is None
        else args.max_joint_delta
    )
    gripper_threshold = (
        config.vla.threadhold
        if args.gripper_threshold is None
        else args.gripper_threshold
    )
    action_rate_hz = (
        float(config.dataset.lerobot_fps)
        if args.action_rate_hz is None
        else args.action_rate_hz
    )

    client = FlexPiHTTPClient(server_url, timeout_s=timeout_s)
    health = client.health()
    print("FlexPi health:")
    print(json.dumps(health, indent=2, ensure_ascii=False))

    profile = health.get("profile")
    if args.expected_profile is not None and profile != args.expected_profile:
        raise RuntimeError(
            f"FlexPi profile mismatch: expected {args.expected_profile!r}, "
            f"server reports {profile!r}"
        )
    if not health.get("ready"):
        raise RuntimeError("FlexPi server isn't ready")
    if args.health_only:
        return

    task = load_task_file(str(resolve_repo_path(config.task.file)))
    instruction = task.name if args.task is None else args.task
    if not instruction.strip():
        raise ValueError("task instruction must not be empty")

    if args.execute and args.expected_profile is None:
        raise ValueError("--execute requires --expected-profile")
    if args.execute and max_joint_delta is None:
        raise ValueError(
            "Robot execution requires vla.max_joint_delta in the runtime YAML "
            "or --max-joint-delta"
        )
    if args.execute and gripper_threshold is None:
        raise ValueError(
            "Robot execution requires vla.threadhold in the runtime YAML or "
            "--gripper-threshold"
        )
    # if args.execute:
    #     joint_limits_path = resolve_repo_path(config.safety.joint_limits_file)
    #     if not joint_limits_path.is_file():
    #         raise FileNotFoundError(
    #             "Robot execution requires a calibrated joint-limits file: "
    #             f"{joint_limits_path}. Run calibrate-safety-zone first."
    #         )

    print(f"FlexPi profile: {profile}")
    print(f"Instruction: {instruction}")
    print(
        "Resolved runtime parameters: "
        f"server_url={server_url}, timeout_s={timeout_s}, "
        f"execute_horizon={execute_horizon}, "
        f"max_joint_delta={max_joint_delta}, "
        f"gripper_threshold={gripper_threshold}, "
        f"action_rate_hz={action_rate_hz}"
    )
    if args.execute:
        print(
            "Before inference, the robot will move to robot.reset_joints: "
            f"{list(config.robot.reset_joints)}"
        )

    controller = build_flexpi_inference_controller(
        config=config,
        client=client,
        instruction=instruction,
        mode=args.mode,
        inference_steps=args.inference_steps,
        seed=args.seed,
    )
    try:
        if not args.execute:
            response = controller.infer_once()
            actions = np.asarray(response["actions"], dtype=float)
            summary = {
                "profile": response.get("profile"),
                "mode": response.get("mode"),
                "checkpoint_step": response.get("checkpoint_step"),
                "action_shape": list(actions.shape),
                "first_action": actions[0].tolist(),
                "latency_seconds": response.get("latency_seconds"),
            }
            print("Inference preview (no robot command sent):")
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return

        executed = controller.run(
            max_steps=args.max_steps,
            execute_horizon=execute_horizon,
            max_joint_delta=max_joint_delta,
            gripper_threshold=gripper_threshold,
            action_rate_hz=action_rate_hz,
        )
        print(f"Stopped after {executed} robot command(s).")
    except KeyboardInterrupt:
        print("Stopping FlexPi inference control.")
    finally:
        controller.close()


def _flexpi_http_url(configured_url: str) -> str:
    """Reuse the shared YAML address while selecting FlexPi's HTTP protocol."""

    if configured_url.startswith("ws://"):
        return "http://" + configured_url[len("ws://") :]
    if configured_url.startswith("wss://"):
        return "https://" + configured_url[len("wss://") :]
    return configured_url


if __name__ == "__main__":
    main()
