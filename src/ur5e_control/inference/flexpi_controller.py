"""Independent FlexPi HTTP inference loop for the UR5e runtime."""

from __future__ import annotations

import atexit
from collections.abc import Callable, Iterable
from multiprocessing import Process
import time
from typing import Any

import numpy as np

from ur5e_control.inference.controller import _serve_robot, _stop_process
from ur5e_control.inference.flexpi_http import (
    FlexPiHTTPClient,
    FlexPiRequestAdapter,
)
from ur5e_control.runtime.config import RuntimeConfig, resolve_repo_path

CleanupCallback = Callable[[], None]


class FlexPiInferenceController:
    """Close the live observation/action loop through a FlexPi HTTP server."""

    def __init__(
        self,
        *,
        env: Any,
        client: FlexPiHTTPClient,
        request_adapter: FlexPiRequestAdapter,
        instruction: str,
        init_position: Any | None = None,
        mode: str = "action_dit_action_only",
        inference_steps: int = 4,
        seed: int = 42,
        cleanup_callbacks: Iterable[CleanupCallback] = (),
    ) -> None:
        self.env = env
        self.client = client
        self.request_adapter = request_adapter
        self.instruction = instruction
        self.init_position = (
            None
            if init_position is None
            else np.asarray(init_position, dtype=np.float64).reshape(-1)
        )
        if self.init_position is not None:
            if self.init_position.size != 7:
                raise ValueError("init_position must contain 7 values")
            if not np.isfinite(self.init_position).all():
                raise ValueError("init_position must contain only finite values")
            if not 0.0 <= self.init_position[6] <= 1000.0:
                raise ValueError("init_position gripper value must be in [0, 1000]")
        self.mode = mode
        self.inference_steps = int(inference_steps)
        self.seed = int(seed)
        self._cleanup_callbacks = tuple(cleanup_callbacks)

    def infer_once(self) -> dict[str, Any]:
        observation = self.env.get_obs()
        request = self.request_adapter.from_env_obs(
            observation,
            instruction=self.instruction,
        )
        return self.client.infer(
            request,
            mode=self.mode,
            steps=self.inference_steps,
            seed=self.seed,
        )

    def move_to_init_position(self, *, max_step_rad: float = 0.01) -> int:
        """Move the six arm joints smoothly to the configured reset position."""

        if self.init_position is None:
            return 0
        if max_step_rad <= 0:
            raise ValueError("max_step_rad must be positive")

        observation = self.env.get_obs()
        current = np.asarray(
            observation["joint_positions"], dtype=np.float64
        ).reshape(-1)
        if current.size < 6 or not np.isfinite(current[:6]).all():
            raise ValueError(
                "current joint_positions must contain at least 6 finite values"
            )

        target_arm = self.init_position[:6]
        max_delta = float(np.max(np.abs(target_arm - current[:6])))
        interpolation_steps = max(1, int(np.ceil(max_delta / max_step_rad)))
        print(
            "Moving robot to initial position: "
            f"target={self.init_position.tolist()}, "
            f"steps={interpolation_steps}",
            flush=True,
        )
        for arm_target in np.linspace(
            current[:6], target_arm, interpolation_steps + 1
        )[1:]:
            command = np.concatenate((arm_target, self.init_position[6:7]))
            self.env.step(command)
        print("Robot reached the initial position.", flush=True)
        return interpolation_steps

    def run(
        self,
        *,
        max_steps: int,
        execute_horizon: int = 1,
        max_joint_delta: float = 0.1,
        gripper_threshold: float = 0.7,
        action_rate_hz: float = 15.0,
    ) -> int:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if execute_horizon <= 0:
            raise ValueError("execute_horizon must be positive")
        if max_joint_delta <= 0:
            raise ValueError("max_joint_delta must be positive")
        if not 0.0 <= gripper_threshold <= 1.0:
            raise ValueError("gripper_threshold must be in [0, 1]")
        if action_rate_hz <= 0:
            raise ValueError("action_rate_hz must be positive")

        self.move_to_init_position()
        executed_steps = 0
        while executed_steps < max_steps:
            observation = self.env.get_obs()
            request = self.request_adapter.from_env_obs(
                observation,
                instruction=self.instruction,
            )
            response = self.client.infer(
                request,
                mode=self.mode,
                steps=self.inference_steps,
                seed=self.seed,
            )
            actions = np.asarray(response["actions"], dtype=np.float64)
            reference = np.asarray(
                observation["joint_positions"], dtype=np.float64
            ).reshape(-1)[:6]

            for raw_action in actions[:execute_horizon]:
                command = np.asarray(raw_action, dtype=np.float64).copy()
                unclamped = command[:6].copy()
                command[:6] = np.clip(
                    command[:6],
                    reference - max_joint_delta,
                    reference + max_joint_delta,
                )
                if not np.array_equal(command[:6], unclamped):
                    print(
                        "Joint target clamped by max_joint_delta: "
                        f"requested={unclamped.tolist()} "
                        f"commanded={command[:6].tolist()}",
                        flush=True,
                    )
                command[6] = 1000.0 if command[6] >= gripper_threshold else 0.0

                started = time.monotonic()
                print(f"FlexPi command: {command}", flush=True)
                self.env.step(command)
                executed_steps += 1
                reference = command[:6].copy()
                if executed_steps >= max_steps:
                    return executed_steps

                remaining = 1.0 / action_rate_hz - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
        return executed_steps

    def close(self) -> None:
        for callback in reversed(self._cleanup_callbacks):
            callback()


def build_flexpi_inference_controller(
    *,
    config: RuntimeConfig,
    client: FlexPiHTTPClient,
    instruction: str,
    mode: str,
    inference_steps: int,
    seed: int,
) -> FlexPiInferenceController:
    """Build live robot/camera resources without touching the WebSocket path."""

    from ur5e_control.data_collection.teleop.env import RobotEnv
    from ur5e_control.hardware import RealSenseCameraServer
    from ur5e_control.runtime.transport.robot_zmq import ZMQClientRobot

    required_camera_names = {"third_left_top", "wrist"}
    missing_cameras = required_camera_names.difference(
        config.camera.realsense_ids.values()
    )
    if missing_cameras:
        raise ValueError(
            "FlexPi requires logical cameras: "
            + ", ".join(sorted(missing_cameras))
        )

    robot_process = Process(
        target=_serve_robot,
        kwargs={
            "robot_ip": config.robot.ip,
            "gripper_port": config.gripper.port,
            "gripper_position_tolerance": config.gripper.position_tolerance,
            "gripper_min_command_interval_s": (
                config.gripper.min_command_interval_s
            ),
            "gripper_position_read_interval_s": (
                config.gripper.position_read_interval_s
            ),
            "port": config.robot.zmq_port,
            "host": config.robot.zmq_host,
        },
    )
    camera_server = None
    try:
        robot_process.start()
        atexit.register(_stop_process, robot_process)

        # FlexPi's DA3 path consumes RGB only.  Don't start an unused raw-depth
        # stream or introduce dual-D455 emitter interference in this entrypoint.
        camera_server = RealSenseCameraServer(
            hostname=config.camera.hostname,
            realsense_ids=dict(config.camera.realsense_ids),
            fps=config.camera.fps,
            enable_depth=False,
        )
        atexit.register(camera_server.stop)

        robot_client = ZMQClientRobot(
            port=config.robot.zmq_port,
            host=config.robot.zmq_host,
        )
        env = RobotEnv(
            robot_client,
            control_rate_hz=config.robot.control_hz,
            camera_name_id=camera_server.realsense_ids,
            cameras=camera_server.cameras,
            joint_limits=_load_optional_joint_limits(config),
        )
        return FlexPiInferenceController(
            env=env,
            client=client,
            request_adapter=FlexPiRequestAdapter(),
            instruction=instruction,
            init_position=config.robot.reset_joints,
            mode=mode,
            inference_steps=inference_steps,
            seed=seed,
            cleanup_callbacks=(
                camera_server.stop,
                lambda: _stop_process(robot_process),
            ),
        )
    except Exception:
        if camera_server is not None:
            camera_server.stop()
        _stop_process(robot_process)
        raise


def _load_optional_joint_limits(config: RuntimeConfig) -> Any | None:
    path = resolve_repo_path(config.safety.joint_limits_file)
    if not path.is_file():
        print(
            f"Warning: safety joint-limits file not found: {path}; "
            "max_joint_delta remains active.",
            flush=True,
        )
        return None
    from ur5e_control.runtime.safety import load_joint_limits

    return load_joint_limits(path)
