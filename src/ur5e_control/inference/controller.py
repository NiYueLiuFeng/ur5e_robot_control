"""Runtime controller that closes the loop between VLA inference and UR5e."""

from __future__ import annotations

import atexit
from multiprocessing import Process
from pathlib import Path
from typing import Any, Callable, Iterable
import time
import numpy as np

from ur5e_control.inference.action_adapter import ActionAdapter
from ur5e_control.inference.request_adapter import VLARequestAdapter
from ur5e_control.inference.vla_client import VLAClient
from ur5e_control.runtime.config import (
    RuntimeConfig,
    load_runtime_config,
    resolve_repo_path,
)
from ur5e_control.runtime.types import TaskInfo

CleanupCallback = Callable[[], None]


class VLAInferenceController:
    def __init__(
        self,
        *,
        env: Any,
        client: Any,
        task: TaskInfo,
        gripper_threadhold: float,
        request_adapter: VLARequestAdapter,
        action_adapter: ActionAdapter,
        action_space: str = "abs",
        execute_horizon: int = 1,
        max_joint_delta: float | None = None,
        reset_on_start: bool = True,
        sub_task: str | None = None,
        task_name: str | None = None,
        cleanup_callbacks: Iterable[CleanupCallback] = (),
    ) -> None:
        self.env = env
        self.client = client
        self.task = task
        self.request_adapter = request_adapter
        self.action_adapter = action_adapter
        self.action_space = action_space
        self.execute_horizon = int(execute_horizon)
        self.max_joint_delta = max_joint_delta
        self.reset_on_start = reset_on_start
        self.sub_task = sub_task
        self.task_name = task_name
        self.gripper_threadhold = gripper_threadhold
        self._cleanup_callbacks = tuple(cleanup_callbacks)
        if self.execute_horizon <= 0:
            raise ValueError("execute_horizon must be positive")

    def move_to_init_position(self, init_pos: tuple):


        np_init_pos = np.array(init_pos)
        curr_joints = self.env.get_obs()["joint_positions"]
        if np_init_pos.shape == np.array(curr_joints).shape:
            max_delta = (np.abs(curr_joints - init_pos)).max()
            steps = min(int(max_delta / 0.01), 100)
            for jnt in np.linspace(curr_joints, init_pos, steps):
                self.env.step(jnt)
                time.sleep(0.01)

    def run(self, init_pos, max_steps: int | None = None) -> int:
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")

        if self.reset_on_start:
            self.client.reset()

        self.move_to_init_position(init_pos)
        input("press ENTER to start inference control loop...")
        executed_steps = 0

        prev_rtc_leftover = None

        while max_steps is None or executed_steps < max_steps:
            obs = self.env.get_obs()
            request = self.request_adapter.from_env_obs(
                obs,
                task=self.task,
                sub_task=self.sub_task,
                task_name=self.task_name,
                # rtc=self.client.rtc,
            )
            ##rtc
            if prev_rtc_leftover is not None and len(prev_rtc_leftover) > 0:
                    request["rtc"] = {
                        "enabled": True,
                        "prev_chunk_left_over": prev_rtc_leftover,
                        "inference_delay": 0,
                        "execution_horizon": self.client.rtc.RTC_EXECUTION_HORIZON,
                        "max_guidance_weight": self.client.rtc.RTC_MAX_GUIDANCE_WEIGHT,
                        "prefix_attention_schedule": self.client.rtc.RTC_PREFIX_ATTENTION_SCHEDULE,
                    }
            else:
                request["rtc"] = {"enabled": False}
            response = self.client.infer(request)
            actions = self.action_adapter.actions_from_response(
                response,
                current_state=obs["joint_positions"],
                action_space=self.action_space,
                execute_horizon=self.execute_horizon,
                max_joint_delta=self.max_joint_delta,
            )
            ##rtc
            rtc_original_actions = np.asarray(response["rtc_original_actions"], dtype=np.float32)[0]
            assert len(actions) == self.client.rtc.ACTION_CHUNK
            assert rtc_original_actions.shape == (self.client.rtc.ACTION_CHUNK, 7), rtc_original_actions.shape

            ##save to npz
            # np.savez("/home/robot/workspace/ur5e_robot_control-main/input_rtc.npz", state=[request["observation.state"]], 
            # third_image=[request["observation.images.third_person"]], wrist_image=[request["observation.images.wrist"]], 
            # prompt=['task'], repo_id=[request['repo_id']], rtc_enable=request['rtc']['enabled'], 
            # rtc_prev_chunk_left_over=[request['rtc']['prev_chunk_left_over']],
            # rtc_inference_delay=[request['rtc']['inference_delay']], rtc_execution_horizon=[request['rtc']['execution_horizon']], 
            # rtc_max_guidance_weight=[request['rtc']['max_guidance_weight']], 
            # rtc_prefix_attention_schedule=[request['rtc']['prefix_attention_schedule']])
            # tmp = np.load("/home/robot/workspace/ur5e_robot_control-main/input_rtc.npz")

            execute_n = min(self.client.rtc.REPLAN_STEPS, len(actions))
            for action in actions[:execute_n]:
                # start_time = time.time()
                action_np = np.asarray(action.values, dtype=float)
                action_np[-1] = 1.0 if action_np[-1] >= self.gripper_threadhold else 0.0 
                action_np[-1] = action_np[-1]*1000
                print("########action: ", action_np)
                self.env.step(action_np)
                time.sleep(0.001)
                executed_steps += 1
                # print("#######total spend time: ", time.time() - start_time)
                if max_steps is not None and executed_steps >= max_steps:
                    return executed_steps
            prev_rtc_leftover = rtc_original_actions[execute_n:].copy()
        return executed_steps

    def close(self) -> None:
        close_client = getattr(self.client, "close", None)
        if callable(close_client):
            close_client()
        for callback in reversed(self._cleanup_callbacks):
            callback()


def build_vla_inference_controller(
    config_path: str | Path | None = None,
    *,
    runtime_config: RuntimeConfig | None = None,
    task_name: str | None = None,
    sub_task: str | None = None,
) -> VLAInferenceController:
    from ur5e_control.data_collection.tasks import load_task_file
    from ur5e_control.data_collection.teleop.env import RobotEnv
    from ur5e_control.hardware import RealSenseCameraServer
    from ur5e_control.runtime.safety import load_joint_limits
    from ur5e_control.runtime.transport.robot_zmq import ZMQClientRobot

    config = runtime_config or load_runtime_config(config_path)
    # joint_limits_path = resolve_repo_path(config.safety.joint_limits_file)
    # if not joint_limits_path.exists(): ##lwz
    #     raise FileNotFoundError(
    #         f"Safety joint limits file not found: {joint_limits_path}. "
    #         "Run `uv run calibrate-safety-zone` on the edge laptop first."
    #     )

    robot_process = Process(
        target=_serve_robot,
        kwargs={
            "robot_ip": config.robot.ip,
            "gripper_port": config.gripper.port,
            "gripper_position_tolerance": config.gripper.position_tolerance,
            "gripper_min_command_interval_s": (config.gripper.min_command_interval_s),
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

        camera_server = RealSenseCameraServer(
            hostname=config.camera.hostname,
            realsense_ids=dict(config.camera.realsense_ids),
            fps=config.camera.fps,
            enable_depth=config.camera.enable_depth,
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
            # joint_limits=load_joint_limits(joint_limits_path),
            joint_limits=None,  # lwz: disable joint limits for now
        )
        task = load_task_file(str(resolve_repo_path(config.task.file)))
        client = VLAClient(
            server_url=config.vla.server_url,
            api_key=config.vla.api_key,
            timeout_s=config.vla.request_timeout_s,
            rtc=config.rtc,
        )
        return VLAInferenceController(
            env=env,
            client=client,
            task=task,
            gripper_threadhold=config.vla.threadhold,
            request_adapter=VLARequestAdapter(
                repo_id=config.vla.repo_id,
                state_key=config.vla.state_key,
                image_key_map=config.vla.image_key_map,
            ),
            action_adapter=ActionAdapter(expected_dim=7),
            action_space=config.vla.action_space,
            execute_horizon=config.vla.execute_horizon,
            max_joint_delta=config.vla.max_joint_delta,
            reset_on_start=config.vla.reset_on_start,
            sub_task=sub_task,
            task_name=task_name,
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


def _serve_robot(
    *,
    robot_ip: str,
    gripper_port: str,
    gripper_position_tolerance: int,
    gripper_min_command_interval_s: float,
    gripper_position_read_interval_s: float,
    port: int,
    host: str,
) -> None:
    from ur5e_control.hardware import URRobot
    from ur5e_control.runtime.transport.robot_zmq import ZMQServerRobot

    robot = URRobot(
        robot_ip=robot_ip,
        gripper_port=gripper_port,
        gripper_position_tolerance=gripper_position_tolerance,
        gripper_min_command_interval_s=gripper_min_command_interval_s,
        gripper_position_read_interval_s=gripper_position_read_interval_s,
    )
    ZMQServerRobot(robot, port=port, host=host).serve()


def _stop_process(process: Process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
