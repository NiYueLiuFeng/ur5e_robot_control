"""Runtime controller that closes the loop between VLA inference and UR5e."""

from __future__ import annotations

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
    from ur5e_control.runtime.env import build_robot_env

    config = runtime_config or load_runtime_config(config_path)
    bundle = build_robot_env(config)
    try:
        task = load_task_file(str(resolve_repo_path(config.task.file)))
        client = VLAClient(
            server_url=config.vla.server_url,
            api_key=config.vla.api_key,
            timeout_s=config.vla.request_timeout_s,
            rtc=config.rtc,
        )
        return VLAInferenceController(
            env=bundle.env,
            client=client,
            task=task,
            gripper_threadhold=config.vla.threshold,
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
            cleanup_callbacks=(bundle.cleanup,),
        )
    except Exception:
        bundle.close()
        raise
