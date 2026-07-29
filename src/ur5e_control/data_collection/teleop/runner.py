"""UR5e teleoperation data collection runner."""

from __future__ import annotations

import atexit
import time
from multiprocessing import Process
from pathlib import Path

from ur5e_control.data_collection.tasks import load_task_file
from ur5e_control.runtime.config import (
    RuntimeConfig,
    load_runtime_config,
    resolve_repo_path,
)
from ur5e_control.runtime.types import TaskInfo


class TeleopDataCollection:
    def __init__(
        self,
        config_path: str | Path | None = None,
        runtime_config: RuntimeConfig | None = None,
    ):
        import numpy as np
        from ur5e_control.data_collection.teleop.env import RobotEnv
        from ur5e_control.runtime.transport.robot_zmq import ZMQClientRobot
        from ur5e_control.hardware import RealSenseCameraServer
        from ur5e_control.runtime.safety import load_joint_limits

        self.runtime_config = runtime_config or load_runtime_config(config_path)
        self._np = np
        self.hz = self.runtime_config.robot.control_hz
        self.hostname = self.runtime_config.robot.zmq_host
        self.robot_port = self.runtime_config.robot.zmq_port
        self.robot_ip = self.runtime_config.robot.ip
        self.gripper_port = self.runtime_config.gripper.port
        self.camera_hostname = self.runtime_config.camera.hostname
        self.camera_fps = self.runtime_config.camera.fps
        self.reset_joints = np.array(self.runtime_config.robot.reset_joints)
        self.flag_save = False
        self.flag_start_subtask = False
        self.flag_no_save = False

        task_file = resolve_repo_path(self.runtime_config.task.file)
        self.task_info: TaskInfo = load_task_file(str(task_file))
        self.episode_count = self.runtime_config.task.episode_count
        self.lerobot_repo_id = self.runtime_config.dataset.lerobot_repo_id
        self.lerobot_root = str(
            resolve_repo_path(self.runtime_config.dataset.lerobot_root)
        )
        self.lerobot_fps = self.runtime_config.dataset.lerobot_fps
        self.dataset_recorder = None
        joint_limits_path = resolve_repo_path(
            self.runtime_config.safety.joint_limits_file
        )

        # if not joint_limits_path.exists():
        #     raise FileNotFoundError(
        #         f"Safety joint limits file not found: {joint_limits_path}. "
        #         "Run `uv run calibrate-safety-zone` on the edge laptop first."
        #     )
        # self.joint_limits = load_joint_limits(joint_limits_path)

        camera_server = RealSenseCameraServer(
            hostname=self.camera_hostname,
            realsense_ids=dict(self.runtime_config.camera.realsense_ids),
            fps=self.camera_fps,
            enable_depth=self.runtime_config.camera.enable_depth,
        )
        robot_client = ZMQClientRobot(port=self.robot_port, host=self.hostname)
        self.env = RobotEnv(
            robot_client,
            control_rate_hz=self.hz,
            camera_name_id=camera_server.realsense_ids,
            cameras=camera_server.cameras,
            # joint_limits=self.joint_limits,
            joint_limits=None,
        )
        self.listener = self.start_keyboard_listener()

    def launch_robot_server(self):
        from ur5e_control.runtime.transport.robot_zmq import ZMQServerRobot
        from ur5e_control.hardware import URRobot

        robot = URRobot(
            robot_ip=self.robot_ip,
            gripper_port=self.gripper_port,
            gripper_position_tolerance=self.runtime_config.gripper.position_tolerance,
            gripper_min_command_interval_s=(
                self.runtime_config.gripper.min_command_interval_s
            ),
            gripper_position_read_interval_s=(
                self.runtime_config.gripper.position_read_interval_s
            ),
        )
        server = ZMQServerRobot(robot, port=self.robot_port, host=self.hostname)
        print(f"Starting robot server on port {self.robot_port}")
        server.serve()

    def start_robot_process(self):
        process = Process(target=self.launch_robot_server)

        def kill_child_process(process):
            print("Killing child process...")
            process.terminate()
            process.join(timeout=2)

        atexit.register(kill_child_process, process)
        process.start()


    def align_ini_position(self):
        np = self._np

        print("Going to start position")
        start_pos = self.agent.act(self.env.get_obs())
        obs = self.env.get_obs()
        joints = np.array(obs["joint_positions"])

        abs_deltas = np.abs(start_pos[:6] - joints[:6])
        id_max_joint_delta = np.argmax(abs_deltas)
        max_joint_delta = 0.8
        if abs_deltas[id_max_joint_delta] > max_joint_delta:
            id_mask = abs_deltas > max_joint_delta
            ids = np.arange(len(id_mask))[id_mask]
            for i, delta, joint, current_j in zip(
                ids,
                abs_deltas[id_mask],
                start_pos[id_mask],
                joints[id_mask],
            ):
                print(
                    f"joint[{i}]: delta: {delta:4.3f}, "
                    f"leader: {joint:4.3f}, follower: {current_j:4.3f}"
                )
            raise RuntimeError("Joint delta too large")

        if len(start_pos) != len(joints):
            raise RuntimeError(
                f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"
            )

        max_delta = 0.05
        for _ in range(100):
            obs = self.env.get_obs()
            command_joints = self.agent.act(obs)
            current_joints = np.array(obs["joint_positions"])
            delta = command_joints - current_joints
            max_joint_delta = np.abs(delta[:6]).max()
            if max_joint_delta > max_delta:
                delta[:6] = delta[:6] / max_joint_delta * max_delta
            self.env.step(current_joints + delta)

        obs = self.env.get_obs()
        joints = np.array(obs["joint_positions"])
        action = self.agent.act(obs)
        if (action[:6] - joints[:6] > 0.5).any():
            raise RuntimeError("Action is too large after alignment")

    def start_keyboard_listener(self):
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.esc:
                self.flag_save = True
            elif key == keyboard.Key.space:
                self.flag_start_subtask = not self.flag_start_subtask
            elif hasattr(key, 'char') and key.char == 'Q':
                self.flag_no_save = True
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        return listener

    def get_episode_recorder(self, first_obs):
        from ur5e_control.data_collection.lerobot_v3 import (
            LeRobotV3Recorder,
            AsyncLeRobotV3Recorder,
            LeRobotV3RecorderConfig,
        )
        from ur5e_control.data_collection.teleop_recording import (
            infer_image_shapes_from_env_obs,
        )

        if self.dataset_recorder is None:
            config_kwargs = {
                "repo_id": self.lerobot_repo_id,
                "root": self.lerobot_root,
                "fps": self.lerobot_fps,
                "robot_type": "ur5e_pgc140_d455",
            }
            image_shapes = infer_image_shapes_from_env_obs(first_obs)
            if image_shapes:
                config_kwargs["image_shapes"] = image_shapes
            recorder = LeRobotV3Recorder(LeRobotV3RecorderConfig(**config_kwargs))
            if self.runtime_config.dataset.lerobot_async_recording:
                recorder = AsyncLeRobotV3Recorder(
                    recorder,
                    max_queue_size=(
                        self.runtime_config.dataset.lerobot_record_queue_size
                    ),
                )
            self.dataset_recorder = recorder
            atexit.register(self.dataset_recorder.finalize)
        return self.dataset_recorder

    def ur5e_move_to_ini_position(self):
        np = self._np

        curr_joints = self.env.get_obs()["joint_positions"]
        if self.reset_joints.shape == np.array(curr_joints).shape:
            max_delta = (np.abs(curr_joints - self.reset_joints)).max()
            steps = min(int(max_delta / 0.01), 100)
            for jnt in np.linspace(curr_joints, self.reset_joints, steps):
                obs_ = self.env.step(jnt)
                time.sleep(0.05)
            curr_joints_ = self.env.get_obs()["joint_positions"]
            print(curr_joints_)

    def start(self):
        from ur5e_control.data_collection.teleop.agents.factory import (
            create_teleop_agent,
        )
        from ur5e_control.data_collection.teleop_recording import (
            record_lerobot_step,
            select_subtask,
        )

        self.start_robot_process()

        time.sleep(5)
        print("Please press Play on the teach pendant to connect to the ur5e robotic arm.")
        user_input = input("请在ur5上点击播放，然后输入回车")

        self.agent = create_teleop_agent(self.runtime_config.agent)
        self.ur5e_move_to_ini_position()

        for idx in range(self.episode_count):
            runtime_task = self.task_info
            curr_sub_tasks = list(runtime_task.subtasks)
            if self.runtime_config.agent.name=='gello':
                self.align_ini_position()
            self.flag_save = False
            self.flag_no_save = False
            self.flag_start_subtask = False
            obs = self.env.get_obs()
            episode_recorder = self.get_episode_recorder(obs)
            episode_recorder.start_episode(runtime_task)
            input("press ENTER to start record this task")
            episode_start_time = time.time()

            while True:
                start_time = time.time()
                action_ = self.agent.act(obs)
                # print(f"state gripper: {obs["gripper_position"]}   action gripper: {action_[-1]}")

                # action = self.env.apply_safety(action_)
                action = action_
                curr_sub_task = select_subtask(
                    curr_sub_tasks,
                    second_subtask_active=self.flag_start_subtask,
                )
                record_lerobot_step(
                    recorder=episode_recorder,
                    env_obs=obs,
                    action_values=action,
                    task=runtime_task,
                    subtask=curr_sub_task,
                )

                print(f"Action: {action}", f"time: {time.time() - start_time}")
                obs = self.env.step(action)
                time.sleep(0.05)

                if self.flag_no_save:  ###input shift + q
                    self.ur5e_move_to_ini_position()
                    if self.runtime_config.agent.name=='gello':
                        self.align_ini_position()
                    episode_recorder._recorder.dataset.clear_episode_buffer()
                    break

                if self.flag_save: ###input esc
                    print(f"Task {idx} finished")
                    print("Saving LeRobot episode...")
                    episode_recorder.save_episode()
                    self.ur5e_move_to_ini_position()
                    if self.runtime_config.agent.name=='gello':
                        self.align_ini_position()
                    break

        if self.dataset_recorder is not None:
            self.dataset_recorder.finalize()


def main(config_path: str | Path | None = None) -> None:
    collector = TeleopDataCollection(config_path=config_path)
    try:
        collector.start()
    finally:
        listener = getattr(collector, "listener", None)
        if listener is not None:
            listener.stop()