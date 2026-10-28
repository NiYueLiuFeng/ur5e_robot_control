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


def require_new_dataset_root(root: str | Path) -> Path:
    """Reject an existing destination before any robot hardware is started."""

    dataset_root = Path(root)
    if dataset_root.exists():
        raise FileExistsError(
            f"LeRobot dataset root already exists: {dataset_root}. "
            "Choose a new dataset.lerobot.root; this recorder creates new datasets "
            "and does not append to an existing one."
        )
    return dataset_root


def remaining_frame_delay(elapsed_s: float, fps: int) -> float:
    """Return the delay needed to keep the recording loop at the declared FPS."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    return max(0.0, (1.0 / fps) - elapsed_s)


class TeleopDataCollection:
    def __init__(
        self,
        config_path: str | Path | None = None,
        runtime_config: RuntimeConfig | None = None,
    ):
        import numpy as np
        from ur5e_control.data_collection.teleop.env import RobotEnv
        from ur5e_control.data_collection.observations import DEFAULT_IMAGE_KEY_MAP
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
        self.depth_quality_sample_count = (
            self.runtime_config.camera.depth_quality_sample_count
        )
        self.depth_max_invalid_ratio = {
            f"{camera_name}_depth": limit
            for camera_name, limit in (
                self.runtime_config.camera.depth_max_invalid_ratio.items()
            )
        }
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
        require_new_dataset_root(self.lerobot_root)
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
            depth_options=self.runtime_config.camera.depth_options,
        )
        self.camera_calibration = camera_server.get_calibration()
        self.camera_calibration["image_features"] = {
            camera_name: (
                "observation.images."
                + DEFAULT_IMAGE_KEY_MAP.get(
                    (
                        camera_name
                        if camera_name.endswith("_depth")
                        else f"{camera_name}_rgb"
                    ),
                    camera_name,
                )
            )
            for camera_name in self.camera_calibration["intrinsics"]
        }
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
            print("Initializing LeRobot dataset recorder...", flush=True)
            config_kwargs = {
                "repo_id": self.lerobot_repo_id,
                "root": self.lerobot_root,
                "fps": self.lerobot_fps,
                "robot_type": "ur5e_pgc140_d455",
                "camera_calibration": self.camera_calibration,
                "repair_video_timestamps": (
                    self.runtime_config.dataset.lerobot_repair_video_timestamps
                ),
                "depth_min_m": self.runtime_config.dataset.lerobot_depth_min_m,
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
            print("LeRobot dataset recorder ready", flush=True)
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
        # print("Please press Play on the teach pendant to connect to the ur5e robotic arm.")
        # user_input = input("请在ur5上点击播放，然后输入回车")

        self.agent = create_teleop_agent(self.runtime_config.agent)
        self.ur5e_move_to_ini_position()

        for idx in range(self.episode_count):
            runtime_task = self.task_info
            curr_sub_tasks = list(runtime_task.subtasks)
            if self.runtime_config.agent.name == "gello":
                self.align_ini_position()
            self.flag_save = False
            self.flag_no_save = False
            self.flag_start_subtask = False
            obs = self.env.get_obs()
            episode_recorder = None
            if not self.depth_max_invalid_ratio:
                # With no quality gate, finish the expensive first-time
                # LeRobot imports/dataset/writer setup before telling the
                # operator that Enter starts recording.
                episode_recorder = self.get_episode_recorder(obs)
            input("press ENTER to start record this task")
            obs = self.verify_depth_quality(obs)
            if episode_recorder is None:
                # Keep the old safety behavior when a quality gate is enabled:
                # do not create an empty dataset until the gate passes.
                episode_recorder = self.get_episode_recorder(obs)
            episode_recorder.start_episode(runtime_task)
            while True:
                start_time = time.monotonic()
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

                print(
                    f"Action: {action}",
                    f"gripper_target: {action[-1]}",
                    f"gripper_raw: {obs['gripper_position_raw']}",
                    f"gripper_state: {obs['gripper_position']}",
                    f"time: {time.monotonic() - start_time}",
                )
                obs = self.env.step(action)
                delay = remaining_frame_delay(
                    time.monotonic() - start_time,
                    self.lerobot_fps,
                )
                if delay > 0:
                    time.sleep(delay)

                if self.flag_no_save:  # input shift + q
                    self.ur5e_move_to_ini_position()
                    if self.runtime_config.agent.name == "gello":
                        self.align_ini_position()
                    episode_recorder._recorder.dataset.clear_episode_buffer()
                    break

                if self.flag_save:  # input esc
                    print(f"Task {idx} finished")
                    print("Saving LeRobot episode...")
                    episode_recorder.save_episode()
                    self.ur5e_move_to_ini_position()
                    if self.runtime_config.agent.name == "gello":
                        self.align_ini_position()
                    break

        if self.dataset_recorder is not None:
            self.dataset_recorder.finalize()

    def verify_depth_quality(self, first_obs):
        """Reject an episode when raw depth is mostly invalid before encoding."""

        if not self.depth_max_invalid_ratio:
            return first_obs

        from ur5e_control.data_collection.depth_quality import (
            evaluate_depth_quality,
        )

        observations = [first_obs]
        for _ in range(self.depth_quality_sample_count - 1):
            time.sleep(1.0 / self.camera_fps)
            observations.append(self.env.get_obs())

        results = evaluate_depth_quality(
            observations,
            self.depth_max_invalid_ratio,
        )
        failures = []
        print("Depth quality precheck:")
        for result in results:
            status = "ok" if result.ok else "blocked"
            print(
                f"- [{status}] {result.key}: invalid median "
                f"{result.median_invalid_ratio:.1%}, range "
                f"{result.minimum_invalid_ratio:.1%}-"
                f"{result.maximum_invalid_ratio:.1%}, limit {result.limit:.1%}"
            )
            if not result.ok:
                failures.append(result)

        if failures:
            details = ", ".join(
                f"{result.key}={result.median_invalid_ratio:.1%}"
                for result in failures
            )
            raise RuntimeError(
                "Depth quality check failed before recording: "
                f"{details}. Check camera distance, stereo-imager occlusion, "
                "projector interference, exposure, and USB bandwidth."
            )
        return observations[-1]


def main(config_path: str | Path | None = None) -> None:
    collector = TeleopDataCollection(config_path=config_path)
    try:
        collector.start()
    finally:
        listener = getattr(collector, "listener", None)
        if listener is not None:
            listener.stop()
