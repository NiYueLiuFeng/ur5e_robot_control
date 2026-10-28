"""Build a live UR5e environment: robot ZMQ service plus RealSense cameras."""

from __future__ import annotations

import atexit
from dataclasses import dataclass
from multiprocessing import Process
from typing import Any, Callable

from ur5e_control.runtime.config import RuntimeConfig


@dataclass
class RobotEnvBundle:
    """A ready `RobotEnv` plus the callback that releases its processes."""

    env: Any
    cleanup: Callable[[], None]

    def close(self) -> None:
        self.cleanup()


def build_robot_env(config: RuntimeConfig) -> RobotEnvBundle:
    """Start the robot and camera processes and return a connected environment."""

    from ur5e_control.data_collection.teleop.env import RobotEnv
    from ur5e_control.hardware import RealSenseCameraServer
    from ur5e_control.runtime.transport.robot_zmq import ZMQClientRobot

    robot_process = Process(
        target=serve_robot,
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
    robot_process.start()
    atexit.register(stop_process, robot_process)

    camera_server = None
    try:
        camera_server = RealSenseCameraServer(
            hostname=config.camera.hostname,
            realsense_ids=dict(config.camera.realsense_ids),
            fps=config.camera.fps,
            enable_depth=config.camera.enable_depth,
            depth_options=config.camera.depth_options,
        )
        atexit.register(camera_server.stop)

        env = RobotEnv(
            ZMQClientRobot(port=config.robot.zmq_port, host=config.robot.zmq_host),
            control_rate_hz=config.robot.control_hz,
            camera_name_id=camera_server.realsense_ids,
            cameras=camera_server.cameras,
            joint_limits=None,  # lwz: disable joint limits for now
        )
    except Exception:
        if camera_server is not None:
            camera_server.stop()
        stop_process(robot_process)
        raise

    def cleanup() -> None:
        assert camera_server is not None
        camera_server.stop()
        stop_process(robot_process)

    return RobotEnvBundle(env=env, cleanup=cleanup)


def serve_robot(
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


def stop_process(process: Process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
