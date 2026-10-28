"""Runtime configuration for UR5e teleoperation and inference apps."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import Any, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HARDWARE_CONFIG_PATH = REPO_ROOT / "configs/hardware/ur5e_d455.yaml"
DEFAULT_VLA_IMAGE_KEY_MAP = {
    "third_front_rgb": "observation.images.third_front_image",
    "wrist_rgb": "observation.images.wrist_image",
}


@dataclass(frozen=True)
class RobotRuntimeConfig:
    name: str = "ur5e"
    ip: str = "192.168.56.2"
    zmq_host: str = "127.0.0.1"
    zmq_port: int = 6001
    control_hz: int = 100
    reset_joints: tuple[float, ...] = (
        1.56,
        -1.57,
        1.58,
        -1.58,
        -1.57,
        -0.0077,
        1000.0,
    )

    def __post_init__(self) -> None:
        if self.name != "ur5e":
            raise ValueError("robot.name must be 'ur5e'")
        if self.control_hz <= 0:
            raise ValueError("robot.control_hz must be positive")
        if self.zmq_port <= 0:
            raise ValueError("robot.zmq_port must be positive")
        object.__setattr__(
            self,
            "reset_joints",
            tuple(float(value) for value in self.reset_joints),
        )

@dataclass(frozen=True)
class SpaceMouseRuntimeConfig:
    device_path: str | None = None
    translation_scale: float = 0.06
    angle_scale: float = 0.24
    invert_control: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    rotation_mode: str = "euler"
    invert_button: bool = False
    gripper_open: int = 1000
    gripper_closed: int = 0

    def __post_init__(self) -> None:
        if self.translation_scale <= 0:
            raise ValueError("agent.spacemouse.translation_scale must be positive")
        if self.angle_scale <= 0:
            raise ValueError("agent.spacemouse.angle_scale must be positive")
        if len(self.invert_control) != 6:
            raise ValueError("agent.spacemouse.invert_control must contain 6 values")
        if self.rotation_mode not in {"euler", "rpy"}:
            raise ValueError("agent.spacemouse.rotation_mode must be 'euler' or 'rpy'")
        if not 0 <= self.gripper_open <= 1000:
            raise ValueError("agent.spacemouse.gripper_open must be in [0, 1000]")
        if not 0 <= self.gripper_closed <= 1000:
            raise ValueError("agent.spacemouse.gripper_closed must be in [0, 1000]")
        object.__setattr__(
            self,
            "invert_control",
            tuple(float(value) for value in self.invert_control),
        )

@dataclass(frozen=True)
class AgentRuntimeConfig:
    name: str = "gello"
    robot: str = "ur5"
    gello_port: str = "/dev/ttyUSB1"
    verbose: bool = False
    spacemouse: SpaceMouseRuntimeConfig = field(default_factory=SpaceMouseRuntimeConfig)

    def __post_init__(self) -> None:
        if self.name not in {"gello", "spacemouse"}:
            raise ValueError("agent.name must be 'gello' or 'spacemouse'")
        if self.robot not in {"ur5", "ur5e"}:
            raise ValueError("agent.robot must be 'ur5' or 'ur5e'")

@dataclass(frozen=True)
class RtcRuntimeConfig:
    ACTION_CHUNK: float = 15.0
    REPLAN_STEPS: int = 8
    RTC_EXECUTION_HORIZON: int = 8
    RTC_MAX_GUIDANCE_WEIGHT: float = 10.0
    RTC_PREFIX_ATTENTION_SCHEDULE: str = "EXP"

    def __post_init__(self) -> None:
        if self.ACTION_CHUNK <= 0:
            raise ValueError("rtc.ACTION_CHUNK must be positive")
        if self.REPLAN_STEPS <= 0:
            raise ValueError("rtc.REPLAN_STEPS must be positive")
        if self.RTC_EXECUTION_HORIZON <= 0:
            raise ValueError("rtc.RTC_EXECUTION_HORIZON must be positive")
        if self.RTC_MAX_GUIDANCE_WEIGHT <= 0:
            raise ValueError("rtc.RTC_MAX_GUIDANCE_WEIGHT must be positive")
        
@dataclass(frozen=True)
class CameraRuntimeConfig:
    hostname: str = "127.0.0.1"
    fps: int = 30
    enable_depth: bool = False
    depth_quality_sample_count: int = 15
    depth_max_invalid_ratio: Mapping[str, float] = field(default_factory=dict)
    depth_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    realsense_ids: Mapping[str, str] = field(default_factory=dict)
    calibration_dir: str = "configs/cameras"
    intrinsics_files: Mapping[str, str] = field(default_factory=dict)
    extrinsics_file: str = "configs/cameras/extrinsics.yaml"

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("camera.fps must be positive")
        if self.depth_quality_sample_count <= 0:
            raise ValueError("camera.depth_quality.sample_count must be positive")
        ids = {str(serial): str(name) for serial, name in self.realsense_ids.items()}
        if not ids:
            raise ValueError("camera.realsense_ids must define at least one camera")
        # missing_names = {"wrist", "third_front"}.difference(ids.values())
        # if missing_names:
        #     missing = ", ".join(sorted(missing_names))
        #     raise ValueError(
        #         f"camera.realsense_ids must include logical cameras: {missing}"
        #     )
        object.__setattr__(self, "realsense_ids", ids)
        if not self.calibration_dir:
            raise ValueError("camera.calibration_dir must be set")
        if not self.extrinsics_file:
            raise ValueError("camera.extrinsics_file must be set")
        logical_names = set(ids.values())
        depth_options = {
            str(name): dict(_mapping(options))
            for name, options in self.depth_options.items()
        }
        unknown_option_names = set(depth_options).difference(logical_names)
        if unknown_option_names:
            unknown = ", ".join(sorted(unknown_option_names))
            raise ValueError(
                "camera.depth_options contains unknown logical cameras: "
                f"{unknown}"
            )
        allowed_depth_options = {
            "visual_preset",
            "emitter_enabled",
            "laser_power",
        }
        for name, options in depth_options.items():
            unknown_options = set(options).difference(allowed_depth_options)
            if unknown_options:
                unknown = ", ".join(sorted(unknown_options))
                raise ValueError(
                    f"camera.depth_options.{name} contains unknown options: {unknown}"
                )
            if "emitter_enabled" in options and not isinstance(
                options["emitter_enabled"], bool
            ):
                raise ValueError(
                    f"camera.depth_options.{name}.emitter_enabled must be boolean"
                )
            if "laser_power" in options:
                try:
                    options["laser_power"] = float(options["laser_power"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"camera.depth_options.{name}.laser_power must be numeric"
                    ) from exc
        if depth_options and not self.enable_depth:
            raise ValueError("camera.depth_options requires camera.enable_depth")
        object.__setattr__(self, "depth_options", depth_options)
        depth_limits = {
            str(name): float(value)
            for name, value in self.depth_max_invalid_ratio.items()
        }
        unknown_depth_names = set(depth_limits).difference(logical_names)
        if unknown_depth_names:
            unknown = ", ".join(sorted(unknown_depth_names))
            raise ValueError(
                "camera.depth_quality.max_invalid_ratio contains unknown logical "
                f"cameras: {unknown}"
            )
        for name, limit in depth_limits.items():
            if not 0.0 <= limit <= 1.0:
                raise ValueError(
                    "camera.depth_quality.max_invalid_ratio."
                    f"{name} must be in [0, 1]"
                )
        if depth_limits and not self.enable_depth:
            raise ValueError(
                "camera.depth_quality.max_invalid_ratio requires camera.enable_depth"
            )
        object.__setattr__(self, "depth_max_invalid_ratio", depth_limits)
        intrinsics_files = {
            str(name): str(path) for name, path in self.intrinsics_files.items()
        }
        if not intrinsics_files:
            calibration_dir = str(self.calibration_dir).rstrip("/")
            intrinsics_files = {
                name: f"{calibration_dir}/intrinsics_{name}.yaml"
                for name in sorted(logical_names)
            }
        unknown_names = set(intrinsics_files).difference(logical_names)
        if unknown_names:
            unknown = ", ".join(sorted(unknown_names))
            raise ValueError(
                "camera.intrinsics_files contains unknown logical cameras: "
                f"{unknown}"
            )
        object.__setattr__(self, "calibration_dir", str(self.calibration_dir))
        object.__setattr__(self, "intrinsics_files", intrinsics_files)
        object.__setattr__(self, "extrinsics_file", str(self.extrinsics_file))


@dataclass(frozen=True)
class GripperRuntimeConfig:
    gripper_type: str = "dh_pgc_140_50"
    port: str = "/dev/ttyUSB0"
    position_tolerance: int = 10
    min_command_interval_s: float = 0.05
    position_read_interval_s: float = 0.1

    def __post_init__(self) -> None:
        if self.gripper_type != "dh_pgc_140_50":
            raise ValueError("gripper.type must be 'dh_pgc_140_50'")
        if self.position_tolerance < 0:
            raise ValueError("gripper.position_tolerance must be non-negative")
        if self.min_command_interval_s < 0:
            raise ValueError("gripper.min_command_interval_s must be non-negative")
        if self.position_read_interval_s < 0:
            raise ValueError("gripper.position_read_interval_s must be non-negative")


@dataclass(frozen=True)
class DatasetRuntimeConfig:
    lerobot_repo_id: str = "chopintelligence/ur5e-pen-pick-place"
    lerobot_root: str = ".datasets/lerobot/ur5e-pen-pick-place"
    lerobot_fps: int = 30
    lerobot_async_recording: bool = True
    lerobot_record_queue_size: int = 64
    lerobot_repair_video_timestamps: bool = False
    lerobot_depth_min_m: float | None = None

    def __post_init__(self) -> None:
        if self.lerobot_fps <= 0:
            raise ValueError("dataset.lerobot.fps must be positive")
        if self.lerobot_record_queue_size <= 0:
            raise ValueError("dataset.lerobot.record_queue_size must be positive")
        if self.lerobot_depth_min_m is not None and self.lerobot_depth_min_m < 0:
            raise ValueError("dataset.lerobot.depth_min_m must be non-negative")


@dataclass(frozen=True)
class SafetyRuntimeConfig:
    joint_limits_file: str = "configs/safety/ur5e_joint_limits.yaml"

    def __post_init__(self) -> None:
        if not self.joint_limits_file:
            raise ValueError("safety.joint_limits_file must be set")


@dataclass(frozen=True)
class VLARuntimeConfig:
    server_url: str = "ws://127.0.0.1:8000"
    api_key: str | None = None
    repo_id: str = "chopintelligence/ur5e-pen-pick-place"
    state_key: str = "observation.state"
    image_key_map: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_VLA_IMAGE_KEY_MAP)
    )
    action_space: str = "abs"
    execute_horizon: int = 1
    request_timeout_s: float = 5.0
    max_joint_delta: float | None = 0.1
    reset_on_start: bool = True
    threshold: float | None = 0.7

    def __post_init__(self) -> None:
        if not self.server_url:
            raise ValueError("vla.server_url must be set")
        if not self.repo_id:
            raise ValueError("vla.repo_id must be set")
        if not self.state_key:
            raise ValueError("vla.state_key must be set")
        action_space = self.action_space.strip().lower()
        if action_space in {"absolute"}:
            action_space = "abs"
        elif action_space in {"relative"}:
            action_space = "delta"
        if action_space not in {"abs", "delta"}:
            raise ValueError("vla.action_space must be 'abs' or 'delta'")
        if self.execute_horizon <= 0:
            raise ValueError("vla.execute_horizon must be positive")
        if self.request_timeout_s <= 0:
            raise ValueError("vla.request_timeout_s must be positive")
        if self.max_joint_delta is not None and self.max_joint_delta < 0:
            raise ValueError("vla.max_joint_delta must be non-negative")
        object.__setattr__(self, "action_space", action_space)
        object.__setattr__(
            self,
            "image_key_map",
            {str(key): str(value) for key, value in self.image_key_map.items()},
        )


@dataclass(frozen=True)
class TaskRuntimeConfig:
    file: str = "configs/tasks/pen_pick_place.yaml"
    episode_count: int = 300

    def __post_init__(self) -> None:
        if self.episode_count <= 0:
            raise ValueError("task.episode_count must be positive")


@dataclass(frozen=True)
class RuntimeConfig:
    robot: RobotRuntimeConfig = field(default_factory=RobotRuntimeConfig)
    agent: AgentRuntimeConfig = field(default_factory=AgentRuntimeConfig)
    camera: CameraRuntimeConfig = field(default_factory=CameraRuntimeConfig)
    gripper: GripperRuntimeConfig = field(default_factory=GripperRuntimeConfig)
    dataset: DatasetRuntimeConfig = field(default_factory=DatasetRuntimeConfig)
    safety: SafetyRuntimeConfig = field(default_factory=SafetyRuntimeConfig)
    vla: VLARuntimeConfig = field(default_factory=VLARuntimeConfig)
    task: TaskRuntimeConfig = field(default_factory=TaskRuntimeConfig)
    rtc: RtcRuntimeConfig = field(default_factory=RtcRuntimeConfig)
    source_path: str | None = None


def load_runtime_config(
    path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> RuntimeConfig:
    """Load the shared runtime config and apply laptop-specific env overrides."""

    env = os.environ if environ is None else environ
    config_path = resolve_config_path(path, env)
    data = _load_yaml_mapping(config_path)
    dataset = _load_dataset_config(_mapping(data.get("dataset")))
    config = RuntimeConfig(
        robot=_load_robot_config(_mapping(data.get("robot"))),
        agent=_load_agent_config(_mapping(data.get("agent"))),
        camera=_load_camera_config(_mapping(data.get("camera"))),
        gripper=_load_gripper_config(_mapping(data.get("gripper"))),
        dataset=dataset,
        safety=_load_safety_config(_mapping(data.get("safety"))),
        vla=_load_vla_config(_mapping(data.get("vla")), dataset=dataset),
        task=_load_task_config(_mapping(data.get("task"))),
        source_path=str(config_path),
        rtc=_load_rtc_config(_mapping(data.get("rtc"))),
    )
    return _apply_env_overrides(config, env)


def resolve_config_path(
    path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    env = os.environ if environ is None else environ
    raw_path = path or env.get("UR5E_HARDWARE_CONFIG")
    if raw_path is None:
        return DEFAULT_HARDWARE_CONFIG_PATH
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    import yaml  # type: ignore

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def _load_robot_config(data: Mapping[str, Any]) -> RobotRuntimeConfig:
    return RobotRuntimeConfig(
        name=str(data.get("name", "ur5e")),
        ip=str(data.get("ip", "192.168.56.2")),
        zmq_host=str(data.get("zmq_host", "127.0.0.1")),
        zmq_port=_int_value(data.get("zmq_port", 6001), "robot.zmq_port"),
        control_hz=_int_value(data.get("control_hz", 100), "robot.control_hz"),
        reset_joints=_float_tuple(
            data.get("reset_joints", RobotRuntimeConfig().reset_joints),
            "robot.reset_joints",
        ),
    )

def _load_rtc_config(data: Mapping[str, Any]) -> RtcRuntimeConfig:
    if "config" not in data:
        raise ValueError("rtc.config must be defined in the config file")

    config = _mapping(data["config"])
    defaults = RtcRuntimeConfig()
    return RtcRuntimeConfig(
        ACTION_CHUNK=_float_value(
            config.get("ACTION_CHUNK", defaults.ACTION_CHUNK),
            "rtc.config.ACTION_CHUNK",
        ),
        REPLAN_STEPS=_int_value(
            config.get("REPLAN_STEPS", defaults.REPLAN_STEPS),
            "rtc.config.REPLAN_STEPS",
        ),
        RTC_EXECUTION_HORIZON=_int_value(
            config.get(
                "RTC_EXECUTION_HORIZON",
                defaults.RTC_EXECUTION_HORIZON,
            ),
            "rtc.config.RTC_EXECUTION_HORIZON",
        ),
        RTC_MAX_GUIDANCE_WEIGHT=_float_value(
            config.get(
                "RTC_MAX_GUIDANCE_WEIGHT",
                defaults.RTC_MAX_GUIDANCE_WEIGHT,
            ),
            "rtc.config.RTC_MAX_GUIDANCE_WEIGHT",
        ),
        RTC_PREFIX_ATTENTION_SCHEDULE=str(
            config.get(
                "RTC_PREFIX_ATTENTION_SCHEDULE",
                defaults.RTC_PREFIX_ATTENTION_SCHEDULE,
            )
        ),
    )


def _load_agent_config(data: Mapping[str, Any]) -> AgentRuntimeConfig:
    spacemouse_config = _load_spacemouse_config(_mapping(data.get("spacemouse")))

    return AgentRuntimeConfig(
        name=str(data.get("name", "gello")),
        robot=str(data.get("robot", "ur5")),
        gello_port=str(data.get("gello_port", "/dev/ttyUSB1")),
        verbose=_bool_value(data.get("verbose", False), "agent.verbose"),
        spacemouse=spacemouse_config,
    )

def _load_spacemouse_config(data: Mapping[str, Any]) -> SpaceMouseRuntimeConfig:
    device_path = data.get("device_path")
    return SpaceMouseRuntimeConfig(
        device_path=None if device_path is None else str(device_path),
        translation_scale=_float_value(
            data.get("translation_scale", 0.06),
            "agent.spacemouse.translation_scale",
        ),
        angle_scale=_float_value(
            data.get("angle_scale", 0.24),
            "agent.spacemouse.angle_scale",
        ),
        invert_control=_float_tuple(
            data.get("invert_control", SpaceMouseRuntimeConfig().invert_control),
            "agent.spacemouse.invert_control",
        ),
        rotation_mode=str(data.get("rotation_mode", "euler")),
        invert_button=_bool_value(
            data.get("invert_button", False),
            "agent.spacemouse.invert_button",
        ),
        gripper_open=_int_value(
            data.get("gripper_open", 1000),
            "agent.spacemouse.gripper_open",
        ),
        gripper_closed=_int_value(
            data.get("gripper_closed", 0),
            "agent.spacemouse.gripper_closed",
        ),
    )


def _load_camera_config(data: Mapping[str, Any]) -> CameraRuntimeConfig:
    if "realsense_ids" not in data:
        raise ValueError("camera.realsense_ids must be defined in the config file")
    depth_quality = _mapping(data.get("depth_quality"))
    return CameraRuntimeConfig(
        hostname=str(data.get("hostname", "127.0.0.1")),
        fps=_int_value(data.get("fps", 30), "camera.fps"),
        enable_depth=_bool_value(
            data.get("enable_depth", False), "camera.enable_depth"
        ),
        depth_quality_sample_count=_int_value(
            depth_quality.get("sample_count", 15),
            "camera.depth_quality.sample_count",
        ),
        depth_max_invalid_ratio=_float_mapping(
            depth_quality.get("max_invalid_ratio", {}),
            "camera.depth_quality.max_invalid_ratio",
        ),
        depth_options=_nested_mapping(
            data.get("depth_options", {}),
            "camera.depth_options",
        ),
        realsense_ids=_string_mapping(
            data.get("realsense_ids"),
            "camera.realsense_ids",
        ),
        calibration_dir=str(data.get("calibration_dir", "configs/cameras")),
        intrinsics_files=_string_mapping(
            data.get("intrinsics_files", {}),
            "camera.intrinsics_files",
        ),
        extrinsics_file=str(
            data.get("extrinsics_file", "configs/cameras/extrinsics.yaml")
        ),
    )


def _load_gripper_config(data: Mapping[str, Any]) -> GripperRuntimeConfig:
    return GripperRuntimeConfig(
        gripper_type=str(data.get("type", "dh_pgc_140_50")),
        port=str(data.get("port", "/dev/ttyUSB0")),
        position_tolerance=_int_value(
            data.get("position_tolerance", 10),
            "gripper.position_tolerance",
        ),
        min_command_interval_s=_float_value(
            data.get("min_command_interval_s", 0.05),
            "gripper.min_command_interval_s",
        ),
        position_read_interval_s=_float_value(
            data.get("position_read_interval_s", 0.1),
            "gripper.position_read_interval_s",
        ),
    )


def _load_dataset_config(data: Mapping[str, Any]) -> DatasetRuntimeConfig:
    lerobot = _mapping(data.get("lerobot"))
    depth_min_m = lerobot.get("depth_min_m")
    return DatasetRuntimeConfig(
        lerobot_repo_id=str(
            lerobot.get("repo_id", "chopintelligence/ur5e-pen-pick-place")
        ),
        lerobot_root=str(lerobot.get("root", ".datasets/lerobot/ur5e-pen-pick-place")),
        lerobot_fps=_int_value(lerobot.get("fps", 30), "dataset.lerobot.fps"),
        lerobot_async_recording=_bool_value(
            lerobot.get("async_recording", True),
            "dataset.lerobot.async_recording",
        ),
        lerobot_record_queue_size=_int_value(
            lerobot.get("record_queue_size", 64),
            "dataset.lerobot.record_queue_size",
        ),
        lerobot_repair_video_timestamps=_bool_value(
            lerobot.get("repair_video_timestamps", False),
            "dataset.lerobot.repair_video_timestamps",
        ),
        lerobot_depth_min_m=(
            None
            if depth_min_m is None
            else _float_value(depth_min_m, "dataset.lerobot.depth_min_m")
        ),
    )


def _load_safety_config(data: Mapping[str, Any]) -> SafetyRuntimeConfig:
    return SafetyRuntimeConfig(
        joint_limits_file=str(
            data.get("joint_limits_file", "configs/safety/ur5e_joint_limits.yaml")
        ),
    )


def _load_vla_config(
    data: Mapping[str, Any],
    *,
    dataset: DatasetRuntimeConfig,
) -> VLARuntimeConfig:
    api_key = data.get("api_key")
    max_joint_delta = data.get("max_joint_delta", 0.1)
    threshold = data.get("threshold", 0.7)
    return VLARuntimeConfig(
        server_url=str(data.get("server_url", "ws://127.0.0.1:8000")),
        api_key=None if api_key in {None, ""} else str(api_key),
        repo_id=str(data.get("repo_id", dataset.lerobot_repo_id)),
        state_key=str(data.get("state_key", "observation.state")),
        image_key_map=_string_mapping(
            data.get("image_key_map", DEFAULT_VLA_IMAGE_KEY_MAP),
            "vla.image_key_map",
        ),
        action_space=str(data.get("action_space", "abs")),
        execute_horizon=_int_value(
            data.get("execute_horizon", 1),
            "vla.execute_horizon",
        ),
        request_timeout_s=_float_value(
            data.get("request_timeout_s", 5.0),
            "vla.request_timeout_s",
        ),
        max_joint_delta=(
            None
            if max_joint_delta is None
            else _float_value(max_joint_delta, "vla.max_joint_delta")
        ),
        reset_on_start=_bool_value(
            data.get("reset_on_start", True), "vla.reset_on_start"
        ),
        threshold=(
            None if threshold is None else _float_value(threshold, "vla.threshold")
        ),
    )


def _load_task_config(data: Mapping[str, Any]) -> TaskRuntimeConfig:
    return TaskRuntimeConfig(
        file=str(data.get("file", "configs/tasks/pen_pick_place.yaml")),
        episode_count=_int_value(data.get("episode_count", 300), "task.episode_count"),
    )


def _apply_env_overrides(
    config: RuntimeConfig,
    environ: Mapping[str, str],
) -> RuntimeConfig:
    robot = config.robot
    if "UR5E_ROBOT_IP" in environ:
        robot = replace(robot, ip=environ["UR5E_ROBOT_IP"])
    if "UR5E_ROBOT_ZMQ_HOST" in environ:
        robot = replace(robot, zmq_host=environ["UR5E_ROBOT_ZMQ_HOST"])
    if "UR5E_ROBOT_ZMQ_PORT" in environ:
        robot = replace(
            robot,
            zmq_port=_int_value(environ["UR5E_ROBOT_ZMQ_PORT"], "UR5E_ROBOT_ZMQ_PORT"),
        )
    if "UR5E_CONTROL_HZ" in environ:
        robot = replace(
            robot,
            control_hz=_int_value(environ["UR5E_CONTROL_HZ"], "UR5E_CONTROL_HZ"),
        )

    agent = config.agent
    if "UR5E_AGENT_NAME" in environ:
        agent = replace(agent, name=environ["UR5E_AGENT_NAME"])
    if "UR5E_AGENT_ROBOT" in environ:
        agent = replace(agent, robot=environ["UR5E_AGENT_ROBOT"])
    if "UR5E_GELLO_PORT" in environ:
        agent = replace(agent, gello_port=environ["UR5E_GELLO_PORT"])
    spacemouse = agent.spacemouse
    if "UR5E_SPACEMOUSE_DEVICE_PATH" in environ:
        spacemouse = replace(
            spacemouse,
            device_path=environ["UR5E_SPACEMOUSE_DEVICE_PATH"] or None,
        )
    if "UR5E_SPACEMOUSE_TRANSLATION_SCALE" in environ:
        spacemouse = replace(
            spacemouse,
            translation_scale=_float_value(
                environ["UR5E_SPACEMOUSE_TRANSLATION_SCALE"],
                "UR5E_SPACEMOUSE_TRANSLATION_SCALE",
            ),
        )
    if "UR5E_SPACEMOUSE_ANGLE_SCALE" in environ:
        spacemouse = replace(
            spacemouse,
            angle_scale=_float_value(
                environ["UR5E_SPACEMOUSE_ANGLE_SCALE"],
                "UR5E_SPACEMOUSE_ANGLE_SCALE",
            ),
        )
    if "UR5E_SPACEMOUSE_INVERT_CONTROL" in environ:
        spacemouse = replace(
            spacemouse,
            invert_control=_float_tuple(
                environ["UR5E_SPACEMOUSE_INVERT_CONTROL"].split(","),
                "UR5E_SPACEMOUSE_INVERT_CONTROL",
            ),
        )
    if "UR5E_SPACEMOUSE_ROTATION_MODE" in environ:
        spacemouse = replace(
            spacemouse,
            rotation_mode=environ["UR5E_SPACEMOUSE_ROTATION_MODE"],
        )
    if "UR5E_SPACEMOUSE_INVERT_BUTTON" in environ:
        spacemouse = replace(
            spacemouse,
            invert_button=_bool_value(
                environ["UR5E_SPACEMOUSE_INVERT_BUTTON"],
                "UR5E_SPACEMOUSE_INVERT_BUTTON",
            ),
        )
    if "UR5E_SPACEMOUSE_GRIPPER_OPEN" in environ:
        spacemouse = replace(
            spacemouse,
            gripper_open=_int_value(
                environ["UR5E_SPACEMOUSE_GRIPPER_OPEN"],
                "UR5E_SPACEMOUSE_GRIPPER_OPEN",
            ),
        )
    if "UR5E_SPACEMOUSE_GRIPPER_CLOSED" in environ:
        spacemouse = replace(
            spacemouse,
            gripper_closed=_int_value(
                environ["UR5E_SPACEMOUSE_GRIPPER_CLOSED"],
                "UR5E_SPACEMOUSE_GRIPPER_CLOSED",
            ),
        )
    agent = replace(agent, spacemouse=spacemouse)

    camera = config.camera
    if "UR5E_CAMERA_HOSTNAME" in environ:
        camera = replace(camera, hostname=environ["UR5E_CAMERA_HOSTNAME"])
    if "UR5E_CAMERA_FPS" in environ:
        camera = replace(
            camera,
            fps=_int_value(environ["UR5E_CAMERA_FPS"], "UR5E_CAMERA_FPS"),
        )
    if "UR5E_CAMERA_ENABLE_DEPTH" in environ:
        enable_depth = _bool_value(
            environ["UR5E_CAMERA_ENABLE_DEPTH"],
            "UR5E_CAMERA_ENABLE_DEPTH",
        )
        camera = replace(
            camera,
            enable_depth=enable_depth,
            depth_max_invalid_ratio=(
                camera.depth_max_invalid_ratio if enable_depth else {}
            ),
            depth_options=(camera.depth_options if enable_depth else {}),
        )
    if "UR5E_CAMERA_CALIBRATION_DIR" in environ:
        camera = replace(
            camera,
            calibration_dir=environ["UR5E_CAMERA_CALIBRATION_DIR"],
        )
    if "UR5E_CAMERA_EXTRINSICS_FILE" in environ:
        camera = replace(
            camera,
            extrinsics_file=environ["UR5E_CAMERA_EXTRINSICS_FILE"],
        )

    gripper = config.gripper
    if "UR5E_PGC_PORT" in environ:
        gripper = replace(gripper, port=environ["UR5E_PGC_PORT"])
    if "UR5E_PGC_POSITION_TOLERANCE" in environ:
        gripper = replace(
            gripper,
            position_tolerance=_int_value(
                environ["UR5E_PGC_POSITION_TOLERANCE"],
                "UR5E_PGC_POSITION_TOLERANCE",
            ),
        )
    if "UR5E_PGC_MIN_COMMAND_INTERVAL_S" in environ:
        gripper = replace(
            gripper,
            min_command_interval_s=_float_value(
                environ["UR5E_PGC_MIN_COMMAND_INTERVAL_S"],
                "UR5E_PGC_MIN_COMMAND_INTERVAL_S",
            ),
        )
    if "UR5E_PGC_POSITION_READ_INTERVAL_S" in environ:
        gripper = replace(
            gripper,
            position_read_interval_s=_float_value(
                environ["UR5E_PGC_POSITION_READ_INTERVAL_S"],
                "UR5E_PGC_POSITION_READ_INTERVAL_S",
            ),
        )

    dataset = config.dataset
    if "UR5E_LEROBOT_REPO_ID" in environ:
        dataset = replace(dataset, lerobot_repo_id=environ["UR5E_LEROBOT_REPO_ID"])
    if "UR5E_LEROBOT_ROOT" in environ:
        dataset = replace(dataset, lerobot_root=environ["UR5E_LEROBOT_ROOT"])
    if "UR5E_LEROBOT_FPS" in environ:
        dataset = replace(
            dataset,
            lerobot_fps=_int_value(environ["UR5E_LEROBOT_FPS"], "UR5E_LEROBOT_FPS"),
        )
    if "UR5E_LEROBOT_ASYNC_RECORDING" in environ:
        dataset = replace(
            dataset,
            lerobot_async_recording=_bool_value(
                environ["UR5E_LEROBOT_ASYNC_RECORDING"],
                "UR5E_LEROBOT_ASYNC_RECORDING",
            ),
        )
    if "UR5E_LEROBOT_RECORD_QUEUE_SIZE" in environ:
        dataset = replace(
            dataset,
            lerobot_record_queue_size=_int_value(
                environ["UR5E_LEROBOT_RECORD_QUEUE_SIZE"],
                "UR5E_LEROBOT_RECORD_QUEUE_SIZE",
            ),
        )
    if "UR5E_LEROBOT_REPAIR_VIDEO_TIMESTAMPS" in environ:
        dataset = replace(
            dataset,
            lerobot_repair_video_timestamps=_bool_value(
                environ["UR5E_LEROBOT_REPAIR_VIDEO_TIMESTAMPS"],
                "UR5E_LEROBOT_REPAIR_VIDEO_TIMESTAMPS",
            ),
        )

    safety = config.safety
    if "UR5E_JOINT_LIMITS_FILE" in environ:
        safety = replace(
            safety,
            joint_limits_file=environ["UR5E_JOINT_LIMITS_FILE"],
        )

    task = config.task
    if "UR5E_TASK_FILE" in environ:
        task = replace(task, file=environ["UR5E_TASK_FILE"])
    if "UR5E_EPISODE_COUNT" in environ:
        task = replace(
            task,
            episode_count=_int_value(
                environ["UR5E_EPISODE_COUNT"],
                "UR5E_EPISODE_COUNT",
            ),
        )

    vla = config.vla
    if "UR5E_VLA_SERVER_URL" in environ:
        vla = replace(vla, server_url=environ["UR5E_VLA_SERVER_URL"])
    if "UR5E_VLA_API_KEY" in environ:
        api_key = environ["UR5E_VLA_API_KEY"]
        vla = replace(vla, api_key=api_key or None)
    if "UR5E_VLA_REPO_ID" in environ:
        vla = replace(vla, repo_id=environ["UR5E_VLA_REPO_ID"])
    if "UR5E_VLA_ACTION_SPACE" in environ:
        vla = replace(vla, action_space=environ["UR5E_VLA_ACTION_SPACE"])
    if "UR5E_VLA_EXECUTE_HORIZON" in environ:
        vla = replace(
            vla,
            execute_horizon=_int_value(
                environ["UR5E_VLA_EXECUTE_HORIZON"],
                "UR5E_VLA_EXECUTE_HORIZON",
            ),
        )
    if "UR5E_VLA_REQUEST_TIMEOUT_S" in environ:
        vla = replace(
            vla,
            request_timeout_s=_float_value(
                environ["UR5E_VLA_REQUEST_TIMEOUT_S"],
                "UR5E_VLA_REQUEST_TIMEOUT_S",
            ),
        )
    if "UR5E_VLA_MAX_JOINT_DELTA" in environ:
        raw_delta = environ["UR5E_VLA_MAX_JOINT_DELTA"].strip()
        vla = replace(
            vla,
            max_joint_delta=(
                None
                if raw_delta.lower() in {"", "none", "null"}
                else _float_value(raw_delta, "UR5E_VLA_MAX_JOINT_DELTA")
            ),
        )
    if "UR5E_VLA_RESET_ON_START" in environ:
        vla = replace(
            vla,
            reset_on_start=_bool_value(
                environ["UR5E_VLA_RESET_ON_START"],
                "UR5E_VLA_RESET_ON_START",
            ),
        )

    return replace(
        config,
        robot=robot,
        agent=agent,
        camera=camera,
        gripper=gripper,
        dataset=dataset,
        safety=safety,
        vla=vla,
        task=task,
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a mapping, got {type(value).__name__}")
    return value


def _string_mapping(value: Any, label: str) -> Mapping[str, str]:
    data = _mapping(value)
    return {str(key): str(item) for key, item in data.items()}


def _float_mapping(value: Any, label: str) -> Mapping[str, float]:
    data = _mapping(value)
    try:
        return {str(key): float(item) for key, item in data.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must map names to numbers") from exc


def _nested_mapping(value: Any, label: str) -> Mapping[str, Mapping[str, Any]]:
    data = _mapping(value)
    try:
        return {str(key): dict(_mapping(item)) for key, item in data.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must map names to mappings") from exc


def _float_tuple(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list of numbers")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a list of numbers") from exc


def _int_value(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc


def _float_value(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc


def _bool_value(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{label} must be a boolean")
