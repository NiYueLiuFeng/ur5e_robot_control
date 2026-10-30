"""Control the UR5e from a remote policy server over WebSocket.

``--model`` selects the wire protocol and the action contract:

- ``json`` (default): JSON text frames with base64 JPEG images.
- ``fastwam``: FastWAM msgpack frames with raw JPEG bytes and no handshake
  (token goes in the URL).
- ``lingbot``: LingBot-VA msgpack frames using the server's ``__ndarray__``
  array encoding. The server sends metadata on connect, then accepts a
  ``reset`` frame and one-frame ``obs`` requests; the client resets before
  every request because the server's streaming VAE caches frames until reset.
- ``openwam``: OpenWAM JSON text frames with base64 PNG images. The server
  needs a ``ping`` / ``reset`` handshake before the first ``obs``.
- ``pi05`` (alias ``openpi05``): OpenPI 0.5 msgpack frames using the server's
  ``__ndarray__`` array encoding.
- ``flexpi``: Flex-π msgpack-numpy frames with the 13-dim HS13 state.

Execution contract: the UR5e accepts absolute 6-joint + gripper commands, so
only 7-value joint actions are sent to the robot. Every action is validated
(finite, joint range, optional ``vla.max_joint_delta`` per-step limit) and the
gripper is binarized to 0/1000 with ``vla.threshold``. Every UR5 checkpoint
returns absolute joint angles, so it drives the arm directly: FastWAM
``autogen_mixed_ur5``, LingBot-VA ``ur5``, OpenWAM ``ur5`` and OpenPI
``pi05`` (``pi05_huashu`` delivers ``(50, 7)`` absolute joints; the server
already undoes normalization and delta->ABS, so the client must not add the
state back). Non-UR5e ``pi05`` configs must not be executed: ``guodi``
(20-dim) and ``xinsong`` (16-dim) fail the 7-dim check, and ``flexiv`` is a
7-DOF arm whose 7 values are not a UR5e 6-joint + gripper command. ``flexpi``
targets a Flexiv arm and delivers the 13-dim HS13 end-effector state, so it
can only run in preview mode (``--dry-run``).

Each loop: read cameras + state -> send one request -> receive one action
chunk -> validate and execute (or preview) the actions. While the loop runs,
pressing N or T moves the arm back to ``robot.reset_joints``. With
``--save-video`` that reset first asks whether to keep the episode's merged
third-person + wrist recording.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import msgpack
import numpy as np

from ur5e_control.data_collection.tasks import load_task_file
from ur5e_control.inference.websocket import connect_websocket
from ur5e_control.runtime.config import load_runtime_config, resolve_repo_path
from ur5e_control.runtime.env import build_robot_env
from ur5e_control.runtime.episode_video import EpisodeVideoRecorder

MODELS = ("json", "fastwam", "pi05", "lingbot", "flexpi", "openwam")
# 旧名字保留，等价于 pi05。
MODEL_ALIASES = {"openpi05": "pi05"}
# 每条 action 的维度：6 个关节角（rad）+ 1 个夹爪值。
ACTION_DIM = 7
# UR5e 关节角的安全读数范围（rad），超出说明服务端返回了别的语义。
JOINT_LIMIT_RAD = 2.0 * math.pi
# FastWAM 的去噪步数。
NUM_INFERENCE_STEPS = 10
# 每个动作块处理完后等待 20 ms，再发起下一次请求。
CHUNK_WAIT_S = 0.02
# json 协议：服务端图片键名 -> RobotEnv.get_obs() 键名。
IMAGE_FIELDS = {"third_person": "third_left_top_rgb", "wrist": "wrist_rgb"}
# fastwam 协议：服务端相机键名 -> RobotEnv.get_obs() 键名。
FASTWAM_IMAGE_FIELDS = {
    "third_person": "third_left_top_rgb",
    "wrist": "wrist_rgb",
}
# pi05 协议：OpenPI 字段名 -> RobotEnv.get_obs() 键名（HWC RGB uint8）。
PI05_IMAGE_FIELDS = {
    "observation.images.third_person": "third_left_top_rgb",
    "observation.images.wrist": "wrist_rgb",
}
# lingbot 协议：ur5 profile 的 obs_cam_keys（见 wan_va/ur5.py）。
LINGBOT_IMAGE_FIELDS = {
    "observation.images.third_person": "third_left_top_rgb",
    "observation.images.wrist": "wrist_rgb",
}
# flexpi 协议：服务端相机名 -> RobotEnv.get_obs() 键名。
FLEXPI_IMAGE_FIELDS = {
    "cam_high": "third_left_top_rgb",
    "cam_left_wrist": "wrist_rgb",
}
# openwam 协议：服务端固定三个图像槽位；单腕机器人没有右腕，发 null。
OPENWAM_IMAGE_FIELDS = {
    "head_camera": "third_left_top_rgb",
    "left_wrist_camera": "wrist_rgb",
    "right_wrist_camera": None,
}


@dataclass(frozen=True)
class ModelSpec:
    """The action contract a protocol/model combination returns."""

    # "joint"：绝对关节角 + 夹爪，UR5e 可直接执行；"eef"：末端位姿，不能下发。
    representation: str
    # 动作对应的采集帧率；None 表示沿用 robot.control_hz 的节奏。
    action_hz: float | None
    note: str

    @property
    def executable(self) -> bool:
        return self.representation == "joint"


MODEL_SPECS = {
    "json": ModelSpec("joint", None, "7 维绝对关节角 + 夹爪"),
    "fastwam": ModelSpec("joint", 15.0, "7 维绝对关节角 + 夹爪"),
    "lingbot": ModelSpec("joint", 15.0, "7 维绝对关节角 + 夹爪"),
    "openwam": ModelSpec("joint", 15.0, "7 维绝对关节角 + 夹爪"),
    "pi05": ModelSpec("joint", 15.0, "7 维绝对关节角 + 夹爪"),
    "flexpi": ModelSpec("eef", 15.0, "13 维 HS13 绝对末端状态"),
}
# 按这些键时让机械臂回到 robot.reset_joints（复位）。
RESET_KEYS = ("n", "t")


class ResetKeyListener:
    """Set a one-shot flag when N or T is pressed, mirroring the teleop listeners."""

    def __init__(self) -> None:
        self._requested = threading.Event()
        self._listener = None

    def start(self) -> None:
        try:
            from pynput import keyboard
        except Exception as exc:  # pynput needs a display on Linux
            print(f"Reset keys disabled: {exc}")
            return

        def on_press(key: Any) -> None:
            char = getattr(key, "char", None)
            if char is not None and char.lower() in RESET_KEYS:
                self._requested.set()

        try:
            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.start()
        except Exception as exc:
            print(f"Reset keys disabled: {exc}")

    def take(self) -> bool:
        """Return True once per reset request."""
        if not self._requested.is_set():
            return False
        self._requested.clear()
        return True

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


def build_parser() -> argparse.ArgumentParser:
    # 命令行参数都是可选的：不填就用 YAML 配置，填了就覆盖 YAML。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Runtime YAML config. Defaults to configs/hardware/ur5e_d455.yaml.",
    )
    parser.add_argument(
        "--model",
        choices=(*MODELS, *MODEL_ALIASES),
        default=None,
        help="Policy protocol: 'json' (default), 'fastwam', 'lingbot', 'openwam', "
        "'pi05' (alias 'openpi05') or 'flexpi'.",
    )
    parser.add_argument("--server-url", default=None, help="Policy WebSocket URL.")
    parser.add_argument("--api-key", default=None, help="Optional policy API key.")
    parser.add_argument("--task", default=None, help="Task text override.")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Stop after this many actions."
    )
    parser.add_argument(
        "--no-wait", action="store_true", help="Skip the ENTER prompt before the loop."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the returned actions without sending them to the robot.",
    )
    parser.add_argument(
        "--action-hz",
        type=float,
        default=None,
        help="Override the action rate (defaults to vla.action_hz or the model's fps).",
    )
    parser.add_argument(
        "--max-joint-delta",
        default=None,
        help="Max change (rad) per joint between two commands; 'none' disables it.",
    )
    parser.add_argument(
        "--save-video",
        nargs="?",
        const=True,
        default=False,
        type=_str2bool,
        metavar="BOOL",
        help="Record one episode's merged third-person + wrist video and ask "
        "whether to keep it when N/T resets the arm.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_runtime_config(args.config)
    # 只取主循环用到的字段，避免后面到处 config.vla.xxx。
    vla = config.vla
    model = _normalize_model(args.model or vla.model)
    # 任务文本优先用 --task，否则读 task.file 里的 task 字段。
    task = args.task or load_task_file(str(resolve_repo_path(config.task.file))).name
    # 不能执行的模型（EEF 位姿）必须在启动机器人进程之前拒绝。
    dry_run = args.dry_run
    _ensure_executable(model, dry_run=dry_run)
    if dry_run:
        print(f"Dry run: {model} actions are printed, never sent to the robot.")

    action_hz = args.action_hz if args.action_hz is not None else vla.action_hz
    max_joint_delta = (
        _optional_float(args.max_joint_delta)
        if args.max_joint_delta is not None
        else vla.max_joint_delta
    )
    # 视频只在真机执行时录制；目录在录到第一帧时才创建。
    video = (
        EpisodeVideoRecorder(model=model, fps=action_hz or 15.0)
        if args.save_video and not dry_run
        else None
    )

    # 1. 启动机器人 ZMQ 子进程和 RealSense 相机服务。
    bundle = build_robot_env(config)

    # 2. 连接策略服务端；整轮控制复用这一条长连接。
    #    FastWAM 不走 Authorization 头，token 放在 URL 查询参数里。
    server_url = args.server_url or vla.server_url
    api_key = args.api_key or vla.api_key
    if model == "fastwam":
        connection = connect_websocket(
            _url_with_token(server_url, api_key),
            None,
            vla.request_timeout_s,
        )
    else:
        connection = connect_websocket(
            server_url,
            api_key,
            vla.request_timeout_s,
        )
    try:
        # 3. 需要握手的服务先完成元数据/探活，避免动作语义不符时机器人已经动了。
        _prepare_server(connection, model=model, task=task)
        # 4. 先分小步移动到初始位姿，避免上电后一次大幅跳变。
        if not dry_run:
            _move_to_start(bundle.env, config.robot.reset_joints)
            if not args.no_wait:
                # 停顿一下，留给操作员确认现场并把手放到急停上。
                input("press ENTER to start policy control loop...")
        print(
            f"Starting UR5e policy control ({model}). "
            "Press Ctrl+C to stop; press N or T to return to the start pose."
        )
        steps = _run(
            bundle.env,
            connection,
            model=model,
            vla=vla,
            task=task,
            max_steps=args.max_steps,
            control_hz=config.robot.control_hz,
            action_hz=action_hz,
            max_joint_delta=max_joint_delta,
            reset_joints=config.robot.reset_joints,
            video=video,
            dry_run=dry_run,
        )
        print(f"Stopped after handling {steps} action(s).")
    except KeyboardInterrupt:
        print("Stopping UR5e policy control.")
    finally:
        # 未按 y 保存的录像在退出时删除，避免留下半截 mp4。
        if video is not None:
            video.discard()
        # 正常结束、报错或 Ctrl+C 都要断开连接并释放机器人/相机进程。
        connection.close()
        bundle.close()


def _run(
    env: Any,
    connection: Any,
    *,
    model: str,
    vla: Any,
    task: str,
    max_steps: int | None,
    control_hz: float,
    action_hz: float | None = None,
    max_joint_delta: float | None = None,
    reset_joints: Any = None,
    video: EpisodeVideoRecorder | None = None,
    dry_run: bool = False,
) -> int:
    """Run the request loop and return how many actions were handled."""

    model = _normalize_model(model)
    # 即使调用方直接进 _run，也不允许把 EEF 位姿当关节角下发。
    if not dry_run:
        _ensure_executable(model, dry_run=False)
    spec = MODEL_SPECS[model]
    if action_hz is None:
        action_hz = spec.action_hz
    period = 1.0 / action_hz if action_hz else 0.0

    executed = 0
    request_id = 0
    # 按 N/T 回到 robot.reset_joints；dry-run 不动机器人，不需要监听。
    reset_key = ResetKeyListener()
    if reset_joints is not None and not dry_run:
        reset_key.start()
    try:
        while max_steps is None or executed < max_steps:
            if reset_key.take():
                _handle_reset(env, reset_joints, video)
            # 每轮重新取一次观测，交给 get_model_body 组装请求。
            obs = env.get_obs()
            # 限幅参考：本轮开始时的实际关节角，逐条下发后更新为刚下发的目标。
            reference = np.asarray(obs["joint_positions"], dtype=float).reshape(-1)[:6]
            body = get_model_body(
                obs,
                model=model,
                repo_id=vla.repo_id,
                task=task,
                request_id=request_id,
            )
            if model == "lingbot":
                # 服务端的 streaming VAE 缓存只在 reset 时清空，而每块都要重新
                # 编码当前观测；不 reset 的话第二块会报卷积核大于输入。
                _reset_lingbot(connection, task, verbose=False)
            # 一问一答：观测发给服务端，拿回一个动作块。
            response = _request(connection, body, model=model)
            actions = _parse_actions(response, model=model)
            # 单臂 UR5 的动作必须是 7 维；dry-run 只告警，真机执行前直接报错。
            _check_action_shape(actions, model=model, warn_only=dry_run)
            shape_ok = actions.shape[1] == ACTION_DIM
            # 只处理前 execute_horizon 条，剩余动作丢弃，下一轮重新请求。
            actions = actions[: vla.execute_horizon]
            if model != "json" and len(actions):
                print(
                    f"{model} actions: shape={actions.shape} "
                    f"first={np.round(actions[0], 4).tolist()}"
                )
            for action in actions:
                if reset_key.take():
                    # 复位后丢弃剩下的动作，下一轮按新位姿重新请求。
                    _handle_reset(env, reset_joints, video)
                    break
                started = time.monotonic()
                if dry_run:
                    # 预览模式：不碰机器人，只按同样的节奏维持循环。
                    if spec.executable and shape_ok:
                        command = _robot_command(
                            action,
                            vla.threshold,
                            reference=reference,
                            max_joint_delta=max_joint_delta,
                        )
                        reference = command[:6].copy()
                        preview = f"command={np.round(command, 4).tolist()}"
                    else:
                        preview = (
                            "action="
                            f"{np.round(np.asarray(action, dtype=float).reshape(-1), 4).tolist()}"
                        )
                    print(f"  dry-run {executed}: {preview}")
                    time.sleep(period or 1.0 / control_hz)
                else:
                    command = _robot_command(
                        action,
                        vla.threshold,
                        reference=reference,
                        max_joint_delta=max_joint_delta,
                    )
                    # RobotEnv.step 会做安全限位，并按 control_hz 控制下发节奏。
                    env.step(command)
                    if video is not None:
                        # 每条动作后取一帧，recorder 内部按 fps 抽帧。
                        video.add(env.get_obs())
                    if period:
                        # 模型动作按数据集帧率给出，不能按 control_hz 抢跑。
                        remaining = period - (time.monotonic() - started)
                        if remaining > 0:
                            time.sleep(remaining)
                    reference = command[:6].copy()
                executed += 1
                if max_steps is not None and executed >= max_steps:
                    break
            request_id += 1
            # 每个动作块处理完停 20 ms，再向服务端请求下一个 chunk。
            time.sleep(CHUNK_WAIT_S)
    finally:
        reset_key.stop()
    return executed


def get_model_body(
    obs: dict[str, Any],
    *,
    model: str,
    task: str,
    repo_id: str = "",
    request_id: int = 0,
    jpeg_quality: int = 90,
) -> dict[str, Any]:
    """返回固定格式的请求 body。

    图片键名分别是 `IMAGE_FIELDS`、`FASTWAM_IMAGE_FIELDS`、`PI05_IMAGE_FIELDS`、
    `LINGBOT_IMAGE_FIELDS`、`FLEXPI_IMAGE_FIELDS` 和 `OPENWAM_IMAGE_FIELDS`。
    换模型要改字段时，直接改这几个常量。
    """

    model = _normalize_model(model)
    # 7 维：6 个关节角 + 夹爪状态；与相机协议无关。
    state = np.asarray(obs["joint_positions"], dtype=float).reshape(-1).tolist()

    if model == "pi05":
        # OpenPI 0.5 直接收发 numpy 数组（走 msgpack-numpy），不走 JPEG。
        # 字段名里的点号是字典键名本身，不能改成嵌套对象。
        body = {
            "observation.state": np.asarray(
                obs["joint_positions"], dtype=np.float32
            ).reshape(-1),
            "prompt": task,
        }
        for name, obs_key in PI05_IMAGE_FIELDS.items():
            body[name] = _to_hwc_uint8(obs[obs_key])
        return body

    if model == "lingbot":
        # LingBot-VA 的生成请求：obs 是 {cam_key: HWC RGB uint8}。
        return {
            "obs": {
                name: _to_hwc_uint8(obs[obs_key])
                for name, obs_key in LINGBOT_IMAGE_FIELDS.items()
            },
            "prompt": task,
        }

    if model == "flexpi":
        # Flex-π 需要 13 维 HS13 状态（位置、姿态、夹爪宽度、力、力矩）。
        return {
            "state": _flexpi_state(obs),
            "images": {
                name: _to_hwc_uint8(obs[obs_key])
                for name, obs_key in FLEXPI_IMAGE_FIELDS.items()
            },
            "prompt": task,
        }

    if model == "openwam":
        # OpenWAM 是 JSON 文本帧；图像是 base64 PNG 裸串，缺失的相机发 null。
        return {
            "type": "obs",
            "images": {
                name: (None if obs_key is None else _encode_png_b64(obs[obs_key]))
                for name, obs_key in OPENWAM_IMAGE_FIELDS.items()
            },
            "prompt": task,
            "state": state,
        }

    if model == "fastwam":
        # 参考 FastWAM 的 infer 帧：图片是原始 JPEG 字节，不是 base64。
        return {
            "v": 1,
            "type": "infer",
            "id": f"req-{request_id}",
            "prompt": task,
            "images": [
                {
                    "key": key,
                    "format": "jpeg",
                    "data": _encode_jpeg_bytes(obs[obs_key], jpeg_quality),
                }
                for key, obs_key in FASTWAM_IMAGE_FIELDS.items()
            ],
            "state": state,
            "options": {"num_inference_steps": NUM_INFERENCE_STEPS},
        }

    return {
        "repo_id": repo_id,  # 服务端用哪个数据集/策略
        "task": task,  # 自然语言任务描述
        "state": state,
        # 每路相机编码成 base64 JPEG，键名见 IMAGE_FIELDS。
        "images": {
            name: _encode_jpeg(obs[key], jpeg_quality)
            for name, key in IMAGE_FIELDS.items()
        },
    }


def _to_hwc_uint8(image: Any) -> np.ndarray:
    """Validate an RGB `HxWx3` frame and return a contiguous uint8 copy."""

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB image, got shape {rgb.shape}")
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _flexpi_state(obs: dict[str, Any]) -> np.ndarray:
    """Assemble the 13-dim HS13 state Flex-π expects (preview only).

    HS13 is `[pos xyz, rot abc, gripper width, force xyz, torque xy/yz/zx]`.
    The UR5e exposes the pose as an axis-angle TCP pose and the gripper as a
    0..1000 PGC position, so this vector is only representative: the returned
    Flex-π actions target a Flexiv arm and are never sent to the UR5e.
    """

    tcp_pose = np.asarray(obs["tcp_pose"], dtype=float).reshape(-1)
    gripper = np.asarray(obs["gripper_position_raw"], dtype=float).reshape(-1)
    tcp_force = np.asarray(obs["tcp_force"], dtype=float).reshape(-1)
    if tcp_pose.size != 6 or tcp_force.size != 6 or gripper.size != 1:
        raise ValueError(
            "Flex-π state needs a 6-value tcp_pose, 6-value tcp_force and "
            "one gripper_position_raw value"
        )
    return np.concatenate([tcp_pose, gripper, tcp_force]).astype(np.float32)


def _request(connection: Any, body: dict[str, Any], *, model: str) -> dict[str, Any]:
    """Send one request frame and return the parsed response frame."""

    model = _normalize_model(model)

    if model in {"lingbot", "pi05"}:
        # 这两类服务端都用纯 msgpack 收包，数组走它们自己的 `__ndarray__` 帧，
        # 不是 msgpack-numpy；发 msgpack-numpy 数组时服务端会把它当成普通 dict。
        connection.send(_pack_ndarray(body))
        return _unpack_ndarray(connection.recv())

    if model == "flexpi":
        # Flex-π 直接序列化 numpy 数组，需要 msgpack-numpy 编解码。
        connection.send(_pack_numpy(body))
        return _unpack_numpy(connection.recv())

    if model == "fastwam":
        connection.send(_pack(body))
        return _recv_action_frame(connection)

    connection.send(json.dumps(body))
    response = json.loads(connection.recv())
    # 服务端约定：返回必须是 JSON 对象；{"ok": false, "error": ...} 表示失败。
    if not isinstance(response, dict):
        raise RuntimeError(
            f"policy server returned {type(response).__name__}, expected an object"
        )
    if response.get("ok") is False:
        raise RuntimeError(str(response.get("error", "policy server error")))
    if model == "openwam":
        # OpenWAM 的错误帧不会断开连接，但本次请求没有动作可用。
        if response.get("type") == "error":
            raise RuntimeError(
                f"{response.get('code', 'ERROR')}: {response.get('message', '')}"
            )
        if response.get("type") != "action":
            raise RuntimeError(
                f"unexpected OpenWAM frame type: {response.get('type')!r}"
            )
    return response


def _prepare_server(connection: Any, *, model: str, task: str) -> None:
    """Consume connect metadata and run the handshake a server requires."""

    model = _normalize_model(model)

    if model == "openwam":
        # ping 带回 representation / gripper_convention，用来拒绝 EEF 数据集。
        _send_json(connection, {"type": "ping"})
        pong = _read_json(connection)
        if pong.get("type") != "pong":
            raise RuntimeError(f"OpenWAM ping returned {pong!r}, expected pong")
        representation = pong.get("representation")
        print(
            "OpenWAM server: "
            f"representation={representation!r} "
            f"gripper_convention={pong.get('gripper_convention')!r}"
        )
        if representation != "joint":
            raise RuntimeError(
                "OpenWAM server reports representation="
                f"{representation!r}; the UR5e needs the 'ur5' joint dataset"
            )
        # 每个 episode 必须先 reset，服务端清空动作缓冲后回 reset_ack。
        _send_json(connection, {"type": "reset"})
        ack = _read_json(connection)
        if ack.get("type") != "reset_ack":
            raise RuntimeError(f"OpenWAM reset returned {ack!r}, expected reset_ack")
        return

    if model == "lingbot":
        # 连上后第一条是 metadata（frame_chunk_size 等），随后 reset 清理历史。
        # 该服务端收到的数组同样走 `__ndarray__` 编码，见 `_request`。
        metadata = _unpack_ndarray(connection.recv())
        print(f"LingBot-VA metadata: {metadata}")
        _reset_lingbot(connection, task)
        return

    if model == "pi05":
        # OpenPI 服务连接后先发一条 metadata，必须先消费掉。
        metadata = _unpack_ndarray(connection.recv())
        print(f"{model} server metadata: {metadata}")
        return

    if model == "flexpi":
        metadata = _unpack_numpy(connection.recv())
        print(f"{model} server metadata: {metadata}")


def _reset_lingbot(connection: Any, task: str, *, verbose: bool = True) -> None:
    """Clear the LingBot-VA streaming state before the next chunk.

    The server only encodes the observation through its streaming VAE while
    ``frame_st_id`` is 0, and that VAE keeps its per-frame cache between
    calls. A second observation request without an intervening reset reuses
    the cache with a single-frame chunk and dies in ``time_conv`` with
    ``Kernel size can't be greater than actual input size``. Resetting makes
    every chunk an independent first chunk, matching this loop's design where
    each chunk is re-inferred from the current observation.
    """

    connection.send(_pack_ndarray({"reset": True, "prompt": task}))
    response = _unpack_ndarray(connection.recv())
    # 新版服务端的 reset 回包会附带 server_timing（如 infer_ms），这不是错误。
    unexpected = {key: value for key, value in response.items() if key != "server_timing"}
    if unexpected:
        raise RuntimeError(
            "LingBot-VA reset returned unexpected keys "
            f"{sorted(unexpected)}, expected {{}} or server_timing"
        )
    if verbose and response:
        print(f"LingBot-VA reset: {response}")


def _parse_actions(response: dict[str, Any], *, model: str) -> np.ndarray:
    """Read the server action chunk into a validated `(N, D)` array."""

    model = _normalize_model(model)

    if model == "openwam":
        frame = response
        if frame.get("type") != "action":
            raise RuntimeError(f"unexpected OpenWAM frame type: {frame.get('type')!r}")
        actions = np.asarray(frame.get("action", []), dtype=float)
        if actions.ndim == 1:
            actions = actions[None, :]
    elif model == "lingbot":
        actions = _lingbot_chunk(response)
    elif model in {"fastwam", "pi05", "flexpi"}:
        if "actions" in response:
            actions = np.asarray(response["actions"], dtype=float)
        elif "action" in response:
            actions = np.asarray(response["action"], dtype=float)
        else:
            raise KeyError("policy response is missing 'actions'")
        if actions.ndim == 1:
            actions = actions[None, :]
    else:
        if "action" not in response:
            raise KeyError("policy response is missing 'action'")
        actions = np.asarray(response["action"], dtype=float)
        if actions.ndim == 1:
            # 服务端只返回一条 action 时，也补成 (1, 7) 方便统一处理。
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(
                "policy response 'action' must be a list of "
                f"{ACTION_DIM}-value actions, got shape {actions.shape}"
            )

    if actions.ndim != 2:
        raise ValueError(
            f"policy response actions must be 2-D, got shape {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("policy response actions must contain finite values")
    return actions


def _lingbot_chunk(response: dict[str, Any]) -> np.ndarray:
    """Flatten LingBot-VA's `(C, chunk, actions_per_frame)` action block.

    The first latent frame is the model's zero context block, not a prediction
    (``align_actions`` leaves it empty during training and ``_infer`` overwrites
    it with ``action_cond``). Because this client re-infers from the current
    observation on every request, that block must be dropped before execution.
    """

    if "action" not in response:
        raise KeyError("LingBot-VA response is missing 'action'")
    raw = np.asarray(response["action"], dtype=float)
    if raw.ndim == 2:
        return raw
    if raw.ndim != 3:
        raise ValueError(f"LingBot-VA action must be 2-D or 3-D, got {raw.shape}")

    # (action_dim, latent_frames, actions_per_frame) -> (N, action_dim)
    dim, frames, per_frame = raw.shape
    actions = raw.transpose(1, 2, 0).reshape(frames * per_frame, dim)
    # 丢掉第 0 帧（per_frame 条）的上下文块。
    actions = np.ascontiguousarray(actions[per_frame:])
    if not actions.size:
        raise RuntimeError(
            "LingBot-VA returned only the context block; "
            "check frame_chunk_size and the checkpoint"
        )
    return actions


def _check_action_shape(
    actions: np.ndarray, *, model: str, warn_only: bool = False
) -> None:
    """Warn unless the chunk is `(N, 7)`: single-arm UR5 = 6 joints + gripper."""

    if actions.shape[1] == ACTION_DIM:
        return
    hint = ""
    if model == "fastwam":
        hint = (
            " (the charger/whiteboard checkpoint is 13-dim HS13; load "
            "autogen_mixed_ur5 for the UR5e)"
        )
    elif model == "lingbot":
        hint = " (load the LingBot-VA 'ur5' profile/checkpoint)"
    elif model == "openwam":
        hint = " (load the OpenWAM 'ur5' dataset/checkpoint)"
    elif model == "pi05":
        hint = (
            " (load the UR5e checkpoint OpenPI pi05_huashu; flexiv is a 7-DOF "
            "arm and guodi/xinsong are multi-arm)"
        )
    elif model == "flexpi":
        hint = " (Flex-π returns the 13-dim HS13 state)"
    message = (
        f"{model}: expected {ACTION_DIM}-dim single-arm UR5 actions "
        f"(6 joints + gripper), got {actions.shape[1]}{hint}"
    )
    # dry-run 只预览，告警后继续；真正下发前必须停止，否则会下发错误语义。
    if warn_only:
        print(f"WARNING: {message}")
        return
    raise ValueError(message)


def _ensure_executable(model: str, *, dry_run: bool) -> None:
    """Refuse to drive the UR5e with actions that are not joint commands."""

    spec = MODEL_SPECS[model]
    if spec.executable or dry_run:
        return
    raise ValueError(
        f"model {model!r} returns {spec.note}, but the UR5e executes only "
        "absolute 6-joint + gripper commands. Use a UR5 checkpoint (FastWAM "
        "autogen_mixed_ur5, LingBot-VA ur5, OpenWAM ur5, OpenPI pi05_huashu) "
        "or pass --dry-run to inspect the actions without moving the robot."
    )


def _url_with_token(url: str, token: str | None) -> str:
    """Append the FastWAM token as a query parameter."""

    if not token:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}token={quote(token)}"


def _recv_action_frame(connection: Any) -> dict[str, Any]:
    """Read frames until `actions` or `error`, ignoring pong/reset_ack."""

    while True:
        frame = _unpack(connection.recv())
        kind = frame.get("type")
        if kind == "actions":
            return frame
        if kind == "error":
            raise RuntimeError(
                f"{frame.get('code', 'ERROR')}: {frame.get('message', '')} "
                f"{frame.get('detail') or ''}".strip()
            )
        if kind not in {"pong", "reset_ack"}:
            raise RuntimeError(f"unexpected FastWAM frame type: {kind!r}")


def _pack(frame: dict[str, Any]) -> bytes:
    return msgpack.packb(frame, use_bin_type=True)


def _unpack(data: Any) -> dict[str, Any]:
    if isinstance(data, str):
        raise RuntimeError(f"FastWAM server returned text: {data}")
    frame = msgpack.unpackb(data, raw=False, strict_map_key=False)
    if not isinstance(frame, dict):
        raise RuntimeError(f"FastWAM server returned {type(frame).__name__}")
    return frame


def _pack_ndarray(frame: dict[str, Any]) -> bytes:
    """Pack a frame using the OpenPI server's ``__ndarray__`` array encoding.

    The server unpacks with plain msgpack, so msgpack-numpy arrays arrive as
    ordinary dicts there. Its own format is a map with bytes keys holding the
    raw buffer, dtype and shape.
    """

    return msgpack.packb(frame, default=_encode_ndarray, use_bin_type=True)


def _encode_ndarray(array: np.ndarray) -> dict[bytes, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        b"__ndarray__": True,
        b"data": contiguous.tobytes(),
        b"dtype": contiguous.dtype.str,
        b"shape": list(contiguous.shape),
    }


def _unpack_ndarray(data: Any) -> dict[str, Any]:
    if isinstance(data, str):
        raise RuntimeError(f"policy server returned text: {data}")
    frame = msgpack.unpackb(
        data, raw=False, strict_map_key=False, object_hook=_decode_ndarray
    )
    if not isinstance(frame, dict):
        raise RuntimeError(f"policy server returned {type(frame).__name__}")
    return frame


def _decode_ndarray(obj: dict[Any, Any]) -> Any:
    """Rebuild arrays from this server's ``__ndarray__`` or msgpack-numpy maps."""

    def field(*names: str) -> Any:
        for name in names:
            for key in (name, name.encode()):
                if key in obj:
                    return obj[key]
        return None

    if field("__ndarray__"):
        dtype, shape = field("dtype"), field("shape")
    elif field("nd"):
        # 兼容上游 OpenPI / msgpack-numpy 的数组格式。
        dtype, shape = field("type"), field("shape")
    else:
        return obj
    if isinstance(dtype, bytes):
        dtype = dtype.decode()
    return np.frombuffer(field("data"), dtype=np.dtype(dtype)).reshape(shape)


def _pack_numpy(frame: dict[str, Any]) -> bytes:
    import msgpack_numpy

    return msgpack_numpy.packb(frame, use_bin_type=True)


def _unpack_numpy(data: Any) -> dict[str, Any]:
    import msgpack_numpy

    if isinstance(data, str):
        raise RuntimeError(f"policy server returned text: {data}")
    frame = msgpack_numpy.unpackb(data, raw=False, strict_map_key=False)
    if not isinstance(frame, dict):
        raise RuntimeError(f"policy server returned {type(frame).__name__}")
    return frame


def _send_json(connection: Any, frame: dict[str, Any]) -> None:
    connection.send(json.dumps(frame))


def _read_json(connection: Any) -> dict[str, Any]:
    frame = json.loads(connection.recv())
    if not isinstance(frame, dict):
        raise RuntimeError(f"policy server returned {type(frame).__name__}")
    return frame


def _encode_jpeg_bytes(image: Any, quality: int = 90) -> bytes:
    """Encode an RGB `HxWx3` frame as JPEG bytes."""

    return _encode_image_bytes(image, ".jpg", quality)


def _encode_jpeg(image: Any, quality: int = 90) -> str:
    """Encode an RGB `HxWx3` frame as a base64 JPEG string."""

    return base64.b64encode(_encode_jpeg_bytes(image, quality)).decode("ascii")


def _encode_png_b64(image: Any) -> str:
    """Encode an RGB `HxWx3` frame as a base64 PNG string (OpenWAM)."""

    return base64.b64encode(_encode_image_bytes(image, ".png")).decode("ascii")


def _encode_image_bytes(image: Any, extension: str, quality: int = 90) -> bytes:
    import cv2  # imported lazily to keep module imports hardware-free

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB image, got shape {rgb.shape}")

    # cv2 的编码器按 BGR 处理，相机给的是 RGB，这里翻转通道。
    bgr = np.ascontiguousarray(rgb[:, :, ::-1], dtype=np.uint8)
    params = (
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)] if extension == ".jpg" else []
    )
    ok, buffer = cv2.imencode(extension, bgr, params)
    if not ok:
        raise RuntimeError(f"failed to encode camera image as {extension}")
    return buffer.tobytes()


def _robot_command(
    action: np.ndarray,
    threshold: float | None,
    *,
    reference: np.ndarray | None = None,
    max_joint_delta: float | None = None,
) -> np.ndarray:
    """Convert a policy action into a UR5e joint command (gripper 0 or 1000).

    Only absolute 6-joint + gripper actions are accepted: the UR5e is commanded
    with ``servoJ``, so an EEF pose would be silently executed as joint angles.
    """

    command = np.asarray(action, dtype=float).reshape(-1).copy()
    if command.shape != (ACTION_DIM,):
        # fastwam 的 action 维度以服务端为准，下发前必须确认是 6 关节 + 夹爪。
        raise ValueError(
            f"expected a {ACTION_DIM}-value action, got shape {command.shape}"
        )
    if not np.isfinite(command).all():
        raise ValueError("policy action must contain finite values")
    if np.any(np.abs(command[:6]) > JOINT_LIMIT_RAD):
        raise ValueError(
            "policy action is not a UR5e joint command: joint values must be "
            f"within +/-{JOINT_LIMIT_RAD:.3f} rad, got {command[:6].tolist()}"
        )

    if reference is not None and max_joint_delta is not None:
        if max_joint_delta < 0:
            raise ValueError("max_joint_delta must be non-negative")
        reference = np.asarray(reference, dtype=float).reshape(-1)[:6]
        unclamped = command[:6].copy()
        command[:6] = np.clip(
            command[:6],
            reference - float(max_joint_delta),
            reference + float(max_joint_delta),
        )
        if not np.array_equal(command[:6], unclamped):
            print(
                "Joint target clamped by max_joint_delta: "
                f"requested={np.round(unclamped, 4).tolist()} "
                f"commanded={np.round(command[:6], 4).tolist()}"
            )

    # 服务端第 7 维可能是 0~1 概率或 0~1000 原始值，统一二值化成 0/1000。
    limit = 0.5 if threshold is None else float(threshold)
    command[6] = 1000.0 if command[6] >= limit else 0.0
    return command


def _handle_reset(
    env: Any,
    reset_joints: Any,
    video: EpisodeVideoRecorder | None,
) -> None:
    """Ask whether to keep the recorded episode, then re-home the arm."""

    if video is not None and video.has_frames:
        try:
            answer = input("Save the recorded episode video? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer in {"y", "yes"}:
            video.save()
        else:
            video.discard()
    _reset_to_start(env, reset_joints)


def _reset_to_start(env: Any, reset_joints: Any) -> None:
    """Move the arm back to `robot.reset_joints` (N/T key)."""

    print("Reset key pressed: returning to the start pose.")
    _move_to_start(env, reset_joints)


def _move_to_start(env: Any, joints: Any, steps: int = 100) -> None:
    """Move the arm to the configured start pose in small joint steps."""

    target = np.asarray(joints, dtype=float)
    current = np.asarray(env.get_obs()["joint_positions"], dtype=float)
    if current.shape != target.shape:
        # 维度对不上（例如夹爪未接入）时不移动，避免下发错误指令。
        return
    # 在当前位置和目标位姿之间线性插值，分 steps 个小步走过去。
    for point in np.linspace(current, target, steps):
        env.step(point)


def _normalize_model(model: str) -> str:
    normalized = str(model).strip().lower()
    return MODEL_ALIASES.get(normalized, normalized)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def _str2bool(value: Any) -> bool:
    """Let `--save-video true` work like `--save-video`."""

    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


if __name__ == "__main__":
    main()
