"""Control the UR5e from a remote policy server over WebSocket.

Three protocols are supported, selected with ``--model``:

- ``json`` (default): JSON text frames with base64 JPEG images.
- ``fastwam``: FastWAM msgpack frames with raw JPEG bytes and no handshake
  (token goes in the URL). Actions are parsed and logged but NOT sent to the
  robot.
- ``openpi05``: OpenPI 0.5 msgpack-numpy frames with CHW image arrays. Actions
  are parsed and logged but NOT sent to the robot.

Each loop: read cameras + state -> send one request -> receive one action
chunk -> execute (json) or log (fastwam/openpi05) the actions.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from typing import Any
from urllib.parse import quote

import msgpack
import numpy as np

from ur5e_control.data_collection.tasks import load_task_file
from ur5e_control.inference.websocket import connect_websocket
from ur5e_control.runtime.config import load_runtime_config, resolve_repo_path
from ur5e_control.runtime.env import build_robot_env

MODELS = ("json", "fastwam", "openpi05")
# 每条 json action 的维度：6 个关节角（rad）+ 1 个夹爪值。
ACTION_DIM = 7
# FastWAM 的去噪步数。
NUM_INFERENCE_STEPS = 10
# json 协议：服务端图片键名 -> RobotEnv.get_obs() 键名。
IMAGE_FIELDS = {"third_person": "third_left_top_rgb", "wrist": "wrist_rgb"}
# fastwam 协议：服务端相机键名 -> RobotEnv.get_obs() 键名。
FASTWAM_IMAGE_FIELDS = {"front": "third_left_top_rgb", "wrist": "wrist_rgb"}
# openpi05 协议：服务端相机键名 -> RobotEnv.get_obs() 键名。
# 单腕机器人没有左右两路腕部相机，两个键共用同一路 wrist。
OPENPI_IMAGE_FIELDS = {
    "cam_high": "third_left_top_rgb",
    "cam_left_wrist": "wrist_rgb",
    "cam_right_wrist": "wrist_rgb",
}


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
        choices=MODELS,
        default=None,
        help="Policy protocol: 'json' (default), 'fastwam' or 'openpi05'.",
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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_runtime_config(args.config)
    # 只取主循环用到的字段，避免后面到处 config.vla.xxx。
    vla = config.vla
    model = args.model or vla.model
    api_key = args.api_key or vla.api_key
    # 任务文本优先用 --task，否则读 task.file 里的 task 字段。
    task = args.task or load_task_file(str(resolve_repo_path(config.task.file))).name

    # 1. 启动机器人 ZMQ 子进程和 RealSense 相机服务。
    bundle = build_robot_env(config)

    # 2. 连接策略服务端；整轮控制复用这一条长连接。
    #    FastWAM 不走 Authorization 头，token 放在 URL 查询参数里。
    server_url = args.server_url or vla.server_url
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
        # 3. 先分小步移动到初始位姿，避免上电后一次大幅跳变。
        _move_to_start(bundle.env, config.robot.reset_joints)
        if not args.no_wait:
            # 停顿一下，留给操作员确认现场并把手放到急停上。
            input("press ENTER to start policy control loop...")
        print(f"Starting UR5e policy control ({model}). Press Ctrl+C to stop.")
        steps = _run(
            bundle.env,
            connection,
            model=model,
            vla=vla,
            task=task,
            max_steps=args.max_steps,
            control_hz=config.robot.control_hz,
        )
        print(f"Stopped after handling {steps} action(s).")
    except KeyboardInterrupt:
        print("Stopping UR5e policy control.")
    finally:
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
) -> int:
    """Run the request loop and return how many actions were handled."""

    executed = 0
    request_id = 0
    while max_steps is None or executed < max_steps:
        # 每轮重新取一次观测，交给 get_model_body 组装请求。
        body = get_model_body(
            env.get_obs(),
            model=model,
            repo_id=vla.repo_id,
            task=task,
            request_id=request_id,
        )
        # 一问一答：观测发给服务端，拿回一个动作块。
        response = _request(connection, body, model=model)
        # 只处理前 execute_horizon 条，剩余动作丢弃，下一轮重新请求。
        actions = _parse_actions(response, model=model)[: vla.execute_horizon]
        if model == "fastwam" and len(actions):
            # 仅解析模式：打印形状和第一条，方便确认链路通了。
            print(
                f"FastWAM actions: shape={actions.shape} "
                f"first={np.round(actions[0], 4).tolist()}"
            )
        elif model == "openpi05" and len(actions):
            print(
                f"OpenPI05 actions: shape={actions.shape} "
                f"first={np.round(actions[0], 4).tolist()}"
            )
        for action in actions:
            if model != "json":
                # 非 json 协议仅解析，不向机器人下发动作；按控制周期维持节奏。
                time.sleep(1.0 / control_hz)
            else:
                # RobotEnv.step 会做安全限位，并按 control_hz 控制下发节奏。
                env.step(_robot_command(action, vla.threshold))
            executed += 1
            if max_steps is not None and executed >= max_steps:
                break
        request_id += 1
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
    """返回固定格式的请求 body：json、fastwam 或 openpi05。

    图片键名分别是 `IMAGE_FIELDS`、`FASTWAM_IMAGE_FIELDS` 和
    `OPENPI_IMAGE_FIELDS`。换模型要改字段时，直接改这几个常量。
    """

    # 7 维：6 个关节角 + 夹爪状态；与相机协议无关。
    state = np.asarray(obs["joint_positions"], dtype=float).reshape(-1).tolist()

    if model == "openpi05":
        # OpenPI 0.5 直接收发 numpy 数组（走 msgpack-numpy），不走 JPEG。
        return {
            "images": {
                name: _to_chw_uint8(obs[obs_key])
                for name, obs_key in OPENPI_IMAGE_FIELDS.items()
            },
            "state": np.asarray(obs["joint_positions"], dtype=np.float32).reshape(-1),
            "prompt": task,
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


def _to_chw_uint8(image: Any) -> np.ndarray:
    """Convert an RGB `HxWx3` frame to OpenPI's CHW uint8 layout."""

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB image, got shape {rgb.shape}")
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.uint8, copy=False)


def _request(connection: Any, body: dict[str, Any], *, model: str) -> dict[str, Any]:
    """Send one request frame and return the parsed response frame."""

    if model == "openpi05":
        # OpenPI 的 msgpack 帧里是 numpy 数组，需要 msgpack-numpy 编解码。
        import msgpack_numpy

        connection.send(msgpack_numpy.packb(body, use_bin_type=True))
        response = msgpack_numpy.unpackb(connection.recv())
        if not isinstance(response, dict):
            raise RuntimeError(
                f"policy server returned {type(response).__name__}, expected an object"
            )
        return response

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
    return response


def _parse_actions(response: dict[str, Any], *, model: str) -> np.ndarray:
    """Read the server action chunk into a validated `(N, D)` array."""

    if model in {"fastwam", "openpi05"}:
        # 维度由服务端模型决定，这里不写死。
        actions = np.asarray(response["actions"], dtype=float)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2:
            raise ValueError(
                f"policy response 'actions' must be 2-D, got shape {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("policy response 'actions' must contain finite values")
        return actions

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
    if not np.isfinite(actions).all():
        raise ValueError("policy response 'action' must contain finite values")
    return actions


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


def _encode_jpeg_bytes(image: Any, quality: int = 90) -> bytes:
    """Encode an RGB `HxWx3` frame as JPEG bytes."""

    import cv2  # imported lazily to keep module imports hardware-free

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB image, got shape {rgb.shape}")

    # cv2 的 JPEG 编码器按 BGR 处理，相机给的是 RGB，这里翻转通道。
    bgr = np.ascontiguousarray(rgb[:, :, ::-1], dtype=np.uint8)
    ok, buffer = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("failed to encode camera image as JPEG")
    return buffer.tobytes()


def _encode_jpeg(image: Any, quality: int = 90) -> str:
    """Encode an RGB `HxWx3` frame as a base64 JPEG string."""

    return base64.b64encode(_encode_jpeg_bytes(image, quality)).decode("ascii")


def _robot_command(action: np.ndarray, threshold: float | None) -> np.ndarray:
    """Convert a policy action into a UR5e joint command (gripper 0 or 1000)."""

    command = np.asarray(action, dtype=float).copy()
    # 服务端第 7 维可能是 0~1 概率或 0~1000 原始值，统一二值化成 0/1000。
    limit = 0.5 if threshold is None else float(threshold)
    command[6] = 1000.0 if command[6] >= limit else 0.0
    return command


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


if __name__ == "__main__":
    main()
