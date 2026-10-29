# 用 WebSocket 控制 UR5e 机器人

`ws-control` 是一个极简的主进程：它读取相机图像和机械臂状态，通过
**JSON over WebSocket** 发给远程策略服务，收到 `action` 后直接下发到 UR5e。

```text
相机 + UR5e 状态
      │  {"images": {...}, "state": [...]}
      ▼
 remote policy server  ──►  {"action": [[...], [...]]}
      │
      ▼
 UR5e servoJ 执行
```

和旧入口 `run-vla-inference`（CloudRobo-VLA msgpack 协议）互不影响，两者可并存。

---

## 1. 准备环境

```bash
cd ur5e_robot_control-main
uv sync --no-dev
source .venv/bin/activate
```

确认硬件在线（不启动策略，只自检）：

```bash
# 采集每路 D455 一段短视频，输出到 .datasets/camera_check/
uv run check-cameras --config configs/hardware/ur5e_d455.yaml

# 检查依赖、串口、相机配置，以及 UR5e TCP 连通性
uv run check-recording-ready \
  --config configs/hardware/ur5e_d455.yaml \
  --check-robot-tcp
```

---

## 2. 修改配置

编辑 `configs/hardware/ur5e_d455.yaml`：

```yaml
robot:
  ip: "192.168.56.2"          # UR5e IP
  control_hz: 100             # 每条 action 的最小执行间隔 = 1/100 s
  reset_joints: [1.56, -1.57, 1.58, -1.58, -1.57, -0.0077, 1000.0]

camera:
  realsense_ids:
    "353522301461": third_left_top
    "333422304095": wrist

vla:
  server_url: "ws://127.0.0.1:8000"   # 策略服务地址
  api_key:                            # 可留空
  repo_id: "pick_place_blocks_150_train1_v3.0"
  execute_horizon: 15                 # 每次响应处理多少条 action
  request_timeout_s: 5.0              # 连接超时
  threshold: 0.7                      # 夹爪二值化阈值
  model: "json"                       # 协议：json 或 fastwam

task:
  file: "configs/tasks/pick_place_pen_block.yaml"
```

`ws-control` **只使用**上面这些字段。旧的 `vla.image_key_map`、`vla.state_key`、
`vla.action_space`、`vla.max_joint_delta`、`vla.reset_on_start` 是给
`run-vla-inference` 用的，新接口不读取它们。

请求体里的 `state` 和图片字段不放进配置，直接写在
`src/apps/run_ws_control.py` 顶部：

```python
IMAGE_FIELDS = {"third_person": "third_left_top_rgb", "wrist": "wrist_rgb"}
FASTWAM_IMAGE_FIELDS = {"third_person": "third_left_top_rgb", "wrist": "wrist_rgb"}
```

`state` 固定取 `obs["joint_positions"]`。换模型要改字段时，改这几行即可。

---

## 3. 接口约定

### 3.1 请求（客户端 → 服务端）

一帧文本 JSON，每次循环发一次：

```json
{
  "repo_id": "pick_place_blocks_150_train1_v3.0",
  "task": "Pick up the pens and the blocks from the table ...",
  "state": [1.56, -1.57, 1.58, -1.58, -1.57, -0.0077, 1.0],
  "images": {
    "third_person": "<base64 编码的 JPEG>",
    "wrist": "<base64 编码的 JPEG>"
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `state` | 7 维：6 个关节角（rad）+ 夹爪状态（0/1） |
| `images` | RGB 图像，JPEG + base64；键名见 `IMAGE_FIELDS` |
| `task` | 来自 `task.file` 的 `task` 字段，可用 `--task` 覆盖 |
| `repo_id` | 来自 `vla.repo_id` |

### 3.2 响应（服务端 → 客户端）

```json
{"action": [[j1, j2, j3, j4, j5, j6, gripper], [j1, j2, j3, j4, j5, j6, gripper]]}
```

- `action` 是 `N × 7` 的数组，客户端只执行前 `vla.execute_horizon` 条。
- 前 6 维按**绝对关节角**下发（`servoJ`）。
- 第 7 维是夹爪，`>= vla.threshold` 记为张开（1000），否则闭合（0）。
  如果你的服务端直接输出 `0~1000`，把 `threshold` 改成 `500` 即可。
- 服务端返回 `{"ok": false, "error": "..."}` 时客户端会抛错并停止。

---

## 4. 先用 mock 服务端联调

不接策略模型也能验证链路。新建 `mock_policy_server.py`：

```python
"""返回当前状态的 mock 策略：让机械臂保持原位。"""

import json

from websockets.sync.server import serve


def handler(ws):
    for message in ws:
        request = json.loads(message)
        state = request["state"]
        ws.send(json.dumps({"action": [state, state]}))


with serve(handler, "0.0.0.0", 8000) as server:
    print("mock policy listening on ws://0.0.0.0:8000")
    server.serve_forever()
```

启动：

```bash
uv run python mock_policy_server.py
```

这个 mock 会把当前关节角原样返回，机械臂只保持不动，适合验证连接、图像和字段。

---

## 5. 启动主进程

```bash
uv run ws-control \
  --config configs/hardware/ur5e_d455.yaml \
  --server-url ws://127.0.0.1:8000 \
  --max-steps 20
```

运行顺序：

1. 启动 UR5e ZMQ 子进程和 RealSense 相机子进程；
2. 按 `robot.reset_joints` 分 100 步移动到初始位姿；
3. 终端提示 `press ENTER to start policy control loop...`，按回车开始；
4. 循环：取观测 → 请求 → 执行 action；
5. `Ctrl+C` 退出，或达到 `--max-steps` 自动停止。

### 命令行参数

| 参数 | 说明 |
| --- | --- |
| `--config` | 运行配置，默认 `configs/hardware/ur5e_d455.yaml` |
| `--model` | 协议：`json`（默认）、`fastwam` 或 `openpi05`，覆盖 `vla.model` |
| `--server-url` | 覆盖 `vla.server_url` |
| `--api-key` | 覆盖 `vla.api_key` |
| `--task` | 覆盖任务描述文本 |
| `--max-steps` | 处理多少条 action 后停止，不填则一直运行 |
| `--no-wait` | 跳过 ENTER 确认，直接开始 |

首次上真机建议：

```bash
uv run ws-control --max-steps 5
```

确认方向和幅度正确后，再逐步放开 `--max-steps` 和 `vla.execute_horizon`。

---

## 6. 修改接口字段

整个入口就是一个文件
`src/apps/run_ws_control.py`，从上到下是：参数 → 配置 → 建环境 → 连服务端 →
`_run` 主循环。协议相关只有三个小函数：

```python
def get_model_body(
    obs, *, model, task, repo_id="", request_id=0, jpeg_quality=90,
) -> dict
def _parse_actions(response, *, model) -> np.ndarray
def _request(connection, body, *, model) -> dict
```

常见改法：

**改图片字段名 / 增加一路相机**

直接改文件顶部的常量，左边是发给服务端的键名，右边是
`RobotEnv.get_obs()` 里的 key：

```python
IMAGE_FIELDS = {
    "third_person": "third_left_top_rgb",
    "wrist": "wrist_rgb",
    "wrist_depth": "wrist_depth",     # 新增一路
}
```

**增加请求字段**

```python
return {
    "repo_id": repo_id,
    "task": task,
    "state": [float(v) for v in obs["joint_positions"]],
    "tcp_pose": [float(v) for v in obs["tcp_pose"]],   # 新增
    "images": {...},
}
```

**改响应解析**（例如服务端包了一层 `data`）

```python
def _parse_actions(response, *, model):
    response = response.get("data", response)
    ...
```

字段改动不影响主循环和硬件层，改这三处即可。

---

## 7. FastWAM 协议（`--model fastwam`）

`fastwam` 走 FastWAM 的 msgpack WebSocket 协议，帧格式参考
`FastWAM/docs/readme_infer.md`。与默认 `json` 的差异：

| | `json` | `fastwam` |
| --- | --- | --- |
| 传输 | JSON 文本帧 | msgpack 二进制帧 |
| 建连 | 直接请求 | 直接发 `infer`，不做握手 |
| 图片 | base64 JPEG | 原始 JPEG 字节 |
| state | `obs["joint_positions"]` | `obs["joint_positions"]` |
| 动作 | 二值化夹爪后下发 | 二值化夹爪后下发（与 `json` 相同） |

启动：

```bash
uv run ws-control \
  --config configs/hardware/ur5e_d455.yaml \
  --model fastwam \
  --server-url ws://127.0.0.1:8765/v1/ws \
  --api-key fastwam-dev-token \
  --max-steps 5


python -m src.apps.run_ws_control \
  --config configs/hardware/ur5e_d455.yaml \
  --model fastwam \
  --server-url ws://localhost:8765/v1/ws \
  --api-key fastwam-dev-token \
  --max-steps 5
```

请求体字段映射：

| 请求字段 | 来源 |
| --- | --- |
| `prompt` | `task.file` 的 `task`，或 `--task` |
| `images[i].key` | `FASTWAM_IMAGE_FIELDS` 左侧 |
| `images[i].data` | `FASTWAM_IMAGE_FIELDS` 右侧 obs 键编码出的 JPEG 字节 |
| `state` | `obs["joint_positions"]` 平铺后的列表 |
| `options.num_inference_steps` | `run_ws_control.py` 的 `NUM_INFERENCE_STEPS`（默认 10） |

字段都固定在 `run_ws_control.py` 顶部常量里。`action` 先校验为二维有限值
数组；下发前再要求每条形如 `[6 个关节角, 夹爪]`（7 维），否则报错停止。
每一块动作执行 `vla.execute_horizon` 条，夹爪用 `vla.threshold` 二值化成
0/1000，与控制 `json` 协议完全一致。收到 action 后会打印
`FastWAM actions: shape=(N, D) first=[...]` 便于核对。

### 7.1 OpenPI 0.5（`--model openpi05`）

走 OpenPI 的 msgpack-numpy 协议，body 里直接是 numpy 数组（不是 JPEG），
需要安装 `msgpack-numpy`（已加入项目依赖）：

| 请求字段 | 来源 |
| --- | --- |
| `images["cam_high"]` | `third_left_top_rgb`，转成 `(3, H, W)` uint8 |
| `images["cam_left_wrist"]` | `wrist_rgb`（单腕机器人两路腕部共用） |
| `images["cam_right_wrist"]` | `wrist_rgb` |
| `state` | 7 维关节状态（6 关节 + 夹爪）float32 |
| `prompt` | `task.file` 的 `task`，或 `--task` |

```bash
uv run ws-control \
  --config configs/hardware/ur5e_d455.yaml \
  --model openpi05 \
  --server-url ws://127.0.0.1:8000 \
  --api-key fastwam-dev-token \
  --max-steps 5
```

响应读取 `actions`（`(N, 7)`，维度以服务端返回为准），和 `fastwam` 一样
只解析并打印 `OpenPI05 actions: ...`，不会下发到机器人。

---

## 8. 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| `Configured RealSense cameras are not connected: ...` | 配置里的序列号与实际相机不符，`lsusb` 或 `rs-enumerate-devices` 查看 |
| `ConnectionRefusedError` / 连接超时 | 服务端未启动、端口错误，或 `--server-url` 写错 |
| `JSONDecodeError` | 服务端返回了二进制/msgpack；本入口只接受 JSON 文本帧 |
| `policy response 'action' must be a list of 7-value actions` | 返回维度不是 `N×7`，检查 `parse_actions` |
| 夹爪不动作或方向相反 | 调整 `vla.threshold`（0~1 输出用 0.7，0~1000 输出用 500） |
| 机械臂运动幅度过大 | 降低 `vla.execute_horizon`，并确认服务端输出的是绝对关节角 |
| 终端无输出 | 检查是否停在 `press ENTER` 提示，或加 `--no-wait` |

---

## 9. 安全提示

- 真机运行前确认急停按钮可用，手不要离开急停。
- 首次运行务必 `--max-steps` 取小值（如 5），并观察 `reset_joints` 移动过程。
- 默认按 `control_hz: 100` 逐条下发 `servoJ`，不要在网络延迟高时增大
  `execute_horizon`。
- 策略返回绝对关节角，网络中断时最后一条指令会被保持；确认服务端有超时保护。
