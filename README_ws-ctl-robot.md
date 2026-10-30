# 用 WebSocket 控制 UR5e 机器人

`ws-control` 是一个极简的主进程：它读取相机图像和机械臂状态，通过
**WebSocket** 发给远程策略服务，收到动作块后校验并下发到 UR5e。

```text
相机 + UR5e 状态
      │  {"images": {...}, "state": [...]}
      ▼
 remote policy server  ──►  {"action": [[...], [...]]}
      │
      ▼
 校验 → 限幅 → 二值化夹爪 → UR5e servoJ 执行
```

除了默认的 `json` 协议，它还适配了：

| `--model` | 服务 | 传输 | 动作语义 | 能否直接执行 |
| --- | --- | --- | --- | --- |
| `json` | 通用 mock / 自建服务 | JSON 文本 | 6 关节 + 夹爪 | ✅ |
| `fastwam` | FastWAM | msgpack | 6 关节 + 夹爪（`autogen_mixed_ur5`） | ✅ |
| `lingbot` | LingBot-VA | msgpack（`__ndarray__`） | 6 关节 + 夹爪（`ur5` profile） | ✅ |
| `openwam` | OpenWAM | JSON 文本 | 6 关节 + 夹爪（`ur5` 数据集） | ✅ |
| `pi05`（旧名 `openpi05`） | OpenPI 0.5 | msgpack（`__ndarray__`） | 绝对关节角 + 夹爪（`pi05_huashu`） | ✅ |
| `flexpi` | Flex-π | msgpack-numpy | 13 维 HS13 末端状态 | ❌ 仅 `--dry-run` |

UR5e 的 `servoJ` 只接受**绝对关节角**，所以 `flexpi` 这类输出末端位姿的
检查点绝不能被当成关节角下发。程序启动时就会拒绝这类组合，除非显式加
`--dry-run`（只打印动作、不动机器人）。OpenPI 的 `pi05_huashu` 返回的是
`(N, 7)` 绝对关节角，可直接执行；`guodi`（20 维）/ `xinsong`（16 维）是多臂
检查点，会被 7 维校验拦下；`flexiv` 虽是 7 维，但那是 7 关节 Flexiv 臂，
不是 UR5e 的 6 关节 + 夹爪。

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

`lingbot` / `pi05` 用服务端自己的 `__ndarray__` 编码（纯 msgpack，不是
msgpack-numpy）；`flexpi` 需要 `msgpack-numpy`（已在项目依赖里）；`openwam`
用 `cv2` 编码 PNG。

---

## 2. 修改配置

编辑 `configs/hardware/ur5e_d455.yaml`：

```yaml
robot:
  ip: "192.168.56.2"          # UR5e IP
  control_hz: 100             # env.step 的最小周期
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
  request_timeout_s: 600.0            # 连接超时
  threshold: 0.7                      # 夹爪二值化阈值
  max_joint_delta: 0.1                # 每条指令单关节最大变化（rad）
  action_hz:                          # 动作下发帧率，留空用协议默认值
  model: "json"                       # 协议，见上表

task:
  file: "configs/tasks/pick_place_pen_block.yaml"
```

`ws-control` **只使用**上面这些字段。旧的 `vla.image_key_map`、`vla.state_key`、
`vla.action_space`、`vla.reset_on_start` 是给 `run-vla-inference` 用的，
新接口不读取它们。

`action_hz` 留空时：`json` 跟随 `robot.control_hz`（保持旧行为），其余模型
按数据集的 15 Hz 下发。模型是在 15 fps 数据上训练的，按 100 Hz 抢跑会让
同一段轨迹在时间上缩短 6 倍，所以不要随手调高。

请求体里的 `state` 和图片字段不放进配置，直接写在
`src/apps/run_ws_control.py` 顶部：

```python
IMAGE_FIELDS = {"third_person": "third_left_top_rgb", "wrist": "wrist_rgb"}
FASTWAM_IMAGE_FIELDS = {...}
PI05_IMAGE_FIELDS = {...}
LINGBOT_IMAGE_FIELDS = {...}
FLEXPI_IMAGE_FIELDS = {...}
OPENWAM_IMAGE_FIELDS = {...}
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

### 3.3 下发前的校验

对每个模型、每条动作都会做（`_check_action_shape` / `_robot_command`）：

1. 返回的动作块宽度必须是 **7**（单臂 UR5 = 6 关节 + 夹爪）；不是 7 维时
   打印 `WARNING: ... expected 7-dim single-arm UR5 actions, got D`，真机执行
   前直接报错停止，`--dry-run` 下只告警并继续打印；
2. 每条动作必须全部是有限值；
3. 6 个关节值必须落在 ±2π rad 内，否则判定为“不是关节指令”并停止；
4. 用上一帧目标值做参考，单关节变化超过 `vla.max_joint_delta` 时截断
   （YAML 写成空值或传 `--max-joint-delta none` 关闭）；
5. 夹爪按 `vla.threshold` 二值化成 0/1000。

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

1. 校验模型的动作是否可执行（EEF 模型在这里就被拒绝）；
2. 启动 UR5e ZMQ 子进程和 RealSense 相机子进程；
3. 连接策略服务端，完成所在协议需要的握手；
4. 按 `robot.reset_joints` 分 100 步移动到初始位姿；
5. 终端提示 `press ENTER to start policy control loop...`，按回车开始；
6. 循环：取观测 → 请求 → 校验 → 执行 action；
7. 运行中按 **N** 或 **T**，机械臂回到 `robot.reset_joints`（复位），并丢弃
   当前动作块剩余动作、按新位姿重新请求；加了 `--save-video` 时，复位前会先
   询问是否保存本次 episode 的录像（见 5.1）；
8. `Ctrl+C` 退出，或达到 `--max-steps` 自动停止。

复位键需要图形环境（`pynput`）；在无显示器的 SSH 会话里监听会失效，程序只
打印 `Reset keys disabled: ...` 并继续运行。`--dry-run` 下不启动复位键。

### 命令行参数

| 参数 | 说明 |
| --- | --- |
| `--config` | 运行配置，默认 `configs/hardware/ur5e_d455.yaml` |
| `--model` | 协议：`json`、`fastwam`、`lingbot`、`openwam`、`pi05`（旧名 `openpi05`）、`flexpi`，覆盖 `vla.model` |
| `--server-url` | 覆盖 `vla.server_url` |
| `--api-key` | 覆盖 `vla.api_key` |
| `--task` | 覆盖任务描述文本 |
| `--max-steps` | 处理多少条 action 后停止，不填则一直运行 |
| `--no-wait` | 跳过 ENTER 确认，直接开始 |
| `--dry-run` | 只请求并打印动作，不下发；EEF 模型必须用它 |
| `--action-hz` | 覆盖动作下发帧率（Hz） |
| `--max-joint-delta` | 覆盖单关节限幅（rad），`none` 表示关闭 |
| `--save-video [BOOL]` | 录制单次 episode 的录像，`true`/不带值开启，默认关闭（见 5.1） |

### 5.1 保存 episode 视频

```bash
uv run ws-control \
  --config configs/hardware/ur5e_d455.yaml \
  --server-url ws://127.0.0.1:8000 \
  --save-video true
```

- 每条 action 执行后取一帧，把第三人称（`third_left_top_rgb`）和腕部
  （`wrist_rgb`）两路画面左右拼接到同一张图上，按动作帧率写盘。
- 路径：`./save_video/<model>/<YYMMDD-HH:MM>/video_<num>.mp4`，同一分钟内按
  `video_1.mp4`、`video_2.mp4` 递增。
- 按 **N** / **T** 复位时，终端会询问
  `Save the recorded episode video? [y/N]`：回答 `y` 保留文件，其他（含直接
  回车）丢弃并删除半截文件，然后机械臂才回初始位姿。下一段 episode 从头录制。
- `Ctrl+C` / `--max-steps` 结束时不会保存未确认的录像，避免留下损坏的 mp4。
- `--dry-run` 下不录制（没有真实动作）。

首次上真机建议：

```bash
uv run ws-control --max-steps 5
```

确认方向和幅度正确后，再逐步放开 `--max-steps` 和 `vla.execute_horizon`。

---

## 6. 修改接口字段

整个入口就是一个文件
`src/apps/run_ws_control.py`，从上到下是：参数 → 配置 → 校验 → 建环境 →
连服务端 → 握手 → `_run` 主循环。协议相关只有四个小函数：

```python
def get_model_body(
    obs, *, model, task, repo_id="", request_id=0, jpeg_quality=90,
) -> dict
def _prepare_server(connection, *, model, task) -> None
def _request(connection, body, *, model) -> dict
def _parse_actions(response, *, model) -> np.ndarray
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

字段改动不影响主循环和硬件层，改这几处即可。

---

## 7. 各模型协议

所有模型的共同点：动作块先校验为二维有限值数组，且宽度必须是 **7**
（单臂 UR5 = 6 关节 + 夹爪），否则告警；再按 `vla.execute_horizon` 截断。
每条动作都经过 §3.3 的校验、限幅和二值化。
每次收到动作块会打印一行
`<model> actions: shape=(N, D) first=[...]` 便于核对。

### 7.1 FastWAM（`--model fastwam`）

走 FastWAM 的 msgpack WebSocket 协议，帧格式参考
`FastWAM/docs/readme_infer.md`。与默认 `json` 的差异：

| | `json` | `fastwam` |
| --- | --- | --- |
| 传输 | JSON 文本帧 | msgpack 二进制帧 |
| 建连 | 直接请求 | 直接发 `infer`，不做握手 |
| 鉴权 | `Authorization` 头 | token 放在 URL 查询参数 |
| 图片 | base64 JPEG | 原始 JPEG 字节 |
| 动作 | 二值化夹爪后下发 | 同 `json`，按 15 Hz 下发 |

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

可执行的检查点是 `autogen_mixed_ur5`（7 维、0/1 夹爪、15 fps）。
`charger_white` 系列是 13 维 HS13，会被维度检查拒绝并提示。

### 7.2 LingBot-VA（`--model lingbot`）

纯 msgpack 协议，body 里的 numpy 数组用服务端自己的 `__ndarray__` 格式编码
（`{b"__ndarray__": True, b"data": <原始字节>, b"dtype": "|u1", b"shape": [...]}`），
不是 msgpack-numpy——该服务端用纯 msgpack 解包，msgpack-numpy 数组会变成普通
dict，服务端 `torch.from_numpy` 会报 `numpy.object_` 类型错误：

1. 建连后服务端先发一条 metadata（`frame_chunk_size` 等；当前部署回 `{}`），客户端消费掉；
2. 客户端发 `{"reset": true, "prompt": ...}`，服务端回 `{}`，或只带
   `server_timing`（如 `{"server_timing": {"infer_ms": 76.9}}`），两种都接受；
3. 之后每轮发观测，`obs` 里的图像同样用 `__ndarray__` 数组。

每轮观测请求前都要再发一次 `reset`。服务端只在 `frame_st_id == 0` 时编码观测，
而 streaming VAE 的 `feat_cache` 会跨调用保留；连续两次不发 reset 的观测请求，
第二次会拿单帧输入去拼缓存，在 `time_conv` 报
`Kernel size can't be greater than actual input size`。本入口每块都从当前观测
重新推理、不用 `compute_kv_cache`，所以 reset 让每块都等价于独立的第一块
（每块多约 0.1–0.2 s）。

| 请求字段 | 来源 |
| --- | --- |
| `obs["observation.images.third_person"]` | `third_left_top_rgb`，HWC RGB uint8 |
| `obs["observation.images.wrist"]` | `wrist_rgb`，HWC RGB uint8 |
| `prompt` | `task.file` 的 `task`，或 `--task` |

响应 `action` 形状是 `(7, chunk, actions_per_frame)`。客户端先转成
`(N, 7)`，**丢弃第 0 帧的 context 块**（训练时对齐到全零，推理时由
`_infer` 用 `action_cond` 覆盖，不是预测值），再按 `execute_horizon` 执行。
本入口每轮都从当前观测重新推理、不调用 `compute_kv_cache`，所以每块都
需要丢弃这一帧。

可执行的是 `ur5` profile / `ur5_joint30_v1` 检查点；`huashu`、`guodi`、
`xinsong` 的 profile 维度不同，会在维度检查处报错并提示换检查点。

### 7.3 OpenWAM（`--model openwam`）

JSON 文本协议，带探活和重置握手：

1. 发 `{"type": "ping"}`，收 `pong`，打印 `representation` 和
   `gripper_convention`；`representation` 不是 `joint` 时直接报错；
2. 发 `{"type": "reset"}`，必须收到 `reset_ack`；
3. 之后每轮发 `obs`。

| 请求字段 | 来源 |
| --- | --- |
| `images["head_camera"]` | `third_left_top_rgb`，base64 PNG |
| `images["left_wrist_camera"]` | `wrist_rgb`，base64 PNG |
| `images["right_wrist_camera"]` | 固定 `null`（单腕机器人） |
| `state` | 7 维关节状态 |
| `prompt` | `task.file` 的 `task`，或 `--task` |

响应为 `{"type": "action", "action": ...}`；`{"type": "error", ...}` 会
抛错并停止。使用 `ur5` 数据集（7 维、夹爪 0 闭 / 1 开、15 Hz）；
`hs` / `flexiv` 是 10 维 EEF，不能用。

### 7.4 OpenPI 0.5（`--model pi05`）

msgpack 协议，body 里的 numpy 数组用服务端自己的 `__ndarray__` 格式编码
（`{b"__ndarray__": True, b"data": <原始字节>, b"dtype": "<u1", b"shape": [...]}`），
不是 msgpack-numpy——该服务端用纯 msgpack 解包，msgpack-numpy 数组会被当成
普通 dict。建连后必须先消费服务端的 metadata 帧，之后每轮收发：

| 请求字段 | 来源 |
| --- | --- |
| `observation.images.third_person` | `third_left_top_rgb`，HWC RGB uint8 |
| `observation.images.wrist` | `wrist_rgb`，HWC RGB uint8 |
| `observation.state` | 7 维关节状态（6 关节 + 夹爪）float32 |
| `prompt` | `task.file` 的 `task`，或 `--task` |

```bash
uv run ws-control \
  --config configs/hardware/ur5e_d455.yaml \
  --model pi05 \
  --server-url ws://127.0.0.1:8000 \
  --dry-run \
  --max-steps 5
```

服务端返回的是**绝对关节角**（`response["actions"]`，`pi05_huashu` 为
`(50, 7)`），不是末端位姿，所以和 `lingbot` 一样可以直接下发：去掉
`--dry-run` 就会驱动 UR5e。OpenPI 已经在服务端完成反归一化和 Delta→ABS
还原，客户端**不要**再把 state 加回去。

动作宽度不是 7 时按普通 7 维校验失败处理：`guodi`（20 维）和 `xinsong`
（16 维）是多臂 checkpoints，真机执行前会直接报错。`flexiv` 的 7 维对应
7 关节 Flexiv 臂，形状检查能通过但语义不是 UR5e 的 6 关节 + 夹爪，不要
在 UR5e 上执行它。首次上真机先 `--dry-run --max-steps 5` 确认
`first=[...]` 里的关节值在 ±2π 内，再用 `--max-steps` 小步数试跑。

### 7.5 Flex-π（`--model flexpi`）

msgpack-numpy 协议，同样先消费 metadata。请求体是 HS13 的 13 维状态：

| 请求字段 | 来源 |
| --- | --- |
| `state[0:6]` | `obs["tcp_pose"]`（位置 xyz + 轴角） |
| `state[6]` | `obs["gripper_position_raw"]`（0~1000） |
| `state[7:13]` | `obs["tcp_force"]`（力 xyz + 力矩） |
| `images["cam_high"]` | `third_left_top_rgb` |
| `images["cam_left_wrist"]` | `wrist_rgb` |
| `prompt` | `task.file` 的 `task`，或 `--task` |

Flex-π 的检查点面向 Flexiv 机械臂，13 维是末端状态而不是 UR5e 关节角，
所以同样只能 `--dry-run`。

---

## 8. 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| `Configured RealSense cameras are not connected: ...` | 配置里的序列号与实际相机不符，`lsusb` 或 `rs-enumerate-devices` 查看 |
| `ConnectionRefusedError` / 连接超时 | 服务端未启动、端口错误，或 `--server-url` 写错 |
| `JSONDecodeError` | 服务端返回了二进制/msgpack；确认 `--model` 与协议一致 |
| `returns ... but the UR5e executes only absolute 6-joint` | 选了 EEF 模型；换 UR5 检查点，或加 `--dry-run` |
| `expected 7-dim single-arm UR5 actions ..., got ...` | 返回动作不是 7 维（单臂 = 6 关节 + 夹爪）；加载了别的机器人/任务的检查点（如 13 维 HS13）。`--dry-run` 下只告警不停止 |
| `policy response 'action' must be a list of 7-value actions` | `json`/`openwam` 返回维度不是 `N×7` |
| `unexpected OpenWAM frame type` | 服务端没按 `ping → reset → obs` 的约定回复 |
| 夹爪不动作或方向相反 | 调整 `vla.threshold`（0~1 输出用 0.7，0~1000 输出用 500） |
| 机械臂运动幅度过大 | 降低 `execute_horizon`、调小 `max_joint_delta`，并确认服务端输出绝对关节角 |
| 终端无输出 | 检查是否停在 `press ENTER` 提示，或加 `--no-wait` |

---

## 9. 安全提示

- 真机运行前确认急停按钮可用，手不要离开急停。
- 运行中随时可按 **N / T** 让机械臂分小步回到 `robot.reset_joints`，不会跳变。
- 首次运行务必 `--max-steps` 取小值（如 5），并观察 `reset_joints` 移动过程。
- 换模型先用 `--dry-run` 看 `actions` 的数值范围，再上真机。
- 默认按 `control_hz: 100` 逐条下发 `servoJ`；模型动作按其训练帧率
  （15 Hz）节流，不要调高。网络延迟高时不要增大 `execute_horizon`。
- 策略返回绝对关节角，网络中断时最后一条指令会被保持；确认服务端有超时保护。
