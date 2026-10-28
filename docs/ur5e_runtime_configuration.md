# UR5e Runtime 使用教程

本文档面向当前仓库的最新结构，覆盖 UR5e + 大寰 PGC-140-50 夹爪 +
双 RealSense D455 的相机校准、数据采集前硬件验证、安全区设置、遥操作数据
采集和 VLA 模型推理控制。

所有命令默认从仓库根目录执行：

```bash
cd /home/chopin/ur5e_robot_control
```

默认配置文件是：

```bash
configs/hardware/ur5e_d455.yaml
```

## 一次完整上线流程

第一次在端侧笔记本运行时，建议按这个顺序执行：

1. 安装依赖并配置硬件 YAML。
2. 运行 `check-recording-ready` 检查 Python 依赖、串口路径、相机配置和 UR TCP。
3. 运行 `check-cameras` 录制每个 D455 视角 5 秒 MP4，确认视频流正常。
4. 运行 `calibrate-cameras` 生成 D455 内参和外参。
5. 运行 `calibrate-safety-zone` 生成关节安全区。
6. 运行 `record-teleop` 使用 GELLO 或 SpaceMouse 采集 LeRobot v3 数据。
7. 启动 CloudRobo-VLA server 后，运行 `run-vla-inference` 控制真机执行模型推理动作。

相机视频流由当前 Python RealSense 代码直接读取，不需要先启动 ROS 或
`roslaunch`。

当前运行时的真实数据流如下：

1. `record-teleop` 或 `run-vla-inference` 启动 UR5e + PGC 的 robot ZMQ server。
2. `RealSenseCameraServer` 根据 YAML 中的 D455 序列号为每个视角启动一个
   camera process。
3. 每个 camera process 后台持续读取最新 RGB 帧，client 请求时返回 latest
   cached frame。
4. 控制循环读取 robot observation、最新 RGB 图像和 agent/VLA action。
5. 机械臂前 6 个关节经过安全区 clamp 后再发送给 UR5e；夹爪命令保留。
6. 数据采集时，LeRobot v3 frame 通过有界队列交给后台 writer thread。

目前图像数据仍通过 pickle-over-ZMQ 返回，已经降低了取帧阻塞，但还没有
切换到 shared memory ring buffer。

## 深度视频时间戳修复与质量门禁

对已有 LeRobot v3 数据先执行只读审计：

```bash
uv run repair-video-timestamps \
  --root .datasets/lerobot/pick_place_block_150_train_v3.0
```

审计确认只有 episode 边界偏移、episode 内部仍为固定 FPS 后，再应用修复：

```bash
uv run repair-video-timestamps \
  --root .datasets/lerobot/pick_place_block_150_train_v3.0 \
  --apply
```

写入前会生成 `*.parquet.pre-timestamp-repair.bak` 备份。不要通过增大
LeRobot 的 timestamp tolerance 绕过错误；如果工具报告 episode 内部不是固定
FPS，应重新编码对应视频文件。

新数据在 `dataset.lerobot.repair_video_timestamps: true` 时会在 finalize 后自动
按真实 PTS 校正深度视频 metadata。`dataset.lerobot.depth_min_m: 0.0` 用于让
RealSense 的无效值 0 mm 在视频解码后仍保持为 0。

启用 `camera.depth_quality` 后，每个 episode 开始前会检查多帧原始 Z16：

```yaml
camera:
  enable_depth: true
  depth_quality:
    sample_count: 15
    max_invalid_ratio:
      wrist: 0.50
      third_left_top: 0.35
  depth_options:
    wrist:
      visual_preset: high_density
```

无效像素中位数超过阈值时录制会被阻止。诊断双 D455 红外干扰时，可临时对
其中一路设置 `emitter_enabled: false` 做对照；不要用 hole filling 掩盖大面积
无效深度。

## 安装和硬件配置

安装基础依赖：

```bash
uv sync --no-dev
source .venv/bin/activate
```

GELLO 的 DynamixelSDK 仍来自子模块：

```bash
git submodule update --init third_party/DynamixelSDK
uv pip install -e third_party/DynamixelSDK/python
```

如果使用 SpaceMouse 遥操作，还需要可选依赖和 MuJoCo 模型子模块：

```bash
uv sync --no-dev --extra spacemouse
git submodule update --init third_party/mujoco_menagerie
```

查找稳定串口路径：

```bash
ls -l /dev/serial/by-id/
```

优先把 `agent.gello_port` 和 `gripper.port` 配成 `/dev/serial/by-id/...`，
不要长期依赖 `/dev/ttyUSB*`，因为重启或重新插拔后编号可能变化。

查找 RealSense D455 序列号：

```bash
rs-enumerate-devices
```

然后更新 `configs/hardware/ur5e_d455.yaml`：

```yaml
robot:
  ip: "192.168.56.2"

agent:
  name: gello
  gello_port: "/dev/serial/by-id/<gello-device>"

camera:
  fps: 30
  enable_depth: false
  realsense_ids:
    "<third-person-d455-serial>": third_front
    "<wrist-d455-serial>": wrist

gripper:
  port: "/dev/serial/by-id/<pgc-device>"
```

当前代码要求相机逻辑名包含 `third_front` 和 `wrist`。这两个名字会影响
数据集图像字段和 VLA 请求字段，不建议随意改名。
默认关闭 D455 depth stream，数据采集和 VLA 推理只读取 RGB；只有明确需要
深度图时才把 `camera.enable_depth` 改为 `true`。

## 配置加载规则

所有 runtime CLI 的配置加载顺序一致：

1. 命令行 `--config <path>`
2. 环境变量 `UR5E_HARDWARE_CONFIG=<path>`
3. 默认 `configs/hardware/ur5e_d455.yaml`

相对路径会按仓库根目录解析，绝对路径保持原样。

常用命令可以显式传配置：

```bash
uv run check-recording-ready --config configs/hardware/ur5e_d455.yaml
uv run check-cameras --config configs/hardware/ur5e_d455.yaml
uv run calibrate-safety-zone --config configs/hardware/ur5e_d455.yaml --margin 0.02
uv run record-teleop --config configs/hardware/ur5e_d455.yaml
uv run run-vla-inference --config configs/hardware/ur5e_d455.yaml
```

也可以先导出环境变量：

```bash
export UR5E_HARDWARE_CONFIG=configs/hardware/ur5e_d455.yaml
```

## 数据采集前硬件验证

如果是第一次连接 GELLO，或重新安装过 Dynamixel 编码器，可以先检查 GELLO
关节偏置：

```bash
uv run calibrate-gello --port /dev/serial/by-id/<gello-device>
```

这个命令会打印每个 GELLO 关节的 offset 建议值，以及夹爪开合角度参考。
默认 `start_joints` 和 `joint_signs` 已按当前 UR5e/GELLO 配置设置；需要
自定义时先运行 `uv run calibrate-gello --help` 查看 tyro 参数格式。它不会
移动 UR5e 真机。

先做不启动相机流和真机运动的基础检查：

```bash
uv run check-recording-ready --config configs/hardware/ur5e_d455.yaml
```

这个命令会检查：

- Python 环境中是否能 import `numpy`、`pyrealsense2`、`rtde_control`、
  `rtde_receive`、`zmq`、`lerobot` 等依赖。
- `agent.name: gello` 时 GELLO 串口路径是否存在。
- `agent.name: spacemouse` 时 SpaceMouse 依赖和 MuJoCo UR5e 模型是否存在。
- PGC 夹爪串口路径是否存在。
- YAML 中是否配置了 RealSense 序列号。

需要额外确认 UR 控制器 RTDE 端口是否可达时，加上：

```bash
uv run check-recording-ready \
  --config configs/hardware/ur5e_d455.yaml \
  --check-robot-tcp
```

`--check-robot-tcp` 会探测 `robot.ip:30004`。如果失败，先检查笔记本网口
IP、UR 控制柜 IP 和防火墙。

接着检查 D455 实际视频流：

```bash
uv run check-cameras --config configs/hardware/ur5e_d455.yaml
```

默认会按 `camera.fps` 录制 5 秒视频，每个逻辑视角保存一个 MP4：

```bash
.datasets/camera_check/<YYYYMMDD-HHMMSS>/third_front.mp4
.datasets/camera_check/<YYYYMMDD-HHMMSS>/wrist.mp4
```

也可以指定时长、FPS 和输出目录：

```bash
uv run check-cameras \
  --config configs/hardware/ur5e_d455.yaml \
  --duration 5 \
  --fps 30 \
  --output-dir .datasets/camera_check
```

在开始数据采集或模型推理前都建议跑一次 `check-cameras`。它会直接启动
Python RealSense reader，不依赖 ROS。

## 相机校准

相机校准输出默认写到：

```bash
configs/cameras/
```

这些 YAML 与具体相机安装位置、镜头和机器人工作台有关，已被 git 忽略。
默认标定板是棋盘格：

```text
columns = 7
rows = 6
square_size = 0.025 m
```

如果使用 Charuco 板，可以在命令里加：

```bash
--pattern charuco \
--charuco-squares-x 7 \
--charuco-squares-y 5 \
--charuco-square-length 0.04 \
--charuco-marker-length 0.03
```

### 1. 采集两个 D455 的内参

先标定腕部相机：

```bash
uv run calibrate-cameras --config configs/hardware/ur5e_d455.yaml \
  intrinsics \
  --camera wrist \
  --samples 20 \
  --interval 1.0
```

再标定第三视角相机：

```bash
uv run calibrate-cameras --config configs/hardware/ur5e_d455.yaml \
  intrinsics \
  --camera third_front \
  --samples 20 \
  --interval 1.0
```

默认输出：

```bash
configs/cameras/intrinsics_wrist.yaml
configs/cameras/intrinsics_third_front.yaml
```

采样时让标定板覆盖图像中不同位置和角度。程序会按 `--interval` 自动读取
帧，检测不到标定板的帧会跳过。

### 2. 标定腕部相机手眼外参

腕部相机安装在末端，需要估计 `tool_T_wrist`：

```bash
uv run calibrate-cameras --config configs/hardware/ur5e_d455.yaml \
  hand-eye \
  --camera wrist \
  --samples 12 \
  --interval 1.0 \
  --method tsai
```

这个命令会启动 UR robot ZMQ server 并读取 TCP pose，同时从腕部相机读取
标定板姿态。采样时移动机械臂到多个不同姿态，让标定板持续可见。

如果你已经单独启动了 robot ZMQ server，可加：

```bash
--use-existing-robot-server
```

### 3. 标定第三视角相机到机器人 base 的外参

第三视角相机固定在外部，需要估计 `base_T_third_front`：

```bash
uv run calibrate-cameras --config configs/hardware/ur5e_d455.yaml \
  base-extrinsics \
  --camera third_front \
  --samples 12 \
  --interval 1.0
```

默认假设标定板固定在 tool 坐标系原点。如果你有单独测量的
`tool_T_target`，可以传：

```bash
--tool-target-transform configs/cameras/tool_T_target.yaml
```

### 4. 标定两个相机之间的外参

用于估计 `wrist_T_third_front`：

```bash
uv run calibrate-cameras --config configs/hardware/ur5e_d455.yaml \
  stereo-extrinsics \
  --reference-camera wrist \
  --camera third_front \
  --samples 12 \
  --interval 1.0
```

采样时标定板需要同时出现在两个相机画面中。

### 5. 校准后检查

校准完成后应有这些文件：

```bash
configs/cameras/intrinsics_wrist.yaml
configs/cameras/intrinsics_third_front.yaml
configs/cameras/extrinsics.yaml
```

再次录制相机检查视频，确认两个视角没有接反：

```bash
uv run check-cameras --config configs/hardware/ur5e_d455.yaml
```

## 安全区设置

安全区文件默认写到：

```bash
configs/safety/ur5e_joint_limits.yaml
```

它只限制前 6 个 UR5e 关节，夹爪命令会保留。数据采集和 VLA 推理都会加载
这个文件；缺失时会直接报错，避免没有边界条件就控制真机。

运行校准：

```bash
uv run calibrate-safety-zone \
  --config configs/hardware/ur5e_d455.yaml \
  --margin 0.02
```

校准过程会使用当前 `agent.name` 指定的遥操作方式：

- `gello`: 用 GELLO leader arm 移动机器人。
- `spacemouse`: 用 SpaceMouse 控制机器人。

操作方式：

1. 启动命令后，把机器人移动到期望工作空间的各个边界。
2. 程序会持续记录当前 6 个关节的最小值和最大值。
3. 完成后按 `Esc`，或在终端按 `Ctrl-C`。
4. 程序把观测到的 min/max 写成 `configs/safety/ur5e_joint_limits.yaml`。

`--margin 0.02` 表示在记录到的 min/max 内侧各收缩 0.02 rad，更保守。
第一次真机调试建议保留 margin。

## 遥操作数据采集

数据采集入口：

```bash
uv run record-teleop --config configs/hardware/ur5e_d455.yaml
```

`record-teleop` 目前只有 `--config` 一个 CLI 参数。episode 数量、任务文件、
LeRobot repo id、dataset root、FPS、是否异步写入、队列长度以及 GELLO /
SpaceMouse 模式都从 YAML 或环境变量读取。

当前数据会直接保存为 LeRobot dataset v3，不再保存 npz，也不再保留 npz
转换路径。默认输出目录：

```bash
.datasets/lerobot/ur5e-pen-pick-place
```

这个目录已被 git 忽略。
默认采集为 RGB-only，不保存 D455 depth。LeRobot 写入默认通过后台 writer
thread 异步执行，减少视频编码/写盘对遥操作控制循环的阻塞。

### GELLO 和 SpaceMouse 切换

GELLO：

```yaml
agent:
  name: gello
  robot: ur5
  gello_port: "/dev/serial/by-id/<gello-device>"
```

SpaceMouse：

```yaml
agent:
  name: spacemouse
  robot: ur5
  spacemouse:
    device_path: "/dev/hidraw0"
```

也可以临时用环境变量切换：

```bash
UR5E_AGENT_NAME=spacemouse uv run record-teleop --config configs/hardware/ur5e_d455.yaml
```

### 采集时的运行逻辑

`record-teleop` 会：

1. 启动 UR5e robot ZMQ server。
2. 启动 Python RealSense camera server，读取 YAML 中的 D455 序列号。
3. 加载 `configs/safety/ur5e_joint_limits.yaml`。
4. 创建 GELLO 或 SpaceMouse agent。
5. 将机器人移动到 `robot.reset_joints`。
6. 对齐 leader/follower 初始位置。
7. 按 episode 保存 LeRobot v3 数据。

创建相机流时会直接通过 RealSense SDK 读取当前 color/depth stream 的内参及
stream 间外参，不依赖 `configs/cameras/*.yaml`。这些参数会注册为 LeRobot
feature，并随每一帧写入 parquet。字段名分别使用
`observation.camera_intrinsics.<camera>.*` 和
`observation.camera_extrinsics.<transform>.matrix`；其中图像宽高存放在
`observation.camera_intrinsics.<camera>.resolution`，避免被 LeRobot 的统计
模块误判成图像张量。

每个 D455 camera server 会持续读取最新 RGB 帧；采集循环请求的是缓存中的
latest frame，而不是在控制路径上等待相机现场采一帧。相机和机器人 ZMQ
client 都设置了有界超时，并会在 timeout/socket error 后重建 REQ socket，
避免后续重试卡在旧的 REQ/REP 状态里。

每条数据的核心字段来源：

- live env observation 中的 `joint_positions`: 从 UR5e robot ZMQ client 读取，
  当前为 6 个 UR5e 关节位置加 1 个 PGC 夹爪位置。
- live env observation 中的 `wrist_rgb` / `third_front_rgb`: 从 RealSense
  camera server 读取的 RGB 图像；默认没有 `*_depth` key。
- LeRobot `observation.state`: 由 live `joint_positions` 写入。
- LeRobot `observation.tcp_pose`: 由 UR RTDE `getActualTCPPose()` 写入，格式为
  `[x, y, z, rx, ry, rz]`；平移单位为米，姿态为旋转向量（弧度）。
- LeRobot `observation.images.wrist`: 由 live `wrist_rgb` 写入。
- LeRobot `observation.images.third_person`: 由 live `third_front_rgb` 写入。
- LeRobot `action`: 从当前 teleop agent 输出，经安全区 clamp 后写入数据集。
- `task` 和 `subtask`: 从 `configs/tasks/pen_pick_place.yaml` 读取。

也就是说，实时控制循环使用的是 `wrist_rgb` / `third_front_rgb`，而保存到
LeRobot dataset 后的图像 feature 名是 `observation.images.wrist` /
`observation.images.third_person`。如果以后改任务或训练 schema，需要同时检查
`src/ur5e_control/data_collection/observations.py` 中的 LeRobot 图像映射，以及
YAML 里的 `vla.image_key_map`。

交互按键：

- 终端提示后按 `Enter` 开始当前 episode。
- 采集中按 `Space` 在第一个和第二个 subtask 标签之间切换。
- 采集中按 `Esc` 保存当前 episode，并回到初始位。

## VLA 模型推理控制

VLA 推理入口：

```bash
uv run run-vla-inference --config configs/hardware/ur5e_d455.yaml
```

它会连接已经启动的 CloudRobo-VLA WebSocket server。当前客户端协议与
`WebSocketModelServer` 对齐，发送 msgpack 请求，不是 HTTP JSON。

推理前建议先执行：

```bash
uv run check-recording-ready \
  --config configs/hardware/ur5e_d455.yaml \
  --check-robot-tcp

uv run check-cameras --config configs/hardware/ur5e_d455.yaml
```

然后用小步数启动 VLA 控制：

```bash
uv run run-vla-inference \
  --config configs/hardware/ur5e_d455.yaml \
  --server-url ws://<vla-server-host>:8000 \
  --max-steps 20
```

确认动作方向和幅度安全后，再逐步放大 `--max-steps`，最后才去掉这个参数。

常用参数：

```bash
uv run run-vla-inference \
  --config configs/hardware/ur5e_d455.yaml \
  --server-url ws://<vla-server-host>:8000 \
  --repo-id chopintelligence/ur5e-pen-pick-place \
  --action-space abs \
  --execute-horizon 1 \
  --max-joint-delta 0.1 \
  --max-steps 100
```

字段含义：

- `--server-url`: VLA server WebSocket 地址。
- `--api-key`: server 启动了 API key 时使用。
- `--repo-id`: 发给 CloudRobo-VLA request preprocessor 的数据集/schema id。
- `--task`: 临时覆盖 task 文本。
- `--sub-task`: 临时设置 `sub_task`。
- `--action-space`: `abs` 表示模型输出绝对关节目标；`delta` 表示模型输出
  相对当前关节的增量。
- `--execute-horizon`: 每次 server 返回 action chunk 后执行多少步。
- `--max-joint-delta`: 每条命令允许的最大关节变化，单位 rad。
- `--max-steps`: 执行固定步数后退出，真机调试强烈建议使用。
- `--no-reset`: 启动时不向 server 发送 reset。

VLA request 默认字段映射来自 YAML：

```yaml
vla:
  state_key: "observation.state"
  image_key_map:
    third_front_rgb: "observation.images.third_front_image"
    wrist_rgb: "observation.images.wrist_image"
```

这里左侧的 `third_front_rgb` / `wrist_rgb` 是 live env observation key，
右侧是发送给 CloudRobo-VLA server 的 request key。它们不要求和 LeRobot
dataset 里的 `observation.images.third_person` / `observation.images.wrist`
完全同名，关键是要和 VLA server 端的数据 mixture schema 对齐。

如果 CloudRobo-VLA server 使用了不同 dataset mixture schema，需要同步修改
`vla.image_key_map` 和 `vla.repo_id`。

后续低延迟优化 TODO：

- 将图像数组从 pickle-over-ZMQ 迁移到 shared memory ring buffer，ZMQ 只传
  frame id、timestamp、shape 和 dtype。
- 增加独立 UR servo watchdog loop，由它固定频率消费最新安全 action。
- 增加相机 frame age、robot command round trip、recorder queue depth 和
  control-loop p95/p99 timing 指标。

## 重要配置字段参考

### `robot`

```yaml
robot:
  name: ur5e
  ip: "192.168.56.2"
  zmq_host: "127.0.0.1"
  zmq_port: 6001
  control_hz: 100
  reset_joints: [1.56, -1.57, 1.58, -1.58, -1.57, -0.0077, 1000.0]
```

`reset_joints` 是 6 个机械臂关节加 1 个夹爪命令。

### `camera`

```yaml
camera:
  hostname: "127.0.0.1"
  fps: 30
  enable_depth: false
  calibration_dir: "configs/cameras"
  realsense_ids:
    "333422300669": third_front
    "353522301461": third_left_top
    "333422304095": wrist
  intrinsics_files:
    third_front: "configs/cameras/intrinsics_third_front.yaml"
    wrist: "configs/cameras/intrinsics_wrist.yaml"
  extrinsics_file: "configs/cameras/extrinsics.yaml"
```

`enable_depth: false` 是默认值，此时 RealSense 只启 color stream，env obs
中不会产生 `*_depth` key。`realsense_ids` 必须在 YAML 中配置，代码里不再
写死 D455 序列号。

### `gripper`

```yaml
gripper:
  type: dh_pgc_140_50
  port: "/dev/ttyUSB1"
  position_tolerance: 10
  min_command_interval_s: 0.05
  position_read_interval_s: 0.1
```

PGC Modbus 链路比机械臂控制环慢。`position_tolerance` 和
`min_command_interval_s` 用于减少重复夹爪命令，降低长时间采集时夹爪卡住
的概率。

### `dataset`

```yaml
dataset:
  lerobot:
    repo_id: "chopintelligence/ur5e-pen-pick-place"
    root: ".datasets/lerobot/ur5e-pen-pick-place"
    fps: 30
    async_recording: true
    record_queue_size: 64
```

`root` 默认放在 `.datasets/` 下，不会提交到 git。`async_recording` 会把
`add_frame()` 放到后台 writer thread 中执行；`record_queue_size` 是有界
队列长度，队列满说明写盘/编码已经跟不上采集速度。

### `safety`

```yaml
safety:
  joint_limits_file: "configs/safety/ur5e_joint_limits.yaml"
```

数据采集和 VLA 推理都依赖这个文件。

### `vla`

```yaml
vla:
  server_url: "ws://127.0.0.1:8000"
  api_key:
  repo_id: "chopintelligence/ur5e-pen-pick-place"
  state_key: "observation.state"
  image_key_map:
    third_front_rgb: "observation.images.third_front_image"
    wrist_rgb: "observation.images.wrist_image"
  action_space: "abs"
  execute_horizon: 1
  request_timeout_s: 5.0
  max_joint_delta: 0.1
  reset_on_start: true
```

第一次真机推理建议保持：

```yaml
execute_horizon: 1
max_joint_delta: 0.1
```

## 环境变量覆盖

常用临时覆盖项：

| 环境变量 | 覆盖字段 |
| --- | --- |
| `UR5E_HARDWARE_CONFIG` | 配置文件路径 |
| `UR5E_ROBOT_IP` | `robot.ip` |
| `UR5E_ROBOT_ZMQ_HOST` | `robot.zmq_host` |
| `UR5E_ROBOT_ZMQ_PORT` | `robot.zmq_port` |
| `UR5E_CONTROL_HZ` | `robot.control_hz` |
| `UR5E_AGENT_NAME` | `agent.name` |
| `UR5E_AGENT_ROBOT` | `agent.robot` |
| `UR5E_GELLO_PORT` | `agent.gello_port` |
| `UR5E_SPACEMOUSE_DEVICE_PATH` | `agent.spacemouse.device_path` |
| `UR5E_SPACEMOUSE_TRANSLATION_SCALE` | `agent.spacemouse.translation_scale` |
| `UR5E_SPACEMOUSE_ANGLE_SCALE` | `agent.spacemouse.angle_scale` |
| `UR5E_SPACEMOUSE_INVERT_CONTROL` | `agent.spacemouse.invert_control`，逗号分隔 6 个数 |
| `UR5E_SPACEMOUSE_ROTATION_MODE` | `agent.spacemouse.rotation_mode` |
| `UR5E_SPACEMOUSE_INVERT_BUTTON` | `agent.spacemouse.invert_button` |
| `UR5E_SPACEMOUSE_GRIPPER_OPEN` | `agent.spacemouse.gripper_open` |
| `UR5E_SPACEMOUSE_GRIPPER_CLOSED` | `agent.spacemouse.gripper_closed` |
| `UR5E_CAMERA_HOSTNAME` | `camera.hostname` |
| `UR5E_CAMERA_FPS` | `camera.fps` |
| `UR5E_CAMERA_ENABLE_DEPTH` | `camera.enable_depth` |
| `UR5E_CAMERA_CALIBRATION_DIR` | `camera.calibration_dir` |
| `UR5E_CAMERA_EXTRINSICS_FILE` | `camera.extrinsics_file` |
| `UR5E_PGC_PORT` | `gripper.port` |
| `UR5E_PGC_POSITION_TOLERANCE` | `gripper.position_tolerance` |
| `UR5E_PGC_MIN_COMMAND_INTERVAL_S` | `gripper.min_command_interval_s` |
| `UR5E_PGC_POSITION_READ_INTERVAL_S` | `gripper.position_read_interval_s` |
| `UR5E_LEROBOT_REPO_ID` | `dataset.lerobot.repo_id` |
| `UR5E_LEROBOT_ROOT` | `dataset.lerobot.root` |
| `UR5E_LEROBOT_FPS` | `dataset.lerobot.fps` |
| `UR5E_LEROBOT_ASYNC_RECORDING` | `dataset.lerobot.async_recording` |
| `UR5E_LEROBOT_RECORD_QUEUE_SIZE` | `dataset.lerobot.record_queue_size` |
| `UR5E_JOINT_LIMITS_FILE` | `safety.joint_limits_file` |
| `UR5E_VLA_SERVER_URL` | `vla.server_url` |
| `UR5E_VLA_API_KEY` | `vla.api_key` |
| `UR5E_VLA_REPO_ID` | `vla.repo_id` |
| `UR5E_VLA_ACTION_SPACE` | `vla.action_space` |
| `UR5E_VLA_EXECUTE_HORIZON` | `vla.execute_horizon` |
| `UR5E_VLA_REQUEST_TIMEOUT_S` | `vla.request_timeout_s` |
| `UR5E_VLA_MAX_JOINT_DELTA` | `vla.max_joint_delta`，可用 `none` 关闭 |
| `UR5E_VLA_RESET_ON_START` | `vla.reset_on_start` |
| `UR5E_TASK_FILE` | `task.file` |
| `UR5E_EPISODE_COUNT` | `task.episode_count` |

示例：

```bash
export UR5E_GELLO_PORT=/dev/serial/by-id/usb-FTDI_GELLO-if00-port0
export UR5E_PGC_PORT=/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0
export UR5E_ROBOT_IP=192.168.56.2
export UR5E_LEROBOT_ROOT=/data/lerobot/ur5e-pen-pick-place
```

## 常见问题

### `check-recording-ready` 提示串口 missing

先确认设备是否插入：

```bash
ls -l /dev/serial/by-id/
```

再把 YAML 中的 `agent.gello_port` 或 `gripper.port` 改成稳定路径。

### D455 视频文件为空或打开失败

先确认序列号配置正确，再运行：

```bash
rs-enumerate-devices
uv run check-cameras --config configs/hardware/ur5e_d455.yaml --duration 5 --fps 30
```

如果两个视角保存反了，交换 `camera.realsense_ids` 中的逻辑名。

### 相机校准采样不够

程序只保存检测到标定板的帧。增加 `--samples` 前，先让标定板覆盖更多图像
区域并减少反光。必要时增大 `--interval`，给移动标定板或机械臂留出时间。

### 安全区文件不存在

先运行：

```bash
uv run calibrate-safety-zone --config configs/hardware/ur5e_d455.yaml --margin 0.02
```

没有安全区文件时，`record-teleop` 和 `run-vla-inference` 不会继续控制真机。

### PGC 夹爪长时间采集后卡住

提高这两个值可以减少 Modbus 写入频率：

```yaml
gripper:
  position_tolerance: 20
  min_command_interval_s: 0.08
```

如果仍然频繁卡住，优先检查 USB 转串口质量、供电和接地，再降低控制频率。

### VLA 动作方向不对或幅度过大

先使用小步数和保守限幅：

```bash
uv run run-vla-inference \
  --config configs/hardware/ur5e_d455.yaml \
  --server-url ws://<vla-server-host>:8000 \
  --action-space abs \
  --execute-horizon 1 \
  --max-joint-delta 0.05 \
  --max-steps 10
```

如果 server 输出的是 delta action，把 `--action-space` 改成 `delta`。
