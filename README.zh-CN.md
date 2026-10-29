# UR5e Robot Control（中文说明）

[English README](README.md)

本仓库是 Chop Intelligence UR5e 数据采集与 VLA 机器人执行的边缘运行时仓库。

仓库刻意只面向一套物理硬件栈：

- Universal Robots UR5e 从臂
- DH Robotics PGC-140-50 夹爪
- GELLO 主臂或 SpaceMouse 三维鼠标遥操作
- 两路 Intel RealSense D455 相机：腕部视角与第三人称视角
- 默认直接录制 RGB-only 的 LeRobot dataset v3
- 用于真机控制的 CloudRobo-VLA WebSocket 推理客户端

旧的多机器人示例、ROS 包、以仿真为主的示例、其他录制后端以及 npz 转换路径
都已移除，以便边缘笔记本上的工作流保持精简、可维护。

## 当前状态

已实现：

- 从 `configs/hardware/ur5e_d455.yaml` 加载运行时配置
- 硬件预检
- RealSense D455 5 秒 MP4 视频流检查（默认只用 RGB）
- 相机内参、手眼、基座外参和双目外参标定辅助工具
- 关节空间安全区域标定
- GELLO 与 SpaceMouse 遥操作
- 基于有界异步写入线程的直接 LeRobot v3 数据集录制
- CloudRobo-VLA msgpack/WebSocket 客户端、请求适配器、动作适配器和
  UR5e 执行回路
- 机器人与相机服务之间的 ZMQ 进程边界，带超时、请求套接字恢复和
  最新帧相机缓存
- 新增 `ws-control`：支持 JSON / FastWAM / LingBot-VA / OpenWAM / OpenPI 0.5 /
  Flex-π 协议的策略控制入口，且只执行 7 维单臂关节动作

重要验证事项：

- 本仓库的自动化测试不连接 UR5e / GELLO / D455 / PGC 硬件，真机运动回路
  仍需在连接硬件的边缘笔记本上做受控验证。
- 首次真机运行请保持保守的 VLA 默认值：`execute_horizon: 1`、
  `max_joint_delta: 0.1`。
- 先用较小的 `--max-steps` 运行 VLA 控制，确认动作方向和尺度后，再逐步扩大运行规模。

## 快速开始

在仓库根目录执行：

```bash
cd <仓库根目录>
```

安装运行时环境：

```bash
uv sync --no-dev
source .venv/bin/activate
```

初始化 GELLO 使用的 Dynamixel SDK 子模块：

```bash
git submodule update --init third_party/DynamixelSDK
uv pip install -e third_party/DynamixelSDK/python
```

SpaceMouse 支持是可选的：

```bash
uv sync --no-dev --extra spacemouse
git submodule update --init third_party/mujoco_menagerie
```

开发环境：

```bash
uv sync --dev
```

## 运行时配置

同一个 YAML 配置驱动硬件检查、相机标定、安全区域标定、遥操作录制和 VLA 推理。
运行时会为 UR 机器人启动一个本地 ZMQ 服务进程，并按 `camera.realsense_ids`
中配置的每个 D455 序列号启动一个 RealSense 相机进程；不需要 ROS。

`record-teleop`、`run-vla-inference` 和 `ws-control` 共享同一条实时链路：

1. 启动 UR5e + PGC 机器人桥接进程。
2. 根据 `camera.realsense_ids` 启动 D455 相机服务。
3. 在每个相机进程中持续缓存最新的 RGB 帧。
4. 在控制回路中读取机器人状态、最新图像和遥操作/VLA 动作。
5. 下发 UR5e 指令前，用标定好的安全限位裁剪机械臂关节。

默认配置文件：

```bash
configs/hardware/ur5e_d455.yaml
```

需要在其中填写边缘笔记本的实际硬件参数：

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

dataset:
  lerobot:
    root: ".datasets/lerobot/ur5e-pen-pick-place"
    async_recording: true
    record_queue_size: 64

vla:
  execute_horizon: 1
  max_joint_delta: 0.1
  threshold: 0.7
  model: "json"
```

配置加载顺序：

1. `--config <path>`
2. `UR5E_HARDWARE_CONFIG=<path>`
3. `configs/hardware/ur5e_d455.yaml`

配置文件中的相对路径都相对仓库根目录解析。

完整操作指南见：

[docs/ur5e_runtime_configuration.md](docs/ur5e_runtime_configuration.md)

## CLI 命令

| 命令 | 用途 |
| --- | --- |
| `uv run calibrate-gello` | 打印 GELLO 关节偏移建议值 |
| `uv run check-recording-ready` | 检查依赖、串口设备、相机配置，可选检查 UR TCP |
| `uv run check-cameras` | 为每路配置的 D455 视角录制一段短视频 |
| `uv run calibrate-cameras` | 标定 D455 内参/外参 |
| `uv run calibrate-safety-zone` | 通过遥操作生成 UR5e 关节安全限位 |
| `uv run record-teleop` | 采集 LeRobot v3 遥操作数据 |
| `uv run run-vla-inference` | 通过 CloudRobo-VLA WebSocket 服务控制 UR5e |
| `uv run ws-control` | 通过 WebSocket 策略服务（JSON / msgpack / msgpack-numpy）控制 UR5e |

## 硬件联调

查找稳定的串口路径：

```bash
ls -l /dev/serial/by-id/
```

查找 RealSense 序列号：

```bash
rs-enumerate-devices
```

运行预检：

```bash
uv run check-recording-ready --config configs/hardware/ur5e_d455.yaml
```

当边缘笔记本已连接到 UR 控制器网络时：

```bash
uv run check-recording-ready \
  --config configs/hardware/ur5e_d455.yaml \
  --check-robot-tcp
```

在数据采集或 VLA 推理前检查相机视频流：

```bash
uv run check-cameras --config configs/hardware/ur5e_d455.yaml
```

MP4 文件会写入：

```bash
.datasets/camera_check/<timestamp>/
```

相机运行时直接从 Python 读取 RealSense D455 视频流，不需要 ROS。默认关闭深度；
除非设置 `camera.enable_depth: true`，相机检查、遥操作录制和 VLA 推理都只使用 RGB 帧。

## 相机标定

生成的标定文件写入 `configs/cameras/`，因为它们依赖实际相机安装位置，已被 git 忽略。

典型流程：

```bash
uv run calibrate-cameras --config configs/hardware/ur5e_d455.yaml \
  intrinsics --camera wrist --samples 20

uv run calibrate-cameras --config configs/hardware/ur5e_d455.yaml \
  intrinsics --camera third_front --samples 20

uv run calibrate-cameras --config configs/hardware/ur5e_d455.yaml \
  hand-eye --camera wrist --samples 12

uv run calibrate-cameras --config configs/hardware/ur5e_d455.yaml \
  base-extrinsics --camera third_front --samples 12

uv run calibrate-cameras --config configs/hardware/ur5e_d455.yaml \
  stereo-extrinsics --reference-camera wrist --camera third_front --samples 12
```

默认输出文件：

```bash
configs/cameras/intrinsics_wrist.yaml
configs/cameras/intrinsics_third_front.yaml
configs/cameras/extrinsics.yaml
```

## 安全区域

数据采集和 VLA 推理需要一份已标定的安全限位文件：

```bash
uv run calibrate-safety-zone \
  --config configs/hardware/ur5e_d455.yaml \
  --margin 0.02
```

使用配置好的遥操作输入让机器人走遍期望的工作空间边界，然后按 `Esc` 或
`Ctrl-C` 保存：

```bash
configs/safety/ur5e_joint_limits.yaml
```

安全裁剪作用于 UR5e 的前六个机械臂关节，夹爪指令保持不变。

## 数据采集

仓库直接写入 LeRobot v3，不再保存 npz episode。默认录制只使用 RGB，并通过异步
LeRobot 写入线程写盘，避免数据集写入直接阻塞遥操作控制回路。

默认数据集根目录：

```bash
.datasets/lerobot/ur5e-pen-pick-place
```

`.datasets/` 已被 git 忽略。

运行遥操作数据采集：

```bash
uv run record-teleop --config configs/hardware/ur5e_d455.yaml
```

`record-teleop` 从 YAML 读取 episode 数量、任务文件、LeRobot repo id、数据集根目录、
FPS、异步写入开关和遥操作模式。默认任务文件是
`configs/tasks/pen_pick_place.yaml`。

录制过程中：

- 按提示按 `Enter` 开始一个 episode。
- 按 `Space` 在配置好的子任务标签之间切换。
- 按 `Esc` 结束并保存当前 episode。

D455 相机桥接进程会持续缓存每路相机的最新帧。遥操作回路读取最新缓存帧，而不是在
每个控制周期强制重新采集硬件图像，从而减少相机抖动。相机和机器人 ZMQ 客户端也
使用有界超时，并在超时后重建请求套接字，避免重试继承失效的 REQ/REP 状态。

实时环境观测使用逻辑相机名作为键：

```text
wrist_rgb
third_front_rgb
joint_positions
gripper_position
```

写入 LeRobot 帧之前，录制适配器会把这些图像键映射为数据集特征名：

```text
wrist_rgb          -> observation.images.wrist
third_front_rgb -> observation.images.third_person
joint_positions    -> observation.state
safe action         -> action
```

遥操作模式通过 `agent.name` 选择：

```yaml
agent:
  name: gello
```

或：

```yaml
agent:
  name: spacemouse
```

## VLA 控制

`run-vla-inference` 从已经运行的 CloudRobo-VLA WebSocket 服务控制机器人。
客户端发送与 `WebSocketModelServer` 兼容的 msgpack 请求。

推理前先运行：

```bash
uv run check-recording-ready \
  --config configs/hardware/ur5e_d455.yaml \
  --check-robot-tcp

uv run check-cameras --config configs/hardware/ur5e_d455.yaml
```

先用受控的短运行开始：

```bash
uv run run-vla-inference \
  --config configs/hardware/ur5e_d455.yaml \
  --server-url ws://<vla-server-host>:8000 \
  --execute-horizon 1 \
  --max-joint-delta 0.1 \
  --max-steps 20
```

只有当服务端提供 delta 关节动作时才使用 `--action-space delta`。默认的 `abs`
模式会把模型动作当作绝对关节目标。

默认请求映射在 YAML 中配置：

```yaml
vla:
  repo_id: "chopintelligence/ur5e-pen-pick-place"
  state_key: "observation.state"
  image_key_map:
    third_front_rgb: "observation.images.third_front_image"
    wrist_rgb: "observation.images.wrist_image"
  action_space: "abs"
  execute_horizon: 1
  max_joint_delta: 0.1
```

VLA 推理会直接从实时环境观测中读取图像键，并映射到上面配置的 CloudRobo-VLA
请求键。这与 LeRobot 写入器的 `wrist` / `third_person` 特征名是两套命名。

如果 VLA 服务端使用不同的 CloudRobo 数据集 mixture schema，请同时修改
`vla.repo_id` 和 `vla.image_key_map`。

## JSON 策略控制（ws-control）

`ws-control` 是面向自定义策略服务的更小的 JSON 控制回路。每一步上传实时观测：

```json
{
  "repo_id": "...",
  "task": "...",
  "state": [j1, j2, j3, j4, j5, j6, gripper],
  "images": {"third_person": "<base64 jpeg>", "wrist": "<base64 jpeg>"}
}
```

服务端返回一个动作块：

```json
{"action": [[j1, j2, j3, j4, j5, j6, gripper], [j1, j2, j3, j4, j5, j6, gripper]]}
```

如需调整实际报文格式，修改 `src/apps/run_ws_control.py`
（`get_model_body` / `_prepare_server` / `_request` / `_parse_actions`）。

```bash
uv run ws-control \
  --config configs/hardware/ur5e_d455.yaml \
  --server-url ws://<policy-host>:8000 \
  --max-steps 20
```

如果 `ws-control` 控制台脚本不在 `PATH` 中，可以直接以模块方式运行同一个入口
（`uv run` 会以可编辑模式安装项目，因此 `src/` 可被导入）：

```bash
uv run python -m apps.run_ws_control \
  --config configs/hardware/ur5e_d455.yaml \
  --server-url ws://<policy-host>:8000 \
  --max-steps 20
```

不通过 `uv` 运行时，先把 `src/` 加入 `PYTHONPATH`：

```bash
PYTHONPATH=src python -m apps.run_ws_control \
  --config configs/hardware/ur5e_d455.yaml \
  --server-url ws://<policy-host>:8000 \
  --max-steps 20
```

`apps.run_ws_control:main` 就是 `pyproject.toml` 中注册为 `ws-control` 控制台脚本的
同一个函数，两种方式的参数完全一致。

每个响应会执行 `vla.execute_horizon` 条动作。夹爪值使用 `vla.threshold` 二值化后
以 0 或 1000 下发。

每条动作在下发前都会校验：必须是 7 维（单臂 UR5 = 6 关节 + 夹爪）有限值，
6 个关节值在 ±2π rad 内，且单关节每次变化不超过 `vla.max_joint_delta`。
模型动作按数据集帧率（15 Hz）节流，而不是 100 Hz 控制频率。

`--model` 还可选择其他模型的交付协议：

| `--model` | 传输 | 动作语义 | 能否直接执行 |
| --- | --- | --- | --- |
| `json` | JSON 文本 | 6 关节 + 夹爪 | ✅ |
| `fastwam` | msgpack | 6 关节 + 夹爪（`autogen_mixed_ur5`） | ✅ |
| `lingbot` | msgpack-numpy | 6 关节 + 夹爪（`ur5` profile） | ✅ |
| `openwam` | JSON 文本 | 6 关节 + 夹爪（`ur5` 数据集） | ✅ |
| `pi05`（旧名 `openpi05`） | msgpack-numpy | 绝对末端位姿 + 夹爪 | ❌ 仅 `--dry-run` |
| `flexpi` | msgpack-numpy | 13 维 HS13 末端状态 | ❌ 仅 `--dry-run` |

`fastwam` 把原始 JPEG 字节和关节状态直接作为 `infer` 帧发送（无握手，token
追加在 URL 中）。`lingbot` 先消费连接元数据，再发
`{"reset": true, "prompt": ...}`，并把 `(dim, chunk, per_frame)` 响应里的
零上下文动作块丢掉。`openwam` 先完成 `ping` → `reset` 握手，`representation`
不是 `joint` 时直接拒绝，图像用 base64 PNG。`pi05` 和 `flexpi` 消费 OpenPI
元数据帧、图像为 HWC uint8；它们的检查点面向其他机械臂，输出末端位姿，
所以不加 `--dry-run` 会在连接机器人之前报错。

无论哪个模型，返回动作不是 7 维时都会打印
`WARNING: ... expected 7-dim single-arm UR5 actions, got D`；真机执行前直接
停止，`--dry-run` 下只告警并继续预览。

字段映射位于 `run_ws_control.py` 的 `IMAGE_FIELDS`、`FASTWAM_IMAGE_FIELDS`、
`PI05_IMAGE_FIELDS`、`LINGBOT_IMAGE_FIELDS`、`FLEXPI_IMAGE_FIELDS`、
`OPENWAM_IMAGE_FIELDS`。

更完整的 `ws-control` 操作说明（含硬件自检、各协议报文和排错）见
[README_ws-ctl-robot.md](README_ws-ctl-robot.md)。

协议的选择方式：

```yaml
vla:
  model: "json"      # json（默认）、fastwam、lingbot、openwam、pi05、flexpi
```

也可以在命令行用 `--model` 覆盖 YAML；换模型先用 `--dry-run` 确认动作数值。

## 项目结构

```text
configs/cameras/              生成的 D455 标定 YAML
configs/hardware/             UR5e + D455 + PGC 运行时 YAML
configs/safety/               生成的关节安全限位
configs/tasks/                语言任务与子任务元数据
docs/                         操作指南与实现计划
src/apps/                     CLI 入口
src/ur5e_control/calibration/ 相机标定采集、IO、求解与坐标变换
src/ur5e_control/data_collection/
                              LeRobot v3 录制与遥操作运行器
src/ur5e_control/data_collection/teleop/gello/
                              GELLO 主臂后端
src/ur5e_control/data_collection/teleop/spacemouse/
                              SpaceMouse 后端
src/ur5e_control/hardware/    UR5e、PGC 夹爪与 RealSense 适配器
src/ur5e_control/inference/   CloudRobo-VLA 客户端、请求/动作适配器、控制器、ws-control 传输
src/ur5e_control/runtime/     共享配置、安全、传输与数据契约，以及机器人环境构建
tests/                        不依赖硬件的单元测试
third_party/                  DynamixelSDK 与可选的 MuJoCo menagerie 子模块
```

## 开发检查

默认测试路径不依赖硬件：

```bash
uv lock --check
uv run python -m unittest discover -s tests -v
uv run python -m compileall src tests
```

如果 `uv run` 会同步体积较大的可选依赖，可直接使用已有的虚拟环境：

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m compileall src tests
```

真机、夹爪、GELLO 和相机行为必须在连接硬件的边缘笔记本上验证。

后续低延迟优化方向：

- 把通过 ZMQ pickle 传输的图像数组改为共享内存环形缓冲区。
- 增加独立的 UR servo 看门狗回路，消费最新的安全目标动作。
- 增加相机帧龄、机器人指令往返、录制队列深度以及控制回路 p95/p99 时延指标。

## 许可证

本仓库沿用上游 GELLO 的 MIT 许可证，详见 [LICENSE](LICENSE)。
