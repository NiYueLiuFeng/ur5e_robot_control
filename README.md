# UR5e Robot Control

UR5e Robot Control is the edge-runtime repository for Chop Intelligence UR5e
data collection and VLA-controlled robot execution.

中文说明：[README.zh-CN.md](README.zh-CN.md)

The repository is intentionally scoped to one physical stack:

- Universal Robots UR5e follower arm
- DH Robotics PGC-140-50 gripper
- GELLO leader arm or SpaceMouse 3D mouse teleoperation
- Two Intel RealSense D455 cameras: wrist view and third-person view
- Direct RGB-only LeRobot dataset v3 recording by default
- CloudRobo-VLA WebSocket inference client for real-robot control

Legacy multi-robot demos, ROS packages, simulation-first examples, alternate
recording backends, and npz conversion paths have been removed so the edge
laptop workflow stays small and maintainable.

## Current Status

Implemented:

- Runtime configuration from `configs/hardware/ur5e_d455.yaml`
- Hardware preflight checks
- RealSense D455 5 second MP4 stream checks, RGB-only by default
- Camera intrinsics, hand-eye, base extrinsics, and stereo extrinsics helpers
- Joint-space safety-zone calibration
- GELLO and SpaceMouse teleoperation
- Direct LeRobot v3 dataset recording with a bounded async writer
- CloudRobo-VLA msgpack/WebSocket client, request adapter, action adapter, and
  UR5e execution loop
- ZMQ process boundaries for robot and camera services, with bounded timeouts,
  request-socket recovery, and latest-frame camera caching

Important validation TODO:

- This server is not connected to UR5e / GELLO / D455 / PGC hardware, so the
  real robot motion loop still needs guarded validation on the edge laptop.
- Keep the conservative VLA defaults for the first real-machine run:
  `execute_horizon: 1` and `max_joint_delta: 0.1`.
- Run VLA control with a small `--max-steps` value first, confirm action
  direction and scale, then gradually expand the run.

## Quick Start

From the repository root:

```bash
cd /home/chopin/ur5e_robot_control
```

Install the runtime environment:

```bash
uv sync --no-dev
source .venv/bin/activate
```

Initialize the GELLO Dynamixel SDK submodule:

```bash
git submodule update --init third_party/DynamixelSDK
uv pip install -e third_party/DynamixelSDK/python
```

SpaceMouse support is optional:

```bash
uv sync --no-dev --extra spacemouse
git submodule update --init third_party/mujoco_menagerie
```

For development:

```bash
uv sync --dev
```

## Runtime Config

The same YAML config drives hardware checks, camera calibration, safety
calibration, teleoperation recording, and VLA inference. The runtime starts a
local UR robot ZMQ server process plus one RealSense camera process per
configured D455 serial number; ROS is not required.

At a high level, `record-teleop` and `run-vla-inference` share this live path:

1. Start the UR5e + PGC robot bridge.
2. Start D455 camera servers from `camera.realsense_ids`.
3. Continuously cache the newest RGB frame in each camera process.
4. Read robot state, latest images, and teleop/VLA actions in the control loop.
5. Clamp arm joints with the calibrated safety limits before commanding UR5e.

The default config is:

```bash
configs/hardware/ur5e_d455.yaml
```

Set the edge laptop values in this file:

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
```

Configuration loading order:

1. `--config <path>`
2. `UR5E_HARDWARE_CONFIG=<path>`
3. `configs/hardware/ur5e_d455.yaml`

Relative paths inside the config are resolved from the repository root.

The full operator guide is here:

[docs/ur5e_runtime_configuration.md](docs/ur5e_runtime_configuration.md)

## CLI Commands

| Command | Purpose |
| --- | --- |
| `uv run calibrate-gello` | Print GELLO joint offset suggestions |
| `uv run check-recording-ready` | Check dependencies, serial devices, camera config, optional UR TCP |
| `uv run check-cameras` | Record one short MP4 per configured D455 view |
| `uv run calibrate-cameras` | Calibrate D455 intrinsics/extrinsics |
| `uv run calibrate-safety-zone` | Generate UR5e joint safety limits through teleoperation |
| `uv run record-teleop` | Collect LeRobot v3 teleoperation episodes |
| `uv run run-vla-inference` | Run UR5e control from a CloudRobo-VLA WebSocket server |
| `uv run ws-control` | Run UR5e control from a JSON-over-WebSocket policy server |

## Hardware Bring-Up

Find stable serial paths:

```bash
ls -l /dev/serial/by-id/
```

Find RealSense serial numbers:

```bash
rs-enumerate-devices
```

Run preflight checks:

```bash
uv run check-recording-ready --config configs/hardware/ur5e_d455.yaml
```

When the edge laptop is connected to the UR controller network:

```bash
uv run check-recording-ready \
  --config configs/hardware/ur5e_d455.yaml \
  --check-robot-tcp
```

Check camera streams before data collection or VLA inference:

```bash
uv run check-cameras --config configs/hardware/ur5e_d455.yaml
```

This records MP4 files under:

```bash
.datasets/camera_check/<timestamp>/
```

The camera runtime reads RealSense D455 streams directly from Python. ROS is not
required. Depth is disabled by default; camera checks, teleoperation recording,
and VLA inference use RGB frames only unless `camera.enable_depth: true` is set.

## Camera Calibration

Generated calibration files are written under `configs/cameras/` and ignored by
git because they depend on the physical camera mounts.

Typical sequence:

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

Default output files:

```bash
configs/cameras/intrinsics_wrist.yaml
configs/cameras/intrinsics_third_front.yaml
configs/cameras/extrinsics.yaml
```

## Safety Zone

Data collection and VLA inference require a calibrated safety file:

```bash
uv run calibrate-safety-zone \
  --config configs/hardware/ur5e_d455.yaml \
  --margin 0.02
```

Move the robot through the desired workspace boundary with the configured
teleoperation input, then press `Esc` or `Ctrl-C` to save:

```bash
configs/safety/ur5e_joint_limits.yaml
```

Safety clamping applies to the first six UR5e arm joints. The gripper command is
preserved.

## Data Collection

The repository writes LeRobot v3 directly. It no longer saves npz episodes.
By default, recording is RGB-only and uses an async LeRobot writer thread so
dataset writes do not sit directly on the teleoperation control path.

Default dataset root:

```bash
.datasets/lerobot/ur5e-pen-pick-place
```

`.datasets/` is ignored by git.

Run teleoperation data collection:

```bash
uv run record-teleop --config configs/hardware/ur5e_d455.yaml
```

`record-teleop` takes its episode count, task file, LeRobot repo id, dataset
root, FPS, async-writer setting, and teleoperation mode from YAML. The default
task file is `configs/tasks/pen_pick_place.yaml`.

During recording:

- Press `Enter` when prompted to start an episode.
- Press `Space` to switch between configured subtask labels.
- Press `Esc` to finish and save the current episode.

The D455 camera bridge continuously captures the newest frame in each camera
process. The teleoperation loop requests the latest cached frame instead of
forcing a fresh hardware capture on every control step, reducing camera-induced
jitter. Camera and robot ZMQ clients also use bounded timeouts and recreate
their request sockets after timeout errors so retries do not inherit a stale
REQ/REP state.

The live environment observation uses camera keys from the logical camera names:

```text
wrist_rgb
third_front_rgb
joint_positions
gripper_position
```

Before writing a LeRobot frame, the current recording adapter maps those image
keys to dataset feature names:

```text
wrist_rgb          -> observation.images.wrist
third_front_rgb -> observation.images.third_person
joint_positions    -> observation.state
safe action         -> action
```

Teleoperation mode is selected with `agent.name`:

```yaml
agent:
  name: gello
```

or:

```yaml
agent:
  name: spacemouse
```

## VLA Control

`run-vla-inference` controls the robot from an already-running CloudRobo-VLA
WebSocket server. The client sends msgpack requests compatible with
`WebSocketModelServer`.

Before inference, run:

```bash
uv run check-recording-ready \
  --config configs/hardware/ur5e_d455.yaml \
  --check-robot-tcp

uv run check-cameras --config configs/hardware/ur5e_d455.yaml
```

Start with a short guarded run:

```bash
uv run run-vla-inference \
  --config configs/hardware/ur5e_d455.yaml \
  --server-url ws://<vla-server-host>:8000 \
  --execute-horizon 1 \
  --max-joint-delta 0.1 \
  --max-steps 20
```

Use `--action-space delta` only if the server is serving delta joint actions.
The default `abs` mode treats model actions as absolute joint targets.

The default request mapping is configured in YAML:

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

For VLA inference, image keys are read directly from the live env observation
and mapped into the CloudRobo-VLA request keys configured above. This is
separate from the LeRobot writer's `wrist` / `third_person` feature names.

If the VLA server uses a different CloudRobo dataset mixture schema, update
`vla.repo_id` and `vla.image_key_map` together.

## JSON Policy Control

`ws-control` is a smaller, JSON-only loop for a custom policy server. Each step
uploads the live observation:

```json
{
  "repo_id": "...",
  "task": "...",
  "state": [j1, j2, j3, j4, j5, j6, gripper],
  "images": {"third_person": "<base64 jpeg>", "wrist": "<base64 jpeg>"}
}
```

The server replies with an action chunk:

```json
{"action": [[j1, j2, j3, j4, j5, j6, gripper], [j1, j2, j3, j4, j5, j6, gripper]]}
```

Adjust the wire format in `src/apps/run_ws_control.py`
(`get_model_body` / `_parse_actions` / `_request`).

```bash
uv run ws-control \
  --config configs/hardware/ur5e_d455.yaml \
  --server-url ws://<policy-host>:8000 \
  --max-steps 20
```

If the console script is not on your `PATH`, run the same entry point as a
module (`uv run` installs the project in editable mode, so `src/` is
importable):

```bash
uv run python -m apps.run_ws_control \
  --config configs/hardware/ur5e_d455.yaml \
  --server-url ws://<policy-host>:8000 \
  --max-steps 20
```

Without `uv`, put `src/` on the path first:

```bash
PYTHONPATH=src python -m apps.run_ws_control \
  --config configs/hardware/ur5e_d455.yaml \
  --server-url ws://<policy-host>:8000 \
  --max-steps 20
```

`apps.run_ws_control:main` is the function registered as the `ws-control`
console script in `pyproject.toml`, so both forms accept the same options.

`vla.execute_horizon` actions are executed per response. The gripper value is
binarized with `vla.threshold` and sent as 0 or 1000.

Use `--model fastwam` (or `vla.model: fastwam`) to speak FastWAM's
msgpack/WebSocket protocol. That path sends raw JPEG bytes and a joint-state
vector directly as an `infer` frame (no handshake; the token is appended to the
URL), then executes the action chunk on the robot exactly like the `json` path
(gripper binarized with `vla.threshold`). Request fields are fixed in
`run_ws_control.py` (`IMAGE_FIELDS`, `FASTWAM_IMAGE_FIELDS`).

Use `--model openpi05` for OpenPI 0.5's msgpack-numpy protocol. The body carries
CHW uint8 image arrays, a 7-value float32 single-arm state and a `prompt`;
returned actions are parsed and logged without commanding the robot. Its field
mapping lives in `OPENPI_IMAGE_FIELDS`.

## Project Layout

```text
configs/cameras/              Generated D455 calibration YAML
configs/hardware/             UR5e + D455 + PGC runtime YAML
configs/safety/               Generated joint safety limits
configs/tasks/                Language task and subtask metadata
docs/                         Operator guide and implementation plans
src/apps/                     CLI entry points
src/ur5e_control/calibration/ Camera calibration capture, IO, solvers, transforms
src/ur5e_control/data_collection/
                              LeRobot v3 recording and teleoperation runner
src/ur5e_control/data_collection/teleop/gello/
                              GELLO leader-arm backend
src/ur5e_control/data_collection/teleop/spacemouse/
                              SpaceMouse backend
src/ur5e_control/hardware/    UR5e, PGC gripper, and RealSense adapters
src/ur5e_control/inference/   CloudRobo-VLA client, request/action adapters, controller
src/ur5e_control/runtime/     Shared config, safety, transport, and data contracts
tests/                        Hardware-free unit tests
third_party/                  DynamixelSDK and optional MuJoCo menagerie submodules
```

## Development Checks

The default test path is hardware-free:

```bash
uv lock --check
uv run python -m unittest discover -s tests -v
uv run python -m compileall src tests
```

On machines where `uv run` would sync large optional stacks, use the existing
virtual environment directly:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m compileall src tests
```

Real robot, gripper, GELLO, and camera behavior must be validated on the edge
laptop connected to the hardware.

Future low-latency work:

- Move image arrays from pickle-over-ZMQ to shared memory ring buffers.
- Add an independent UR servo watchdog loop that consumes the latest safe
  target action.
- Add latency metrics for camera frame age, robot command round trip, recorder
  queue depth, and control-loop p95/p99 timing.

## License

This repository keeps the upstream GELLO MIT license. See [LICENSE](LICENSE).
