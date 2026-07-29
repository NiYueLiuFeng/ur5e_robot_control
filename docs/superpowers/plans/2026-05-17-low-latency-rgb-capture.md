# Low Latency RGB Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce teleoperation capture latency and jitter while making RealSense D455 depth capture disabled by default.

**Architecture:** Keep ZMQ as the process-boundary transport, but change camera servers from request-time capture to continuous latest-frame capture. Keep the control loop data model RGB-only by default, and move LeRobot `add_frame()` work behind a bounded async recorder so disk/video encoding work does not sit directly on the control path.

**Tech Stack:** Python multiprocessing/threading, pyzmq, pyrealsense2, numpy, LeRobot v3, unittest.

---

### Task 1: Runtime Config for RGB-Only Capture

**Files:**
- Modify: `src/ur5e_control/runtime/config.py`
- Modify: `configs/hardware/ur5e_d455.yaml`
- Test: `tests/test_runtime_config.py`

- [x] Add `camera.enable_depth: false` to the YAML defaults.
- [x] Add `CameraRuntimeConfig.enable_depth: bool = False`.
- [x] Load `camera.enable_depth` from YAML.
- [x] Add `UR5E_CAMERA_ENABLE_DEPTH` environment override.
- [x] Test that depth is disabled by default and can be enabled explicitly.

### Task 2: RGB-Only RealSense and Env Observations

**Files:**
- Modify: `src/ur5e_control/hardware/cameras/base.py`
- Modify: `src/ur5e_control/hardware/cameras/realsense.py`
- Modify: `src/ur5e_control/data_collection/teleop/env.py`
- Test: `tests/test_realsense_rgb_only.py`
- Test: `tests/test_teleop_env_camera_depth.py`

- [x] Add `enable_depth` to `RealSenseCamera`.
- [x] When `enable_depth=False`, only enable the color stream.
- [x] Make camera `read()` return `(rgb, None)` when depth is disabled.
- [x] Store `*_depth` in `RobotEnv.get_obs()` only when the camera returns a depth array.
- [x] Test with a fake `pyrealsense2` module that depth stream is not enabled by default.
- [x] Test that env observations contain RGB keys but no depth keys when depth is `None`.

### Task 3: Latest-Frame Camera Bridge and ZMQ Hardening

**Files:**
- Modify: `src/ur5e_control/hardware/cameras/zmq.py`
- Modify: `src/ur5e_control/hardware/cameras/realsense.py`
- Modify: `src/ur5e_control/runtime/transport/robot_zmq.py`
- Test: `tests/test_zmq_camera_latest_frame.py`

- [x] Add a `LatestFrameCache` helper that continuously stores the newest camera frame.
- [x] Use the cache inside `ZMQServerCamera` so client reads return the newest frame instead of forcing a fresh hardware read.
- [x] Add `RCVTIMEO`, `SNDTIMEO`, and `LINGER=0` to camera and robot ZMQ sockets.
- [x] Rebuild camera and robot REQ sockets after timeout/socket errors so retries are not poisoned by ZMQ state.
- [x] Raise clear runtime errors on camera client timeouts.
- [x] Test that latest-frame cache returns the newest captured frame and handles `depth=None`.
- [x] Test that camera and robot clients reset sockets after request timeouts.

### Task 4: Async LeRobot Recording

**Files:**
- Modify: `src/ur5e_control/data_collection/lerobot_v3.py`
- Modify: `src/ur5e_control/data_collection/teleop/runner.py`
- Modify: `src/ur5e_control/runtime/config.py`
- Modify: `configs/hardware/ur5e_d455.yaml`
- Test: `tests/test_lerobot_recording.py`
- Test: `tests/test_runtime_config.py`

- [x] Add `dataset.lerobot.async_recording: true` and `dataset.lerobot.record_queue_size: 64`.
- [x] Add an `AsyncLeRobotV3Recorder` wrapper with a bounded queue and writer thread.
- [x] Ensure `save_episode()` drains queued frames before saving.
- [x] Ensure `finalize()` drains and stops the writer thread.
- [x] Propagate writer-thread exceptions to the main thread.
- [x] Wrap the recorder in teleop collection when async recording is enabled.

### Task 5: Docs, Verification, Commit

**Files:**
- Modify: `README.md`
- Modify: `docs/ur5e_runtime_configuration.md`

- [x] Document RGB-only default and how to re-enable depth.
- [x] Document latest-frame camera behavior and async recording.
- [x] Keep shared-memory image transport and independent UR servo watchdog as future TODO.
- [x] Run targeted tests.
- [x] Run full hardware-free tests.
- [x] Run compile, format, lock, and diff checks.
- [x] Commit and push the branch.
