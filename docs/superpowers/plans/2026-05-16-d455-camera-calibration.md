# D455 Camera Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native `calibrate-cameras` workflow for D455 intrinsics, wrist hand-eye, third-person base extrinsics, and dual-camera extrinsics.

**Architecture:** Add a focused `ur5e_control.calibration` package for pattern detection, SE(3) transforms, solvers, capture, and YAML IO. Expose the package through one CLI entry point that follows the existing `src/apps` style and runtime config loader.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, PyYAML, pyrealsense2, existing UR5e runtime config and ZMQ camera stack.

---

### Task 1: Calibration Data Model And Transform Math

**Files:**
- Create: `src/ur5e_control/calibration/__init__.py`
- Create: `src/ur5e_control/calibration/types.py`
- Create: `src/ur5e_control/calibration/transforms.py`
- Test: `tests/test_calibration_transforms.py`

- [ ] **Step 1: Write failing transform tests**

```python
def test_ur_pose_vector_round_trips_through_matrix():
    pose = [0.1, -0.2, 0.3, 0.0, 0.0, 1.57079632679]
    matrix = matrix_from_ur_pose_vector(pose)
    recovered = ur_pose_vector_from_matrix(matrix)
    self.assertSequenceAlmostEqual(recovered[:3], pose[:3])
    self.assertAlmostEqual(recovered[5], pose[5], places=6)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `python -m unittest tests.test_calibration_transforms -v`

- [ ] **Step 3: Implement transform helpers**

Implement matrix construction, inversion, composition, quaternion conversion,
rotation-vector conversion, and transform averaging with NumPy.

- [ ] **Step 4: Run transform tests and confirm they pass**

Run: `python -m unittest tests.test_calibration_transforms -v`

### Task 2: Solver Layer

**Files:**
- Create: `src/ur5e_control/calibration/solvers.py`
- Test: `tests/test_calibration_solvers.py`

- [ ] **Step 1: Write failing synthetic extrinsics tests**

```python
def test_base_extrinsics_solver_recovers_fixed_camera_pose():
    result = solve_base_extrinsics(samples, tool_T_target=identity_matrix())
    assert_matrix_close(result.matrix, expected_base_T_camera)
```

- [ ] **Step 2: Run the solver tests and confirm they fail**

Run: `python -m unittest tests.test_calibration_solvers -v`

- [ ] **Step 3: Implement base, stereo, and hand-eye solvers**

Use direct SE(3) composition for base/stereo solvers and OpenCV
`calibrateHandEye` for wrist hand-eye calibration.

- [ ] **Step 4: Run solver tests and confirm they pass**

Run: `python -m unittest tests.test_calibration_solvers -v`

### Task 3: Pattern Detection And Intrinsics

**Files:**
- Create: `src/ur5e_control/calibration/patterns.py`
- Extend: `src/ur5e_control/calibration/solvers.py`
- Test: `tests/test_calibration_patterns.py`

- [ ] **Step 1: Write failing geometry tests**

```python
def test_chessboard_object_points_follow_row_major_order():
    pattern = ChessboardPattern(columns=3, rows=2, square_size_m=0.04)
    points = chessboard_object_points(pattern)
    self.assertEqual(points[4].tolist(), [0.04, 0.04, 0.0])
```

- [ ] **Step 2: Run the geometry tests and confirm they fail**

Run: `python -m unittest tests.test_calibration_patterns -v`

- [ ] **Step 3: Implement target geometry and OpenCV detection**

Add chessboard detection and optional ChArUco detection that raises a clear
dependency error when `cv2.aruco` is unavailable.

- [ ] **Step 4: Run pattern tests and confirm they pass**

Run: `python -m unittest tests.test_calibration_patterns -v`

### Task 4: YAML IO And Runtime Config

**Files:**
- Create: `src/ur5e_control/calibration/io.py`
- Modify: `src/ur5e_control/runtime/config.py`
- Modify: `configs/hardware/ur5e_d455.yaml`
- Test: `tests/test_calibration_io.py`
- Extend: `tests/test_runtime_config.py`

- [ ] **Step 1: Write failing IO/config tests**

```python
def test_calibration_config_defaults_to_configs_cameras():
    config = load_runtime_config(environ={})
    self.assertEqual(config.camera.calibration_dir, "configs/cameras")
```

- [ ] **Step 2: Run IO/config tests and confirm they fail**

Run: `python -m unittest tests.test_calibration_io tests.test_runtime_config -v`

- [ ] **Step 3: Implement YAML writers and config fields**

Add `camera.calibration_dir`, `camera.intrinsics_files`, and
`camera.extrinsics_file` with validation and repo-relative path resolution.

- [ ] **Step 4: Run IO/config tests and confirm they pass**

Run: `python -m unittest tests.test_calibration_io tests.test_runtime_config -v`

### Task 5: Live Capture And CLI

**Files:**
- Create: `src/ur5e_control/calibration/capture.py`
- Create: `src/apps/calibrate_cameras.py`
- Modify: `src/ur5e_control/hardware/robots/ur.py`
- Modify: `pyproject.toml`
- Extend: `tests/test_runtime_layering.py`

- [ ] **Step 1: Write failing CLI import tests**

```python
def test_app_modules_import_without_starting_hardware(self):
    self.assertTrue(importlib.import_module("apps.calibrate_cameras"))
```

- [ ] **Step 2: Run import tests and confirm they fail**

Run: `python -m unittest tests.test_runtime_layering -v`

- [ ] **Step 3: Implement CLI and capture helpers**

Add subcommands for `intrinsics`, `hand-eye`, `base-extrinsics`, and
`stereo-extrinsics`. Keep hardware imports inside command execution paths.

- [ ] **Step 4: Run import tests and confirm they pass**

Run: `python -m unittest tests.test_runtime_layering -v`

### Task 6: Documentation And Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/ur5e_runtime_configuration.md`

- [ ] **Step 1: Document calibration workflow**

Add concise bring-up commands and explain where outputs are written.

- [ ] **Step 2: Run focused unit tests**

Run: `python -m unittest tests.test_calibration_transforms tests.test_calibration_solvers tests.test_calibration_patterns tests.test_calibration_io tests.test_runtime_config tests.test_runtime_layering -v`

- [ ] **Step 3: Run full test suite if the environment has completed dependency installation**

Run: `uv run pytest`
