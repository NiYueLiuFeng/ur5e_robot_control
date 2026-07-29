# D455 Camera Calibration Design

## Goal

Add a repository-native calibration workflow for the UR5e + dual RealSense D455
setup. The workflow supports camera intrinsics, wrist hand-eye extrinsics,
third-person camera-to-robot-base extrinsics, and dual-camera extrinsics without
requiring ROS.

## Architecture

Calibration lives in a new `ur5e_control.calibration` package with pure,
testable modules for pattern geometry, SE(3) math, solvers, capture helpers, and
YAML IO. A single `calibrate-cameras` CLI in `src/apps` exposes subcommands for
the four calibration modes and reuses the existing runtime config and RealSense
camera mapping.

## Calibration Target

The default target is a standard chessboard because the existing dependency
stack already includes `opencv-python-headless` through LeRobot. The
implementation also supports ChArUco when OpenCV ArUco support is available in
the environment.

## Supported Modes

### Intrinsics

The operator captures multiple frames for one logical camera, detects the
configured target, runs OpenCV camera calibration, and writes per-camera
intrinsics:

- image size
- camera matrix
- distortion coefficients
- reprojection error
- sample count

### Wrist Hand-Eye

The wrist D455 observes a target fixed in the robot workspace while the operator
moves the arm through varied poses. Each sample stores `base_T_tool` from the
robot and `camera_T_target` from the image. OpenCV hand-eye calibration returns
`tool_T_camera`.

### Base Extrinsics

The fixed third-person D455 observes a target attached to the robot TCP. Each
sample stores `base_T_tool`, `tool_T_target`, and `camera_T_target`. The solver
estimates `base_T_camera` per sample and averages the results.

### Dual-Camera Extrinsics

Both cameras observe the same target at the same time. Each sample stores
`reference_camera_T_target` and `camera_T_target`; the solver estimates the
transform from the reference camera to the other camera and averages samples.

## Data Flow

1. Load runtime config from `configs/hardware/ur5e_d455.yaml`.
2. Start the existing RealSense camera server when live capture is requested.
3. Capture color frames and optional robot TCP pose.
4. Detect calibration target corners.
5. Solve intrinsics or extrinsics.
6. Write YAML under `configs/cameras/` by default.

## Output Format

Generated YAML stores transforms as both a 4x4 matrix and decomposed
translation/quaternion fields. Intrinsics are stored separately from extrinsics
so data collection can load only the pieces it needs.

## Error Handling

The CLI validates that enough samples were captured, that required cameras exist
in `camera.realsense_ids`, and that robot TCP poses are present for robot-based
extrinsic solvers. Detection failures are reported as skipped samples rather
than silently poisoning the solve.

## Testing

Unit tests cover SE(3) math, transform averaging, extrinsic solvers with
synthetic data, YAML formatting, app import behavior, and runtime config parsing
for the new camera calibration output paths. Hardware-specific behavior is kept
behind injectable capture interfaces so it can be validated on the edge laptop.
