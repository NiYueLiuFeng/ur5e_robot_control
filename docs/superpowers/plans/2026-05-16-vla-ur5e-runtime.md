# VLA UR5e Runtime Plan

## Goal

Make `run-vla-inference` usable for UR5e + D455 + PGC control against an already-running CloudRobo-VLA WebSocket server.

## Plan

- [ ] Add tests for the CloudRobo-VLA WebSocket/msgpack protocol, live observation request mapping, VLA action chunk parsing, and the inference controller loop.
- [ ] Implement a small WebSocket client compatible with `WebSocketModelServer` without importing the CloudRobo-VLA repo.
- [ ] Implement a request adapter that maps `RobotEnv.get_obs()` output into the flat CloudRobo-VLA request schema.
- [ ] Extend the action adapter to accept `actions` chunks, validate dimensions, support absolute and delta action spaces, and limit large joint jumps before safety clamping.
- [ ] Add a hardware controller that starts the robot/camera runtime, loads safety limits, calls the VLA server, and executes actions through `RobotEnv`.
- [ ] Replace the scaffold CLI with real `run-vla-inference` arguments and runtime config defaults.
- [ ] Add VLA config to `configs/hardware/ur5e_d455.yaml` and dependency metadata.
- [ ] Run unit tests, compile checks, lock checks, and diff checks.
