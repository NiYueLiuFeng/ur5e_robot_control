"""DH PGC-140-50 gripper adapter."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable


class PGCGripper:
    """Minimal Modbus adapter for the DH PGC-140-50 gripper."""

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        sdk_path: str | None = None,
        init_timeout_s: float = 10.0,
        force: int = 100,
        speed: int = 100,
        position_tolerance: int = 10,
        min_command_interval_s: float = 0.05,
        position_read_interval_s: float = 0.1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.port = port
        self.baudrate = int(baudrate)
        self.current_speed = int(speed)
        self.current_force = int(force)
        self.position_tolerance = max(0, int(position_tolerance))
        self.min_command_interval_s = max(0.0, float(min_command_interval_s))
        self.position_read_interval_s = max(0.0, float(position_read_interval_s))
        self._clock = clock
        self._last_target_position: int | None = None
        self._last_command_time = float("-inf")
        self._cached_position: int | None = None
        self._last_position_read_time = float("-inf")
        module = _load_dh_gripper_module(sdk_path)
        self._gripper = module.dh_modbus_gripper()
        self._gripper.open(self.port, self.baudrate)
        self._gripper.Initialization()
        self._wait_initialized(timeout_s=init_timeout_s)
        self._gripper.SetTargetForce(self.current_force)
        self._gripper.SetTargetSpeed(self.current_speed)

    def grip(self, position: int, speed: int | None = None, force: int | None = None):
        target_position = _clamp_pgc_position(position)
        if speed is not None:
            self.current_speed = int(speed)
            self._gripper.SetTargetSpeed(self.current_speed)
        if force is not None:
            self.current_force = int(force)
            self._gripper.SetTargetForce(self.current_force)

        now = self._clock()
        if not self._should_send_position(target_position, now):
            return

        self._gripper.SetTargetPosition(target_position)
        self._last_target_position = target_position
        self._last_command_time = now
        self._last_position_read_time = float("-inf")

    def get_gripper_position(self) -> int:
        now = self._clock()
        if (
            self._cached_position is not None
            and now - self._last_position_read_time < self.position_read_interval_s
        ):
            return self._cached_position

        position = int(self._gripper.GetCurrentPosition())
        self._cached_position = position
        self._last_position_read_time = now
        return position

    def _should_send_position(self, target_position: int, now: float) -> bool:
        if self._last_target_position is None:
            return True
        if now - self._last_command_time < self.min_command_interval_s:
            return False
        return (
            abs(target_position - self._last_target_position)
            >= self.position_tolerance
        )

    def _wait_initialized(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        while self._gripper.GetInitState() != 1:
            if time.time() >= deadline:
                raise TimeoutError(
                    f"PGC gripper on {self.port} did not initialize within {timeout_s}s"
                )
            time.sleep(0.2)


def _default_sdk_path() -> Path:
    """Repo-vendored SDK: src/dh_gripper, next to the ur5e_control package."""
    return Path(__file__).resolve().parents[3] / "dh_gripper"


def _load_dh_gripper_module(sdk_path: str | None = None) -> Any:
    candidates = [
        sdk_path or os.environ.get("UR5E_DH_GRIPPER_SDK_PATH"),
        str(_default_sdk_path()),
    ]
    # Prefer the first directory that actually contains the module; a stale or
    # unrelated UR5E_DH_GRIPPER_SDK_PATH (e.g. another project's SDK) must not
    # shadow the vendored copy.
    path = next(
        (
            str(candidate)
            for candidate in candidates
            if candidate and (Path(candidate) / "dh_modbus_gripper.py").is_file()
        ),
        None,
    )
    if path and path not in sys.path:
        sys.path.insert(0, path)

    try:
        import dh_modbus_gripper
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import dh_modbus_gripper. Install the DH gripper SDK or set "
            "UR5E_DH_GRIPPER_SDK_PATH to its Python module directory. "
            f"Original error: {exc}"
        ) from exc
    return dh_modbus_gripper


def _clamp_pgc_position(position: int) -> int:
    return min(max(int(position), 0), 1000)
