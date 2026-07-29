"""Safe preflight checks for teleoperation recording."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import shlex
import socket
from typing import Callable, Iterable, List, Mapping


FindSpec = Callable[[str], object]

LEROBOT_RECORDING_IMPORTS = (
    "numpy",
    "pynput",
    "pyrealsense2",
    "rtde_control",
    "rtde_receive",
    "zmq",
    "lerobot",
)
SPACEMOUSE_IMPORTS = ("dm_control", "pyspacemouse", "quaternion")


@dataclass(frozen=True)
class PreflightResult:
    name: str
    ok: bool
    detail: str


def check_imports(
    module_names: Iterable[str],
    find_spec: FindSpec = importlib.util.find_spec,
) -> List[PreflightResult]:
    results = []
    for module_name in module_names:
        spec = find_spec(module_name)
        if spec is None:
            results.append(
                PreflightResult(
                    name=f"import {module_name}",
                    ok=False,
                    detail="not importable in this Python environment",
                )
            )
        else:
            results.append(
                PreflightResult(name=f"import {module_name}", ok=True, detail="ok")
            )
    return results


def check_path_exists(path: str, label: str) -> PreflightResult:
    exists = Path(path).exists()
    return PreflightResult(
        name=f"{label} {path}",
        ok=exists,
        detail="exists" if exists else "missing",
    )


def check_tcp_connect(host: str, port: int, timeout_s: float = 1.0) -> PreflightResult:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return PreflightResult(
                name=f"tcp {host}:{port}",
                ok=True,
                detail="reachable",
            )
    except OSError as exc:
        return PreflightResult(
            name=f"tcp {host}:{port}",
            ok=False,
            detail=str(exc),
        )


def check_configured_cameras(realsense_ids: Mapping[str, str]) -> PreflightResult:
    if not realsense_ids:
        return PreflightResult(
            name="configured realsense cameras",
            ok=False,
            detail="no camera serials configured",
        )

    camera_summary = ", ".join(
        f"{serial}={name}" for serial, name in sorted(realsense_ids.items())
    )
    return PreflightResult(
        name="configured realsense cameras",
        ok=True,
        detail=camera_summary,
    )


def collect_recording_preflight(
    gello_port: str,
    robot_ip: str,
    agent_name: str = "gello",
    check_robot_tcp: bool = False,
    gripper_port: str | None = None,
    spacemouse_device_path: str | None = None,
    realsense_ids: Mapping[str, str] | None = None,
    find_spec: FindSpec = importlib.util.find_spec,
) -> List[PreflightResult]:
    imports = list(LEROBOT_RECORDING_IMPORTS)
    if agent_name == "spacemouse":
        imports.extend(SPACEMOUSE_IMPORTS) # type: ignore

    results = check_imports(imports, find_spec=find_spec)
    if agent_name == "gello":
        results.append(check_path_exists(gello_port, "gello port"))
    elif agent_name == "spacemouse":
        from ur5e_control.data_collection.teleop.spacemouse.ur5e_model import UR5E_XML

        results.append(check_path_exists(str(UR5E_XML), "spacemouse mujoco model"))
        if spacemouse_device_path:
            results.append(
                check_path_exists(spacemouse_device_path, "spacemouse device")
            )
    if gripper_port:
        results.append(check_path_exists(gripper_port, "pgc gripper port"))
    if realsense_ids is not None:
        results.append(check_configured_cameras(realsense_ids))
    if check_robot_tcp:
        results.append(check_tcp_connect(robot_ip, 30004))
    return results


def build_lerobot_record_command(
    repo_id: str,
    root: str,
    fps: int,
    config_path: str | None = None,
) -> str:
    env = {
        "UR5E_LEROBOT_REPO_ID": repo_id,
        "UR5E_LEROBOT_ROOT": root,
        "UR5E_LEROBOT_FPS": str(fps),
    }
    if config_path:
        env["UR5E_HARDWARE_CONFIG"] = config_path
    env_prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in env.items()
    )
    return f"{env_prefix} uv run record-teleop"


def format_preflight_report(results: Iterable[PreflightResult]) -> str:
    lines = []
    for result in results:
        status = "ok" if result.ok else "blocked"
        lines.append(f"[{status}] {result.name}: {result.detail}")
    return "\n".join(lines)


def preflight_has_blockers(results: Iterable[PreflightResult]) -> bool:
    return any(not result.ok for result in results)
