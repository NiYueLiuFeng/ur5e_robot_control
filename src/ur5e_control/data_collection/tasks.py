"""Task configuration loading for collection and inference runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ur5e_control.runtime.types import TaskInfo


def load_task_file(path: str) -> TaskInfo:
    """Load a task YAML file into a stable runtime contract."""

    data = _load_yaml_like(Path(path))
    name = data.get("task")
    task_type = data.get("type", "long_horizon_task")
    subtasks = data.get("sub_tasks", [])

    if not isinstance(name, str) or not name:
        raise ValueError(f"{path} must define a non-empty 'task' string")
    if not isinstance(task_type, str) or not task_type:
        raise ValueError(f"{path} must define a non-empty 'type' string")
    if not isinstance(subtasks, list) or not all(
        isinstance(item, str) for item in subtasks
    ):
        raise ValueError(f"{path} must define 'sub_tasks' as a string list")

    return TaskInfo(name=name, task_type=task_type, subtasks=tuple(subtasks))


def _load_yaml_like(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        import yaml  # type: ignore
    except ImportError:
        return _load_simple_yaml(path)

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping")
    return loaded


def _load_simple_yaml(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    current_list_key = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("- "):
            if current_list_key is None:
                raise ValueError(f"{path} has a list item without a key")
            list_value = data.setdefault(current_list_key, [])
            if not isinstance(list_value, list):
                raise ValueError(f"{path} key {current_list_key!r} is not a list")
            list_value.append(_strip_yaml_string(line[2:].strip()))
            continue

        current_list_key = None
        if ":" not in line:
            raise ValueError(f"{path} contains unsupported YAML line: {raw_line!r}")

        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if not value:
            data[key] = []
            current_list_key = key
        else:
            data[key] = _strip_yaml_string(value)

    return data


def _strip_yaml_string(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value

