"""Shared safety checks for robot actions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ur5e_control.runtime.types import RobotAction


@dataclass(frozen=True)
class JointLimits:
    """Joint-space safety boundary for the six UR5e arm joints."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    margin: float = 0.0
    enabled: bool = True

    def __post_init__(self) -> None:
        lower = _float_tuple(self.lower, "lower")
        upper = _float_tuple(self.upper, "upper")
        margin = float(self.margin)
        if len(lower) != 6:
            raise ValueError("joint lower limits must contain 6 values")
        if len(upper) != 6:
            raise ValueError("joint upper limits must contain 6 values")
        if margin < 0:
            raise ValueError("joint limit margin must be non-negative")
        for index, (low, high) in enumerate(zip(lower, upper)):
            if low > high:
                raise ValueError(f"joint[{index}] lower limit is greater than upper")
            if low + margin > high - margin:
                raise ValueError(f"joint[{index}] margin leaves no valid range")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "margin", margin)
        object.__setattr__(self, "enabled", bool(self.enabled))

    def clamp_action(
        self,
        action_values: Sequence[float],
    ) -> tuple[tuple[float, ...], bool]:
        """Clamp the first six action values and preserve trailing values."""

        values = tuple(float(value) for value in action_values)
        if not self.enabled:
            return values, False
        if len(values) < 6:
            raise ValueError("joint action must contain at least 6 values")

        clamped_values = list(values)
        was_clamped = False
        for index in range(6):
            low = self.lower[index] + self.margin
            high = self.upper[index] - self.margin
            original = clamped_values[index]
            clamped = min(max(original, low), high)
            if clamped != original:
                was_clamped = True
                clamped_values[index] = clamped
        return tuple(clamped_values), was_clamped

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "enabled": self.enabled,
            "lower": list(self.lower),
            "upper": list(self.upper),
            "margin": self.margin,
        }

    def save(self, path: str | Path) -> None:
        save_joint_limits(path, self)


def validate_action_dimension(action: RobotAction, expected_dim: int) -> RobotAction:
    """Validate command length and numeric finiteness."""

    actual_dim = len(action.values)
    if actual_dim != expected_dim:
        raise ValueError(f"action expected {expected_dim} values, got {actual_dim}")

    if any(not math.isfinite(value) for value in action.values):
        raise ValueError("action contains non-finite values")

    return action


def load_joint_limits(path: str | Path) -> JointLimits:
    import yaml

    candidate = Path(path).expanduser()
    data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{candidate} must contain a YAML mapping")
    return JointLimits(
        enabled=_bool_value(data.get("enabled", True), "enabled"),
        lower=_sequence_value(data.get("lower"), "lower"),
        upper=_sequence_value(data.get("upper"), "upper"),
        margin=float(data.get("margin", 0.0)),
    )


def save_joint_limits(path: str | Path, limits: JointLimits) -> None:
    import yaml

    candidate = Path(path).expanduser()
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(
        yaml.safe_dump(dict(limits.to_mapping()), sort_keys=False),
        encoding="utf-8",
    )


def write_joint_limits(
    path: str | Path,
    lower: Sequence[float],
    upper: Sequence[float],
    margin: float = 0.0,
    enabled: bool = True,
) -> JointLimits:
    limits = JointLimits(
        lower=tuple(float(value) for value in lower),
        upper=tuple(float(value) for value in upper),
        margin=margin,
        enabled=enabled,
    )
    limits.save(path)
    return limits


def _float_tuple(value: Sequence[float], label: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in value)
    except TypeError as exc:
        raise ValueError(f"joint {label} limits must be a list of numbers") from exc
    if any(not math.isfinite(item) for item in values):
        raise ValueError(f"joint {label} limits must contain finite numbers")
    return values


def _sequence_value(value: Any, label: str) -> Sequence[float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list of numbers")
    return value


def _bool_value(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{label} must be a boolean")
