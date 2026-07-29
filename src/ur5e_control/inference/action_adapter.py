"""Adapt VLA server responses into validated robot actions."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ur5e_control.runtime.safety import validate_action_dimension
from ur5e_control.runtime.types import RobotAction


class ActionAdapter:
    def __init__(self, expected_dim: int = 7):
        self.expected_dim = expected_dim

    def from_response(self, response: Any) -> RobotAction:
        return self.actions_from_response(response, execute_horizon=1)[0]

    def actions_from_response(
        self,
        response: Any,
        *,
        current_state: Any | None = None,
        action_space: str = "abs",
        execute_horizon: int = 1,
        max_joint_delta: float | None = None,
    ) -> tuple[RobotAction, ...]:
        if execute_horizon <= 0:
            raise ValueError("execute_horizon must be positive")

        chunk = _normalize_action_chunk(_extract_action(response), self.expected_dim)
        selected = chunk[:execute_horizon]
        if selected.size == 0:
            raise ValueError("VLA response did not contain executable actions")

        actions = []
        for raw_action in selected:
            action = _prepare_action(
                raw_action,
                current_state=current_state,
                action_space=action_space,
                max_joint_delta=max_joint_delta,
            )
            actions.append(
                validate_action_dimension(
                    RobotAction(values=tuple(action)),
                    # RobotAction(values=action),
                    # action,
                    expected_dim=self.expected_dim,
                )
            )
        return tuple(actions)
        # return np.array(actions)


def _normalize_action_chunk(raw_action: Any, expected_dim: int) -> np.ndarray:
    actions = np.asarray(raw_action, dtype=float)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    elif actions.ndim == 3:
        if actions.shape[0] != 1:
            raise ValueError(
                "VLA batched action response must have batch dimension 1, "
                f"got {actions.shape[0]}"
            )
        actions = actions[0]

    if actions.ndim != 2:
        raise ValueError("VLA response actions must be a 1D, 2D, or 3D array")
    if actions.shape[1] != expected_dim:
        raise ValueError(
            f"VLA action dimension mismatch: expected {expected_dim}, "
            f"got {actions.shape[1]}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("VLA response actions must contain finite numeric values")
    return actions


def _prepare_action(
    raw_action: np.ndarray,
    *,
    current_state: Any | None,
    action_space: str,
    max_joint_delta: float | None,
) -> np.ndarray:
    normalized_space = _normalize_action_space(action_space)
    action = np.asarray(raw_action, dtype=float).copy()
    state = None
    if current_state is not None:
        state = np.asarray(current_state, dtype=float).reshape(-1)
        if state.size < 6:
            raise ValueError("current_state must contain at least 6 arm joint values")

    if normalized_space == "delta":
        if state is None:
            raise ValueError("current_state is required for delta VLA actions")
        action[:6] = state[:6] + action[:6]

    # if max_joint_delta is not None:
    #     if max_joint_delta < 0:
    #         raise ValueError("max_joint_delta must be non-negative")
    #     if state is not None:
    #         action[:6] = np.clip(
    #             action[:6],
    #             state[:6] - float(max_joint_delta),
    #             state[:6] + float(max_joint_delta),
    #         )

    return action


def _normalize_action_space(action_space: str) -> str:
    normalized = action_space.strip().lower()
    if normalized in {"abs", "absolute"}:
        return "abs"
    if normalized in {"delta", "relative"}:
        return "delta"
    raise ValueError("action_space must be 'abs' or 'delta'")


def _extract_action(response: Any) -> Any:
    if isinstance(response, Mapping):
        if "actions" in response:
            return response["actions"]
        if "action" in response:
            return response["action"]
        if "payload" in response:
            return _extract_action(response["payload"])
        raise ValueError("VLA response is missing 'actions'")
    return response
