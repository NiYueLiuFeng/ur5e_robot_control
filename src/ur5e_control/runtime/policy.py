"""Protocols for components that can be wired to the robot runtime."""

from __future__ import annotations

from typing import Protocol

from ur5e_control.runtime.types import RobotAction, RobotObservation, TaskInfo


class Policy(Protocol):
    def act(self, observation: RobotObservation, task: TaskInfo) -> RobotAction:
        ...


class RobotController(Protocol):
    def get_observation(self) -> RobotObservation:
        ...

    def step(self, action: RobotAction) -> RobotObservation:
        ...


class EpisodeRecorder(Protocol):
    def start_episode(self, task: TaskInfo) -> None:
        ...

    def add_frame(self, observation: RobotObservation, action: RobotAction) -> None:
        ...

    def save_episode(self) -> str:
        ...

