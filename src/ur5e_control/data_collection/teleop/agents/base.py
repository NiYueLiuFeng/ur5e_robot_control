from typing import Any, Dict, Protocol


class Agent(Protocol):
    def act(self, obs: Dict[str, Any]) -> Any:
        """Returns an action given an observation.

        Args:
            obs: observation from the environment.

        Returns:
            action: action to take on the environment.
        """
        raise NotImplementedError
