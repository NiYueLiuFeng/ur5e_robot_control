"""Synchronous client for an already-running CloudRobo-VLA WebSocket server."""

from __future__ import annotations

from typing import Any, Callable

from ur5e_control.inference.protocol import Packer, unpackb

ConnectionFactory = Callable[[str, str | None, float], Any]


class VLAClient:
    _COMMAND_KEY = "__cloudrobovla_command__"

    def __init__(
        self,
        server_url: str,
        api_key: str | None = None,
        timeout_s: float = 5.0,
        connection_factory: ConnectionFactory | None = None,
        rtc: Any | None = None,
    ) -> None:
        self.server_url = server_url
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self._connection_factory = connection_factory or _connect_websocket
        self._packer = Packer()
        self._connection = self._connection_factory(
            self.server_url,
            self.api_key,
            self.timeout_s,
        )
        self._server_metadata = self._recv_msgpack()
        self.rtc = rtc

    def get_server_metadata(self) -> dict[str, Any]:
        if isinstance(self._server_metadata, dict):
            return dict(self._server_metadata)
        return {}

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self._round_trip({self._COMMAND_KEY: "infer", "obs": obs})

    def reset(self) -> dict[str, Any]:
        return self._round_trip({self._COMMAND_KEY: "reset"})

    def close(self) -> None:
        close = getattr(self._connection, "close", None)
        if callable(close):
            close()

    def _round_trip(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._connection.send(self._packer.pack(payload))
        response = self._recv_msgpack()
        if isinstance(response, dict) and "ok" in response:
            if response.get("ok") is True:
                inner = response.get("payload", {})
                return inner if isinstance(inner, dict) else {"payload": inner}
            error = response.get("error", {})
            if isinstance(error, dict):
                raise RuntimeError(str(error.get("message", "VLA server error")))
            raise RuntimeError("VLA server error")
        if not isinstance(response, dict):
            raise RuntimeError(f"VLA server returned {type(response).__name__}")
        return response

    def _recv_msgpack(self) -> Any:
        response = self._connection.recv()
        if isinstance(response, str):
            raise RuntimeError(f"VLA server returned text error: {response}")
        return unpackb(response)


def _connect_websocket(server_url: str, api_key: str | None, timeout_s: float) -> Any:
    try:
        import websockets.sync.client  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "websockets is required for CloudRobo-VLA inference. "
            "Install project dependencies with `uv sync`."
        ) from exc

    headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
    kwargs: dict[str, Any] = {
        "compression": None,
        "max_size": None,
        "open_timeout": timeout_s,
    }
    if headers:
        kwargs["additional_headers"] = headers

    try:
        return websockets.sync.client.connect(server_url, proxy=None, **kwargs)
    except TypeError:
        if "additional_headers" in kwargs:
            kwargs["extra_headers"] = kwargs.pop("additional_headers")
        kwargs.pop("open_timeout", None)
        return websockets.sync.client.connect(server_url, **kwargs)
