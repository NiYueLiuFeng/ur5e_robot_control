"""Open a synchronous WebSocket connection to a policy server."""

from __future__ import annotations

from typing import Any


def connect_websocket(server_url: str, api_key: str | None, timeout_s: float) -> Any:
    """Connect to `server_url`, attaching an `Api-Key` header when given."""

    try:
        import websockets.sync.client  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "websockets is required for policy control. "
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
