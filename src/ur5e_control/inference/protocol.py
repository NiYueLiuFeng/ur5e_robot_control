"""CloudRobo-VLA WebSocket msgpack protocol helpers."""

from __future__ import annotations

import functools
from typing import Any

import numpy as np


def pack_array(obj: Any) -> Any:
    """Pack numpy values in the format used by CloudRobo-VLA servers."""

    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj: dict[Any, Any]) -> Any:
    ndarray_key = _lookup_key(obj, b"__ndarray__", "__ndarray__")
    if ndarray_key is not None:
        data = obj[_lookup_key(obj, b"data", "data")]
        dtype = np.dtype(obj[_lookup_key(obj, b"dtype", "dtype")])
        shape = obj[_lookup_key(obj, b"shape", "shape")]
        return np.ndarray(buffer=data, dtype=dtype, shape=shape)

    generic_key = _lookup_key(obj, b"__npgeneric__", "__npgeneric__")
    if generic_key is not None:
        dtype = np.dtype(obj[_lookup_key(obj, b"dtype", "dtype")])
        return dtype.type(obj[_lookup_key(obj, b"data", "data")])

    return obj


def Packer() -> Any:  # noqa: N802 - mirror CloudRobo-VLA protocol name.
    return _msgpack().Packer(default=pack_array)


def packb(value: Any) -> bytes:
    return _msgpack().packb(value, default=pack_array)


def unpackb(value: bytes) -> Any:
    return _msgpack().unpackb(value, object_hook=unpack_array)


def _lookup_key(obj: dict[Any, Any], bytes_key: bytes, str_key: str) -> Any:
    if bytes_key in obj:
        return bytes_key
    if str_key in obj:
        return str_key
    return None


@functools.lru_cache(maxsize=1)
def _msgpack() -> Any:
    try:
        import msgpack  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "msgpack is required for CloudRobo-VLA WebSocket inference. "
            "Install project dependencies with `uv sync`."
        ) from exc
    return msgpack
