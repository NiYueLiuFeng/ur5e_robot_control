from typing import Optional, Protocol, Tuple

import numpy as np


class CameraDriver(Protocol):
    """Camera protocol.

    A protocol for a camera driver used by the ZMQ camera bridge.
    """

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:  # type: ignore
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.

        Returns:
            np.ndarray: The color image.
            Optional[np.ndarray]: The depth image when enabled.
        """
