"""Calibration target descriptions and OpenCV detection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ur5e_control.calibration.types import IntrinsicsSample


@dataclass(frozen=True)
class ChessboardPattern:
    columns: int = 7
    rows: int = 6
    square_size_m: float = 0.025

    def __post_init__(self) -> None:
        if self.columns <= 1:
            raise ValueError("columns must be greater than 1")
        if self.rows <= 1:
            raise ValueError("rows must be greater than 1")
        if self.square_size_m <= 0:
            raise ValueError("square_size_m must be positive")


@dataclass(frozen=True)
class CharucoPattern:
    squares_x: int = 7
    squares_y: int = 5
    square_length_m: float = 0.04
    marker_length_m: float = 0.03
    dictionary: str = "DICT_4X4_50"

    def __post_init__(self) -> None:
        if self.squares_x <= 1:
            raise ValueError("squares_x must be greater than 1")
        if self.squares_y <= 1:
            raise ValueError("squares_y must be greater than 1")
        if self.square_length_m <= 0:
            raise ValueError("square_length_m must be positive")
        if self.marker_length_m <= 0:
            raise ValueError("marker_length_m must be positive")
        if self.marker_length_m >= self.square_length_m:
            raise ValueError("marker_length_m must be smaller than square_length_m")


@dataclass(frozen=True)
class PatternDetection:
    object_points: np.ndarray
    image_points: np.ndarray
    image_size: tuple[int, int]

    def as_intrinsics_sample(self) -> IntrinsicsSample:
        return IntrinsicsSample(
            object_points=self.object_points,
            image_points=self.image_points,
            image_size=self.image_size,
        )


def chessboard_object_points(pattern: ChessboardPattern) -> np.ndarray:
    grid = np.mgrid[0 : pattern.columns, 0 : pattern.rows].T.reshape(-1, 2)
    points = np.zeros((pattern.columns * pattern.rows, 3), dtype=np.float32)
    points[:, :2] = grid.astype(np.float32) * float(pattern.square_size_m)
    return points


def detect_chessboard(image, pattern: ChessboardPattern) -> PatternDetection | None:
    import cv2

    gray = _to_gray(image, cv2)
    found, corners = cv2.findChessboardCorners(gray, (pattern.columns, pattern.rows))
    if not found:
        return None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    image_points = corners.reshape(-1, 2)
    return PatternDetection(
        object_points=chessboard_object_points(pattern),
        image_points=image_points.astype(np.float32),
        image_size=(int(gray.shape[1]), int(gray.shape[0])),
    )


def detect_charuco(image, pattern: CharucoPattern) -> PatternDetection | None:
    import cv2

    aruco = _aruco_module(cv2)
    dictionary, board = create_charuco_board(cv2, pattern)
    gray = _to_gray(image, cv2)
    parameters = aruco.DetectorParameters()
    corners, ids, _rejected = aruco.detectMarkers(
        gray,
        dictionary,
        parameters=parameters,
    )
    if ids is None or len(ids) == 0:
        return None
    count, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        corners,
        ids,
        gray,
        board,
    )
    if count is None or count < 4 or charuco_ids is None:
        return None
    object_points = _charuco_object_points(board, charuco_ids.reshape(-1))
    return PatternDetection(
        object_points=object_points,
        image_points=charuco_corners.reshape(-1, 2).astype(np.float32),
        image_size=(int(gray.shape[1]), int(gray.shape[0])),
    )


def create_charuco_board(cv2, pattern: CharucoPattern):
    aruco = _aruco_module(cv2)
    dictionary_id = getattr(aruco, pattern.dictionary, None)
    if dictionary_id is None:
        raise ValueError(f"unknown ArUco dictionary: {pattern.dictionary}")
    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    if hasattr(aruco, "CharucoBoard"):
        try:
            board = aruco.CharucoBoard(
                (pattern.squares_x, pattern.squares_y),
                pattern.square_length_m,
                pattern.marker_length_m,
                dictionary,
            )
        except TypeError:
            board = aruco.CharucoBoard_create(
                pattern.squares_x,
                pattern.squares_y,
                pattern.square_length_m,
                pattern.marker_length_m,
                dictionary,
            )
    else:
        board = aruco.CharucoBoard_create(
            pattern.squares_x,
            pattern.squares_y,
            pattern.square_length_m,
            pattern.marker_length_m,
            dictionary,
        )
    return dictionary, board


def _charuco_object_points(board, ids: np.ndarray) -> np.ndarray:
    if hasattr(board, "getChessboardCorners"):
        corners = board.getChessboardCorners()
    else:
        corners = board.chessboardCorners
    return np.asarray(corners, dtype=np.float32)[ids.astype(int)]


def _aruco_module(cv2):
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise RuntimeError(
            "OpenCV ArUco support is required for ChArUco calibration. "
            "Install opencv-contrib-python-headless."
        )
    return aruco


def _to_gray(image, cv2):
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    raise ValueError("calibration image must be grayscale or RGB")
