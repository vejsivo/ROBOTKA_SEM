import cv2 as cv
import cv2.aruco as aruco
import numpy as np
from core.se3 import SE3
from core.so3 import SO3


def detect_aruco_centers(*, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """detects aruco markers in the image, returns center of each marker and coresponding id from the aruco marker library"""
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(image)

    if ids is None:
        return np.empty((0,), dtype=np.int32), np.empty((0, 2), dtype=np.float32)

    centers = np.array([c[0].mean(axis=0) for c in corners], dtype=np.float32)
    ids = ids.flatten().astype(np.int32)

    return ids, centers


def detect_object_center(*, image: np.ndarray) -> np.ndarray | None:
    """detects object center from image"""
    ids, centers = detect_aruco_centers(image=image)
    if centers.shape[0]  != 2:
        raise ValueError("Image contains more or less than two aruco markers or the aruco markers are occluded")

    return centers.mean(axis=0)




