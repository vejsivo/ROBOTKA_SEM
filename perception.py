import cv2 as cv
import cv2.aruco as aruco 
import numpy as np


def detect_aruco_centers(*, image) -> list:
    """detects all arucos in the image, return list in the form [[aruco ID, [center_x, center_y]], [..., ...] ...]"""
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(image)  
    result = []

    if ids is None:
        return result

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        pts = marker_corners[0]         
        center = pts.mean(axis=0)       
        result.append([int(marker_id), center.tolist()])
    return result

def detect_object_center(*, image):
    print(np.array(detect_aruco_centers(image = image)[:, 1]))

    if len(centers) != 2:
        raise ValueError
    
    object_center = centers.mean(axis = 0)

    return object_center

  