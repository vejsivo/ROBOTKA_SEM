import cv2 as cv
import cv2.aruco as aruco
import numpy as np
from core.se3 import SE3
from core.so3 import SO3

from kinematics import fk, ik, generate_flat_poses, follow_path


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

def detect_aruco_corners(*, image: np.ndarray):
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(image)
    return corners, ids


def calculate_aruco_location(*, corners: np.ndarray) -> SE3:
    center = corners.mean(axis=0)
    s = np.array([corners[2], corners[3]]).mean(axis=0)
    direction = s - center
    angle = np.arctan2(direction[1], direction[0])

    Rz = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])
    rotation = SO3(Rz)

    translation = np.array([center[0], center[1], 0.04])
    pose = SE3(translation=translation, rotation=rotation)

    return pose


def detect_object_center(*, image: np.ndarray) -> np.ndarray | None:
    """detects object center from image"""
    ids, centers = detect_aruco_centers(image=image)
    if centers.shape[0]  != 2:
        raise ValueError("Image contains more or less than two aruco markers or the aruco markers are occluded")

    return centers.mean(axis=0)

def detect_object_location(*, image: np.ndarray, H: np.ndarray) -> SE3 | None:
    corners, ids = detect_aruco_corners(image=image)
    if ids is None or len(ids) == 0:
        return None

    marker_id = int(ids[0])
    if marker_id == 1:
        offset = np.array([0.04, 0.03, 0.0])
    elif marker_id == 2:
        offset = np.array([-0.04, -0.03, 0.0])
    else:
        offset = np.zeros(3)

    # Transform marker corners from image → plane coordinates
    pts = corners[0][0]                        # (4, 2)
    plane_pts = np.array([apply_homography(H, pt) for pt in pts])

    # SE3 pose of ArUco marker in world coordinates
    aruco_coords = calculate_aruco_location(corners=plane_pts)

    # SE3 transform from marker → object
    offset_pose = SE3(translation=offset, rotation=SO3(np.diag([1, 1, 1])))

    # Compose: world ← marker ← object
    object_pose_world = aruco_coords * offset_pose

    return object_pose_world




        



#from toolbox
#images array, hoop_positions array of SE3
def find_hoop_homography(images, hoop_positions) -> np.ndarray:
    """
    Find homography based on images containing the hoop and the hoop positions from array
    """
    
    images = np.asarray(images) #convert to numpy array (1200, 1920, 3) * N

    processed_images = images_processing(images, thresh_val=120, blur_ksize=(9,9))

    circle = [find_circle(img, min_radius=50, max_radius=200) for img in processed_images] 
    
    #Fix types
    #hoop position is type SE3
    circle_pos = np.array([[c[0], c[1]] for c in circle]) #image points (xc,yc)
    hoop_pos = np.array([x.translation[:2] for x in hoop_positions]) #hoop positions in plane P (x,y)
    H, mask = cv.findHomography(circle_pos, hoop_pos, method=cv.RANSAC)

    return H #homography


def images_processing(images,
                    thresh_val=100,
                    blur_ksize=(9, 9)
):
    """
    Threshold and blur úrocessing for N images
    images: np.ndarray with shape (N, 1200, 1920, 3)
    Returns: list of image arrays (or None)
    """
    processed_images = []

    for i, img in enumerate(images):
        # Convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Apply simple binary threshold
        _, th = cv.threshold(gray, thresh_val, 255, cv.THRESH_BINARY)

        # Optionally blur to reduce noise
        blur = cv.GaussianBlur(th, blur_ksize, 2)
        processed_images.append(blur)

    return processed_images

def find_circle(image,min_radius, max_radius):
    """
    Find circle in the image using HoughCircles.
    Returns: one circle(x, y, radius) or None if not found
    """
    # Use HoughCircles to detect circles
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=100, param2=30, minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles = np.uint16(np.around(circles)) 
        # Return the first detected circle !! 
        return circles[0][0]  # (x, y, radius)
    else:
        return None

def find_homography_in_height(robot, height):
    """
    Find homography matrix for given height using robot to capture images
    """
    poses = generate_flat_poses(
        robot=robot,
        xmax=0.5,
        xmin=0.4,
        ymax=0.15,
        ymin=-0.15,
        ysteps=4,
        xsteps=4,
        height=height   
    )

    q_array = follow_path(robot = robot, path = poses)
    images = []
    for q in q_array:
        robot.move_to_q(q)
        robot.wait_for_motion_stop()
        img = robot.grab_image()
        images.append(img)
        
    H = find_hoop_homography(images, poses)
    return H

def apply_homography(H: np.ndarray, pt_uv: np.ndarray) -> np.ndarray:
    """pt_uv: [u,v] pixel (same image used to estimate H)
       returns: [X,Y] in the plane coordinates used for hoop_pos"""
    u, v = float(pt_uv[0]), float(pt_uv[1])
    ph = H @ np.array([u, v, 1.0])
    X = ph[0] / ph[2]
    Y = ph[1] / ph[2]
    return np.array([X, Y], dtype=float)

def apply_homography_inv(H: np.ndarray, pt_xy: np.ndarray) -> np.ndarray:
    """pt_xy: [X, Y] in plane coordinates
       returns: [u, v] pixel coordinates in the image"""
    X, Y = float(pt_xy[0]), float(pt_xy[1])
    Hinv = np.linalg.inv(H)
    ph = Hinv @ np.array([X, Y, 1.0])
    u = ph[0] / ph[2]
    v = ph[1] / ph[2]
    return np.array([u, v], dtype=float)


