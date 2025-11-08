import cv as cv
import cv.aruco as aruco
import numpy as np


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



#from toolbox
def find_hoop_homography(images: ArrayLike, hoop_positions: List[dict]) -> np.ndarray:
    """
    Find homography based on images containing the hoop and the hoop positions from array
    """
    
    images = np.asarray(images) #convert to numpy array (1200, 1920, 3) * N

    processed_images = images_processing(images, thresh_val=120, blur_ksize=(9,9))

    circle = [find_circle(img, min_radius=50, max_radius=200) for img in processed_images] 
    
    #Fix types
    circle_pos = np.array([[c[0], c[1]] for c in circle]) #image points (xc,yc)
    hoop_pos = np.array([x["translation_vector"] for x in hoop_positions])[:, :2] #hoop positions in plane P (x,y)
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
