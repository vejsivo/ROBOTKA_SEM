import cv2
import numpy as np

from ctu_crs import CRS93

# Initialize the robot interface
robot = CRS93()
#robot.initialize()

# Grab an image from the camera
# This will automatically connect to and open the camera on the first call
img = robot.grab_image()

# 'img' is a NumPy array containing the image data.
# You can now process it, for example, using a library like OpenCV.
if img is not None:
    print(f"Successfully grabbed image with shape: {img.shape}")
    out = img
    if out.dtype != np.uint8:
        out = np.clip(out * 255, 0, 255).astype(np.uint8)

    cv2.imwrite("image_RD.png", out)
    print(f"Image saved as 'image.png'.")
# Don't forget to close the connection when you're done
robot.close()
