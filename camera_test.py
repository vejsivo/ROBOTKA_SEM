from config import config as conf
from kinematics import fk, ik, generate_flat_poses
from perception import find_hoop_homography, detect_object_center
from ctu_crs import CRS93, CRS97
from core.se3 import SE3
from core.so3 import SO3
import numpy as np
import time

def initialize_robot():
    robot_type = conf.get("robot_type")

    if robot_type == "CRS97":
        robot = CRS97()
        robot.initialize()
    elif robot_type == "CRS93":
        robot = CRS93()
        robot.initialize()
    elif robot_type == "no_robot":
        robot = CRS93(tty_dev=None)
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")

    
    return robot


def end_robot(robot):
    if robot != None:
        robot.soft_home()
        robot.close()


def main():
    robot = initialize_robot()

    if conf.get("robot_type") != "no_robot":
        end_robot(robot)

                    
    poses = generate_flat_poses(
    robot=robot,
    xmax=0.7,
    xmin=0.4,
    ymax=0.15,
    ymin=-0.15,
    ysteps=3,
    xsteps=4,
    height=0.15
    )

    images = []
    for pose in poses:
        sols = ik(position=pose, robot=robot)
        robot.move_to_q(sols[0])
        # Grab an image from the camera
        # This will automatically connect to and open the camera on the first call
        img = robot.grab_image()
        images.append(img)
    print(f"Captured {len(images)} images.")
    
    H = find_hoop_homography(images, poses)
    
    object_center_tocam = detect_object_center()
    
    print("Homography matrix H:\n", H)
    print("Object center in camera frame:", object_center_tocam)

    if conf.get("robot_type") != "no_robot":
        end_robot(robot)

    # Grab an image from the camera
    # This will automatically connect to and open the camera on the first call
    img = robot.grab_image()


if __name__ == "__main__":
    main()
