from config import config as conf
from kinematics import fk, ik, generate_flat_poses
from perception import apply_homography, find_homography_in_height,detect_object_center
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
        robot = CRS97(tty_dev=None)
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")

    
    return robot


def end_robot(robot):
    if robot != None:
        robot.soft_home()
        robot.close()


def main():
    robot = initialize_robot()
    
    rot = SO3(np.diag([-1, 1, -1]))
    trans = np.array([0.45, -0.1, 0.15])

    q = np.array([0.04164258, -1.38375224, -0.61048516, 3.14159265, 1.14735526, -1.52915375])
    normal_fk = SE3.from_homogeneous(robot.fk(q))
    ee_fk = fk(q = q, robot=robot)

    pos = SE3(trans, rot)
    
    #HOMOGRAPHY TEST
    z = 0.15
    H = find_homography_in_height(robot=robot, target_height=z)
    print("Homography matrix:")
    print(H)

    #test zpracovani homografie
    aruco_image = robot.grab_image()
    obj_px = detect_object_center(image=aruco_image)          # [u,v]
    XY_plane = apply_homography(H, obj_px)             # [X,Y] on the plane

    #move robot to the position
    pos = np.array([XY_plane[0], XY_plane[1], z])
    T_obj = SE3(pos, SO3())  

    sols = ik(position=T_obj, robot=robot)
    robot.move_to_q(sols[0])

    if conf.get("robot_type") != "no_robot":
        end_robot(robot)


if __name__ == "__main__":
    main()