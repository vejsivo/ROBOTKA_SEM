from config import config as conf
from kinematics import fk, ik, generate_flat_poses, follow_path, offset_path

from ctu_crs import CRS93, CRS97
from core.se3 import SE3
from core.so3 import SO3
import numpy as np
import time
from puzzle_paths.puzzle_A import path_A
from puzzle_paths.puzzle_B import path_B
from puzzle_paths.puzzle_C import path_C
from puzzle_paths.puzzle_D import path_D
from puzzle_paths.puzzle_E import path_E
from perception import detect_object_location

def initialize_robot():
    robot_type = conf.get("robot_type")

    if robot_type == "CRS97":
        robot = CRS97()
        robot.initialize(home = False)
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
    

    H = conf.get("homo004")

    image = robot.grab_image()
    object_loc = detect_object_location(image=image, H=H) 
    print('object')
    print(object_loc)
    poses = offset_path(path=path_B, offset = object_loc)

    qs = follow_path(robot=robot, path=poses)
    for q in qs:
        robot.move_to_q(q)

    if conf.get("robot_type") != "no_robot":
        end_robot(robot)


if __name__ == "__main__":
    main()
