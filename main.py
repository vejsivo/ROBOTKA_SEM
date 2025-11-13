from config import config as conf
from kinematics import fk, ik, generate_flat_poses, follow_path

from ctu_crs import CRS93, CRS97
from core.se3 import SE3
from core.so3 import SO3
import numpy as np
import time
from puzzle_paths.puzzle_A import path_A

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
    
    poses = path_A
    
    qs = follow_path(robot=robot, path=poses)
    for q in qs:
        robot.move_to_q(q)
    
    
        
    if conf.get("robot_type") != "no_robot":
        end_robot(robot)


if __name__ == "__main__":
    main()
