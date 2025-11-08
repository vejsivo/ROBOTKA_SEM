from config import config as conf
from kinematics import fk, ik, generate_flat_poses, follow_path, random_rot

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
    poses = []
    for i in range(20):
        poses.append(SE3(np.array([0.4, -0.1, 0.2]), random_rot()))

    path = follow_path(poses)
    for q in path:
        robot.move_to_q(q)
        time.sleep(1)
        



    
    
    
    
        
    if conf.get("robot_type") != "no_robot":
        end_robot(robot)


if __name__ == "__main__":
    main()
