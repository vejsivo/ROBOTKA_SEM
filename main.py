from config import config as conf
from kinematics import fk, ik, generate_flat_poses, follow_path

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
    
    R = np.diag([1.0, 1.0, -1.0])
    rot = SO3(R)

    # choose 5 points along x = y inside the bounds
    n = 5
    x_vals = np.linspace(0.4, 0.7, n)
    y_vals = np.linspace(-0.15, 0.15, n)
    z = 0.3

    poses = [SE3(translation=np.array([x, y, z]), rotation=rot)
            for x, y in zip(x_vals, y_vals)]
    
    qs = follow_path(robot=robot, path=poses)
    for q in qs:
        robot.move_to_q(q)
    
    
        
    if conf.get("robot_type") != "no_robot":
        end_robot(robot)


if __name__ == "__main__":
    main()
