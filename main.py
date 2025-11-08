from config import config as conf
from kinematics import fk, ik
from ctu_crs import CRS93, CRS97
from core.se3 import SE3
from core.so3 import SO3
import numpy as np

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
    trans = np.array([0.6, 0.15, 0.15])

    pos = SE3(trans, rot)

    sols = ik(position=pos, robot=robot)
    for sol in sols:
        robot.move_to_q(sol)

    
    if conf.get("robot_type") != "no_robot":
        end_robot(robot)


if __name__ == "__main__":
    main()
