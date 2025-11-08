from config import config as conf
from kinematics import fk, ik
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
    trans = np.array([0.6, 0.15, 0.15])

    q = np.array([0.04164258, -1.38375224, -0.61048516, 3.14159265, 1.14735526, -1.52915375])
    normal_fk = SE3.from_homogeneous(robot.fk(q))
    ee_fk = fk(q = q, robot=robot)

    print(normal_fk,"normal")
    print()
    print(ee_fk, "ee_fk")
    print()
    

    pos = SE3(trans, rot)

    robot.move_to_q(q)
    time.sleep(100)
        
        




    
    if conf.get("robot_type") != "no_robot":
        end_robot(robot)


if __name__ == "__main__":
    main()
