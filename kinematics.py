import numpy as np
from ctu_crs import CRS93, CRS97
from config import config as conf
from core.se3 import SE3
from core.so3 import SO3


def fk(*, q: np.ndarray, robot) -> SE3:
    '''returns the postition of the end effector given joint configuration in absolute coordinates'''
    robot_fk = SE3.from_homogeneous(robot.fk(q))

    #position of the hoop center relative to end effector
    v = np.array([0.125, 0, 0]) 
    T = SE3(translation=v, rotation=SO3()) 

    fk = robot_fk * T
    return fk

def ik(*, poss )
    