import numpy as np
from ctu_crs import CRS93, CRS97
from config import config as conf
from core.se3 import SE3
from core.so3 import SO3


#robot end effector to stick transformation, will have to be set according to the robot type but i believe this is the correct transform
_T_es = SE3(translation=np.array([-0.125, 0.0, 0.0]), rotation=SO3())
_T_se = _T_es.inverse()    

def fk(*, q: np.ndarray, robot) -> SE3:
    '''returns the postition of the end effector given joint configuration in absolute coordinates'''
    robot_fk = SE3.from_homogeneous(robot.fk(q))
    T = SE3(translation=_T_es.translation, rotation=SO3()) 
    fk = robot_fk * T
    return fk

def q_valid(*, q: np.ndarray) -> bool:
    min = conf["joint_limit_min"]
    max = conf["joint_limit_max"]
    return np.all((q >= min) & (q <= max))

def ik(*, position: SE3, robot) -> list[np.ndarray]:
    '''takes in a desired position of the end effector, return multiple possible solutions for that position 
    possibly rotated around the z axis going through the center of the hoop'''
    ik = []
    num_steps = 20
    thetas = np.linspace(0, 2*np.pi, num_steps, endpoint=False)
    for theta in thetas:
        axis = np.array([0, 0, 1])
        T = SE3(position.translation, position.rotation * SO3.exp(axis * theta) )
        T_effector = T * _T_se
        sols = robot.ik(T_effector.homogeneous())
        ik.extend(sols)

    for i in range(len(ik) - 1, -1, -1):
        if not q_valid(q = ik[i]):
            del ik[i]
    return ik

def follow_path(*, path: list[SE3]):
    "TODO try implementing a backward pass dynamic programming solver that minimizes total cost (differences in join positions between jumps)"
