import numpy as np
from ctu_crs import CRS93

robot = CRS93()
robot.initialize()

q0 = robot.q_home
for i in range(len(q0)):
    q = q0.copy()
    q[i] += np.deg2rad(10)
    robot.move_to_q(q)

robot.close()
