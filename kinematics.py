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
    if len(ik) == 0:
        return None
    return ik

def follow_path(*, path: list[SE3], robot):
    #construct the sol list
    grid =[]
    for position in path:
        sols = ik(position=position, robot=robot)
        assert sols is not None
        grid.append(sols)

    #construct the cost list
    costs = [[np.inf for _ in inner] for inner in grid]
    for i in range(len(costs[-1])):
        costs[-1][i] = 1

    #construct the actions list
    actions = [[None for _ in inner] for inner in grid]

    for i in range(len(grid) - 2, -1, -1):
        sols = grid[i]
        for j in range(len(sols)):
            state = sols[j]
            best_cost = np.inf
            best_action = None
            for k in range(len(grid[i + 1])):
                next_state = grid[i + 1][k]
                static_cost = costs[i+1][k]
                travel_cost = np.linalg.norm(next_state - state)
                total = static_cost + travel_cost
                if total < best_cost:
                    best_cost = total
                    best_action = k
            costs[i][j] = best_cost
            actions[i][j] = best_action
    
    path_idx = [np.argmin(costs[0])]
    for i in range(len(actions) - 1):
        next_idx = actions[i][path_idx[-1]]
        path_idx.append(next_idx)

    optimal_path = [grid[i][path_idx[i]] for i in range(len(grid))]
    return optimal_path

import numpy as np
from core.se3 import SE3
from core.so3 import SO3

def puzzle_path_rotating(x0=0.4, y0=-0.1, z0=0.2,
                         r=0.05, run_chamfer=0.015, rise_chamfer=0.015,
                         up1=0.03, up2=0.06,
                         top_len=0.05,
                         n_up1=4, n_chamfer=4, n_up2=7, n_arc=12, n_top=6):
    pts = []

    # 1) vertical up 30 mm
    for s in np.linspace(0.0, up1, n_up1, endpoint=True):
        pts.append(np.array([x0, y0, z0 + s]))

    # 2) 45° chamfer: +x,+z by 15 mm each
    for t in np.linspace(0.0, 1.0, n_chamfer, endpoint=True):
        pts.append(np.array([x0 + t*run_chamfer, y0, z0 + up1 + t*rise_chamfer]))

    # 3) vertical up additional 60 mm
    S = np.array([x0 + run_chamfer, y0, z0 + up1 + rise_chamfer])
    for s in np.linspace(0.0, up2, n_up2, endpoint=True):
        pts.append(S + np.array([0.0, 0.0, s]))

    # 4) quarter-circle of radius r bending toward +x (tangent +z -> +x)
    S_end = pts[-1]
    Cx, Cz = S_end[0] + r, S_end[2]
    for a in np.linspace(0.0, np.pi/2, n_arc, endpoint=True):
        x = Cx - r * np.cos(a)
        z = Cz + r * np.sin(a)
        pts.append(np.array([x, y0, z]))

    # 5) straight top along +x for 50 mm
    T_start = pts[-1]
    for s in np.linspace(0.0, top_len, n_top, endpoint=True):
        pts.append(T_start + np.array([s, 0.0, 0.0]))

    # remove duplicates
    out = []
    for p in pts:
        if not out or np.linalg.norm(p - out[-1]) > 1e-9:
            out.append(p)
    pts = out

    # build SE3 list with rotation following tangent
    poses = []
    up = np.array([0, 1, 0])  # fixed "side" or "up" direction (world Y)
    for i in range(len(pts)):
        if i < len(pts) - 1:
            tangent = pts[i + 1] - pts[i]
        else:
            tangent = pts[i] - pts[i - 1]
        tangent /= np.linalg.norm(tangent)

        z_axis = tangent                # direction of motion
        x_axis = np.cross(up, z_axis)   # lateral axis
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R = np.column_stack((x_axis, y_axis, z_axis))
        poses.append(SE3(translation=pts[i], rotation=SO3(R)))

    return poses





def generate_flat_poses(*, robot, xmax:float, xmin: float, ymax: float, ymin: float, ysteps: float, xsteps:float, height: float) -> SE3:
    x_vals = np.linspace(xmin, xmax, xsteps)
    y_vals = np.linspace(ymin, ymax, ysteps)
    z = height

    R = np.diag([-1.0, 1.0, -1.0])
    rot = SO3(R)

    poses = []
    for x in x_vals:
        for y in y_vals:
            t = np.array([x, y, z])
            poses.append(SE3(translation=t, rotation=rot))
    return poses