import numpy as np
from ctu_crs import CRS93, CRS97
from config import config as conf
from core.se3 import SE3
from core.so3 import SO3
import heapq


#robot end effector to stick transformation, will have to be set according to the robot type but i believe this is the correct transform
_T_es = SE3(translation=np.array([-0.135, 0.0, 0.0]), rotation=SO3(np.diag([-1, 1, -1])))
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

    ik = np.asarray(ik)
    ref = ik[0]
    dist = np.linalg.norm(ik - ref, axis=1)
    order = np.argsort(dist)

    ik_sorted = ik[order]
    return ik_sorted

def follow_path(*, path: list[SE3], robot):
    #construct the sol list
    grid =[]
    for position in path:
        sols = ik(position=position, robot=robot)
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

def follow_path_chatgpt(*, path: list, robot):
    grid = []
    for position in path:
        sols = ik(position=position, robot=robot)
        grid.append(sols)

    n_layers = len(grid)
    total_nodes = sum(len(layer) for layer in grid)
    node_index = []
    offset = 0
    for layer in grid:
        node_index.append(range(offset, offset + len(layer)))
        offset += len(layer)
    adj = {idx: [] for idx in range(total_nodes)}
    for layer_idx, layer in enumerate(grid):
        for i, state_a in enumerate(layer):
            for j, state_b in enumerate(layer):
                if i == j:
                    continue
                cost = np.linalg.norm(state_a - state_b)
                adj[node_index[layer_idx][i]].append((node_index[layer_idx][j], cost))
    for i in range(n_layers - 1):
        for j, state_a in enumerate(grid[i]):
            for k, state_b in enumerate(grid[i + 1]):
                cost = np.linalg.norm(state_a - state_b)
                adj[node_index[i][j]].append((node_index[i + 1][k], cost))
    start_nodes = list(node_index[0])
    goal_nodes = list(node_index[-1])

    dist = {idx: np.inf for idx in adj}
    prev = {idx: None for idx in adj}
    pq = []

    for s in start_nodes:
        dist[s] = 0.0
        heapq.heappush(pq, (0.0, s))
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    goal = min(goal_nodes, key=lambda g: dist[g])
    seq = []
    u = goal
    while u is not None:
        seq.append(u)
        u = prev[u]
    seq.reverse()
    optimal_path = []
    for idx in seq:
        for layer_idx, layer_range in enumerate(node_index):
            if idx in layer_range:
                local_idx = idx - layer_range.start
                optimal_path.append(grid[layer_idx][local_idx])
                break

    return optimal_path



def generate_flat_poses(*, robot, xmax:float, xmin: float, ymax: float, ymin: float, ysteps: float, xsteps:float, height: float) -> SE3:
    x_vals = np.linspace(xmin, xmax, xsteps)
    y_vals = np.linspace(ymin, ymax, ysteps)
    z = height

    R = np.diag([1.0, 1.0, 1.0])
    rot = SO3(R)

    poses = []
    for x in x_vals:
        for y in y_vals:
            t = np.array([x, y, z])
            poses.append(SE3(translation=t, rotation=rot))
    return poses


def offset_path(*, path: list[SE3], offset: SE3) -> list[SE3]:
    result = []
    for pose in path:
        result.append(pose * offset)
    
    return result
