import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.path import Path
import matplotlib.path as mpath

# Assuming these classes are defined elsewhere
from habitat_llm.planner.miqp_planner.SingleIntegrator import SingleIntegrator
from habitat_llm.planner.miqp_planner.RTA import RTA
from habitat_llm.planner.miqp_planner.Swarm import Swarm
from habitat_llm.planner.miqp_planner.GlobalVarsManager import GlobalVarsManager
from habitat_llm.planner.miqp_planner.task_utils.tools_util import clamp, get_task_assignment, print_info, plot_quad, plot_fov, read_mud_file

plt.rcParams['font.sans-serif'] = ['SimHei']  # SimHei font for CJK plot labels
plt.rcParams['axes.unicode_minus'] = False  # fix minus sign rendering with CJK fonts

global_vars = GlobalVarsManager()

# Numbers and dimensions
n_r = 5          # number of robots
n_t = 4          # Mission type: Navi/explore/manipulation/Wait
n_c = 8          # capability categories
n_f = 4          # feature dims: locomotion/transport/manipulation/communication-compute
n_x = 3          # state dims: x,y,theta
n_u = 3          # controlInput dimension: linear velocity/Angular velocity/Monitoring direction adjustment

# 8capability categories:
# (1) MotorCategory sub-ability: explore, navi, pick, place, wait（5 in total)
# (2) find_action（context awareness）（1indivual)
# (3) communication, computation（2indivual)

# Robots' features and capabilities
A = np.zeros((n_f, n_r))
A[:, 0] = [1, 1, 1, 1]  # robot 1: full capability
A[:, 1] = [1, 1, 1, 1]  # robot 2: full capability
A[:, 2] = [1, 1, 0, 0]  # robot 3: locomotion+transport specialist
A[:, 3] = [1, 0, 1, 0]  # robot 4: locomotion+manipulation specialist
A[:, 4] = [1, 0, 0, 1]  # robot 5: locomotion+communication-compute specialist

T = np.zeros((n_t, n_c))

T[0, 0:2] = [1, 1]      # task 1: Navi - needs locomotion and transport
                        # capabilities 1-2 (Motor navi/pick)
T[1, 2:4] = [1, 1]      # task 2: explore - needs manipulation and communication
                        # capabilities 3-4 (find_action and communication-compute)
T[2, 1:3] = [1, 1]      # Mission 3: Manipulation - Requires Manipulation and transportation capabilities
                        # capabilities 2-3 (Motor place and manipulation)
T[3, 0] = 0             # task 4: wait - no capability required
                        # capability 1 (basic Motor)

# mapping Hs (8 capabilities → 4 features)
Hs = [np.zeros((1, n_f)) for _ in range(n_c)]

# Motor capabilities (first 5) map to locomotion/transport features
for i in range(5):
    Hs[i][0, 0] = 1     # locomotion feature supports Motor capability
    Hs[i][0, 1] = 1     # transport feature supports Motor capability
    
Hs[5][0, 2] = 1        # ManipulationFunction support find_action
Hs[6][0, 3] = 1        # Communication computing capabilities support communication capabilities
Hs[7][0, 3] = 1        # Communication computing capabilities support computing capabilities

# weight matrix ws
ws = [np.eye(1) for _ in range(n_c)]  # Default unit weight
ws[0] = 2 * np.eye(1)  # Strengthen the weight of walking ability
ws[2] = 1.5 * np.eye(1) # Strengthen the weight of Manipulation capabilities

scenario_params = {
    'A': A,
    'T': T,
    'Hs': Hs,
    'ws': ws
}
updated_scenario_params = {'A': A.copy()}

# Robot model
def f(x):
    return 0 * x

def g(x):
    return np.eye(n_x)

def sys_dyn(x, u):
    return f(x) + g(x) @ u

scenario_params['robot_dyn'] = {
    'f': f,
    'g': g,
    'n_x': n_x,
    'n_u': n_u
}

# Tasks
scenario_params['tasks'] = [None] * n_t

# Global variables
p_start = np.array([1, -0.6])
p_goal = np.array([-1, 0.6])
t_start = 2
delta_t = 60
p_transport_t = None
poi = np.array([0, 1])
G = None
s = None

# Transport function
# TODO: p_transportIt should also be necessary to introduce the state design of World Graph
def p_transport(t, vars_dict=None):
    # Get the variable value (preferably use the passed in dictionary, otherwise use global vars manager)
    if vars_dict is not None:
        p_start_val = vars_dict.get('p_start', global_vars.p_start)
        p_goal_val = vars_dict.get('p_goal', global_vars.p_goal)
        t_start_val = vars_dict.get('t_start', global_vars.t_start)
        delta_t_val = vars_dict.get('delta_t', global_vars.delta_t)
    else:
        p_start_val = global_vars.p_start
        p_goal_val = global_vars.p_goal
        t_start_val = global_vars.t_start
        delta_t_val = global_vars.delta_t
    
    if t < t_start_val:
        return p_start_val
    elif t > t_start_val + delta_t_val:
        return p_goal_val
    else:
        x = clamp(1 - (t - t_start_val) / delta_t_val, 0, 1) * p_start_val[0] + clamp((t - t_start_val) / delta_t_val, 0, 1) * p_goal_val[0]
        y = clamp(1 - (t - t_start_val)**8 / delta_t_val**8, 0, 1) * p_start_val[1] + clamp((t - t_start_val)**8 / delta_t_val**8, 0, 1) * p_goal_val[1]
        return np.array([x, y])

def p_transport_time_derivative(t, vars_dict=None):
    # Get the variable value (preferably use the passed in dictionary, otherwise use global vars manager)
    if vars_dict is not None:
        p_start_val = vars_dict.get('p_start', global_vars.p_start)
        p_goal_val = vars_dict.get('p_goal', global_vars.p_goal)
        t_start_val = vars_dict.get('t_start', global_vars.t_start)
        delta_t_val = vars_dict.get('delta_t', global_vars.delta_t)
    else:
        p_start_val = global_vars.p_start
        p_goal_val = global_vars.p_goal
        t_start_val = global_vars.t_start
        delta_t_val = global_vars.delta_t
    
    if t < t_start_val or t > t_start_val + delta_t_val:
        return np.array([0, 0])
    else:
        dx_dt = (p_goal_val[0] - p_start_val[0]) / delta_t_val
        dy_dt = 8 * (t - t_start_val)**7 / delta_t_val**8 * (p_goal_val[1] - p_start_val[1])
        return np.array([dx_dt, dy_dt])

# 1. navigation(Navi)Task
def navi_function(x_i, t, i, vars_dict=None):
    # Use vars_dict first, otherwise use global vars manager
    p_transport_t = vars_dict.get('p_transport_t', global_vars.p_transport_t)
    return -0.5 * np.linalg.norm(x_i[0:2] - p_transport_t)**2

def navi_gradient(x_i, t, i, vars_dict=None):
    p_transport_t = vars_dict.get('p_transport_t', global_vars.p_transport_t)
    gradient = np.zeros(3)
    gradient[0:2] = -2 * (x_i[0:2] - p_transport_t)
    return gradient

def navi_time_derivative(x_i, t, i, vars_dict=None):
    p_transport_t = vars_dict.get('p_transport_t', global_vars.p_transport_t)
    dp_dt = p_transport_time_derivative(t, vars_dict)
    return -2 * np.dot(x_i[0:2] - p_transport_t, dp_dt)

# 2. exploration mission
def explore_function(x_i, t, i, vars_dict=None):
    G = vars_dict.get('G', global_vars.G)
    if G is None or G.shape[1] <= i:
        return 0.0
    pos = np.clip(x_i[0:2], [-1.8, -1.2], [1.8, 1.2])
    target = np.clip(G[:, i], [-1.8, -1.2], [1.8, 1.2])
    pos_error = -np.linalg.norm(pos - target)**2 / (1.8**2 + 1.2**2)
    desired_angle = np.arctan2(target[1] - pos[1], target[0] - pos[0])
    angle_diff = np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
    angle_error = -0.5 * angle_diff**2
    return pos_error + angle_error

def explore_gradient(x_i, t, i, vars_dict=None):
    G = vars_dict.get('G', global_vars.G)
    if G is None or G.shape[1] <= i:
        return np.zeros(3)
    gradient = np.zeros(3)
    pos = np.clip(x_i[0:2], [-1.8, -1.2], [1.8, 1.2])
    target = np.clip(G[:, i], [-1.8, -1.2], [1.8, 1.2])
    gradient[0:2] = -2 * (pos - target) / (1.8**2 + 1.2**2)
    desired_angle = np.arctan2(target[1] - pos[1], target[0] - pos[0])
    gradient[2] = -np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
    return gradient

def explore_time_derivative(x_i, t, i, vars_dict=None):
    # G = vars_dict.get('G', global_vars.G)
    # if G is None or G.shape[1] <= i:
    #     return np.zeros(3)
    # dp_dt = p_transport_time_derivative(t, vars_dict)
    # return -2 * np.dot(x_i[0:2] - G[:, i], dp_dt)
    return 0

# 3. manipulationTask
def manipulation_function(x_i, t, i, vars_dict=None):
    poi = vars_dict.get('poi', global_vars.poi)
    dist_to_target = np.linalg.norm(x_i[0:2] - poi)
    pos_error = -(dist_to_target - 0.3)**2
    angle_error = -0.5 * (x_i[2] - np.arctan2(poi[1] - x_i[1], poi[0] - x_i[0]))**2
    return pos_error + angle_error

def manipulation_gradient(x_i, t, i, vars_dict=None):
    poi = vars_dict.get('poi', global_vars.poi)
    gradient = np.zeros(3)
    dist_to_target = np.linalg.norm(x_i[0:2] - poi)
    if dist_to_target > 1e-6:
        gradient[0:2] = -2 * (dist_to_target - 0.3) * (x_i[0:2] - poi) / dist_to_target
    target_angle = np.arctan2(poi[1] - x_i[1], poi[0] - x_i[0])
    gradient[2] = -(x_i[2] - target_angle)
    return gradient

def manipulation_time_derivative(x_i, t, i, vars_dict=None):
    """manipulationTask time derivative"""
    return 0  # Assume that the manipulation target is static

# 4. Waiting for tasks
def wait_function(x_i, t, i, vars_dict=None):
    """The objective function of the waiting task"""
    # Maintain current position, minimize speed
    return -0.02 * np.linalg.norm(x_i[0:2])**2  # Slightly inclined to return to the starting point

def wait_gradient(x_i, t, i, vars_dict=None):
    """Gradient of waiting tasks"""
    gradient = np.zeros(3)
    gradient[0:2] = -0.2 * x_i[0:2]  # Produce a small force towards the origin
    return gradient

def wait_time_derivative(x_i, t, i, vars_dict=None):
    """Time derivative of waiting tasks"""
    return 0  # Static tasks

# Add tasks to scenario_params
scenario_params['tasks'][:] = [
    {
        'function': navi_function,
        'gradient': navi_gradient,
        'time_derivative': navi_time_derivative
    },
    {
        'function': explore_function,
        'gradient': explore_gradient,
        'time_derivative': explore_time_derivative
    },
    {
        'function': manipulation_function,
        'gradient': manipulation_gradient,
        'time_derivative': manipulation_time_derivative
    },
    {
        'function': wait_function,
        'gradient': wait_gradient,
        'time_derivative': wait_time_derivative
    }
]

# Initialize robots
robots = [None] * n_r
for i in range(n_r):
    # Assuming SingleIntegrator class is defined elsewhere
    robots[i] = SingleIntegrator()

# Area
environment = np.array([[1.8, 1.2], [-1.8, 1.2], [-1.8, -1.2], [1.8, -1.2]]).T

# Define a simplified density function
def simple_density(x, y):
    # Use a simple Gaussian distribution as the density function
    return np.exp(-0.5 * ((x/1.8)**2 + (y/1.2)**2))

# Initialize Swarm, using simplified density function
s = Swarm(robots=robots, environment=environment, densityFunction=simple_density)

# Optimization parameters
opt_params = {
    'l': 1e-6,  # relative weight delta/u in the cost
    'kappa': 1e6,  # scale between tasks with different priorities
    'delta_max': 1e3,  # delta_max
    # 'n_r_bounds': np.array([[1, 1], [3, 3]]),  # row i is min and max number of robots for task i
    'n_r_bounds': np.array([
        [1, 2],  # NaviThe mission requires 1-2 robots
        [1, 2],  # Exploration missions require 1-2 robots
        [1, 1],  # manipulationThe mission requires 1 robot
        [0, 1]   # Waiting for tasks 0-1 robots
    ]),
    'gamma': lambda x: 5*x  # class K function for task execution
}

# Disturbance parameters
robot_exo_dist = 4
task_exo_dist = 2
# Assuming x_mud and y_mud are loaded from a file
# In Python we will create simple shapes for visualization
x_mud = np.array([-0.5, 0.5, 0.5, -0.5, -0.5])
y_mud = np.array([-0.5, -0.5, 0.5, 0.5, -0.5])

t_endogenous = 15

# Initialize simulation
DT = 0.1
x = np.zeros((3, n_r))
x[0, :] = 1.65 * np.ones(n_r)
x[1, :] = np.linspace(-1, 1, n_r)
x[2, :] = np.zeros(n_r)

# Assuming RTA class is defined elsewhere
rta = RTA(scenario_params, opt_params)

# Helper functions

def exogenous_disturbance(x, alpha, i, vars_dict=None):
    if vars_dict is not None:
        robot_exo_dist_val = vars_dict.get('robot_exo_dist', global_vars.robot_exo_dist)
        task_exo_dist_val = vars_dict.get('task_exo_dist', global_vars.task_exo_dist)
        x_mud_val = vars_dict.get('x_mud', global_vars.x_mud)
        y_mud_val = vars_dict.get('y_mud', global_vars.y_mud)
    else:
        robot_exo_dist_val = global_vars.robot_exo_dist
        task_exo_dist_val = global_vars.task_exo_dist
        x_mud_val = global_vars.x_mud
        y_mud_val = global_vars.y_mud
    
    # Check if point is in polygon (mud area)
    path = Path(np.column_stack((x_mud_val, y_mud_val)))
    in_mud = path.contains_point((x[0, i], x[1, i]))
    
    return (i == robot_exo_dist_val - 1) and (alpha[task_exo_dist_val - 1, i] > 0) and in_mud


# Mud area
x_mud, y_mud = read_mud_file('mud.txt')

p_transport_t = p_transport(0) 

# Initialize variables for simulation and plotting
max_iter = 800
x_traj = np.zeros((n_x, n_r, max_iter + 1))
x_traj[:, :, 0] = x
P_traj = np.column_stack((np.nan, np.nan))  # Initialize with NaN
h_quad = []
h_fov = []
P_traj = np.column_stack((p_transport(0), p_transport(0)))
# Main simulation loop
for iter in range(max_iter):
    t = iter * DT
    global_vars.p_transport_t = p_transport(t)
    vars_dict = global_vars.get_all_vars()
    
    # Update transport position and trajectory
    p_transport_t = p_transport(t)
    P_traj = np.column_stack((P_traj, p_transport_t.reshape(2, 1)))
    
    # Make sure the robot position is within the environment boundaries
    x[0, :] = np.clip(x[0, :], -1.8, 1.8)
    x[1, :] = np.clip(x[1, :], -1.2, 1.2)
    x[2, :] = np.mod(x[2, :], 2*np.pi)  # Limit the angle to[0, 2π]within range
    
    # Update swarm poses
    poses = np.vstack((x[0:2, :], np.zeros((1, n_r))))
    s.setPoses(poses)
    
    G, _, VC = s.coverageControl()
    global_vars.G = G  # Update global variable manager
    
    # Solve task assignment MIQP
    alpha, u, delta, time_to_synthesize_controller, opt_sol_info = rta.solve_miqp(x, t)
    
    # Reshape solutions
    alpha = alpha.reshape(n_t, n_r, order="F")
    u = u.reshape(n_u, n_r, order="F")
    delta = delta.reshape(n_t, n_r, order="F")
    task_assignment = get_task_assignment(alpha)
    
    for i in range(n_r):
        # Handle endogenous disturbance
        # if t >= t_endogenous and updated_scenario_params['A'][2, 0] != 0:
        #     # updated_scenario_params['A'][2, 0] = 0
        #     rta.set_scenario_params(updated_scenario_params)
        
        # Handle exogenous disturbance
        dx = sys_dyn(x[:, i], u[:, i]) * DT
        print(f"Robot{i}: dx: {dx}, New Position: x[:, i] -> {x[:, i]}")
        x_sim_i = x[:, i].copy() + dx
        if exogenous_disturbance(x, alpha, i, vars_dict=global_vars.get_all_vars()):
            x[2, i] += 0.95 * dx[2]
            x[:, i] += 0.05 * dx
        else:
            x[:, i] += dx
        
        # Update specializations
        S = rta.get_specializations()
        for j in range(n_t):
            vars_dict = global_vars.get_all_vars()
            before_val = scenario_params['tasks'][j]['function'](x_sim_i, t, i, vars_dict=vars_dict)
            after_val = scenario_params['tasks'][j]['function'](x[:, i], t, i, vars_dict=vars_dict)
            Dh_ij = after_val - before_val
            S[j, i] = max(0, S[j, i] + 10 * alpha[j, i] * Dh_ij)
        
        rta.set_specializations(S)
    
    # Save trajectory
    x_traj[:, :, iter + 1] = x

    # Calculate task function values
    h = np.zeros(n_t)
    for i in range(n_r):
        if task_assignment[i] >= 0:
            task_idx = task_assignment[i]
            h[task_idx] += scenario_params['tasks'][task_idx]['function'](
                x[:, i], t, i, vars_dict=vars_dict
            )
    
    # Print information about current state
    print_info(t, time_to_synthesize_controller, task_assignment, u, delta, S, t_endogenous)

# plt.show()  # Keep the final plot visible