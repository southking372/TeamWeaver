import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.path import Path
import matplotlib.path as mpath

from habitat_llm.planner.miqp_planner.SingleIntegrator import SingleIntegrator
from habitat_llm.planner.miqp_planner.RTA import RTA
from habitat_llm.planner.miqp_planner.Swarm import Swarm
from habitat_llm.planner.miqp_planner.GlobalVarsManager import GlobalVarsManager
from habitat_llm.planner.miqp_planner.task_utils.scenario_params import ScenarioConfig
from habitat_llm.planner.miqp_planner.task_utils.opt_params import OptimizationConfig
from habitat_llm.planner.miqp_planner.task_utils.disturbance import DisturbanceConfig
from habitat_llm.planner.miqp_planner.task_utils.transport_module import TransportTask
from habitat_llm.planner.miqp_planner.task_utils.coverage_module import CoverageControl
from habitat_llm.planner.miqp_planner.optimizer.adaptive_optimizer import AdaptiveTaskOptimizer
from habitat_llm.planner.miqp_planner.task_utils.tools_util import get_task_assignment, print_info, read_mud_file

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
global_vars = GlobalVarsManager()

# Numbers and dimensions
n_r = 5
n_t = 2  # 1: transport and 2: perimeter defense
n_c = 2  # 1: locomotion and 2: monitoring
n_f = 3  # 1: wheels, 2: propellers, 3: camera
n_x = 3
n_u = 3

# Initialize managers
scenario_manager = ScenarioConfig()
opt_manager = OptimizationConfig()

# Access scenario parameters
scenario_params = scenario_manager.get_scenario_params()
updated_scenario_params = scenario_manager.get_updated_scenario_params()

def f(x):
    return np.zeros_like(x)
def g(x):
    return np.eye(n_x)
def sys_dyn(x, u):    
    return f(x) + g(x) @ u

# Global variables
global_vars.p_start = np.array([1, -0.6])
global_vars.p_goal = np.array([-1, 0.6])
global_vars.t_start = 2
global_vars.delta_t = 60
global_vars.poi = np.array([0, 1])
G = None
s = None
p_transport_t = None

# Initialize robots
robots = [None] * n_r
for i in range(n_r):
    robots[i] = SingleIntegrator()
environment = np.array([[1.8, 1.2], [-1.8, 1.2], [-1.8, -1.2], [1.8, -1.2]]).T

# Assuming Swarm class is defined elsewhere
s = Swarm(robots=robots, environment=environment, densityFunction=CoverageControl.phi_perimeter)

# Optimization parameters
opt_params = opt_manager.get_opt_params()

# Disturbance parameters
x_mud, y_mud = read_mud_file('mud.txt')
global_vars.robot_exo_dist = 4
global_vars.task_exo_dist = 2
global_vars.x_mud = x_mud
global_vars.y_mud = y_mud
global_vars.t_endogenous = 15
disturbance_manager = DisturbanceConfig(global_vars)

# Initialize simulation
DT = 0.1
x = np.zeros((3, n_r))
x[0, :] = 1.65 * np.ones(n_r)
x[1, :] = np.linspace(-1, 1, n_r)
x[2, :] = np.zeros(n_r)
global_vars.x = x 

# Assuming RTA class is defined elsewhere
rta = RTA(scenario_params, opt_params)
task_optimizer = AdaptiveTaskOptimizer(scenario_params, global_vars)

if __name__ == "__main__":

    # Initialize variables for simulation and plotting
    max_iter = 800
    x_traj = np.zeros((n_x, n_r, max_iter + 1))
    x_traj[:, :, 0] = x
    P_traj = np.zeros((2, max_iter + 1))
    P_traj[:, 0] = p_transport_t
    P_traj_valid_points = 0

    # Main simulation loop
    for iter in range(max_iter+1):
        t = iter * DT
        global_vars.x = x 
        p_transport_t = TransportTask.p_transport(t)
        global_vars.p_transport_t = p_transport_t
        
        if t >= global_vars.t_start:
            if P_traj_valid_points < P_traj.shape[1]:
                P_traj[:, P_traj_valid_points] = p_transport_t
                P_traj_valid_points += 1
            else:
                print("Warning: P_traj array full.")
        
        poses = np.vstack((x[0:2, :], np.zeros((1, n_r))))
        s.setPoses(poses)
        
        # Get coverage control information
        G, _, VC = s.coverageControl()
        global_vars.G = G  # Update global variable manager
        
        # Solve task allocation MIQP
        alpha, u, delta, time_to_synthesize_controller, opt_sol_info = rta.solve_miqp(x, t)
        
        # Reshape solutions
        alpha = alpha.reshape(n_t, n_r, order="F")
        u = u.reshape(n_u, n_r, order="F")
        delta = delta.reshape(n_t, n_r, order="F")
        task_assignment = get_task_assignment(alpha)
        
        # ===== Adaptive Function Optimizer =====
        optimization_updated = task_optimizer.optimize_tasks(x, t, task_assignment)
        if optimization_updated:
            print(f"Task Function has been Adaptive Optimized on t={t:.1f}s")
        
        for i in range(n_r):
            # Handle endogenous disturbance
            # if t >= t_endogenous and updated_scenario_params['A'][2, 0] != 0:
            #     # updated_scenario_params['A'][2, 0] = 0
            #     rta.set_scenario_params(updated_scenario_params)
            
            # Calculate and limit state update
            dx = sys_dyn(x[:, i], u[:, i]) * DT
            # print(f"Robot{i}: dx: {dx}, New Position: x[:, i] -> {x[:, i]}")
            x_sim_i = x[:, i].copy() + dx
            if disturbance_manager.check_exogenous_disturbance(x, alpha, i):
                x[2, i] += 0.95 * dx[2]
                x[:, i] += 0.05 * dx
            else:
                x[:, i] += dx
            
            # Update specialization matrix
            S = rta.get_specializations()
            for j in range(n_t):
                vars_dict = global_vars.get_all_vars()
                before_val = scenario_params['tasks'][j]['function'](x_sim_i, t, i, vars_dict=vars_dict)
                after_val = scenario_params['tasks'][j]['function'](x[:, i], t, i, vars_dict=vars_dict)
                Dh_ij = after_val - before_val
                S[j, i] = max(0, S[j, i] + 10 * alpha[j, i] * Dh_ij)
            rta.set_specializations(S)
        
        x_traj[:, :, iter + 1] = x

        # Calculate task function values
        h = np.zeros(n_t)
        vars_dict = global_vars.get_all_vars()
        for i in range(n_r):
            if task_assignment[i] > 0: 
                task_idx = task_assignment[i] - 1
                h[task_idx] = h[task_idx] + scenario_params['tasks'][task_idx]['function'](x[:, i], t, i, vars_dict=vars_dict)
        print_info(t, time_to_synthesize_controller, task_assignment, u, delta, S)