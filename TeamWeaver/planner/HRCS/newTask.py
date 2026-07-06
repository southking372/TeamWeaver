import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.path import Path
import matplotlib.path as mpath

#Import base classes
from class_def.SingleIntegrator import SingleIntegrator
from class_def.task_module.RTA_task import RTA
from class_def.task_module.Swarm_task import Swarm
from class_def.task_module.GlobalVarsManager_task import GlobalVarsManager_task
from task_utils.tools_util import clamp, get_task_assignment, print_info, plot_quad, plot_fov, read_mud_file

#Import modular components
from task_utils.task_module.scenario_params_task import ScenarioConfigTask
from task_utils.task_module.opt_params_task import OptimizationConfigTask
from task_utils.task_module.control_module import ControlConfig
from task_utils.task_module.simulation_module import SimulationConfig
from task_utils.task_module.visualization_module import VisualizationConfig
from task_utils.task_module.environment_module import EnvironmentConfig
from task_utils.task_module.robot_features_module import RobotFeaturesConfig
from task_utils.task_module.robot_dynamics_module import RobotDynamicsConfig
from task_utils.task_module.task_config_module import TaskConfig
from task_utils.task_module.disturbance_module import DisturbanceConfig

#Import task module
from task_utils.task_module.explore_module import ExploreTask
from task_utils.task_module.manipulation_module import ManipulationTask, ManipulationPhase
from task_utils.task_module.wait_module import WaitTask
from task_utils.task_module.navi_module import NaviTask

plt.rcParams['font.sans-serif'] = ['SimHei']  # SimHei font for CJK plot labels
plt.rcParams['axes.unicode_minus'] = False  # fix minus sign rendering with CJK fonts

global_vars = GlobalVarsManager_task()
scenario_config = ScenarioConfigTask(n_u=2)
opt_config = OptimizationConfigTask()
control_config = ControlConfig()
control_params = control_config.get_control_params()

new_u_max = np.array([0.5, 2.5]) #Maximum linear speed 0.5,Angular speed maximum 2.5rad/s
control_config.update_control_params('u_max', new_u_max)
print(f"The robot maximum speed limit has been updated to: {new_u_max}")
control_params = control_config.get_control_params()

simulation_config = SimulationConfig()
visualization_config = VisualizationConfig()
environment_config = EnvironmentConfig()
robot_features_config = RobotFeaturesConfig()
robot_dynamics_config = RobotDynamicsConfig(n_u=2)
task_config = TaskConfig()
disturbance_config = DisturbanceConfig(global_vars)
scenario_params = scenario_config.get_scenario_params()
updated_scenario_params = scenario_config.get_updated_scenario_params()
global_task_vars = scenario_config.get_global_task_vars()
global_vars.register_vars_from_dict(global_task_vars)

p_goal = global_vars.get_var('p_goal')
opt_params = opt_config.get_opt_params()
control_params = control_config.get_control_params()

simulation_params = simulation_config.get_sim_params()
DT = simulation_params['DT']
max_iter = simulation_params['max_iter']
env_params = environment_config.get_env_params()
environment = env_params['environment']

#Get robot characteristics
robot_features = robot_features_config.get_robot_features()
n_r = robot_features['n_r']
n_f = robot_features['n_f']
robot_dynamics = robot_dynamics_config.get_robot_dynamics()
n_x = robot_dynamics['n_x']
n_u = robot_dynamics['n_u']
sys_dyn = robot_dynamics['sys_dyn']
task_config_params = task_config.get_task_params()
n_t = task_config_params['n_t']
n_c = task_config_params['n_c']
robots = [None] * n_r
for i in range(n_r):
    robots[i] = SingleIntegrator()
if n_u != 2:
    print(f"warn: RobotDynamicsConfigreturnedcontrolDimensionsn_u={n_u}, expected to be 2.")

def simple_density(x, y):
    return np.exp(-0.5 * ((x/1.8)**2 + (y/1.2)**2))

#initializationSwarm, using a simplified density function
s = Swarm(robots=robots, environment=environment, densityFunction=simple_density)
global_vars.s = s  #storageSwarmobject to global variable
rta = RTA(scenario_params, opt_params)

x = simulation_config.get_initial_states()
global_vars.x = x  #Store initial state in global variables
x_traj = simulation_config.get_trajectory()

fig = visualization_config.setup_plot()
ax = plt.gca()

replan_interval = 300 #Replan every 300 iterations(3s)

print("Make initial task assignments...")
t_initial = 0.0
alpha, u, delta, time_to_synthesize_controller, opt_sol_info = rta.solve_miqp(x, t_initial)
alpha = alpha.reshape(n_t, n_r, order="F")
u = u.reshape(n_u, n_r, order="F")
delta = delta.reshape(n_t, n_r, order="F")
task_assignment = get_task_assignment(alpha)
print(f"Initial task assignment completed: {task_assignment}")

for iter in range(max_iter):
    t = iter * DT

    #Update time-related task variables(includemanipulationPhase timer and wait timer)
    global_vars.update_task_timer(DT)

    #Make sure the robot position is within the environment boundaries
    # x[0, :] = np.clip(x[0, :], -1.8, 1.8)
    # x[1, :] = np.clip(x[1, :], -1.2, 1.2)
    # x[2, :] = np.mod(x[2, :], 2*np.pi)  #Limit the angle to[0, 2π]within range
    # global_vars.x = x  #Update robot status in global variables

    #renewSwarmattitude
    poses = np.vstack((x[0:2, :], np.zeros((1, n_r))))
    s.setPoses(poses)

    #Execute coveragecontrol, getVoronoicenter of mass
    G, _, VC = s.coverageControl()
    global_vars.G = G  #Update the global variable managerVoronoicenter of mass

    #Update exploration target status
    robot_positions = x[0:2, :].T  #Extract all robotsposition
    updated_targets = global_vars.update_exploration_status(robot_positions)

    #--- Task allocation by frequency ---
    if iter % replan_interval == 0:
        print(f"\n--- Iter {iter}:Replan tasks ---")

        # --- Debug:Output the utility function value of each robot task ---
        print(f"Iter {iter}:Calculate task utility value...")
        current_vars_dict = global_vars.get_all_vars()
        task_names = [task.get('name', f'Task_{idx}') for idx, task in enumerate(scenario_params['tasks'])] #Get task name
        for i in range(n_r):
            print(f"  Robot {i} (Pos: [{x[0, i]:.2f}, {x[1, i]:.2f}], Theta: {x[2, i]:.2f}):")
            #Check if robot is the owner
            is_current_holding = current_vars_dict.get('is_holding', False) and current_vars_dict.get('holding_robot_id') == i
            print(f"    Is Holding: {is_current_holding}")
            for j in range(n_t):
                try:
                    #Use current robot statex[:, i]and timetCalculate utility value
                    utility_value = scenario_params['tasks'][j]['function'](x[:, i], t, i, vars_dict=current_vars_dict)
                    print(f"    - {task_names[j]} (Index {j}): Utility = {utility_value:.2f}")
                except Exception as e:
                    print(f"    - {task_names[j]} (Index {j}): Error calculating utility: {e}")
        # --- End Debug ---

        #Solve task assignmentMIQP
        alpha_new, u_new, delta_new, time_to_synthesize_controller, opt_sol_info = rta.solve_miqp(x, t)

        #Reshape the solution and update global variables
        alpha = alpha_new.reshape(n_t, n_r, order="F")
        u = u_new.reshape(n_u, n_r, order="F") #Note: hereuyesMIQPsolution, possible and actualcontrolInputs are different
        delta = delta_new.reshape(n_t, n_r, order="F")
        task_assignment = get_task_assignment(alpha)
        print(f"Iter {iter}:New task assignment: {task_assignment}")

        #Update wait time (accumulate wait time if wait task is assigned, reset otherwise)
        #Note: It may be more appropriate to move this logic to check every iteration, depending on the specific needs
        # wait_task_idx = 3 #Assumption 3 is the waiting task index
        # assigned_to_wait = any(ta == wait_task_idx for ta in task_assignment)
        # if not assigned_to_wait:
        #     global_vars.reset_wait_timer_for_all() #Need a function to reset the waiting time of all robots

    # --------------------------

    #--- renewmanipulationTask (checked every iteration) ---
    manipulation_task_idx = 2 #Assumption 2 ismanipulationTask index
    manipulating_robot_indices = [i for i, task in enumerate(task_assignment) if task == manipulation_task_idx]

    for robot_idx in manipulating_robot_indices:
        #Check if you can proceed to the next stage(Enter based on distance and exit at a fixed time)
        advanced = global_vars.check_and_advance_manipulation_phase(robot_idx)
    # -------------------------------------

    #--- Use the visualization module to print debugging information for all tasks ---
    # visualization_config.print_task_debug_info(x, t, task_assignment, global_vars, iter) #Comment it out temporarily to reduce output

    #Save the current state of each robot for use in calculationsdelta H
    x_prev = x.copy()

    #--- Application Movementcontrol(Executed on every iteration) ---
    for i in range(n_r):
        #Get the current task assignment and global variables
        current_task = task_assignment[i] #Use current (possibly from last planning) task assignments
        vars_dict = global_vars.get_all_vars()

        #Call the corresponding task according to the task allocationapply_motion_controlmethod
        try:
            if current_task == 0:  #Navigation tasks
                x[:, i] = NaviTask.apply_motion_control(x[:, i], t, i, vars_dict, DT)
                # if iter % 10 == 0: print(f"Iter {iter}:robot{i}Perform navigation tasksmovement")
            elif current_task == 1:  #exploration mission
                x[:, i] = ExploreTask.apply_motion_control(x[:, i], t, i, vars_dict, DT)
                # if iter % 10 == 0: print(f"Iter {iter}:robot{i}Execute exploration missionmovement")
            elif current_task == 2:  # manipulationTask
                x[:, i] = ManipulationTask.apply_motion_control(x[:, i], t, i, vars_dict, DT)
                # if iter % 10 == 0: print(f"Iter {iter}:robot{i}implementmanipulationTaskmovement,stage: {global_vars.manipulation_phase.name}")
            elif current_task == 3: #Waiting for tasks
                x[:, i] = WaitTask.apply_motion_control(x[:, i], t, i, vars_dict, DT)
                # if iter % 10 == 0: print(f"Iter {iter}:robot{i}Execute waiting tasks")
            else:
                #For unknown mission types or default cases, base dynamics may be applied or left unchanged
                # dx = sys_dyn(x[:, i], u[:, i]) * DT #Note: hereufromMIQP, may not be optimalcontrolenter
                # x[:, i] += dx #useMIQPofuMay cause unexpected behavior, comment out
                print(f"Iter {iter}:robot{i}Task{current_task}no specific sportcontrol, keep the status")
                pass #keep current statusx[:, i]constant

        except Exception as e:
            print(f"Iter {iter}:robot{i}Task{current_task}sportscontrolError: {e}")
            #Option to maintain state or apply base dynamics on error
            # dx = sys_dyn(x[:, i], u[:, i]) * DT #Likewise, useMIQPofuThere may be a problem
            # x[:, i] += dx
            pass #Maintain state on error

    #--- Update the specialization matrix (performed every iteration, using thealpha） ---
    S = rta.get_specializations()
    for i in range(n_r): #Traversing robots
        for j in range(n_t): #Traverse tasks
            vars_dict = global_vars.get_all_vars() #Get the latestphase/progressglobal variables
            #Calculated using saved previous statedelta H
            #Note: passed in herex[:, i]yes*alreadymovementPass*status
            try:
                before_val = scenario_params['tasks'][j]['function'](x_prev[:, i], t, i, vars_dict=vars_dict)
                after_val = scenario_params['tasks'][j]['function'](x[:, i], t, i, vars_dict=vars_dict) #usemovementthe final state
                Dh_ij = after_val - before_val
                #use currentalpha (From the latest planning)to updateS
                S[j, i] = max(0, S[j, i] + 10 * alpha[j, i] * Dh_ij)
            except Exception as e:
                 print(f"Iter {iter}:update bot{i}Task{j}Error while specializing: {e}")

    rta.set_specializations(S)
    # -----------------------------------------------------------------------

    #Save track
    simulation_config.update_trajectory(iter, x)

    #Update visualization
    plt.clf()
    ax = plt.gca()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True)
    
    #Draw environment boundaries
    border = Rectangle((-1.8, -1.2), 3.6, 2.4, edgecolor=[0.5, 0.5, 0.5], linewidth=5, facecolor='none')
    ax.add_patch(border)
    
    #drawing robot
    for i in range(n_r):
        robot_type = 'wheeled'  #Default is wheeled robot
        
        #Get the robot'smanipulationStages and Holding States
        phase = None
        is_holding = False
        
        #Check if the bot is executingmanipulationTask
        if task_assignment[i] == 2:  #Assumption 2 ismanipulationindex of task
            #Get currentmanipulationstage
            if hasattr(global_vars, 'manipulation_phase'):
                phase = global_vars.manipulation_phase.name
                
                #Check if it is the holder
                if hasattr(global_vars, 'is_holding') and hasattr(global_vars, 'holding_robot_id'):
                    is_holding = global_vars.is_holding and global_vars.holding_robot_id == i
        
        #Use new visualization methods
        visualization_config.plot_robot_with_phase(ax, x, i, robot_type, phase, is_holding)
    
    #Draw static navigation target pointsp_goal
    if p_goal is not None:
        ax.plot(p_goal[0], p_goal[1], 'k*', markersize=12, label='Nav Goal')
    
    #Draw exploration goals
    exploration_targets = global_vars.exploration_targets
    if exploration_targets:
        #Draw legend only once
        plotted_explored = False
        plotted_unexplored = False
        for target in exploration_targets:
            is_explored = target.get('explored', False)
            color = 'g' if is_explored else 'r'
            label = None
            if is_explored and not plotted_explored:
                label = 'Explored Target'
                plotted_explored = True
            elif not is_explored and not plotted_unexplored:
                 label = 'Unexplored Target'
                 plotted_unexplored = True
            ax.plot(target['position'][0], target['position'][1], color + 'o', markersize=8, label=label)
    
    #drawmanipulationTarget
    target_obj_pos = global_vars.target_object_position
    target_rec_pos = global_vars.target_receptacle_position
    plotted_obj = False
    plotted_rec = False
    
    #Update target location information in visualization module
    visualization_config.set_target_positions(target_obj_pos, target_rec_pos)
    
    #Check if there is a robot holding the object
    is_any_robot_holding = hasattr(global_vars, 'is_holding') and global_vars.is_holding
    
    if target_obj_pos is not None:
        #If there is a robot holding the object, the target object should appear as"Obtained"state
        if is_any_robot_holding:
            label = 'Object Target (Obtained)' if not plotted_obj else None
            ax.plot(target_obj_pos[0], target_obj_pos[1], 'rs', markersize=8, label=label)
        else:
            label = 'Object Target' if not plotted_obj else None
            ax.plot(target_obj_pos[0], target_obj_pos[1], 'bs', markersize=8, label=label)
        plotted_obj = True
        
    if target_rec_pos is not None:
        label = 'Receptacle Target' if not plotted_rec else None
        ax.plot(target_rec_pos[0], target_rec_pos[1], 'ms', markersize=8, label=label)
        plotted_rec = True
    
    #Update task text and current stage text
    visualization_config.update_task_text(ax, t, task_assignment)
    manipulation_phase = global_vars.manipulation_phase
    if manipulation_phase:
        phase_text = f"manipulationstage: {manipulation_phase.name}"
        ax.text(-1.8, 1.35, phase_text, fontsize=10)
        
        #Add status information of held objects
        if hasattr(global_vars, 'is_holding') and hasattr(global_vars, 'holding_robot_id'):
            if global_vars.is_holding:
                holding_text = f"holding object:robot{global_vars.holding_robot_id}"
                ax.text(-1.8, 1.25, holding_text, fontsize=10, color='red')
            else:
                holding_text = "holding object:none"
                ax.text(-1.8, 1.25, holding_text, fontsize=10)
    
    #Show waiting time
    wait_time = global_vars.wait_elapsed_time
    wait_threshold = global_vars.wait_step_threshold
    wait_text = f"waiting time: {wait_time:.1f}s / {wait_threshold:.1f}s"
    wait_progress = f"schedule: {min(1.0, wait_time/wait_threshold):.0%}"
    ax.text(1.0, 1.35, wait_text, fontsize=10)
    ax.text(1.0, 1.25, wait_progress, fontsize=10)
    
    #Calculate the percentage of targets explored
    if exploration_targets:
        explored_count = sum(1 for target in exploration_targets if target.get('explored', False))
        total_count = len(exploration_targets)
        explore_text = f"Exploration progress: {explored_count}/{total_count}"
        explore_progress = f"Finish: {explored_count/total_count:.0%}"
        ax.text(0.0, 1.35, explore_text, fontsize=10)
        ax.text(0.0, 1.25, explore_progress, fontsize=10)
    
    #Use the visualization module to draw each task information
    visualization_config.visualize_manipulation_phases(ax, x, task_assignment, global_vars)
    visualization_config.visualize_navi_task(ax, x, task_assignment, p_goal)
    visualization_config.visualize_explore_task(ax, x, task_assignment, global_vars)
    
    #Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) #Remove duplicates
    ax.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
    
    plt.draw()
    #--- Increase visualization pause time ---
    plt.pause(0.1) #Increased from 0.01 to 0.1 seconds
    # ---------------------------

    #Print current status information(Reduce printing frequency)
    if iter % 10 == 0:
         print(f"T={t:.1f}s | Assign: {task_assignment} | Manip Phase: {global_vars.manipulation_phase.name} | Wait: {global_vars.wait_elapsed_time:.1f}s")

#Show final graph
plt.show()
