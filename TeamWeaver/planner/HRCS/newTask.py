import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.path import Path
import matplotlib.path as mpath

# 导入基础类
from class_def.SingleIntegrator import SingleIntegrator
from class_def.task_module.RTA_task import RTA
from class_def.task_module.Swarm_task import Swarm
from class_def.task_module.GlobalVarsManager_task import GlobalVarsManager_task
from task_utils.tools_util import clamp, get_task_assignment, print_info, plot_quad, plot_fov, read_mud_file

# 导入模块化组件
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

# 导入任务模块
from task_utils.task_module.explore_module import ExploreTask
from task_utils.task_module.manipulation_module import ManipulationTask, ManipulationPhase
from task_utils.task_module.wait_module import WaitTask
from task_utils.task_module.navi_module import NaviTask

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

global_vars = GlobalVarsManager_task()
scenario_config = ScenarioConfigTask(n_u=2)
opt_config = OptimizationConfigTask()
control_config = ControlConfig()
control_params = control_config.get_control_params()

new_u_max = np.array([0.5, 2.5]) # 线速度最大0.5, 角速度最大2.5 rad/s
control_config.update_control_params('u_max', new_u_max)
print(f"机器人最大速度限制已更新为: {new_u_max}")
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

# 获取机器人特征
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
    print(f"警告: RobotDynamicsConfig 返回的控制维度 n_u={n_u}，预期为 2。")

def simple_density(x, y):
    return np.exp(-0.5 * ((x/1.8)**2 + (y/1.2)**2))

# 初始化Swarm，使用简化的密度函数
s = Swarm(robots=robots, environment=environment, densityFunction=simple_density)
global_vars.s = s  # 存储Swarm对象到全局变量
rta = RTA(scenario_params, opt_params)

x = simulation_config.get_initial_states()
global_vars.x = x  # 存储初始状态到全局变量
x_traj = simulation_config.get_trajectory()

fig = visualization_config.setup_plot()
ax = plt.gca()

replan_interval = 300 # 每 300 次迭代重新规划一次 (3s)

print("进行初始任务分配...")
t_initial = 0.0
alpha, u, delta, time_to_synthesize_controller, opt_sol_info = rta.solve_miqp(x, t_initial)
alpha = alpha.reshape(n_t, n_r, order="F")
u = u.reshape(n_u, n_r, order="F")
delta = delta.reshape(n_t, n_r, order="F")
task_assignment = get_task_assignment(alpha)
print(f"初始任务分配完成: {task_assignment}")

for iter in range(max_iter):
    t = iter * DT

    # 更新与时间相关的任务变量 (包括操作阶段计时器和等待计时器)
    global_vars.update_task_timer(DT)

    # 确保机器人位置在环境边界内
    # x[0, :] = np.clip(x[0, :], -1.8, 1.8)
    # x[1, :] = np.clip(x[1, :], -1.2, 1.2)
    # x[2, :] = np.mod(x[2, :], 2*np.pi)  # 将角度限制在[0, 2π]范围内
    # global_vars.x = x  # 更新全局变量中的机器人状态

    # 更新Swarm姿态
    poses = np.vstack((x[0:2, :], np.zeros((1, n_r))))
    s.setPoses(poses)

    # 执行覆盖控制，获取Voronoi质心
    G, _, VC = s.coverageControl()
    global_vars.G = G  # 更新全局变量管理器中的Voronoi质心

    # 更新探索目标状态
    robot_positions = x[0:2, :].T  # 提取所有机器人的位置
    updated_targets = global_vars.update_exploration_status(robot_positions)

    # --- 按频率进行任务分配 ---
    if iter % replan_interval == 0:
        print(f"\n--- Iter {iter}: 重新规划任务 ---")

        # --- Debug: 输出各机器人任务效用函数值 ---
        print(f"Iter {iter}: 计算任务效用值...")
        current_vars_dict = global_vars.get_all_vars()
        task_names = [task.get('name', f'Task_{idx}') for idx, task in enumerate(scenario_params['tasks'])] # 获取任务名称
        for i in range(n_r):
            print(f"  Robot {i} (Pos: [{x[0, i]:.2f}, {x[1, i]:.2f}], Theta: {x[2, i]:.2f}):")
            # 检查机器人是否为持有者
            is_current_holding = current_vars_dict.get('is_holding', False) and current_vars_dict.get('holding_robot_id') == i
            print(f"    Is Holding: {is_current_holding}")
            for j in range(n_t):
                try:
                    # 使用当前的机器人状态 x[:, i] 和时间 t 计算效用值
                    utility_value = scenario_params['tasks'][j]['function'](x[:, i], t, i, vars_dict=current_vars_dict)
                    print(f"    - {task_names[j]} (Index {j}): Utility = {utility_value:.2f}")
                except Exception as e:
                    print(f"    - {task_names[j]} (Index {j}): Error calculating utility: {e}")
        # --- End Debug ---

        # 求解任务分配MIQP
        alpha_new, u_new, delta_new, time_to_synthesize_controller, opt_sol_info = rta.solve_miqp(x, t)

        # 重塑解决方案并更新全局变量
        alpha = alpha_new.reshape(n_t, n_r, order="F")
        u = u_new.reshape(n_u, n_r, order="F") # 注意：这里的u是MIQP的解，可能与实际控制输入不同
        delta = delta_new.reshape(n_t, n_r, order="F")
        task_assignment = get_task_assignment(alpha)
        print(f"Iter {iter}: 新任务分配: {task_assignment}")

        # 更新等待时间（如果分配了等待任务，则累积等待时间，否则重置）
        # 注意：这个逻辑移到每次迭代都检查可能更合适，取决于具体需求
        # wait_task_idx = 3 # 假设 3 是等待任务索引
        # assigned_to_wait = any(ta == wait_task_idx for ta in task_assignment)
        # if not assigned_to_wait:
        #     global_vars.reset_wait_timer_for_all() # 需要一个重置所有机器人等待时间的函数

    # --------------------------

    # --- 更新操作任务（每次迭代都检查） ---
    manipulation_task_idx = 2 # 假设 2 是操作任务索引
    manipulating_robot_indices = [i for i, task in enumerate(task_assignment) if task == manipulation_task_idx]

    for robot_idx in manipulating_robot_indices:
        # 检查是否可以进入下一阶段 (基于距离进入，固定时间退出)
        advanced = global_vars.check_and_advance_manipulation_phase(robot_idx)
    # -------------------------------------

    # --- 使用可视化模块打印所有任务的调试信息 ---
    # visualization_config.print_task_debug_info(x, t, task_assignment, global_vars, iter) # 暂时注释掉，减少输出

    # 保存每个机器人的当前状态，用于计算delta H
    x_prev = x.copy()

    # --- 应用运动控制（每次迭代都执行）---
    for i in range(n_r):
        # 获取当前任务分配和全局变量
        current_task = task_assignment[i] # 使用当前（可能来自上次规划）的任务分配
        vars_dict = global_vars.get_all_vars()

        # 根据任务分配调用相应的apply_motion_control方法
        try:
            if current_task == 0:  # 导航任务
                x[:, i] = NaviTask.apply_motion_control(x[:, i], t, i, vars_dict, DT)
                # if iter % 10 == 0: print(f"Iter {iter}: 机器人{i}执行导航任务移动")
            elif current_task == 1:  # 探索任务
                x[:, i] = ExploreTask.apply_motion_control(x[:, i], t, i, vars_dict, DT)
                # if iter % 10 == 0: print(f"Iter {iter}: 机器人{i}执行探索任务移动")
            elif current_task == 2:  # 操作任务
                x[:, i] = ManipulationTask.apply_motion_control(x[:, i], t, i, vars_dict, DT)
                # if iter % 10 == 0: print(f"Iter {iter}: 机器人{i}执行操作任务移动，阶段: {global_vars.manipulation_phase.name}")
            elif current_task == 3: # 等待任务
                x[:, i] = WaitTask.apply_motion_control(x[:, i], t, i, vars_dict, DT)
                # if iter % 10 == 0: print(f"Iter {iter}: 机器人{i}执行等待任务")
            else:
                # 对于未知的任务类型或默认情况，可能应用基础动力学或保持不动
                # dx = sys_dyn(x[:, i], u[:, i]) * DT # 注意：这里的 u 来自 MIQP，可能不是最优控制输入
                # x[:, i] += dx # 使用 MIQP 的 u 可能导致非预期行为，注释掉
                print(f"Iter {iter}: 机器人{i}任务 {current_task} 无特定运动控制，保持状态")
                pass # 保持当前状态 x[:, i] 不变

        except Exception as e:
            print(f"Iter {iter}: 机器人{i} 任务 {current_task} 运动控制出错: {e}")
            # 出错时可以选择保持状态或应用基础动力学
            # dx = sys_dyn(x[:, i], u[:, i]) * DT # 同样，使用 MIQP 的 u 可能有问题
            # x[:, i] += dx
            pass # 出错时保持状态

    # --- 更新专业化矩阵（每次迭代都执行，使用来自上次规划的 alpha） ---
    S = rta.get_specializations()
    for i in range(n_r): # 遍历机器人
        for j in range(n_t): # 遍历任务
            vars_dict = global_vars.get_all_vars() # 获取包含最新 phase/progress 的全局变量
            # 使用保存的先前状态计算delta H
            # 注意：这里传入的 x[:, i] 是 *已经移动过* 的状态
            try:
                before_val = scenario_params['tasks'][j]['function'](x_prev[:, i], t, i, vars_dict=vars_dict)
                after_val = scenario_params['tasks'][j]['function'](x[:, i], t, i, vars_dict=vars_dict) # 使用移动后的状态
                Dh_ij = after_val - before_val
                # 使用当前的 alpha (来自最近一次规划) 来更新 S
                S[j, i] = max(0, S[j, i] + 10 * alpha[j, i] * Dh_ij)
            except Exception as e:
                 print(f"Iter {iter}: 更新机器人 {i} 任务 {j} 专业化时出错: {e}")

    rta.set_specializations(S)
    # -----------------------------------------------------------------------

    # 保存轨迹
    simulation_config.update_trajectory(iter, x)

    # 更新可视化
    plt.clf()
    ax = plt.gca()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # 绘制环境边界
    border = Rectangle((-1.8, -1.2), 3.6, 2.4, edgecolor=[0.5, 0.5, 0.5], linewidth=5, facecolor='none')
    ax.add_patch(border)
    
    # 绘制机器人
    for i in range(n_r):
        robot_type = 'wheeled'  # 默认为轮式机器人
        
        # 获取机器人的操作阶段和持有状态
        phase = None
        is_holding = False
        
        # 检查机器人是否在执行操作任务
        if task_assignment[i] == 2:  # 假设2是操作任务的索引
            # 获取当前操作阶段
            if hasattr(global_vars, 'manipulation_phase'):
                phase = global_vars.manipulation_phase.name
                
                # 检查是否是持有者
                if hasattr(global_vars, 'is_holding') and hasattr(global_vars, 'holding_robot_id'):
                    is_holding = global_vars.is_holding and global_vars.holding_robot_id == i
        
        # 使用新的可视化方法
        visualization_config.plot_robot_with_phase(ax, x, i, robot_type, phase, is_holding)
    
    # 绘制静态导航目标点 p_goal
    if p_goal is not None:
        ax.plot(p_goal[0], p_goal[1], 'k*', markersize=12, label='Nav Goal')
    
    # 绘制探索目标
    exploration_targets = global_vars.exploration_targets
    if exploration_targets:
        # 只绘制一次图例
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
    
    # 绘制操作目标
    target_obj_pos = global_vars.target_object_position
    target_rec_pos = global_vars.target_receptacle_position
    plotted_obj = False
    plotted_rec = False
    
    # 更新可视化模块中的目标位置信息
    visualization_config.set_target_positions(target_obj_pos, target_rec_pos)
    
    # 检查是否有机器人持有物体
    is_any_robot_holding = hasattr(global_vars, 'is_holding') and global_vars.is_holding
    
    if target_obj_pos is not None:
        # 如果有机器人持有物体，则目标物体应该显示为"已获取"状态
        if is_any_robot_holding:
            label = 'Object Target (已获取)' if not plotted_obj else None
            ax.plot(target_obj_pos[0], target_obj_pos[1], 'rs', markersize=8, label=label)
        else:
            label = 'Object Target' if not plotted_obj else None
            ax.plot(target_obj_pos[0], target_obj_pos[1], 'bs', markersize=8, label=label)
        plotted_obj = True
        
    if target_rec_pos is not None:
        label = 'Receptacle Target' if not plotted_rec else None
        ax.plot(target_rec_pos[0], target_rec_pos[1], 'ms', markersize=8, label=label)
        plotted_rec = True
    
    # 更新任务文本和当前阶段文本
    visualization_config.update_task_text(ax, t, task_assignment)
    manipulation_phase = global_vars.manipulation_phase
    if manipulation_phase:
        phase_text = f"操作阶段: {manipulation_phase.name}"
        ax.text(-1.8, 1.35, phase_text, fontsize=10)
        
        # 添加持有物体状态信息
        if hasattr(global_vars, 'is_holding') and hasattr(global_vars, 'holding_robot_id'):
            if global_vars.is_holding:
                holding_text = f"持有物体: 机器人 {global_vars.holding_robot_id}"
                ax.text(-1.8, 1.25, holding_text, fontsize=10, color='red')
            else:
                holding_text = "持有物体: 无"
                ax.text(-1.8, 1.25, holding_text, fontsize=10)
    
    # 显示等待时间
    wait_time = global_vars.wait_elapsed_time
    wait_threshold = global_vars.wait_step_threshold
    wait_text = f"等待时间: {wait_time:.1f}s / {wait_threshold:.1f}s"
    wait_progress = f"进度: {min(1.0, wait_time/wait_threshold):.0%}"
    ax.text(1.0, 1.35, wait_text, fontsize=10)
    ax.text(1.0, 1.25, wait_progress, fontsize=10)
    
    # 计算已探索目标百分比
    if exploration_targets:
        explored_count = sum(1 for target in exploration_targets if target.get('explored', False))
        total_count = len(exploration_targets)
        explore_text = f"探索进度: {explored_count}/{total_count}"
        explore_progress = f"完成: {explored_count/total_count:.0%}"
        ax.text(0.0, 1.35, explore_text, fontsize=10)
        ax.text(0.0, 1.25, explore_progress, fontsize=10)
    
    # 使用可视化模块绘制各任务信息
    visualization_config.visualize_manipulation_phases(ax, x, task_assignment, global_vars)
    visualization_config.visualize_navi_task(ax, x, task_assignment, p_goal)
    visualization_config.visualize_explore_task(ax, x, task_assignment, global_vars)
    
    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # 去重
    ax.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
    
    plt.draw()
    # --- 增加可视化暂停时间 ---
    plt.pause(0.1) # 从 0.01 增加到 0.1 秒
    # ---------------------------

    # 打印当前状态信息 (减少打印频率)
    if iter % 10 == 0:
         print(f"T={t:.1f}s | Assign: {task_assignment} | Manip Phase: {global_vars.manipulation_phase.name} | Wait: {global_vars.wait_elapsed_time:.1f}s")

# 显示最终图形
plt.show()