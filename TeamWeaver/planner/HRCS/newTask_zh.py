# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.path import Path
import matplotlib.path as mpath

# Ensure HRCS and habitat_llm namespace are on sys.path when run directly
_hrcs_dir = os.path.dirname(os.path.abspath(__file__))
_teamweaver_root = os.path.abspath(os.path.join(_hrcs_dir, '..', '..', '..'))
if _hrcs_dir not in sys.path:
    sys.path.insert(0, _hrcs_dir)
if _teamweaver_root not in sys.path:
    sys.path.insert(0, _teamweaver_root)
_habitat_llm_link = os.path.join(_teamweaver_root, 'habitat_llm')
if not os.path.exists(_habitat_llm_link):
    try:
        os.symlink(_teamweaver_root, _habitat_llm_link, target_is_directory=True)
    except OSError:
        pass

from class_def.SingleIntegrator import SingleIntegrator
from class_def.RTA_task import RTA
from class_def.Swarm_task import Swarm
from class_def.GlobalVarsManager_task import GlobalVarsManager_task
from sys_module.tools_util import clamp, get_task_assignment, print_info, plot_quad, plot_fov, read_mud_file
from sys_module.debug_logger import setup_debug_logging, teardown_debug_logging

from params_module.scenario_params_task import ScenarioConfigTask
from params_module.opt_params_task import OptimizationConfigTask
from sys_module.control_module import ControlConfig
from sys_module.simulation_module import SimulationConfig
from sys_module.visualization_module import VisualizationConfig
from sys_module.environment_module import EnvironmentConfig
from sys_module.robot_features_module import RobotFeaturesConfig
from sys_module.robot_dynamics_module import RobotDynamicsConfig
from sys_module.task_config_module import TaskConfig
from sys_module.disturbance_module import DisturbanceConfig

from task_module.explore_module import ExploreTask
from task_module.manipulation_module import ManipulationTask, ManipulationPhase
from task_module.wait_module import WaitTask
from task_module.navi_module import NaviTask

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="HRCS 2D multi-robot MIQP simulation demo (newTask.py)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: detailed MIQP analysis + tee all console output to a log file",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Debug log file path (default: planner/HRCS/logs/newTask_debug_<timestamp>.log)",
    )
    return parser.parse_args()


def main(debug_mode=False, log_file=None):
    log_path = setup_debug_logging(debug_mode, log_file)

    try:
        if debug_mode:
            print(f"[DEBUG] Debug mode ON — verbose MIQP analysis enabled")
        else:
            print("Running in normal mode (use --debug for detailed MIQP logs)")

        global_vars = GlobalVarsManager_task()
        scenario_config = ScenarioConfigTask(n_u=2)
        opt_config = OptimizationConfigTask()
        control_config = ControlConfig()

        new_u_max = np.array([0.5, 2.5])
        control_config.update_control_params('u_max', new_u_max)
        print(f"The robot maximum speed limit has been updated to: {new_u_max}")

        scenario_params = scenario_config.get_scenario_params()
        updated_scenario_params = scenario_config.get_updated_scenario_params()
        global_task_vars = scenario_config.get_global_task_vars()
        global_vars.register_vars_from_dict(global_task_vars)

        # Use scenario_params dimensions (must match RTA / MIQP)
        n_r = scenario_params['A'].shape[1]
        n_t = scenario_params['T'].shape[0]
        n_c = scenario_params['T'].shape[1]
        n_f = scenario_params['A'].shape[0]

        simulation_config = SimulationConfig(n_r=n_r)
        visualization_config = VisualizationConfig()
        environment_config = EnvironmentConfig()
        robot_dynamics_config = RobotDynamicsConfig(n_u=2)
        disturbance_config = DisturbanceConfig(global_vars)

        p_goal = global_vars.get_var('p_goal')
        opt_params = opt_config.get_opt_params()
        control_params = control_config.get_control_params()

        simulation_params = simulation_config.get_sim_params()
        DT = simulation_params['DT']
        max_iter = simulation_params['max_iter']
        env_params = environment_config.get_env_params()
        environment = env_params['environment']

        robot_dynamics = robot_dynamics_config.get_robot_dynamics()
        n_x = robot_dynamics['n_x']
        n_u = robot_dynamics['n_u']
        sys_dyn = robot_dynamics['sys_dyn']

        robots = [SingleIntegrator() for _ in range(n_r)]
        if n_u != 2:
            print(f"warn: RobotDynamicsConfig returned n_u={n_u}, expected 2.")

        def simple_density(x, y):
            return np.exp(-0.5 * ((x / 1.8) ** 2 + (y / 1.2) ** 2))

        s = Swarm(robots=robots, environment=environment, densityFunction=simple_density)
        global_vars.s = s
        rta = RTA(scenario_params, opt_params, debug_mode=debug_mode)

        x = simulation_config.get_initial_states()
        global_vars.x = x
        x_traj = simulation_config.get_trajectory()

        fig = visualization_config.setup_plot()
        ax = plt.gca()

        replan_interval = 300

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
            global_vars.update_task_timer(DT)

            poses = np.vstack((x[0:2, :], np.zeros((1, n_r))))
            s.setPoses(poses)

            G, _, VC = s.coverageControl()
            global_vars.G = G

            robot_positions = x[0:2, :].T
            global_vars.update_exploration_status(robot_positions)

            if iter % replan_interval == 0:
                print(f"\n--- Iter {iter}: Replan tasks ---")

                if debug_mode:
                    print(f"Iter {iter}: Calculate task utility value...")
                    current_vars_dict = global_vars.get_all_vars()
                    task_names = [
                        task.get('name', f'Task_{idx}')
                        for idx, task in enumerate(scenario_params['tasks'])
                    ]
                    for i in range(n_r):
                        print(f"  Robot {i} (Pos: [{x[0, i]:.2f}, {x[1, i]:.2f}], Theta: {x[2, i]:.2f}):")
                        is_current_holding = (
                            current_vars_dict.get('is_holding', False)
                            and current_vars_dict.get('holding_robot_id') == i
                        )
                        print(f"    Is Holding: {is_current_holding}")
                        for j in range(n_t):
                            try:
                                utility_value = scenario_params['tasks'][j]['function'](
                                    x[:, i], t, i, vars_dict=current_vars_dict
                                )
                                print(f"    - {task_names[j]} (Index {j}): Utility = {utility_value:.2f}")
                            except Exception as e:
                                print(f"    - {task_names[j]} (Index {j}): Error calculating utility: {e}")

                alpha_new, u_new, delta_new, time_to_synthesize_controller, opt_sol_info = rta.solve_miqp(x, t)

                alpha = alpha_new.reshape(n_t, n_r, order="F")
                u = u_new.reshape(n_u, n_r, order="F")
                delta = delta_new.reshape(n_t, n_r, order="F")
                task_assignment = get_task_assignment(alpha)
                print(f"Iter {iter}: New task assignment: {task_assignment}")

            manipulation_task_idx = 2
            manipulating_robot_indices = [
                i for i, task in enumerate(task_assignment) if task == manipulation_task_idx
            ]
            for robot_idx in manipulating_robot_indices:
                global_vars.check_and_advance_manipulation_phase(robot_idx)

            x_prev = x.copy()

            for i in range(n_r):
                current_task = task_assignment[i]
                vars_dict = global_vars.get_all_vars()
                try:
                    if current_task == 0:
                        x[:, i] = NaviTask.apply_motion_control(x[:, i], t, i, vars_dict, DT)
                    elif current_task == 1:
                        x[:, i] = ExploreTask.apply_motion_control(x[:, i], t, i, vars_dict, DT)
                    elif current_task == 2:
                        x[:, i] = ManipulationTask.apply_motion_control(x[:, i], t, i, vars_dict, DT)
                    elif current_task == 3:
                        x[:, i] = WaitTask.apply_motion_control(x[:, i], t, i, vars_dict, DT)
                    else:
                        if debug_mode:
                            print(f"Iter {iter}: robot {i} task {current_task} — no motion control, holding state")
                except Exception as e:
                    print(f"Iter {iter}: robot {i} task {current_task} motion control error: {e}")

            S = rta.get_specializations()
            for i in range(n_r):
                for j in range(n_t):
                    vars_dict = global_vars.get_all_vars()
                    try:
                        before_val = scenario_params['tasks'][j]['function'](x_prev[:, i], t, i, vars_dict=vars_dict)
                        after_val = scenario_params['tasks'][j]['function'](x[:, i], t, i, vars_dict=vars_dict)
                        Dh_ij = after_val - before_val
                        S[j, i] = max(0, S[j, i] + 10 * alpha[j, i] * Dh_ij)
                    except Exception as e:
                        if debug_mode:
                            print(f"Iter {iter}: update bot {i} task {j} specialization error: {e}")

            rta.set_specializations(S)
            simulation_config.update_trajectory(iter, x)

            plt.clf()
            ax = plt.gca()
            ax.set_xlim(-2, 2)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            ax.grid(True)

            border = Rectangle((-1.8, -1.2), 3.6, 2.4, edgecolor=[0.5, 0.5, 0.5], linewidth=5, facecolor='none')
            ax.add_patch(border)

            for i in range(n_r):
                robot_type = 'wheeled'
                phase = None
                is_holding = False
                if task_assignment[i] == 2:
                    if hasattr(global_vars, 'manipulation_phase'):
                        phase = global_vars.manipulation_phase.name
                        if hasattr(global_vars, 'is_holding') and hasattr(global_vars, 'holding_robot_id'):
                            is_holding = global_vars.is_holding and global_vars.holding_robot_id == i
                visualization_config.plot_robot_with_phase(ax, x, i, robot_type, phase, is_holding)

            if p_goal is not None:
                ax.plot(p_goal[0], p_goal[1], 'k*', markersize=12, label='Nav Goal')

            exploration_targets = global_vars.exploration_targets
            if exploration_targets:
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

            target_obj_pos = global_vars.target_object_position
            target_rec_pos = global_vars.target_receptacle_position
            plotted_obj = False
            plotted_rec = False
            visualization_config.set_target_positions(target_obj_pos, target_rec_pos)
            is_any_robot_holding = hasattr(global_vars, 'is_holding') and global_vars.is_holding

            if target_obj_pos is not None:
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

            visualization_config.update_task_text(ax, t, task_assignment)
            manipulation_phase = global_vars.manipulation_phase
            if manipulation_phase:
                phase_text = f"manipulation stage: {manipulation_phase.name}"
                ax.text(-1.8, 1.35, phase_text, fontsize=10)
                if hasattr(global_vars, 'is_holding') and hasattr(global_vars, 'holding_robot_id'):
                    if global_vars.is_holding:
                        ax.text(-1.8, 1.25, f"holding object: robot {global_vars.holding_robot_id}", fontsize=10, color='red')
                    else:
                        ax.text(-1.8, 1.25, "holding object: none", fontsize=10)

            wait_time = global_vars.wait_elapsed_time
            wait_threshold = global_vars.wait_step_threshold
            ax.text(1.0, 1.35, f"waiting time: {wait_time:.1f}s / {wait_threshold:.1f}s", fontsize=10)
            ax.text(1.0, 1.25, f"schedule: {min(1.0, wait_time / wait_threshold):.0%}", fontsize=10)

            if exploration_targets:
                explored_count = sum(1 for target in exploration_targets if target.get('explored', False))
                total_count = len(exploration_targets)
                ax.text(0.0, 1.35, f"Exploration progress: {explored_count}/{total_count}", fontsize=10)
                ax.text(0.0, 1.25, f"Finish: {explored_count / total_count:.0%}", fontsize=10)

            visualization_config.visualize_manipulation_phases(ax, x, task_assignment, global_vars)
            visualization_config.visualize_navi_task(ax, x, task_assignment, p_goal)
            visualization_config.visualize_explore_task(ax, x, task_assignment, global_vars)

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)

            plt.draw()
            plt.pause(0.1)

            if iter % 10 == 0:
                print(f"T={t:.1f}s | Assign: {task_assignment} | Manip Phase: {global_vars.manipulation_phase.name} | Wait: {global_vars.wait_elapsed_time:.1f}s")

        plt.show()

        if log_path:
            print(f"[DEBUG] Log saved to: {log_path}")

    finally:
        teardown_debug_logging()


if __name__ == "__main__":
    args = parse_args()
    main(debug_mode=args.debug, log_file=args.log_file)
