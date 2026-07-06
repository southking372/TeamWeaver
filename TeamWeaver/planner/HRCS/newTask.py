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
from matplotlib.patches import Rectangle

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
from sys_module.tools_util import get_task_assignment
from sys_module.debug_logger import setup_debug_logging, teardown_debug_logging

from params_module.scenario_params_task import ScenarioConfigTask
from params_module.opt_params_task import OptimizationConfigTask
from sys_module.control_module import ControlConfig
from sys_module.simulation_module import SimulationConfig
from sys_module.visualization_module_eng import VisualizationConfigEng
from sys_module.environment_module import EnvironmentConfig
from sys_module.robot_dynamics_module import RobotDynamicsConfig
from sys_module.disturbance_module import DisturbanceConfig

from task_module.explore_module import ExploreTask
from task_module.manipulation_module import ManipulationTask
from task_module.wait_module import WaitTask
from task_module.navi_module import NaviTask

plt.rcParams['axes.unicode_minus'] = False

# he-hrcs/newTask.py: replan every 300 iterations @ DT=0.1 → specialization S accumulates between replans
REPLAN_PERIOD_S = 30.0
WAIT_RESET_PERIOD_S = 5.0
SPECIALIZATION_GAIN = 10.0

# 1-based task indices for the 4-task demo (Navigate/Explore/Manipulation/Wait)
TASK_NAVI = 1
TASK_EXPLORE = 2
TASK_MANIPULATION = 3
TASK_WAIT = 4


def parse_args():
    parser = argparse.ArgumentParser(description="HRCS 2D multi-robot MIQP simulation demo")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed MIQP analysis and tee console output to a log file",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Debug log path (default: planner/HRCS/logs/newTask_debug_<timestamp>.log)",
    )
    return parser.parse_args()


def apply_motion_for_task(current_task, x, t, robot_idx, vars_dict, dt):
    if current_task == TASK_NAVI:
        return NaviTask.apply_motion_control(x[:, robot_idx], t, robot_idx, vars_dict, dt)
    if current_task == TASK_EXPLORE:
        return ExploreTask.apply_motion_control(x[:, robot_idx], t, robot_idx, vars_dict, dt)
    if current_task == TASK_MANIPULATION:
        return ManipulationTask.apply_motion_control(x[:, robot_idx], t, robot_idx, vars_dict, dt)
    if current_task == TASK_WAIT:
        return WaitTask.apply_motion_control(x[:, robot_idx], t, robot_idx, vars_dict, dt)
    return x[:, robot_idx]


def enforce_single_manipulation_assignment(task_assignment, alpha, n_t, n_r, global_vars):
    """At most one robot may be assigned Manipulation (Pick is exclusive)."""
    assignment = np.array(task_assignment, copy=True)
    alpha_out = alpha.copy()
    manip_indices = [i for i in range(n_r) if assignment[i] == TASK_MANIPULATION]
    if len(manip_indices) <= 1:
        return assignment, alpha_out

    executor = global_vars.select_manipulation_executor(manip_indices)
    for i in manip_indices:
        if i != executor:
            assignment[i] = TASK_WAIT
            alpha_out[:, i] = 0.0
            alpha_out[TASK_WAIT - 1, i] = 1.0
    return assignment, alpha_out


def apply_assignment_persistence(task_assignment, alpha, n_t, n_r, global_vars):
    """
    Keep in-progress manipulation assignments across replans (he-hrcs behaviour:
    assignment persists between MIQP cycles; lock robots mid-manipulation).
    """
    from task_module.manipulation_module import ManipulationPhase

    assignment = np.array(task_assignment, copy=True)
    alpha_out = alpha.copy()
    phase = getattr(global_vars, 'manipulation_phase', None)
    in_manipulation = phase in (
        ManipulationPhase.NAV_OBJ,
        ManipulationPhase.PICK,
        ManipulationPhase.NAV_REC,
        ManipulationPhase.PLACE,
    )
    holding = getattr(global_vars, 'is_holding', False)
    holding_robot = getattr(global_vars, 'holding_robot_id', None)
    manipulation_done = getattr(global_vars, 'manipulation_completed', False)

    for i in range(n_r):
        keep_manipulation = False
        if not manipulation_done and assignment[i] == TASK_MANIPULATION and in_manipulation:
            keep_manipulation = True
        if not manipulation_done and holding and holding_robot == i:
            keep_manipulation = True
        if keep_manipulation:
            assignment[i] = TASK_MANIPULATION
            alpha_out[:, i] = 0.0
            alpha_out[TASK_MANIPULATION - 1, i] = 1.0

    return enforce_single_manipulation_assignment(assignment, alpha_out, n_t, n_r, global_vars)


def main(debug_mode=False, log_file=None):
    log_path = setup_debug_logging(debug_mode, log_file)

    try:
        if debug_mode:
            print("[DEBUG] Debug mode ON")
        else:
            print("Running in normal mode (use --debug for detailed MIQP logs)")

        global_vars = GlobalVarsManager_task()
        scenario_config = ScenarioConfigTask(n_r=5, n_t=4, n_c=8, n_f=5, n_u=2)
        opt_config = OptimizationConfigTask(n_r=5, n_t=4)
        opt_config.update_robot_bounds(2, 0, 1)  # Manipulation: at most one robot (Pick exclusive)
        control_config = ControlConfig()

        new_u_max = np.array([0.5, 2.5])
        control_config.update_control_params('u_max', new_u_max)
        print(f"Robot max velocity limit updated to: {new_u_max}")

        scenario_params = scenario_config.get_scenario_params()
        global_vars.register_vars_from_dict(scenario_config.get_global_task_vars())

        # 4-task demo: task 3=Manipulation, task 4=Wait (override Pick/Place from PARTNR template)
        scenario_params['tasks'][2] = {
            'function': ManipulationTask.manipulation_function,
            'gradient': ManipulationTask.manipulation_gradient,
            'time_derivative': ManipulationTask.manipulation_time_derivative,
            'name': 'Manipulation',
        }
        scenario_params['tasks'][3] = {
            'function': WaitTask.wait_function,
            'gradient': WaitTask.wait_gradient,
            'time_derivative': WaitTask.wait_time_derivative,
            'name': 'Wait',
        }
        scenario_params['T'][3, :] = 0  # Wait requires no capability

        # 5 robots × 5 features, all full capability for the demo
        scenario_params['A'] = np.ones((5, 5))

        n_r = scenario_params['A'].shape[1]
        n_t = scenario_params['T'].shape[0]
        n_u = scenario_params['robot_dyn']['n_u']

        simulation_config = SimulationConfig(n_r=n_r)
        visualization_config = VisualizationConfigEng()
        environment_config = EnvironmentConfig()
        robot_dynamics_config = RobotDynamicsConfig(n_u=n_u)
        disturbance_config = DisturbanceConfig(global_vars)

        p_goal = global_vars.get_var('p_goal')
        opt_params = opt_config.get_opt_params()

        simulation_params = simulation_config.get_sim_params()
        DT = simulation_params['DT']
        max_iter = simulation_params['max_iter']
        environment = environment_config.get_env_params()['environment']
        replan_interval = max(1, int(REPLAN_PERIOD_S / DT))
        wait_reset_interval = max(1, int(WAIT_RESET_PERIOD_S / DT))

        robots = [SingleIntegrator() for _ in range(n_r)]

        def simple_density(x_pos, y_pos):
            return np.exp(-0.5 * ((x_pos / 1.8) ** 2 + (y_pos / 1.2) ** 2))

        s = Swarm(robots=robots, environment=environment, densityFunction=simple_density)
        global_vars.s = s
        rta = RTA(scenario_params, opt_params, debug_mode=debug_mode)

        x = simulation_config.get_initial_states()
        global_vars.x = x

        fig = visualization_config.setup_plot()
        ax = plt.gca()

        print("Performing initial task assignment...")
        alpha, u, delta, _, opt_sol_info = rta.solve_miqp(x, 0.0)
        alpha = alpha.reshape(n_t, n_r, order="F")
        u = u.reshape(n_u, n_r, order="F")
        delta = delta.reshape(n_t, n_r, order="F")
        task_assignment = get_task_assignment(alpha)
        task_assignment, alpha = enforce_single_manipulation_assignment(
            task_assignment, alpha, n_t, n_r, global_vars
        )
        print(f"Initial task assignment complete: {task_assignment}")
        print(f"Replan every {REPLAN_PERIOD_S:.0f}s ({replan_interval} steps); wait timer resets every {WAIT_RESET_PERIOD_S:.0f}s")

        for iter in range(max_iter):
            t = iter * DT
            global_vars.set_var('t', t)
            global_vars.update_task_timer(DT)

            poses = np.vstack((x[0:2, :], np.zeros((1, n_r))))
            s.setPoses(poses)

            G, _, _ = s.coverageControl()
            global_vars.G = G
            global_vars.update_exploration_status(x[0:2, :].T)

            # Wait timer: reset every 5s (independent of MIQP replan)
            if iter > 0 and iter % wait_reset_interval == 0:
                global_vars.reset_wait_timer()

            # --- Task assignment by frequency (he-hrcs: every 300 iterations) ---
            if iter % replan_interval == 0:
                print(f"\n--- Iter {iter}: Replanning tasks ---")
                if debug_mode:
                    current_vars_dict = global_vars.get_all_vars()
                    task_names = [task.get('name', f'Task_{idx}') for idx, task in enumerate(scenario_params['tasks'])]
                    for i in range(n_r):
                        print(f"  Robot {i} (Pos: [{x[0, i]:.2f}, {x[1, i]:.2f}], Theta: {x[2, i]:.2f}):")
                        for j in range(n_t):
                            try:
                                utility_value = scenario_params['tasks'][j]['function'](
                                    x[:, i], t, i, vars_dict=current_vars_dict
                                )
                                print(f"    - {task_names[j]} (Index {j}): Utility = {utility_value:.2f}")
                            except Exception as e:
                                print(f"    - {task_names[j]} (Index {j}): Error calculating utility: {e}")

                alpha_new, u_new, delta_new, _, _ = rta.solve_miqp(x, t)
                alpha = alpha_new.reshape(n_t, n_r, order="F")
                u = u_new.reshape(n_u, n_r, order="F")
                delta = delta_new.reshape(n_t, n_r, order="F")
                task_assignment = get_task_assignment(alpha)
                task_assignment, alpha = apply_assignment_persistence(
                    task_assignment, alpha, n_t, n_r, global_vars
                )
                print(f"Iter {iter}: New task assignment: {task_assignment}")

            manipulating_robot_indices = [
                i for i, task in enumerate(task_assignment) if task == TASK_MANIPULATION
            ]
            task_assignment, alpha = enforce_single_manipulation_assignment(
                task_assignment, alpha, n_t, n_r, global_vars
            )
            manipulating_robot_indices = [
                i for i, task in enumerate(task_assignment) if task == TASK_MANIPULATION
            ]
            executor = global_vars.select_manipulation_executor(manipulating_robot_indices)
            if executor is not None and not global_vars.manipulation_completed:
                global_vars.check_and_advance_manipulation_phase(executor)

            x_prev = x.copy()
            for i in range(n_r):
                current_task = task_assignment[i]
                vars_dict = global_vars.get_all_vars()
                try:
                    x[:, i] = apply_motion_for_task(current_task, x, t, i, vars_dict, DT)
                except Exception as e:
                    print(f"Iter {iter}: Robot {i} task {current_task} motion control error: {e}")

            # --- Update specialization matrix every iteration (he-hrcs persistence via S → P) ---
            S = rta.get_specializations()
            for i in range(n_r):
                for j in range(n_t):
                    vars_dict = global_vars.get_all_vars()
                    try:
                        before_val = scenario_params['tasks'][j]['function'](x_prev[:, i], t, i, vars_dict=vars_dict)
                        after_val = scenario_params['tasks'][j]['function'](x[:, i], t, i, vars_dict=vars_dict)
                        Dh_ij = after_val - before_val
                        S[j, i] = np.maximum(0, S[j, i] + SPECIALIZATION_GAIN * alpha[j, i] * Dh_ij)
                    except Exception as e:
                        if debug_mode:
                            print(f"Iter {iter}: specialization update error robot {i} task {j}: {e}")
            rta.set_specializations(S)
            simulation_config.update_trajectory(iter, x)

            global_vars.update_navi_completion(x, task_assignment, TASK_NAVI)
            if global_vars.all_demo_tasks_completed():
                print(
                    f"\nAll demo tasks completed at T={t:.1f}s: "
                    f"Navigate={global_vars.navi_completed}, "
                    f"Explore={global_vars.explore_completed}, "
                    f"Manipulation={global_vars.manipulation_completed} "
                    f"(Pick={global_vars.pick_succeeded}, Place={global_vars.place_completed})"
                )
                plt.clf()
                ax = plt.gca()
                ax.set_xlim(-2, 2)
                ax.set_ylim(-1.5, 1.5)
                ax.set_aspect('equal')
                ax.grid(True)
                ax.add_patch(Rectangle((-1.8, -1.2), 3.6, 2.4, edgecolor=[0.5, 0.5, 0.5], linewidth=5, facecolor='none'))
                for i in range(n_r):
                    phase = None
                    is_holding = False
                    if task_assignment[i] == TASK_MANIPULATION and hasattr(global_vars, 'manipulation_phase'):
                        phase = global_vars.manipulation_phase.name
                        if hasattr(global_vars, 'is_holding') and hasattr(global_vars, 'holding_robot_id'):
                            is_holding = global_vars.is_holding and global_vars.holding_robot_id == i
                    visualization_config.plot_robot_with_phase(ax, x, i, 'wheeled', phase, is_holding)
                visualization_config.display_info_panel(ax, t, global_vars, task_assignment, global_vars.exploration_targets)
                ax.text(0, -1.35, "All tasks completed — simulation stopped.", fontsize=11, ha='center',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.9))
                plt.draw()
                plt.pause(0.5)
                break

            plt.clf()
            ax = plt.gca()
            ax.set_xlim(-2, 2)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            ax.grid(True)
            ax.add_patch(Rectangle((-1.8, -1.2), 3.6, 2.4, edgecolor=[0.5, 0.5, 0.5], linewidth=5, facecolor='none'))

            for i in range(n_r):
                phase = None
                is_holding = False
                if task_assignment[i] == TASK_MANIPULATION and hasattr(global_vars, 'manipulation_phase'):
                    phase = global_vars.manipulation_phase.name
                    if hasattr(global_vars, 'is_holding') and hasattr(global_vars, 'holding_robot_id'):
                        is_holding = global_vars.is_holding and global_vars.holding_robot_id == i
                visualization_config.plot_robot_with_phase(ax, x, i, 'wheeled', phase, is_holding)

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
            visualization_config.set_target_positions(target_obj_pos, target_rec_pos)
            is_any_robot_holding = hasattr(global_vars, 'is_holding') and global_vars.is_holding

            if target_obj_pos is not None:
                if is_any_robot_holding:
                    ax.plot(target_obj_pos[0], target_obj_pos[1], 'rs', markersize=8, label='Object Target (Acquired)')
                else:
                    ax.plot(target_obj_pos[0], target_obj_pos[1], 'bs', markersize=8, label='Object Target')

            if target_rec_pos is not None:
                ax.plot(target_rec_pos[0], target_rec_pos[1], 'ms', markersize=8, label='Receptacle Target')

            visualization_config.display_info_panel(ax, t, global_vars, task_assignment, exploration_targets)
            visualization_config.visualize_manipulation_phases(ax, x, task_assignment, global_vars)
            visualization_config.visualize_navi_task(ax, x, task_assignment, p_goal)
            visualization_config.visualize_explore_task(ax, x, t, task_assignment, global_vars)

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=9)
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.draw()
            plt.pause(0.1)

            if iter % 10 == 0:
                wait_display = global_vars.get_effective_wait_time()
                print(
                    f"T={t:.1f}s | Assign: {task_assignment} | "
                    f"Manip Phase: {global_vars.manipulation_phase.name} | "
                    f"Wait: {wait_display:.1f}s"
                )

        plt.show()
        if log_path:
            print(f"[DEBUG] Log saved to: {log_path}")

    finally:
        teardown_debug_logging()


if __name__ == "__main__":
    args = parse_args()
    main(debug_mode=args.debug, log_file=args.log_file)
