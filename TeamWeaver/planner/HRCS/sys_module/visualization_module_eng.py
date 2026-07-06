# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

# sys_module/visualization_module_eng.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from task_module.navi_module import NaviTask
from task_module.explore_module import ExploreTask
from task_module.manipulation_module import ManipulationTask, ManipulationPhase


class VisualizationConfigEng:
    """English visualization config for newTask.py demo."""

    def __init__(self):
        self.viz_params = self._initialize_viz_params()
        self.target_object_position = None
        self.target_receptacle_position = None

    def _initialize_viz_params(self):
        return {
            'figure_size': (19.2, 10.8),
            'axis_limits': [-1.8, 1.8, -1.2, 1.2],
            'title': 'Robot Task Assignment and Control Simulation',
            'xlabel': 'X Position',
            'ylabel': 'Y Position',
            'robot_colors': {
                'wheeled': [0.2, 0.4, 0.6],
                'quadrotor': [0.6, 0.2, 0.4],
            },
            'arrow_length': {
                'wheeled': 0.05,
                'quadrotor': 0.2,
            },
            'manipulation_phase_colors': {
                'NAV_OBJ': [0.8, 0.4, 0],
                'PICK': [0, 0.8, 0.4],
                'NAV_REC': [0.4, 0, 0.8],
                'PLACE': [0.4, 0.8, 0],
            },
            'navi_color': [0.0, 0.6, 0.8],
            'explore_color': [0.9, 0.5, 0.0],
        }

    def setup_plot(self):
        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=self.viz_params['figure_size'])
        plt.ion()
        plt.axis('equal')
        plt.axis(self.viz_params['axis_limits'])
        plt.title(self.viz_params['title'], fontsize=16)
        plt.xlabel(self.viz_params['xlabel'], fontsize=12)
        plt.ylabel(self.viz_params['ylabel'], fontsize=12)
        return fig

    def plot_robot(self, ax, x, i, robot_type='wheeled'):
        color = self.viz_params['robot_colors'][robot_type]
        length = self.viz_params['arrow_length'][robot_type]
        marker = '.' if robot_type == 'wheeled' else '^'
        ax.scatter(x[0, i], x[1, i], s=100, color=color, marker=marker)
        ax.arrow(
            x[0, i], x[1, i],
            length * np.cos(x[2, i]), length * np.sin(x[2, i]),
            head_width=length * 0.4, head_length=length * 0.4,
            fc=color, ec=color, alpha=0.8,
        )
        ax.text(x[0, i], x[1, i] - length, f"{robot_type} {i}", fontsize=10, ha='center', va='center')
        return ax

    def set_target_positions(self, target_object_position, target_receptacle_position):
        self.target_object_position = target_object_position
        self.target_receptacle_position = target_receptacle_position

    def plot_robot_with_phase(self, ax, x, i, robot_type='wheeled', phase=None, is_holding=False):
        self.plot_robot(ax, x, i, robot_type)
        if is_holding:
            ax.scatter(x[0, i], x[1, i], s=160, color='red', marker='o', facecolors='none', linewidth=2)
            if phase == 'NAV_REC' and self.target_receptacle_position is not None:
                ax.plot(
                    [x[0, i], self.target_receptacle_position[0]],
                    [x[1, i], self.target_receptacle_position[1]],
                    'r--', linewidth=1, alpha=0.6,
                )
        return ax

    @staticmethod
    def _effective_wait_time(global_vars):
        """Wait time within the current 5s replan cycle (mod threshold)."""
        elapsed = global_vars.wait_elapsed_time
        threshold = global_vars.wait_step_threshold
        if threshold <= 0:
            return elapsed
        return elapsed % threshold

    def display_info_panel(self, ax, t, global_vars, task_assignment, exploration_targets):
        manipulation_phase = global_vars.manipulation_phase
        phase_str = f"Phase: {manipulation_phase.name}" if manipulation_phase else "Phase: N/A"

        holding_str = "Holding: None"
        if hasattr(global_vars, 'is_holding') and global_vars.is_holding:
            holding_str = f"Holding: Robot {global_vars.holding_robot_id}"

        if exploration_targets:
            explored_count = sum(1 for target in exploration_targets if target.get('explored', False))
            total_count = len(exploration_targets)
            explore_str = f"Exploration: {explored_count}/{total_count} ({explored_count / total_count:.0%})"
        else:
            explore_str = "Exploration: N/A"

        wait_time = self._effective_wait_time(global_vars)
        wait_threshold = global_vars.wait_step_threshold
        wait_progress = min(1.0, wait_time / wait_threshold) if wait_threshold > 0 else 0.0
        wait_str = f"Wait Time: {wait_time:.1f}s / {wait_threshold:.1f}s ({wait_progress:.0%})"

        pick_str = "Pick: pending"
        if hasattr(global_vars, 'pick_succeeded') and global_vars.pick_succeeded:
            pick_str = "Pick: succeeded (holding)" if global_vars.is_holding else "Pick: succeeded"
        if hasattr(global_vars, 'place_completed') and global_vars.place_completed:
            pick_str = "Pick/Place: done"

        task_done_str = "Tasks: "
        if hasattr(global_vars, 'navi_completed'):
            task_done_str += (
                f"Nav={'OK' if global_vars.navi_completed else '--'} | "
                f"Explore={'OK' if global_vars.explore_completed else '--'} | "
                f"Manip={'OK' if global_vars.manipulation_completed else '--'}"
            )

        info_text = (
            f"Time: {t:.1f}s | Task Assignment: {task_assignment}\n"
            f"{phase_str} | {holding_str} | {pick_str}\n"
            f"{task_done_str}\n"
            f"{explore_str} | {wait_str}"
        )
        ax.text(
            -1.75, 1.45, info_text, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
        )

    def visualize_manipulation_phases(self, ax, x, task_assignment, global_vars):
        manipulation_task_idx = 3  # 1-based: Manipulation
        manipulating_robots = [i for i, task in enumerate(task_assignment) if task == manipulation_task_idx]
        if not manipulating_robots:
            return ax

        current_phase = global_vars.manipulation_phase
        if current_phase is None:
            return ax

        phase_color = self.viz_params['manipulation_phase_colors'].get(current_phase.name, [0.5, 0.5, 0.5])
        for robot_idx in manipulating_robots:
            robot_pos = x[0:2, robot_idx]
            ax.text(
                robot_pos[0], robot_pos[1] + 0.15, current_phase.name,
                color=phase_color, fontsize=9, ha='center',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'),
            )
            if current_phase.name == 'NAV_OBJ' and global_vars.target_object_position is not None:
                target_pos = global_vars.target_object_position
                ax.plot([robot_pos[0], target_pos[0]], [robot_pos[1], target_pos[1]], '--', color=phase_color, linewidth=1, alpha=0.6)
            elif current_phase.name == 'NAV_REC' and global_vars.target_receptacle_position is not None:
                target_pos = global_vars.target_receptacle_position
                ax.plot([robot_pos[0], target_pos[0]], [robot_pos[1], target_pos[1]], '--', color=phase_color, linewidth=1, alpha=0.6)
            if current_phase.name in ['PICK', 'PLACE']:
                ax.add_patch(plt.Circle((robot_pos[0], robot_pos[1]), 0.2, color=phase_color, fill=False, linestyle='--', alpha=0.7))
        return ax

    def visualize_navi_task(self, ax, x, task_assignment, p_goal):
        navi_task_idx = 1  # 1-based: Navigate
        navigating_robots = [i for i, task in enumerate(task_assignment) if task == navi_task_idx]
        if not navigating_robots:
            return ax

        navi_color = self.viz_params['navi_color']
        for robot_idx in navigating_robots:
            robot_pos = x[0:2, robot_idx]
            ax.text(
                robot_pos[0], robot_pos[1] + 0.15, "Navigate",
                color=navi_color, fontsize=9, ha='center',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'),
            )
            if p_goal is not None:
                ax.plot([robot_pos[0], p_goal[0]], [robot_pos[1], p_goal[1]], '--', color=navi_color, linewidth=1, alpha=0.6)
                dist_to_goal = np.linalg.norm(robot_pos - p_goal)
                mid_x = (robot_pos[0] + p_goal[0]) / 2
                mid_y = (robot_pos[1] + p_goal[1]) / 2
                ax.text(
                    mid_x, mid_y, f"{dist_to_goal:.2f}m", color=navi_color, fontsize=8,
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.1'),
                )
        return ax

    def visualize_explore_task(self, ax, x, t, task_assignment, global_vars):
        explore_task_idx = 2  # 1-based: Explore
        exploring_robots = [i for i, task in enumerate(task_assignment) if task == explore_task_idx]
        if not exploring_robots or not hasattr(global_vars, 'exploration_targets'):
            return ax

        explore_color = self.viz_params['explore_color']
        action_color = [0.0, 0.6, 0.8]
        exploring_action_info = global_vars.get_var('exploring_action_info', {})
        exploration_action_timers = global_vars.get_var('exploration_action_timers', {})
        exploration_action_duration = global_vars.get_var('exploration_action_duration', 2.0)

        for robot_idx in exploring_robots:
            robot_pos = x[0:2, robot_idx]
            if robot_idx in exploring_action_info:
                current_timer = exploration_action_timers.get(robot_idx, 0)
                progress = min(1.0, current_timer / exploration_action_duration)
                ax.text(
                    robot_pos[0], robot_pos[1] + 0.15, f"Exploring {progress:.0%}",
                    color=action_color, fontsize=8, ha='center',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'),
                )
                scan_radius = 0.1 + 0.05 * np.sin(t * 5)
                ax.add_patch(plt.Circle((robot_pos[0], robot_pos[1]), scan_radius, color=action_color, fill=False, linestyle='-', alpha=0.8, linewidth=1.5))
            else:
                ax.text(
                    robot_pos[0], robot_pos[1] + 0.15, "Explore Nav",
                    color=explore_color, fontsize=9, ha='center',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'),
                )
                unexplored_targets = [target for target in global_vars.exploration_targets if not target.get('explored', False)]
                if unexplored_targets:
                    nearest_target_pos = min(
                        unexplored_targets,
                        key=lambda target: np.sum((robot_pos - target['position']) ** 2),
                    )['position']
                    min_dist = np.linalg.norm(robot_pos - nearest_target_pos)
                    ax.plot([robot_pos[0], nearest_target_pos[0]], [robot_pos[1], nearest_target_pos[1]], '--', color=explore_color, linewidth=1, alpha=0.6)
                    mid_x = (robot_pos[0] + nearest_target_pos[0]) / 2
                    mid_y = (robot_pos[1] + nearest_target_pos[1]) / 2
                    ax.text(
                        mid_x, mid_y, f"{min_dist:.2f}m", color=explore_color, fontsize=8,
                        ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.1'),
                    )
        return ax
