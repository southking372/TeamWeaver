# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

# sys_module/visualization_module.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.path import Path
from task_module.navi_module import NaviTask
from task_module.explore_module import ExploreTask
from task_module.manipulation_module import ManipulationTask, ManipulationPhase

class VisualizationConfig:
    """
    可视化配置类，用于管理可视化参数
    """
    
    def __init__(self):
        self.viz_params = self._initialize_viz_params()
        self.target_object_position = None
        self.target_receptacle_position = None
    
    def _initialize_viz_params(self):
        viz_params = {
            'figure_size': (19.2, 10.8),
            'axis_limits': [-1.8, 1.8, -1.2, 1.2],
            'title': '机器人任务分配与控制模拟',
            'xlabel': 'X 位置',
            'ylabel': 'Y 位置',
            'robot_colors': {
                'wheeled': [0.2, 0.4, 0.6],
                'quadrotor': [0.6, 0.2, 0.4]
            },
            'arrow_length':{
                'wheeled' : 0.05,
                'quadrotor': 0.2,
            },
            'task_colors': {
                'navi': [0, 0.7, 0],
                'explore': [0.7, 0, 0],
                'manipulation': [0, 0, 0.7],
                'wait': [0.7, 0.7, 0]
            },
            'transport_colors': {
                'point': [0.5, 0, 0.5],
                'path': [0.5, 0, 0.5]
            },
            'poi_color': [0.5, 0, 0],
            'mud_color': [0.5, 0.3, 0.05],
            'perimeter_color': [0, 0.75, 0],
            'manipulation_phase_colors': {
                'NAV_OBJ': [0.8, 0.4, 0],
                'PICK': [0, 0.8, 0.4],
                'NAV_REC': [0.4, 0, 0.8],
                'PLACE': [0.4, 0.8, 0]
            },
            'navi_color': [0.0, 0.6, 0.8],
            'explore_color': [0.9, 0.5, 0.0]
        }
        
        return viz_params
    
    def get_viz_params(self):
        return self.viz_params
    
    def update_viz_params(self, param_name, value):
        if param_name in self.viz_params:
            self.viz_params[param_name] = value
        else:
            print(f"警告：参数 '{param_name}' 不存在于可视化参数中")
    
    def setup_plot(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
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
        ax.arrow(x[0, i], x[1, i], length * np.cos(x[2, i]), length * np.sin(x[2, i]), head_width=length * 0.4, head_length=length * 0.4, fc=color, ec=color, alpha=0.8)
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
                ax.plot([x[0, i], self.target_receptacle_position[0]], 
                       [x[1, i], self.target_receptacle_position[1]], 
                       'r--', linewidth=1, alpha=0.6)
        
        return ax
    
    def plot_transport_point(self, ax, p_transport_t):
        color = self.viz_params['transport_colors']['point']
        ax.scatter(p_transport_t[0], p_transport_t[1], s=150, color=color, marker='o', linewidth=3)
        return ax
    
    def plot_transport_path(self, ax, P_traj, valid_points):
        color = self.viz_params['transport_colors']['path']
        ax.plot(P_traj[0, :valid_points], P_traj[1, :valid_points], '--', color=color, linewidth=2)
        return ax
    
    def plot_poi(self, ax, poi):
        color = self.viz_params['poi_color']
        ax.scatter(poi[0], poi[1], s=150, color=color, marker='p', linewidth=3)
        return ax
    
    def plot_mud_area(self, ax, x_mud, y_mud):
        color = self.viz_params['mud_color']
        mud_patch = Polygon(np.column_stack((x_mud, y_mud)), facecolor=color, edgecolor='none', alpha=0.5)
        ax.add_patch(mud_patch)
        return ax
    
    def plot_perimeter(self, ax, perimeter_x, perimeter_y):
        color = self.viz_params['perimeter_color']
        ax.fill(perimeter_x, perimeter_y, color=color, alpha=0.1, edgecolor='none')
        return ax
    
    def update_task_text(self, ax, t, task_assignment):
        task_assignment_str = "\n".join([
            f"时间: {t:.1f}s",
            f"任务分配: {task_assignment}"
        ])
        
        text_objects = [obj for obj in ax.texts if hasattr(obj, 'get_text') and obj.get_text().startswith("时间:")]
        if text_objects:
            text_objects[0].set_text(task_assignment_str)
        else:
            ax.text(-1.7, 1.1, task_assignment_str, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.7))
        
        return ax
        
    def visualize_manipulation_phases(self, ax, x, task_assignment, global_vars):
        manipulation_task_idx = 2
        manipulating_robots = [i for i, task in enumerate(task_assignment) if task == manipulation_task_idx]
        
        if not manipulating_robots:
            return ax
            
        current_phase = global_vars.manipulation_phase
        if current_phase is None:
            return ax
            
        phase_color = self.viz_params['manipulation_phase_colors'].get(current_phase.name, [0.5, 0.5, 0.5])
        
        for robot_idx in manipulating_robots:
            robot_pos = x[0:2, robot_idx]
            
            ax.text(robot_pos[0], robot_pos[1] + 0.15, 
                   f"阶段: {current_phase.name}", 
                   color=phase_color, 
                   fontsize=9, 
                   ha='center', 
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
            
            if current_phase.name == 'NAV_OBJ' and global_vars.target_object_position is not None:
                target_pos = global_vars.target_object_position
                ax.plot([robot_pos[0], target_pos[0]], [robot_pos[1], target_pos[1]], 
                       '--', color=phase_color, linewidth=1, alpha=0.6)
                
            elif current_phase.name == 'NAV_REC' and global_vars.target_receptacle_position is not None:
                target_pos = global_vars.target_receptacle_position
                ax.plot([robot_pos[0], target_pos[0]], [robot_pos[1], target_pos[1]], 
                       '--', color=phase_color, linewidth=1, alpha=0.6)
                
            if current_phase.name in ['PICK', 'PLACE']:
                ax.add_patch(plt.Circle((robot_pos[0], robot_pos[1]), 0.2, 
                                      color=phase_color, fill=False, linestyle='--', alpha=0.7))
                
        legend_x = -1.7
        legend_y = -1.1
        legend_text = f"操作任务执行机器人: {manipulating_robots}"
        ax.text(legend_x, legend_y, legend_text, fontsize=10,
               bbox=dict(facecolor='white', alpha=0.7))
               
        return ax
        
    def visualize_navi_task(self, ax, x, task_assignment, p_goal):
        navi_task_idx = 0
        navigating_robots = [i for i, task in enumerate(task_assignment) if task == navi_task_idx]
        
        if not navigating_robots:
            return ax
        
        navi_color = self.viz_params['navi_color']
        
        for robot_idx in navigating_robots:
            robot_pos = x[0:2, robot_idx]
            
            ax.text(robot_pos[0], robot_pos[1] + 0.15, 
                   "任务: 导航", 
                   color=navi_color, 
                   fontsize=9, 
                   ha='center', 
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
            
            if p_goal is not None:
                ax.plot([robot_pos[0], p_goal[0]], [robot_pos[1], p_goal[1]], 
                       '--', color=navi_color, linewidth=1, alpha=0.6)
                
                dist_to_goal = np.linalg.norm(robot_pos - p_goal)
                dist_text = f"{dist_to_goal:.2f}m"
                mid_x = (robot_pos[0] + p_goal[0]) / 2
                mid_y = (robot_pos[1] + p_goal[1]) / 2
                ax.text(mid_x, mid_y, dist_text, color=navi_color, fontsize=8, 
                       ha='center', va='center', 
                       bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.1'))
        
        return ax
    
    def visualize_explore_task(self, ax, x, task_assignment, global_vars):
        explore_task_idx = 1
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
            robot_id = robot_idx

            is_performing_action = robot_id in exploring_action_info

            if is_performing_action:
                current_timer = exploration_action_timers.get(robot_id, 0)
                progress = min(1.0, current_timer / exploration_action_duration)
                ax.text(robot_pos[0], robot_pos[1] + 0.15,
                       f"探索中... {progress:.0%}",
                       color=action_color,
                       fontsize=8,
                       ha='center',
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
                scan_radius = 0.1 + 0.05 * np.sin(current_timer * 5)
                ax.add_patch(plt.Circle((robot_pos[0], robot_pos[1]), scan_radius,
                                      color=action_color, fill=False, linestyle='-', alpha=0.8, linewidth=1.5))

            else:
                ax.text(robot_pos[0], robot_pos[1] + 0.15,
                       "任务: 导航探索",
                       color=explore_color,
                       fontsize=9,
                       ha='center',
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

                unexplored_targets = [target for target in global_vars.exploration_targets
                                    if not target.get('explored', False)]
                if unexplored_targets:
                    min_dist_sq = float('inf')
                    nearest_target_pos = None
                    for target in unexplored_targets:
                        dist_sq = np.sum((robot_pos - target['position'])**2)
                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                            nearest_target_pos = target['position']

                    if nearest_target_pos is not None:
                        min_dist = np.sqrt(min_dist_sq)
                        ax.plot([robot_pos[0], nearest_target_pos[0]], [robot_pos[1], nearest_target_pos[1]],
                               '--', color=explore_color, linewidth=1, alpha=0.6)
                        dist_text = f"{min_dist:.2f}m"
                        mid_x = (robot_pos[0] + nearest_target_pos[0]) / 2
                        mid_y = (robot_pos[1] + nearest_target_pos[1]) / 2
                        ax.text(mid_x, mid_y, dist_text, color=explore_color, fontsize=8,
                               ha='center', va='center',
                               bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.1'))

        return ax
    
    def print_task_debug_info(self, x, t, task_assignment, global_vars, iter_count):
        if iter_count % 10 != 0:
            return
            
        vars_dict = global_vars.get_all_vars()
        
        navi_task_idx = 0
        navigating_robot_indices = [i for i, task in enumerate(task_assignment) if task == navi_task_idx]
        
        for robot_idx in navigating_robot_indices:
            NaviTask.print_debug_info(x[:, robot_idx], t, robot_idx, vars_dict, iter_count)
        
        explore_task_idx = 1
        exploring_robot_indices = [i for i, task in enumerate(task_assignment) if task == explore_task_idx]
        
        for robot_idx in exploring_robot_indices:
            ExploreTask.print_debug_info(x[:, robot_idx], t, robot_idx, vars_dict, iter_count)
        
        manipulation_task_idx = 2
        manipulating_robot_indices = [i for i, task in enumerate(task_assignment) if task == manipulation_task_idx]
        
        for robot_idx in manipulating_robot_indices:
            current_phase = global_vars.manipulation_phase
            if global_vars.x is not None and robot_idx < global_vars.x.shape[1]:
                robot_pos = global_vars.x[0:2, robot_idx]
                
                if current_phase == ManipulationPhase.NAV_OBJ:
                    target_pos = global_vars.target_object_position
                    dist_thresh = global_vars.pick_dist_thresh
                    dist = np.linalg.norm(robot_pos - target_pos)
                    print(f"DEBUG: Robot {robot_idx} (Task 2) in NAV_OBJ. Dist to Obj: {dist:.2f}m (Thresh: {dist_thresh})")
                
                elif current_phase == ManipulationPhase.NAV_REC:
                    target_pos = global_vars.target_receptacle_position
                    dist_thresh = global_vars.place_dist_thresh
                    dist = np.linalg.norm(robot_pos - target_pos)
                    print(f"DEBUG: Robot {robot_idx} (Task 2) in NAV_REC. Dist to Rec: {dist:.2f}m (Thresh: {dist_thresh})")
                
                elif current_phase == ManipulationPhase.PICK:
                    print(f"DEBUG: Robot {robot_idx} (Task 2) in PICK phase. Holding: {global_vars.is_holding}")
                
                elif current_phase == ManipulationPhase.PLACE:
                    print(f"DEBUG: Robot {robot_idx} (Task 2) in PLACE phase. Holding: {global_vars.is_holding}")
