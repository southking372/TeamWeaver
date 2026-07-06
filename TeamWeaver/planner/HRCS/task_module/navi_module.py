# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

# HRCS/task_module/navi_module.py
import numpy as np
import warnings
from sys_module.tools_util import clamp

class NaviTask:

    @staticmethod
    def get_navi_params(vars_dict):
        if vars_dict is None:
            raise ValueError("NaviTask.get_navi_params A valid vars_dict is required.")

        required_keys = ['p_goal', 'theta_goal', 'dist_thresh', 'orientation_weight']
        missing_keys = [key for key in required_keys if key not in vars_dict]
        if missing_keys:
            raise KeyError(f"NaviTask Required navigation parameter is missing in vars_dict: {missing_keys}")

        p_goal = vars_dict['p_goal']
        theta_goal = vars_dict['theta_goal']
        dist_thresh = vars_dict['dist_thresh']
        orientation_weight = vars_dict['orientation_weight']

        if not isinstance(p_goal, np.ndarray):
            p_goal = np.array(p_goal)
        if p_goal.shape != (2,):
            warnings.warn(f"'p_goal' Missing, use fallback value 0")
            p_goal = np.zeros(2, dtype=np.float32)
            #raise ValueError(f"NaviTask The shape of 'p_goal' received should be (2,), but get {p_goal.shape}")

        return p_goal, theta_goal, dist_thresh, orientation_weight

    @staticmethod
    def navi_function(x_i, t, i, vars_dict):
        """
        Calculate navigation task function value H(x_i)。
        Defined as the negative value of the error, the goal is to maximize this value.
        H(x_i) = -0.5 * ||pos(x_i) - p_goal||^2 - w_orient * angle_diff(theta(x_i), theta_goal)^2 + bonus

        Parameters:
            x_i: Robot status [x, y, theta]
            t: current time (Not used in this static target version)
            i: Robot Index (Not used)
            vars_dict: A dictionary containing navigation parameters ('p_goal', 'theta_goal', 'dist_thresh', 'orientation_weight')

        Returns:
            objective function value
        """
        p_goal, theta_goal, dist_thresh, orientation_weight = NaviTask.get_navi_params(vars_dict)

        robot_pos = x_i[0:2]
        robot_theta = x_i[2]
        pos_error_term = -0.5 * np.linalg.norm(robot_pos - p_goal)**2

        angle_diff = np.arctan2(np.sin(robot_theta - theta_goal), np.cos(robot_theta - theta_goal))
        angle_error_term = -orientation_weight * angle_diff**2
        dist_to_goal = np.linalg.norm(robot_pos - p_goal)
        arrival_bonus = 2.0 if dist_to_goal < dist_thresh else 0.0

        return pos_error_term + angle_error_term + arrival_bonus

    @staticmethod
    def navi_gradient(x_i, t, i, vars_dict):
        """
        Calculate navigation task function H(x_i) Gradient with respect to robot state x_i (dH/dx_i)。

        Returns:
            The gradient of the objective function with respect to the state [dH/dx, dH/dy, dH/dtheta]
        """
        p_goal, theta_goal, dist_thresh, orientation_weight = NaviTask.get_navi_params(vars_dict)

        robot_pos = x_i[0:2]
        robot_theta = x_i[2]

        gradient = np.zeros(3)

        # position gradient: d/dx (-0.5 * ||x - p_goal||^2) = -(x - p_goal)
        gradient[0:2] = -(robot_pos - p_goal)

        # angular gradient: d/dtheta (-w * angle_diff^2) = -2 * w * angle_diff
        # angle_diff = atan2(sin(theta - theta_goal), cos(theta - theta_goal)) ≈ theta - theta_goal (when close)
        # d(angle_diff)/dtheta = 1
        angle_diff = np.arctan2(np.sin(robot_theta - theta_goal), np.cos(robot_theta - theta_goal))
        gradient[2] = -2.0 * orientation_weight * angle_diff

        return gradient

    @staticmethod
    def navi_time_derivative(x_i, t, i, vars_dict):
        # Since p_goal and theta_goal do not change over time, dH/dt = 0
        return 0.0

    @staticmethod
    def print_debug_info(x_i, t, i, vars_dict, iter_count):
        if iter_count % 10 != 0:  # Print every 10 iterations to avoid screen swiping
            return
        try:
            p_goal, theta_goal, dist_thresh, orientation_weight = NaviTask.get_navi_params(vars_dict)
        except (ValueError, KeyError) as e:
            print(f"DEBUG: Robot {i} (Task 0) Invalid navigation parameter: {e}")
            return
            
        robot_pos = x_i[0:2]
        robot_theta = x_i[2]
        
        dist_to_goal = np.linalg.norm(robot_pos - p_goal)
        angle_diff = np.arctan2(np.sin(robot_theta - theta_goal), np.cos(robot_theta - theta_goal))
        pos_error_term = -0.5 * dist_to_goal**2
        angle_error_term = -orientation_weight * angle_diff**2
        arrival_bonus = 2.0 if dist_to_goal < dist_thresh else 0.0
        H_value = pos_error_term + angle_error_term + arrival_bonus
        
        arrived_status = "Arrived" if dist_to_goal < dist_thresh else "Not arrived"
        print(f"DEBUG: Robot {i} (Task 0) Navigating. Distance to target: {dist_to_goal:.2f}m (threshold: {dist_thresh}), state: {arrived_status}")
        print(f"       heading difference: {angle_diff:.2f}rad, position error term: {pos_error_term:.2f}, angle error term: {angle_error_term:.2f}")
        print(f"       Reach reward: {arrival_bonus:.1f}, total function value: {H_value:.2f}")

    @staticmethod
    def apply_motion_control(x_i, t, i, vars_dict, dt):
        try:
            p_goal, theta_goal, dist_thresh, orientation_weight = NaviTask.get_navi_params(vars_dict)
        except (ValueError, KeyError) as e:
            print(f"warn: Navigation task parameters are missing or invalid: {e}")
            return x_i  # Return to original state without changes
            
        # Get current location and heading
        curr_pos = x_i[0:2]
        curr_theta = x_i[2]
        
        # Calculate distance and direction to target
        vec_to_goal = p_goal - curr_pos
        dist_to_goal = np.linalg.norm(vec_to_goal)
        
        if dist_to_goal < dist_thresh:
            angle_diff = np.arctan2(np.sin(curr_theta - theta_goal), np.cos(curr_theta - theta_goal))
            theta_adjust_rate = 0.5
            new_theta = curr_theta - theta_adjust_rate * angle_diff * dt
            return np.array([curr_pos[0], curr_pos[1], new_theta])
            
        desired_theta = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        angle_diff = np.arctan2(np.sin(curr_theta - desired_theta), np.cos(curr_theta - desired_theta))
        
        max_speed = 0.4
        speed_factor = min(1.0, dist_to_goal / 2.0)
        angle_factor = max(0.2, np.cos(angle_diff)**2)
        forward_speed = max_speed * speed_factor * angle_factor
        
        max_angular_speed = 0.8
        angular_speed = -max_angular_speed * np.sin(angle_diff)
        new_pos_x = curr_pos[0] + forward_speed * np.cos(curr_theta) * dt
        new_pos_y = curr_pos[1] + forward_speed * np.sin(curr_theta) * dt
        new_theta = curr_theta + angular_speed * dt
        
        return np.array([new_pos_x, new_pos_y, new_theta])

