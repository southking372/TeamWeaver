# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

# task_utils/task_module/close_module.py
import numpy as np

class CloseTask:
    """CloseTask module - Handles tasks for closing furniture or objects"""
    
    @staticmethod
    def close_function(x, t, robot_id, vars_dict=None):
        """
        Closetask utility function
        
        Parameters:
            x: Robot status [x, y, theta]
            t: time
            robot_id: robot ID
            vars_dict: global vars dict
        
        Returns:
            Utility value; lower means closer to goal
        """
        if vars_dict is None:
            target_furniture_position = np.array([1.0, 0.5])  # Default furniture position
            operation_dist_thresh = 0.2
            furniture_open_state = True  # must turn off, so initial state is on
        else:
            target_furniture_position = vars_dict.get('target_furniture_position', np.array([1.0, 0.5]))
            operation_dist_thresh = vars_dict.get('operation_dist_thresh', 0.2)
            furniture_open_state = vars_dict.get('furniture_open_state', True)
        
        # If the furniture is closed, the task is completed
        if not furniture_open_state:
            return 0.1  # Return small value indicating task is complete
        
        # Calculate distance to target furniture
        robot_pos = x[:2]
        distance_to_furniture = np.linalg.norm(robot_pos - target_furniture_position)
        
        # Use quadratic utility function
        utility = (distance_to_furniture - operation_dist_thresh) ** 2
        
        # If already within the manipulation range, reduce the utility value
        if distance_to_furniture <= operation_dist_thresh:
            utility = 0.1  # near-completion state
        
        return max(0.0, utility)
    
    @staticmethod
    def close_gradient(x, t, robot_id, vars_dict=None):
        """CloseThe gradient of the task utility function"""
        if vars_dict is None:
            target_furniture_position = np.array([1.0, 0.5])
            operation_dist_thresh = 0.2
            furniture_open_state = True
        else:
            target_furniture_position = vars_dict.get('target_furniture_position', np.array([1.0, 0.5]))
            operation_dist_thresh = vars_dict.get('operation_dist_thresh', 0.2)
            furniture_open_state = vars_dict.get('furniture_open_state', True)
        
        # If the furniture is closed, zero gradient
        if not furniture_open_state:
            return np.zeros(3)
        
        robot_pos = x[:2]
        distance_to_furniture = np.linalg.norm(robot_pos - target_furniture_position)
        
        # gradient computation
        gradient = np.zeros(3)
        if distance_to_furniture > 1e-6:  # avoid division by zero
            direction = (robot_pos - target_furniture_position) / distance_to_furniture
            gradient[:2] = 2 * (distance_to_furniture - operation_dist_thresh) * direction
        
        return gradient
    
    @staticmethod
    def close_time_derivative(x, t, robot_id, vars_dict=None):
        """CloseTime derivative of task utility function"""
        # CloseTasks usually do not depend directly on time
        return 0.0
    
    @staticmethod
    def apply_motion_control(x, t, robot_id, vars_dict, dt):
        """
        Motion control applying Close task
        
        Parameters:
            x: current robot state [x, y, theta]
            t: current time
            robot_id: robot ID
            vars_dict: global vars dict
            dt: time step
        
        Returns:
            new robot state
        """
        if vars_dict is None:
            return x
        
        target_furniture_position = vars_dict.get('target_furniture_position', np.array([1.0, 0.5]))
        operation_dist_thresh = vars_dict.get('operation_dist_thresh', 0.2)
        furniture_open_state = vars_dict.get('furniture_open_state', True)
        
        # If the furniture is already closed, stop movement
        if not furniture_open_state:
            return x
        
        robot_pos = x[:2]
        distance_to_furniture = np.linalg.norm(robot_pos - target_furniture_position)
        
        # If it is already within the manipulation range, perform the shutdown action
        if distance_to_furniture <= operation_dist_thresh:
            # Simulation closed successfully
            if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                # use global vars manager
                vars_dict.set_var('furniture_open_state', False)
            else:
                # plain dict
                vars_dict['furniture_open_state'] = False
            
            print(f"Robot {robot_id} successfully closed furniture at position {target_furniture_position}")
            return x
        
        # movement towards target furniture
        direction = target_furniture_position - robot_pos
        distance = np.linalg.norm(direction)
        
        if distance > 1e-6:
            direction = direction / distance
            
            # compute desired heading
            desired_theta = np.arctan2(direction[1], direction[0])
            
            # heading difference
            theta_diff = desired_theta - x[2]
            theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))  # normalize to [-pi, pi]
            
            # control parameters
            linear_speed = 0.3
            angular_speed = 1.0
            
            new_x = x.copy()
            
            # If the heading difference is large, turn first
            if abs(theta_diff) > 0.1:
                new_x[2] += np.sign(theta_diff) * angular_speed * dt
            else:
                # move forward after alignment
                new_x[0] += linear_speed * np.cos(x[2]) * dt
                new_x[1] += linear_speed * np.sin(x[2]) * dt
                # fine-tune heading simultaneously
                new_x[2] += 0.5 * theta_diff * dt
            
            # normalize heading
            new_x[2] = np.arctan2(np.sin(new_x[2]), np.cos(new_x[2]))
            
            return new_x
        
        return x 