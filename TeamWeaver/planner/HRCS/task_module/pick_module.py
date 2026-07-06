# task_utils/task_module/pick_module.py
import numpy as np

class PickTask:
    """Pick task module - handles object pick tasks"""
    
    @staticmethod
    def pick_function(x, t, robot_id, vars_dict=None):
        """
        Pick task utility function
        
        Parameters:
            x:Robot status[x, y, theta]
            t:time
            robot_id: robot ID
            vars_dict: global vars dict
        
        Returns:
            Utility value; lower means closer to goal
        """
        if vars_dict is None:
            target_object_position = np.array([0.8, -0.5])  # default target position
            pick_dist_thresh = 0.15
            is_holding = False
        else:
            target_object_position = vars_dict.get('target_object_position', np.array([0.8, -0.5]))
            pick_dist_thresh = vars_dict.get('pick_dist_thresh', 0.15)
            is_holding = vars_dict.get('is_holding', False)
            holding_robot_id = vars_dict.get('holding_robot_id', None)
            
            # If another robot holds the object, Pick utility is infinite for other robots
            if is_holding and holding_robot_id != robot_id:
                return 1000.0  # Return large value indicating Pick is not needed
        
        # If this robot already holds the object, Pick is complete
        if is_holding and vars_dict and vars_dict.get('holding_robot_id') == robot_id:
            return 0.1  # Return small value indicating task is complete
        
        # Compute distance to target object
        robot_pos = x[:2]
        distance_to_object = np.linalg.norm(robot_pos - target_object_position)
        
        # Use quadratic utility function
        utility = (distance_to_object - pick_dist_thresh) ** 2
        
        # If within pick range, reduce utility
        if distance_to_object <= pick_dist_thresh:
            utility = 0.1  # near-completion state
        
        return max(0.0, utility)
    
    @staticmethod
    def pick_gradient(x, t, robot_id, vars_dict=None):
        """Pick task utility functionthe gradient of"""
        if vars_dict is None:
            target_object_position = np.array([0.8, -0.5])
            pick_dist_thresh = 0.15
            is_holding = False
        else:
            target_object_position = vars_dict.get('target_object_position', np.array([0.8, -0.5]))
            pick_dist_thresh = vars_dict.get('pick_dist_thresh', 0.15)
            is_holding = vars_dict.get('is_holding', False)
            holding_robot_id = vars_dict.get('holding_robot_id', None)
            
            # If another robot already holds the object
            if is_holding and holding_robot_id != robot_id:
                return np.zeros(3)  # zero gradient
        
        # If this robot already holds the object
        if is_holding and vars_dict and vars_dict.get('holding_robot_id') == robot_id:
            return np.zeros(3)  # zero gradient
        
        robot_pos = x[:2]
        distance_to_object = np.linalg.norm(robot_pos - target_object_position)
        
        # gradient computation
        gradient = np.zeros(3)
        if distance_to_object > 1e-6:  # avoid division by zero
            direction = (robot_pos - target_object_position) / distance_to_object
            gradient[:2] = 2 * (distance_to_object - pick_dist_thresh) * direction
        
        return gradient
    
    @staticmethod
    def pick_time_derivative(x, t, robot_id, vars_dict=None):
        """Pick task utility functiontime derivative of"""
        # Pick task usually does not depend directly on time
        return 0.0
    
    @staticmethod
    def apply_motion_control(x, t, robot_id, vars_dict, dt):
        """
        Apply Pick task motion control
        
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
        
        target_object_position = vars_dict.get('target_object_position', np.array([0.8, -0.5]))
        pick_dist_thresh = vars_dict.get('pick_dist_thresh', 0.15)
        is_holding = vars_dict.get('is_holding', False)
        holding_robot_id = vars_dict.get('holding_robot_id', None)
        
        # If another robot already holds the object,stopmovement
        if is_holding and holding_robot_id != robot_id:
            return x
        
        # If this robot already holds the object,stopmovement
        if is_holding and holding_robot_id == robot_id:
            return x
        
        robot_pos = x[:2]
        distance_to_object = np.linalg.norm(robot_pos - target_object_position)
        
        # If within pick range, execute pick action
        if distance_to_object <= pick_dist_thresh:
            # simulate successful pick
            if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                # use global vars manager
                vars_dict.set_var('is_holding', True)
                vars_dict.set_var('holding_robot_id', robot_id)
            else:
                # plain dict
                vars_dict['is_holding'] = True
                vars_dict['holding_robot_id'] = robot_id
            
            print(f"Robot {robot_id} successfully picked object at position {target_object_position}")
            return x
        
        # move toward target object
        direction = target_object_position - robot_pos
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
            
            #ifheading differenceLarger, turn first
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
