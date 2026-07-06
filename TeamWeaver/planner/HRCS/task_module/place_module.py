# task_utils/task_module/place_module.py
import numpy as np

class PlaceTask:
    """Place task module - handles object placement tasks"""
    
    @staticmethod
    def place_function(x, t, robot_id, vars_dict=None):
        """
        Place task utility function
        
        Parameters:
            x:Robot status[x, y, theta]
            t:time
            robot_id: robot ID
            vars_dict: global vars dict
        
        Returns:
            Utility value; lower means closer to goal
        """
        if vars_dict is None:
            target_receptacle_position = np.array([-0.8, 0.7])
            place_dist_thresh = 0.15
            is_holding = False
            holding_robot_id = None
        else:
            target_receptacle_position = vars_dict.get('target_receptacle_position', np.array([-0.8, 0.7]))
            place_dist_thresh = vars_dict.get('place_dist_thresh', 0.15)
            is_holding = vars_dict.get('is_holding', False)
            holding_robot_id = vars_dict.get('holding_robot_id', None)
        
        # Only a robot holding the object can execute Place
        if not is_holding or holding_robot_id != robot_id:
            return 1000.0  # Return large value indicating Place cannot be executed
        
        # Compute distance to target placement location
        robot_pos = x[:2]
        distance_to_receptacle = np.linalg.norm(robot_pos - target_receptacle_position)
        
        # Use quadratic utility function
        utility = (distance_to_receptacle - place_dist_thresh) ** 2
        
        # If within placement range, reduce utility
        if distance_to_receptacle <= place_dist_thresh:
            utility = 0.1  # near-completion state
        
        return max(0.0, utility)
    
    @staticmethod
    def place_gradient(x, t, robot_id, vars_dict=None):
        """Place task utility functionthe gradient of"""
        if vars_dict is None:
            target_receptacle_position = np.array([-0.8, 0.7])
            place_dist_thresh = 0.15
            is_holding = False
            holding_robot_id = None
        else:
            target_receptacle_position = vars_dict.get('target_receptacle_position', np.array([-0.8, 0.7]))
            place_dist_thresh = vars_dict.get('place_dist_thresh', 0.15)
            is_holding = vars_dict.get('is_holding', False)
            holding_robot_id = vars_dict.get('holding_robot_id', None)
        
        # Only a robot holding the object has non-zero gradient
        if not is_holding or holding_robot_id != robot_id:
            return np.zeros(3)
        
        robot_pos = x[:2]
        distance_to_receptacle = np.linalg.norm(robot_pos - target_receptacle_position)
        
        # gradient computation
        gradient = np.zeros(3)
        if distance_to_receptacle > 1e-6:  # avoid division by zero
            direction = (robot_pos - target_receptacle_position) / distance_to_receptacle
            gradient[:2] = 2 * (distance_to_receptacle - place_dist_thresh) * direction
        
        return gradient
    
    @staticmethod
    def place_time_derivative(x, t, robot_id, vars_dict=None):
        """Place task utility functiontime derivative of"""
        # Place task usually does not depend directly on time
        return 0.0
    
    @staticmethod
    def apply_motion_control(x, t, robot_id, vars_dict, dt):
        """
        Apply Place task motion control
        
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
        
        target_receptacle_position = vars_dict.get('target_receptacle_position', np.array([-0.8, 0.7]))
        place_dist_thresh = vars_dict.get('place_dist_thresh', 0.15)
        is_holding = vars_dict.get('is_holding', False)
        holding_robot_id = vars_dict.get('holding_robot_id', None)
        
        # Only a robot holding the object can execute Place
        if not is_holding or holding_robot_id != robot_id:
            return x
        
        robot_pos = x[:2]
        distance_to_receptacle = np.linalg.norm(robot_pos - target_receptacle_position)
        
        # If within placement range, execute place action
        if distance_to_receptacle <= place_dist_thresh:
            # simulate successful placement
            if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                # use global vars manager
                vars_dict.set_var('is_holding', False)
                vars_dict.set_var('holding_robot_id', None)
            else:
                # plain dict
                vars_dict['is_holding'] = False
                vars_dict['holding_robot_id'] = None
            
            print(f"Robot {robot_id} successfully placed object at position {target_receptacle_position}")
            return x
        
        # move toward target placement location
        direction = target_receptacle_position - robot_pos
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
