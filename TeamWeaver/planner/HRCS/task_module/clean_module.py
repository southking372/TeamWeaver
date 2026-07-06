# task_utils/task_module/clean_module.py
import numpy as np

class CleanTask:
    """CleanTask Module - Tasks to handle cleaning objects (Agent 1 exclusive capability)"""
    
    @staticmethod
    def clean_function(x, t, robot_id, vars_dict=None):
        """
        Cleantask utility function
        
        Parameters:
            x: Robot status [x, y, theta]
            t: time
            robot_id: robot ID
            vars_dict: global vars dict
        
        Returns:
            Utility value; lower means closer to goal
        """
        # Agent 0 Not capable of Cleaning
        if robot_id == 0:
            return 1000.0  # Returns a very large value, indicating that Clean cannot be executed.
        
        if vars_dict is None:
            target_object_position = np.array([0.5, -0.3])  # Default object location to be cleaned
            clean_dist_thresh = 0.15
            object_clean_state = False
        else:
            target_object_position = vars_dict.get('target_object_position', np.array([0.5, -0.3]))
            clean_dist_thresh = vars_dict.get('clean_dist_thresh', 0.15)
            object_clean_state = vars_dict.get('object_clean_state', False)
        
        # If the object has been cleaned, the task is completed
        if object_clean_state:
            return 0.1  # Return small value indicating task is complete
        
        # Calculate distance to target object
        robot_pos = x[:2]
        distance_to_object = np.linalg.norm(robot_pos - target_object_position)
        
        # Use quadratic utility function
        utility = (distance_to_object - clean_dist_thresh) ** 2
        
        # If already within the cleaning range, reduce the utility value
        if distance_to_object <= clean_dist_thresh:
            utility = 0.1  # near-completion state
        
        return max(0.0, utility)
    
    @staticmethod
    def clean_gradient(x, t, robot_id, vars_dict=None):
        """CleanThe gradient of the task utility function"""
        # Agent 0 Not capable of Cleaning
        if robot_id == 0:
            return np.zeros(3)
        
        if vars_dict is None:
            target_object_position = np.array([0.5, -0.3])
            clean_dist_thresh = 0.15
            object_clean_state = False
        else:
            target_object_position = vars_dict.get('target_object_position', np.array([0.5, -0.3]))
            clean_dist_thresh = vars_dict.get('clean_dist_thresh', 0.15)
            object_clean_state = vars_dict.get('object_clean_state', False)
        
        # If the object has been cleaned, zero gradient
        if object_clean_state:
            return np.zeros(3)
        
        robot_pos = x[:2]
        distance_to_object = np.linalg.norm(robot_pos - target_object_position)
        
        # gradient computation
        gradient = np.zeros(3)
        if distance_to_object > 1e-6:  # avoid division by zero
            direction = (robot_pos - target_object_position) / distance_to_object
            gradient[:2] = 2 * (distance_to_object - clean_dist_thresh) * direction
        
        return gradient
    
    @staticmethod
    def clean_time_derivative(x, t, robot_id, vars_dict=None):
        """CleanTime derivative of task utility function"""
        # CleanTasks usually do not depend directly on time
        return 0.0
    
    @staticmethod
    def apply_motion_control(x, t, robot_id, vars_dict, dt):
        """
        Motion control applying Clean task
        
        Parameters:
            x: current robot state [x, y, theta]
            t: current time
            robot_id: robot ID
            vars_dict: global vars dict
            dt: time step
        
        Returns:
            new robot state
        """
        # Agent 0 Not capable of Cleaning
        if robot_id == 0:
            print(f"Warning: Agent {robot_id} attempted to perform Clean task but lacks this capability")
            return x
        
        if vars_dict is None:
            return x
        
        target_object_position = vars_dict.get('target_object_position', np.array([0.5, -0.3]))
        clean_dist_thresh = vars_dict.get('clean_dist_thresh', 0.15)
        object_clean_state = vars_dict.get('object_clean_state', False)
        
        # If the object has been cleaned, stop movement
        if object_clean_state:
            return x
        
        robot_pos = x[:2]
        distance_to_object = np.linalg.norm(robot_pos - target_object_position)
        
        # If it is already within the cleaning range, perform the cleaning action
        if distance_to_object <= clean_dist_thresh:
            # Simulated cleaning successful
            if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                # use global vars manager
                vars_dict.set_var('object_clean_state', True)
            else:
                # plain dict
                vars_dict['object_clean_state'] = True
            
            print(f"Robot {robot_id} successfully cleaned object at position {target_object_position}")
            return x
        
        # movement to the target object
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