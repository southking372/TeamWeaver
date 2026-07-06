# task_utils/task_module/open_module.py
import numpy as np

class OpenTask:
    """OpenTask Module - handles the tasks of opening furniture or objects"""
    
    @staticmethod
    def open_function(x, t, robot_id, vars_dict=None):
        """
        Opentask utility function
        
        Parameters:
            x:Robot status[x, y, theta]
            t:time
            robot_id: robot ID
            vars_dict: global vars dict
        
        Returns:
            Utility value; lower means closer to goal
        """
        if vars_dict is None:
            target_furniture_position = np.array([1.0, 0.5])  #Default furniture position
            operation_dist_thresh = 0.2
            furniture_open_state = False
        else:
            target_furniture_position = vars_dict.get('target_furniture_position', np.array([1.0, 0.5]))
            operation_dist_thresh = vars_dict.get('operation_dist_thresh', 0.2)
            furniture_open_state = vars_dict.get('furniture_open_state', False)
        
        #If the furniture is already opened, the task is completed
        if furniture_open_state:
            return 0.1  # Return small value indicating task is complete
        
        #Calculate distance to target furniture
        robot_pos = x[:2]
        distance_to_furniture = np.linalg.norm(robot_pos - target_furniture_position)
        
        # Use quadratic utility function
        utility = (distance_to_furniture - operation_dist_thresh) ** 2
        
        #if already inmanipulationwithin the range, reducing the utility value
        if distance_to_furniture <= operation_dist_thresh:
            utility = 0.1  # near-completion state
        
        return max(0.0, utility)
    
    @staticmethod
    def open_gradient(x, t, robot_id, vars_dict=None):
        """OpenThe gradient of the task utility function"""
        if vars_dict is None:
            target_furniture_position = np.array([1.0, 0.5])
            operation_dist_thresh = 0.2
            furniture_open_state = False
        else:
            target_furniture_position = vars_dict.get('target_furniture_position', np.array([1.0, 0.5]))
            operation_dist_thresh = vars_dict.get('operation_dist_thresh', 0.2)
            furniture_open_state = vars_dict.get('furniture_open_state', False)
        
        #If the furniture has been opened,zero gradient
        if furniture_open_state:
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
    def open_time_derivative(x, t, robot_id, vars_dict=None):
        """OpenTime derivative of task utility function"""
        # OpenTasks usually do not depend directly on time
        return 0.0
    
    @staticmethod
    def apply_motion_control(x, t, robot_id, vars_dict, dt):
        """
applicationOpentask movementcontrol
        
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
        furniture_open_state = vars_dict.get('furniture_open_state', False)
        
        #If the furniture is already opened, stopmovement
        if furniture_open_state:
            return x
        
        robot_pos = x[:2]
        distance_to_furniture = np.linalg.norm(robot_pos - target_furniture_position)
        
        #if already inmanipulationWithin the range, perform the opening action
        if distance_to_furniture <= operation_dist_thresh:
            #Simulation started successfully
            if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                # use global vars manager
                vars_dict.set_var('furniture_open_state', True)
            else:
                # plain dict
                vars_dict['furniture_open_state'] = True
            
            print(f"Robot {robot_id} successfully opened furniture at position {target_furniture_position}")
            return x
        
        #target furnituremovement
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
