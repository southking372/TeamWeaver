# task_utils/task_module/state_manipulation_modules.py
import numpy as np

class FillTask:
    """Fill task module - handles container fill tasks (Agent 1 only)"""
    
    @staticmethod
    def fill_function(x, t, robot_id, vars_dict=None):
        """Fill task utility function"""
        # Agent 0 cannot perform Fill
        if robot_id == 0:
            return 1000.0
        
        if vars_dict is None:
            target_container_position = np.array([0.3, 0.8])
            fill_dist_thresh = 0.15
            container_filled_state = False
        else:
            target_container_position = vars_dict.get('target_container_position', np.array([0.3, 0.8]))
            fill_dist_thresh = vars_dict.get('fill_dist_thresh', 0.15)
            container_filled_state = vars_dict.get('container_filled_state', False)
        
        if container_filled_state:
            return 0.1
        
        robot_pos = x[:2]
        distance_to_container = np.linalg.norm(robot_pos - target_container_position)
        utility = (distance_to_container - fill_dist_thresh) ** 2
        
        if distance_to_container <= fill_dist_thresh:
            utility = 0.1
        
        return max(0.0, utility)
    
    @staticmethod
    def fill_gradient(x, t, robot_id, vars_dict=None):
        """Fill task utility functionthe gradient of"""
        if robot_id == 0:
            return np.zeros(3)
        
        if vars_dict is None:
            target_container_position = np.array([0.3, 0.8])
            fill_dist_thresh = 0.15
            container_filled_state = False
        else:
            target_container_position = vars_dict.get('target_container_position', np.array([0.3, 0.8]))
            fill_dist_thresh = vars_dict.get('fill_dist_thresh', 0.15)
            container_filled_state = vars_dict.get('container_filled_state', False)
        
        if container_filled_state:
            return np.zeros(3)
        
        robot_pos = x[:2]
        distance_to_container = np.linalg.norm(robot_pos - target_container_position)
        
        gradient = np.zeros(3)
        if distance_to_container > 1e-6:
            direction = (robot_pos - target_container_position) / distance_to_container
            gradient[:2] = 2 * (distance_to_container - fill_dist_thresh) * direction
        
        return gradient
    
    @staticmethod
    def fill_time_derivative(x, t, robot_id, vars_dict=None):
        return 0.0
    
    @staticmethod
    def apply_motion_control(x, t, robot_id, vars_dict, dt):
        """Fill task motion control"""
        if robot_id == 0:
            return x
        
        if vars_dict is None:
            return x
        
        target_container_position = vars_dict.get('target_container_position', np.array([0.3, 0.8]))
        fill_dist_thresh = vars_dict.get('fill_dist_thresh', 0.15)
        container_filled_state = vars_dict.get('container_filled_state', False)
        
        if container_filled_state:
            return x
        
        robot_pos = x[:2]
        distance_to_container = np.linalg.norm(robot_pos - target_container_position)
        
        if distance_to_container <= fill_dist_thresh:
            # simulate successful fill
            if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                vars_dict.set_var('container_filled_state', True)
            else:
                vars_dict['container_filled_state'] = True
            
            print(f"Robot {robot_id} successfully filled container at position {target_container_position}")
            return x
        
        # move toward target container
        direction = target_container_position - robot_pos
        distance = np.linalg.norm(direction)
        
        if distance > 1e-6:
            direction = direction / distance
            desired_theta = np.arctan2(direction[1], direction[0])
            theta_diff = desired_theta - x[2]
            theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
            
            linear_speed = 0.3
            angular_speed = 1.0
            new_x = x.copy()
            
            if abs(theta_diff) > 0.1:
                new_x[2] += np.sign(theta_diff) * angular_speed * dt
            else:
                new_x[0] += linear_speed * np.cos(x[2]) * dt
                new_x[1] += linear_speed * np.sin(x[2]) * dt
                new_x[2] += 0.5 * theta_diff * dt
            
            new_x[2] = np.arctan2(np.sin(new_x[2]), np.cos(new_x[2]))
            return new_x
        
        return x

class PourTask:
    """Pour task module - handles liquid pour tasks (Agent 1 only)"""
    
    @staticmethod
    def pour_function(x, t, robot_id, vars_dict=None):
        """Pour task utility function"""
        if robot_id == 0:
            return 1000.0
        
        if vars_dict is None:
            target_receptacle_position = np.array([-0.3, 0.6])
            pour_dist_thresh = 0.15
            pour_completed_state = False
        else:
            target_receptacle_position = vars_dict.get('target_receptacle_position', np.array([-0.3, 0.6]))
            pour_dist_thresh = vars_dict.get('pour_dist_thresh', 0.15)
            pour_completed_state = vars_dict.get('pour_completed_state', False)
        
        if pour_completed_state:
            return 0.1
        
        robot_pos = x[:2]
        distance_to_receptacle = np.linalg.norm(robot_pos - target_receptacle_position)
        utility = (distance_to_receptacle - pour_dist_thresh) ** 2
        
        if distance_to_receptacle <= pour_dist_thresh:
            utility = 0.1
        
        return max(0.0, utility)
    
    @staticmethod
    def pour_gradient(x, t, robot_id, vars_dict=None):
        """Pour task utility functionthe gradient of"""
        if robot_id == 0:
            return np.zeros(3)
        
        if vars_dict is None:
            target_receptacle_position = np.array([-0.3, 0.6])
            pour_dist_thresh = 0.15
            pour_completed_state = False
        else:
            target_receptacle_position = vars_dict.get('target_receptacle_position', np.array([-0.3, 0.6]))
            pour_dist_thresh = vars_dict.get('pour_dist_thresh', 0.15)
            pour_completed_state = vars_dict.get('pour_completed_state', False)
        
        if pour_completed_state:
            return np.zeros(3)
        
        robot_pos = x[:2]
        distance_to_receptacle = np.linalg.norm(robot_pos - target_receptacle_position)
        
        gradient = np.zeros(3)
        if distance_to_receptacle > 1e-6:
            direction = (robot_pos - target_receptacle_position) / distance_to_receptacle
            gradient[:2] = 2 * (distance_to_receptacle - pour_dist_thresh) * direction
        
        return gradient
    
    @staticmethod
    def pour_time_derivative(x, t, robot_id, vars_dict=None):
        return 0.0
    
    @staticmethod
    def apply_motion_control(x, t, robot_id, vars_dict, dt):
        """Pour task motion control"""
        if robot_id == 0:
            return x
        
        if vars_dict is None:
            return x
        
        target_receptacle_position = vars_dict.get('target_receptacle_position', np.array([-0.3, 0.6]))
        pour_dist_thresh = vars_dict.get('pour_dist_thresh', 0.15)
        pour_completed_state = vars_dict.get('pour_completed_state', False)
        
        if pour_completed_state:
            return x
        
        robot_pos = x[:2]
        distance_to_receptacle = np.linalg.norm(robot_pos - target_receptacle_position)
        
        if distance_to_receptacle <= pour_dist_thresh:
            # simulate successful pour
            if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                vars_dict.set_var('pour_completed_state', True)
            else:
                vars_dict['pour_completed_state'] = True
            
            print(f"Robot {robot_id} successfully poured into receptacle at position {target_receptacle_position}")
            return x
        
        # move toward target container
        direction = target_receptacle_position - robot_pos
        distance = np.linalg.norm(direction)
        
        if distance > 1e-6:
            direction = direction / distance
            desired_theta = np.arctan2(direction[1], direction[0])
            theta_diff = desired_theta - x[2]
            theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
            
            linear_speed = 0.3
            angular_speed = 1.0
            new_x = x.copy()
            
            if abs(theta_diff) > 0.1:
                new_x[2] += np.sign(theta_diff) * angular_speed * dt
            else:
                new_x[0] += linear_speed * np.cos(x[2]) * dt
                new_x[1] += linear_speed * np.sin(x[2]) * dt
                new_x[2] += 0.5 * theta_diff * dt
            
            new_x[2] = np.arctan2(np.sin(new_x[2]), np.cos(new_x[2]))
            return new_x
        
        return x

class PowerOnTask:
    """PowerOn task module - handles device power-on tasks (Agent 1 only)"""
    
    @staticmethod
    def poweron_function(x, t, robot_id, vars_dict=None):
        """PowerOn task utility function"""
        if robot_id == 0:
            return 1000.0
        
        if vars_dict is None:
            target_device_position = np.array([0.7, 0.4])
            power_dist_thresh = 0.15
            device_power_state = False
        else:
            target_device_position = vars_dict.get('target_device_position', np.array([0.7, 0.4]))
            power_dist_thresh = vars_dict.get('power_dist_thresh', 0.15)
            device_power_state = vars_dict.get('device_power_state', False)
        
        if device_power_state:
            return 0.1
        
        robot_pos = x[:2]
        distance_to_device = np.linalg.norm(robot_pos - target_device_position)
        utility = (distance_to_device - power_dist_thresh) ** 2
        
        if distance_to_device <= power_dist_thresh:
            utility = 0.1
        
        return max(0.0, utility)
    
    @staticmethod
    def poweron_gradient(x, t, robot_id, vars_dict=None):
        """PowerOn task utility functionthe gradient of"""
        if robot_id == 0:
            return np.zeros(3)
        
        if vars_dict is None:
            target_device_position = np.array([0.7, 0.4])
            power_dist_thresh = 0.15
            device_power_state = False
        else:
            target_device_position = vars_dict.get('target_device_position', np.array([0.7, 0.4]))
            power_dist_thresh = vars_dict.get('power_dist_thresh', 0.15)
            device_power_state = vars_dict.get('device_power_state', False)
        
        if device_power_state:
            return np.zeros(3)
        
        robot_pos = x[:2]
        distance_to_device = np.linalg.norm(robot_pos - target_device_position)
        
        gradient = np.zeros(3)
        if distance_to_device > 1e-6:
            direction = (robot_pos - target_device_position) / distance_to_device
            gradient[:2] = 2 * (distance_to_device - power_dist_thresh) * direction
        
        return gradient
    
    @staticmethod
    def poweron_time_derivative(x, t, robot_id, vars_dict=None):
        return 0.0
    
    @staticmethod
    def apply_motion_control(x, t, robot_id, vars_dict, dt):
        """PowerOn task motion control"""
        if robot_id == 0:
            return x
        
        if vars_dict is None:
            return x
        
        target_device_position = vars_dict.get('target_device_position', np.array([0.7, 0.4]))
        power_dist_thresh = vars_dict.get('power_dist_thresh', 0.15)
        device_power_state = vars_dict.get('device_power_state', False)
        
        if device_power_state:
            return x
        
        robot_pos = x[:2]
        distance_to_device = np.linalg.norm(robot_pos - target_device_position)
        
        if distance_to_device <= power_dist_thresh:
            # simulate successful power-on
            if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                vars_dict.set_var('device_power_state', True)
            else:
                vars_dict['device_power_state'] = True
            
            print(f"Robot {robot_id} successfully powered on device at position {target_device_position}")
            return x
        
        # move toward target device
        direction = target_device_position - robot_pos
        distance = np.linalg.norm(direction)
        
        if distance > 1e-6:
            direction = direction / distance
            desired_theta = np.arctan2(direction[1], direction[0])
            theta_diff = desired_theta - x[2]
            theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
            
            linear_speed = 0.3
            angular_speed = 1.0
            new_x = x.copy()
            
            if abs(theta_diff) > 0.1:
                new_x[2] += np.sign(theta_diff) * angular_speed * dt
            else:
                new_x[0] += linear_speed * np.cos(x[2]) * dt
                new_x[1] += linear_speed * np.sin(x[2]) * dt
                new_x[2] += 0.5 * theta_diff * dt
            
            new_x[2] = np.arctan2(np.sin(new_x[2]), np.cos(new_x[2]))
            return new_x
        
        return x

class PowerOffTask:
    """PowerOff task module - handles device power-off tasks (Agent 1 only)"""
    
    @staticmethod
    def poweroff_function(x, t, robot_id, vars_dict=None):
        """PowerOff task utility function"""
        if robot_id == 0:
            return 1000.0
        
        if vars_dict is None:
            target_device_position = np.array([0.7, 0.4])
            power_dist_thresh = 0.15
            device_power_state = True  # must turn off, so initial state is on
        else:
            target_device_position = vars_dict.get('target_device_position', np.array([0.7, 0.4]))
            power_dist_thresh = vars_dict.get('power_dist_thresh', 0.15)
            device_power_state = vars_dict.get('device_power_state', True)
        
        if not device_power_state:
            return 0.1  # already off
        
        robot_pos = x[:2]
        distance_to_device = np.linalg.norm(robot_pos - target_device_position)
        utility = (distance_to_device - power_dist_thresh) ** 2
        
        if distance_to_device <= power_dist_thresh:
            utility = 0.1
        
        return max(0.0, utility)
    
    @staticmethod
    def poweroff_gradient(x, t, robot_id, vars_dict=None):
        """PowerOff task utility functionthe gradient of"""
        if robot_id == 0:
            return np.zeros(3)
        
        if vars_dict is None:
            target_device_position = np.array([0.7, 0.4])
            power_dist_thresh = 0.15
            device_power_state = True
        else:
            target_device_position = vars_dict.get('target_device_position', np.array([0.7, 0.4]))
            power_dist_thresh = vars_dict.get('power_dist_thresh', 0.15)
            device_power_state = vars_dict.get('device_power_state', True)
        
        if not device_power_state:
            return np.zeros(3)
        
        robot_pos = x[:2]
        distance_to_device = np.linalg.norm(robot_pos - target_device_position)
        
        gradient = np.zeros(3)
        if distance_to_device > 1e-6:
            direction = (robot_pos - target_device_position) / distance_to_device
            gradient[:2] = 2 * (distance_to_device - power_dist_thresh) * direction
        
        return gradient
    
    @staticmethod
    def poweroff_time_derivative(x, t, robot_id, vars_dict=None):
        return 0.0
    
    @staticmethod
    def apply_motion_control(x, t, robot_id, vars_dict, dt):
        """PowerOff task motion control"""
        if robot_id == 0:
            return x
        
        if vars_dict is None:
            return x
        
        target_device_position = vars_dict.get('target_device_position', np.array([0.7, 0.4]))
        power_dist_thresh = vars_dict.get('power_dist_thresh', 0.15)
        device_power_state = vars_dict.get('device_power_state', True)
        
        if not device_power_state:
            return x
        
        robot_pos = x[:2]
        distance_to_device = np.linalg.norm(robot_pos - target_device_position)
        
        if distance_to_device <= power_dist_thresh:
            # simulate successful power-off
            if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                vars_dict.set_var('device_power_state', False)
            else:
                vars_dict['device_power_state'] = False
            
            print(f"Robot {robot_id} successfully powered off device at position {target_device_position}")
            return x
        
        # move toward target device
        direction = target_device_position - robot_pos
        distance = np.linalg.norm(direction)
        
        if distance > 1e-6:
            direction = direction / distance
            desired_theta = np.arctan2(direction[1], direction[0])
            theta_diff = desired_theta - x[2]
            theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
            
            linear_speed = 0.3
            angular_speed = 1.0
            new_x = x.copy()
            
            if abs(theta_diff) > 0.1:
                new_x[2] += np.sign(theta_diff) * angular_speed * dt
            else:
                new_x[0] += linear_speed * np.cos(x[2]) * dt
                new_x[1] += linear_speed * np.sin(x[2]) * dt
                new_x[2] += 0.5 * theta_diff * dt
            
            new_x[2] = np.arctan2(np.sin(new_x[2]), np.cos(new_x[2]))
            return new_x
        
        return x

class RearrangeTask:
    """Rearrange task module - handles object rearrangement (Pick+Place combo)"""
    
    @staticmethod
    def rearrange_function(x, t, robot_id, vars_dict=None):
        """Rearrange task utility function - combines Pick and Place logic"""
        if vars_dict is None:
            target_object_position = np.array([0.8, -0.5])
            target_receptacle_position = np.array([-0.8, 0.7])
            pick_dist_thresh = 0.15
            place_dist_thresh = 0.15
            is_holding = False
            holding_robot_id = None
        else:
            target_object_position = vars_dict.get('target_object_position', np.array([0.8, -0.5]))
            target_receptacle_position = vars_dict.get('target_receptacle_position', np.array([-0.8, 0.7]))
            pick_dist_thresh = vars_dict.get('pick_dist_thresh', 0.15)
            place_dist_thresh = vars_dict.get('place_dist_thresh', 0.15)
            is_holding = vars_dict.get('is_holding', False)
            holding_robot_id = vars_dict.get('holding_robot_id', None)
        
        robot_pos = x[:2]
        
        # If another robot holds the object, this robot cannot execute Rearrange
        if is_holding and holding_robot_id != robot_id:
            return 1000.0
        
        # If this robot holds the object, compute distance to placement location
        if is_holding and holding_robot_id == robot_id:
            distance_to_receptacle = np.linalg.norm(robot_pos - target_receptacle_position)
            utility = (distance_to_receptacle - place_dist_thresh) ** 2
            if distance_to_receptacle <= place_dist_thresh:
                utility = 0.1  # near completion
        else:
            # If not holding object, compute distance to object
            distance_to_object = np.linalg.norm(robot_pos - target_object_position)
            utility = (distance_to_object - pick_dist_thresh) ** 2
            if distance_to_object <= pick_dist_thresh:
                utility = 0.5  # intermediate state; transition to Place phase needed
        
        return max(0.0, utility)
    
    @staticmethod
    def rearrange_gradient(x, t, robot_id, vars_dict=None):
        """Gradient of Rearrange task utility function"""
        if vars_dict is None:
            target_object_position = np.array([0.8, -0.5])
            target_receptacle_position = np.array([-0.8, 0.7])
            pick_dist_thresh = 0.15
            place_dist_thresh = 0.15
            is_holding = False
            holding_robot_id = None
        else:
            target_object_position = vars_dict.get('target_object_position', np.array([0.8, -0.5]))
            target_receptacle_position = vars_dict.get('target_receptacle_position', np.array([-0.8, 0.7]))
            pick_dist_thresh = vars_dict.get('pick_dist_thresh', 0.15)
            place_dist_thresh = vars_dict.get('place_dist_thresh', 0.15)
            is_holding = vars_dict.get('is_holding', False)
            holding_robot_id = vars_dict.get('holding_robot_id', None)
        
        robot_pos = x[:2]
        gradient = np.zeros(3)
        
        #If another robot holds the object,zero gradient
        if is_holding and holding_robot_id != robot_id:
            return gradient
        
        #Calculate the gradient based on the current state
        if is_holding and holding_robot_id == robot_id:
            # Place phase
            distance_to_receptacle = np.linalg.norm(robot_pos - target_receptacle_position)
            if distance_to_receptacle > 1e-6:
                direction = (robot_pos - target_receptacle_position) / distance_to_receptacle
                gradient[:2] = 2 * (distance_to_receptacle - place_dist_thresh) * direction
        else:
            # Pick phase
            distance_to_object = np.linalg.norm(robot_pos - target_object_position)
            if distance_to_object > 1e-6:
                direction = (robot_pos - target_object_position) / distance_to_object
                gradient[:2] = 2 * (distance_to_object - pick_dist_thresh) * direction
        
        return gradient
    
    @staticmethod
    def rearrange_time_derivative(x, t, robot_id, vars_dict=None):
        return 0.0
    
    @staticmethod
    def apply_motion_control(x, t, robot_id, vars_dict, dt):
        """Rearrange task motion control - automatic Pick→Place state machine"""
        if vars_dict is None:
            return x
        
        target_object_position = vars_dict.get('target_object_position', np.array([0.8, -0.5]))
        target_receptacle_position = vars_dict.get('target_receptacle_position', np.array([-0.8, 0.7]))
        pick_dist_thresh = vars_dict.get('pick_dist_thresh', 0.15)
        place_dist_thresh = vars_dict.get('place_dist_thresh', 0.15)
        is_holding = vars_dict.get('is_holding', False)
        holding_robot_id = vars_dict.get('holding_robot_id', None)
        
        robot_pos = x[:2]
        
        # If another robot holds object, stop moving
        if is_holding and holding_robot_id != robot_id:
            return x
        
        # Place phase: The robot holds an object and needs to be placed
        if is_holding and holding_robot_id == robot_id:
            distance_to_receptacle = np.linalg.norm(robot_pos - target_receptacle_position)
            
            if distance_to_receptacle <= place_dist_thresh:
                # execute placement
                if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                    vars_dict.set_var('is_holding', False)
                    vars_dict.set_var('holding_robot_id', None)
                else:
                    vars_dict['is_holding'] = False
                    vars_dict['holding_robot_id'] = None
                
                print(f"Robot {robot_id} completed Rearrange: placed object at {target_receptacle_position}")
                return x
            
            # move toward placement location
            target_position = target_receptacle_position
        else:
            # Pick phase: Need to grab objects
            distance_to_object = np.linalg.norm(robot_pos - target_object_position)
            
            if distance_to_object <= pick_dist_thresh:
                # execute pick
                if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                    vars_dict.set_var('is_holding', True)
                    vars_dict.set_var('holding_robot_id', robot_id)
                else:
                    vars_dict['is_holding'] = True
                    vars_dict['holding_robot_id'] = robot_id
                
                print(f"Robot {robot_id} Rearrange: picked object, now moving to place")
                return x
            
            # move toward pick location
            target_position = target_object_position
        
        # generic motion logic
        direction = target_position - robot_pos
        distance = np.linalg.norm(direction)
        
        if distance > 1e-6:
            direction = direction / distance
            desired_theta = np.arctan2(direction[1], direction[0])
            theta_diff = desired_theta - x[2]
            theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
            
            linear_speed = 0.3
            angular_speed = 1.0
            new_x = x.copy()
            
            if abs(theta_diff) > 0.1:
                new_x[2] += np.sign(theta_diff) * angular_speed * dt
            else:
                new_x[0] += linear_speed * np.cos(x[2]) * dt
                new_x[1] += linear_speed * np.sin(x[2]) * dt
                new_x[2] += 0.5 * theta_diff * dt
            
            new_x[2] = np.arctan2(np.sin(new_x[2]), np.cos(new_x[2]))
            return new_x
        
        return x 
