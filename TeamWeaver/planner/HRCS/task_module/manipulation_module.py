# task_utils/manipulation_module.py
import numpy as np
from enum import Enum

class ManipulationPhase(Enum):
    """
    manipulationStage enum for task
    """
    NAV_OBJ = 0    # Navigate to object
    PICK = 1       # Grab objects
    NAV_REC = 2    # Navigate to target location
    PLACE = 3      # place objects

classManipulationTask:
    """
    manipulationThe implementation class of the task, corresponding to the rearrange combination skill in partnr-planner.
    This is a composite task that includes the steps of navigating to the object, grabbing, navigating to the target location, and placing it.
    """
    
    @staticmethod
    def get_global_vars_dict():
        """
        Get global vars dict from GlobalVarsManager
        """
        global_vars_dict = None
        try:
            # Get GlobalVarsManager instance
            importsys
            global_vars_manager = None
            
            # Search for GlobalVarsManager instances in all modules
            for module_name, module in sys.modules.items():
                if hasattr(module, 'global_vars'):
                    global_vars_manager = getattr(module, 'global_vars')
                    break
            
            if global_vars_manager is not None:
                if hasattr(global_vars_manager, 'get_all_vars'):
                    global_vars_dict = global_vars_manager.get_all_vars()
                elif hasattr(global_vars_manager, 'get_var'):
                    # Build a dictionary containing all variables required by the manipulation task
                    global_vars_dict = {
                        'manipulation_phase': global_vars_manager.get_var('manipulation_phase', ManipulationPhase.NAV_OBJ),
                        'target_object_position': global_vars_manager.get_var('target_object_position', np.array([0.8, -0.5])),
                        'target_receptacle_position': global_vars_manager.get_var('target_receptacle_position', np.array([-0.8, 0.7])),
                        'pick_dist_thresh': global_vars_manager.get_var('pick_dist_thresh', 0.3),
                        'place_dist_thresh': global_vars_manager.get_var('place_dist_thresh', 0.3),
                        'pick_action_value': global_vars_manager.get_var('pick_action_value', 1.0),
                        'place_action_value': global_vars_manager.get_var('place_action_value', 1.0),
                        'is_holding': global_vars_manager.get_var('is_holding', False),
                        'holding_robot_id': global_vars_manager.get_var('holding_robot_id', None)
                    }
                else:
                    print("Global vars manager is missing required methods")
                    global_vars_dict = {}
            else:
                print("Could not find global_vars_manager")
        
        except Exception as e:
            print(f"Error getting global vars manager: {e}")
            global_vars_dict = None
            
        return global_vars_dict
    
    @staticmethod
    def manipulation_function(x_i, t, i, vars_dict=None):
        if vars_dict is None:
            vars_dict = ManipulationTask.get_global_vars_dict()
            if vars_dict is None: return -100

        current_phase = vars_dict.get('manipulation_phase', ManipulationPhase.NAV_OBJ)
        is_holding = vars_dict.get('is_holding', False)
        holding_robot_id = vars_dict.get('holding_robot_id', None) # Get holder ID
        nav_attraction_factor = 10.0 # Significantly increase navigation attractiveness
        
        # Stage weight coefficient, later stages have higher weights
        phase_weights = {
            ManipulationPhase.NAV_OBJ: 1.0,  # Basic weight
            ManipulationPhase.PICK: 1.2,     # Slightly higher weight
            ManipulationPhase.NAV_REC: 1.5,  # higher weight
            ManipulationPhase.PLACE: 2.0     # highest weight
        }
        current_weight = phase_weights.get(current_phase, 1.0)

        if is_holding:
            if holding_robot_id == i:
                if current_phase == ManipulationPhase.NAV_REC:
                    target_pos = vars_dict.get('target_receptacle_position')
                    if target_pos is None: return -0.1 * np.linalg.norm(x_i[0:2])**2
                    pos_error = -nav_attraction_factor * 0.5 * np.linalg.norm(x_i[0:2] - target_pos)**2
                    target_vector = target_pos - x_i[0:2]
                    if np.linalg.norm(target_vector) < 1e-6: desired_angle = x_i[2]
                    else: desired_angle = np.arctan2(target_vector[1], target_vector[0])
                    angle_diff = np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
                    angle_error = -0.3 * angle_diff**2
                    return current_weight * (pos_error + angle_error) + 20.0
                elif current_phase == ManipulationPhase.PLACE:
                    return 50.0 * current_weight
            else:
                if current_phase == ManipulationPhase.NAV_OBJ or current_phase == ManipulationPhase.PICK:
                    return -100.0 
                if current_phase == ManipulationPhase.NAV_REC or current_phase == ManipulationPhase.PLACE:
                    return -100.0 
        
        if (current_phase == ManipulationPhase.NAV_REC or current_phase == ManipulationPhase.PLACE):
            if not is_holding or holding_robot_id != i:
                return -100.0

        if current_phase == ManipulationPhase.NAV_OBJ:
            target_obj = vars_dict.get('target_object_position')
            if target_obj is None: return -0.1 * np.linalg.norm(x_i[0:2])**2
            pos_error = -nav_attraction_factor * 0.5 * np.linalg.norm(x_i[0:2] - target_obj)**2
            target_vector = target_obj - x_i[0:2]
            if np.linalg.norm(target_vector) < 1e-6: desired_angle = x_i[2]
            else: desired_angle = np.arctan2(target_vector[1], target_vector[0])
            angle_diff = np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
            angle_error = -0.3 * angle_diff**2
            return current_weight * (pos_error + angle_error)

        elif current_phase == ManipulationPhase.PICK:
            target_obj = vars_dict.get('target_object_position')
            pick_dist_thresh = vars_dict.get('pick_dist_thresh', 0.3)
            if target_obj is not None:
                dist_to_obj = np.linalg.norm(x_i[0:2] - target_obj)
                if dist_to_obj < pick_dist_thresh:
                    pick_action_value = vars_dict.get('pick_action_value', 1.0)
                    proximity_bonus = 1.0 - (dist_to_obj / pick_dist_thresh)
                    return current_weight * (pick_action_value + proximity_bonus)
                else:
                    return -10.0 * dist_to_obj
            else:
                return -100.0

        elif current_phase == ManipulationPhase.NAV_REC:
            target_pos = vars_dict.get('target_receptacle_position')
            if target_pos is None: return -0.1 * np.linalg.norm(x_i[0:2])**2
            pos_error = -nav_attraction_factor * 0.5 * np.linalg.norm(x_i[0:2] - target_pos)**2
            target_vector = target_pos - x_i[0:2]
            if np.linalg.norm(target_vector) < 1e-6: desired_angle = x_i[2]
            else: desired_angle = np.arctan2(target_vector[1], target_vector[0])
            angle_diff = np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
            angle_error = -0.3 * angle_diff**2
            if is_holding and holding_robot_id == i:
                return current_weight * (pos_error + angle_error) + 20.0
            return current_weight * (pos_error + angle_error)

        elif current_phase == ManipulationPhase.PLACE:
            target_pos = vars_dict.get('target_receptacle_position')
            place_dist_thresh = vars_dict.get('place_dist_thresh', 0.3)
            if target_pos is not None:
                dist_to_rec = np.linalg.norm(x_i[0:2] - target_pos)
                if dist_to_rec < place_dist_thresh:
                    place_action_value = vars_dict.get('place_action_value', 1.0)
                    proximity_bonus = 1.0 - (dist_to_rec / place_dist_thresh)
                    if is_holding and holding_robot_id == i:
                        return current_weight * (place_action_value + proximity_bonus) + 30.0
                    return current_weight * (place_action_value + proximity_bonus)
                else:
                    # If it is not within the range of PLACE, give a negative task function value to guide the robot back to the target position.
                    return -10.0 * dist_to_rec
            else:
                return -100.0
        else:
            print(f"Warning: Unknown manipulation phase {current_phase}")
            return -0.1 * np.linalg.norm(x_i[0:2])**2

    @staticmethod
    def manipulation_gradient(x_i, t, i, vars_dict=None):
        """
        Return:
            The gradient of the objective function with respect to the state [dx, dy, dtheta]
        """
        if vars_dict is None:
            vars_dict = ManipulationTask.get_global_vars_dict()
            if vars_dict is None: return np.zeros(3)

        current_phase = vars_dict.get('manipulation_phase', ManipulationPhase.NAV_OBJ)
        is_holding = vars_dict.get('is_holding', False)
        holding_robot_id = vars_dict.get('holding_robot_id', None)
        gradient = np.zeros(3)
        nav_attraction_factor = 10.0

        if is_holding and (current_phase == ManipulationPhase.NAV_OBJ or current_phase == ManipulationPhase.PICK):
            return gradient
        if (current_phase == ManipulationPhase.NAV_REC or current_phase == ManipulationPhase.PLACE):
            if not is_holding or holding_robot_id != i:
                return gradient

        if current_phase == ManipulationPhase.NAV_OBJ:
            target_obj = vars_dict.get('target_object_position')
            if target_obj is None:
                gradient[0:2] = -0.2 * x_i[0:2]
                return gradient
            gradient[0:2] = -nav_attraction_factor * (x_i[0:2] - target_obj)
            target_vector = target_obj - x_i[0:2]
            if np.linalg.norm(target_vector) > 1e-6:
                desired_angle = np.arctan2(target_vector[1], target_vector[0])
                angle_diff = np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
                gradient[2] = -0.6 * angle_diff # The angular gradient strength is temporarily unchanged

        elif current_phase == ManipulationPhase.PICK:
            target_obj = vars_dict.get('target_object_position')
            pick_dist_thresh = vars_dict.get('pick_dist_thresh', 0.3)
            if target_obj is not None:
                dist_to_obj = np.linalg.norm(x_i[0:2] - target_obj)
                if dist_to_obj < pick_dist_thresh:
                    gradient[0:2] = -0.1 * (x_i[0:2] - target_obj)
                    target_vector = target_obj - x_i[0:2]
                    if np.linalg.norm(target_vector) > 1e-6:
                        desired_angle = np.arctan2(target_vector[1], target_vector[0])
                        angle_diff = np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
                        gradient[2] = -0.3 * angle_diff
                else:
                    gradient[0:2] = -nav_attraction_factor * (x_i[0:2] - target_obj)
                    target_vector = target_obj - x_i[0:2]
                    if np.linalg.norm(target_vector) > 1e-6:
                        desired_angle = np.arctan2(target_vector[1], target_vector[0])
                        angle_diff = np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
                        gradient[2] = -0.6 * angle_diff

        elif current_phase == ManipulationPhase.NAV_REC:
            target_pos = vars_dict.get('target_receptacle_position')
            if target_pos is None:
                gradient[0:2] = -0.2 * x_i[0:2]
                return gradient
            gradient[0:2] = -nav_attraction_factor * (x_i[0:2] - target_pos)
            target_vector = target_pos - x_i[0:2]
            if np.linalg.norm(target_vector) > 1e-6:
                desired_angle = np.arctan2(target_vector[1], target_vector[0])
                angle_diff = np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
                gradient[2] = -0.6 * angle_diff

        elif current_phase == ManipulationPhase.PLACE:
            target_pos = vars_dict.get('target_receptacle_position')
            place_dist_thresh = vars_dict.get('place_dist_thresh', 0.3)
            if target_pos is not None:
                dist_to_rec = np.linalg.norm(x_i[0:2] - target_pos)
                if dist_to_rec < place_dist_thresh:
                    gradient[0:2] = -0.1 * (x_i[0:2] - target_pos)
                    target_vector = target_pos - x_i[0:2]
                    if np.linalg.norm(target_vector) > 1e-6:
                        desired_angle = np.arctan2(target_vector[1], target_vector[0])
                        angle_diff = np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
                        gradient[2] = -0.3 * angle_diff
                else:
                    gradient[0:2] = -nav_attraction_factor * (x_i[0:2] - target_pos)
                    target_vector = target_pos - x_i[0:2]
                    if np.linalg.norm(target_vector) > 1e-6:
                        desired_angle = np.arctan2(target_vector[1], target_vector[0])
                        angle_diff = np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
                        gradient[2] = -0.6 * angle_diff

        return gradient

    @staticmethod
    def manipulation_time_derivative(x_i, t, i, vars_dict=None):
        return 0 

    @staticmethod
    def apply_motion_control(x_i, t, i, vars_dict, dt):
        if vars_dict is None:
            vars_dict = ManipulationTask.get_global_vars_dict()
            if vars_dict is None:
                return x_i
        current_phase = vars_dict.get('manipulation_phase', ManipulationPhase.NAV_OBJ)
        is_holding = vars_dict.get('is_holding', False)
        holding_robot_id = vars_dict.get('holding_robot_id', None)

        # If other robots are carrying and in NAV_REC or PLACE, the current robot does not move
        if is_holding and holding_robot_id != i and (current_phase == ManipulationPhase.NAV_REC or current_phase == ManipulationPhase.PLACE):
            return x_i

        current_pos = x_i[0:2]
        current_theta = x_i[2]

        # --- Unified navigation control logic ---
        target_pos = None
        if current_phase == ManipulationPhase.NAV_OBJ:
            target_pos = vars_dict.get('target_object_position')
        elif current_phase == ManipulationPhase.NAV_REC:
            target_pos = vars_dict.get('target_receptacle_position')

        if target_pos is not None and (current_phase == ManipulationPhase.NAV_OBJ or current_phase == ManipulationPhase.NAV_REC):
            # control parameters
            Kp_linear = 0.5  # Linear speed proportional gain (a little faster)
            Kp_angular = 1.0 # Angular velocity proportional gain (a little faster)
            u_max_list = vars_dict.get('u_max', [0.5, 2.5]) # Get speed limit from global
            max_v = u_max_list[0] # Maximum linear speed
            max_omega = u_max_list[1] # maximum angular velocity
            stop_dist = 0.05  # Stop distance threshold (very close)
            angle_stop_threshold = 0.05 # angle stop threshold (radian)

            # Calculation error
            vec_to_target = target_pos - current_pos
            dist_to_target = np.linalg.norm(vec_to_target)

            # Calculate target angle and heading difference (Make sure vec_to_target is non-zero)
            desired_theta = current_theta # Keep current angle by default
            if dist_to_target > 1e-6:
                desired_theta = np.arctan2(vec_to_target[1], vec_to_target[0])
            angle_diff = np.arctan2(np.sin(desired_theta - current_theta), np.cos(desired_theta - current_theta)) # desired - current

            # If you are very close to the target
            if dist_to_target < stop_dist:
                forward_speed = 0 # Stop moving forward
                # Pcontrolangular velocity to align
                angular_speed = Kp_angular * angle_diff
                # If the angle error is also small, stop rotating
                if abs(angle_diff) < angle_stop_threshold:
                     angular_speed = 0
                # Limit angular velocity
                angular_speed = np.clip(angular_speed, -max_omega, max_omega)

            # --- If the distance is far, control the linear speed and angular speed at the same time ---
            else:
                # Pcontrollinear speed
                forward_speed = Kp_linear * dist_to_target
                # Reduce linear speed based on angular error
                angle_factor = max(0.0, 1.0 - abs(angle_diff) / (np.pi/2)) # heading differenceThe bigger it is, the slower it is
                forward_speed *= angle_factor
                # Limit line speed
                forward_speed = np.clip(forward_speed, 0, max_v)

                # PcontrolAngular velocity
                angular_speed = Kp_angular * angle_diff
                # Limit angular velocity
                angular_speed = np.clip(angular_speed, -max_omega, max_omega)

            # update status
            new_pos_x = current_pos[0] + forward_speed * np.cos(current_theta) * dt
            new_pos_y = current_pos[1] + forward_speed * np.sin(current_theta) * dt
            new_theta = current_theta + angular_speed * dt
            # The angle is limited to [0, 2*pi)
            new_theta = np.mod(new_theta, 2 * np.pi)

            return np.array([new_pos_x, new_pos_y, new_theta])

        elif current_phase == ManipulationPhase.PICK:
            # PICK The stage is usually a momentary action or external control, here it remains motionless
            return x_i

        elif current_phase == ManipulationPhase.PLACE:
            # PLACE The stage is usually a momentary action or external control, here it remains motionless
            return x_i

        # Remain motionless during other unknown stages or when there is no target
        return x_i

    def __init__(self):
        self.manipulation_phase = ManipulationPhase.NAV_OBJ  # Initialization manipulation phase
        self.is_holding = False  # Initialize is_holding state
        self.manipulation_phase_timer = 0  # Initialize the manipulation phase timer
        self.manipulation_action_fixed_duration = 1.0  # Assume a fixed duration of 1 second

    def update_phase(self, new_phase):
        self.manipulation_phase = new_phase
        self.manipulation_phase_timer = 0  # Reset the manipulation phase timer

    def update_timer(self, dt):
        self.manipulation_phase_timer += dt

    def update_is_holding(self):
        # Update is_holding status at the end of the PICK phase
        if self.manipulation_phase == ManipulationPhase.PICK:
            if self.manipulation_phase_timer >= self.manipulation_action_fixed_duration:
                self.is_holding = True  # Update status after successful fetching

        # Update the is_holding status at the end of the PLACE phase
        if self.manipulation_phase == ManipulationPhase.PLACE:
            if self.manipulation_phase_timer >= self.manipulation_action_fixed_duration:
                self.is_holding = False  # Update status after successful placement 