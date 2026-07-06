# task_utils/explore_module.py
import numpy as np

class ExploreTask:
    """
    The implementation class of exploration tasks, the goal is to drive the robot to explore and scan in the environment, Simulate the behavior and measurement of ExploreSkill in partnr-planner
    This behavior is based on the semantic exploration agent(Sem_Exp_Env_Agentwait) -> This part should be aligned with PARTNR
    Function is related to: Navigate to exploration frontiers in the environment(frontiers)
    """

    @staticmethod
    def get_global_vars_dict():
        global_vars_dict = None
        try:
            import sys
            global_vars_manager = None
            for module_name, module in sys.modules.items():
                if hasattr(module, 'global_vars'):
                    global_vars_manager = getattr(module, 'global_vars')
                    break

            if global_vars_manager is not None:
                if hasattr(global_vars_manager, 'get_all_vars'):
                    global_vars_dict = global_vars_manager.get_all_vars()
                    if 'exploration_frontiers' not in global_vars_dict:
                         global_vars_dict['exploration_frontiers'] = []
                elif hasattr(global_vars_manager, 'get_var'):
                     global_vars_dict = {
                         'exploration_frontiers': global_vars_manager.get_var('exploration_frontiers', []),
                         'explored_map': global_vars_manager.get_var('explored_map'),
                         'explore_dist_thresh': global_vars_manager.get_var('explore_dist_thresh', 0.5), # May no longer be used directly, but retained
                         'G': global_vars_manager.get_var('G')
                     }
                else:
                    print("Global vars manager is missing required methods")
                    global_vars_dict = {'exploration_frontiers': []}
            else:
                print("Could not find global_vars_manager")
                global_vars_dict = {'exploration_frontiers': []}
        except Exception as e:
            print(f"Error getting global vars manager: {e}")
            global_vars_dict = {'exploration_frontiers': []}
            
        if 'exploration_frontiers' in global_vars_dict and isinstance(global_vars_dict['exploration_frontiers'], list):
             global_vars_dict['exploration_frontiers'] = np.array(global_vars_dict['exploration_frontiers'])
        if global_vars_dict.get('exploration_frontiers') is None:
            global_vars_dict['exploration_frontiers'] = np.array([])
        return global_vars_dict

    @staticmethod
    def explore_function(x_i, t, i, vars_dict):
        """
        Return:
            Objective function value H
        """
        exploration_frontiers = vars_dict.get('exploration_frontiers')

        if exploration_frontiers is None or exploration_frontiers.shape[0] == 0:
            # print(f"Warning: No exploration frontiers found for ExploreTask.explore_function at t={t}, robot={i}. Assuming exploration complete.")
            # There is no frontier point, and the exploration is considered completed, H=0
            return 0.0

        current_pos = x_i[0:2]
        current_angle = x_i[2]

        # Calculate the square distance from all front points to the current position
        distances_sq = np.sum((exploration_frontiers - current_pos)**2, axis=1)
        nearest_frontier_idx = np.argmin(distances_sq)
        nearest_frontier = exploration_frontiers[nearest_frontier_idx]
        min_dist_sq = distances_sq[nearest_frontier_idx]

        # Calculate position error term (Use distance squared)
        w_pos = 0.5
        pos_error_term = w_pos * min_dist_sq

        target_vector = nearest_frontier - current_pos
        if np.linalg.norm(target_vector) < 1e-6:
            angle_diff = 0.0
        else:
            desired_angle = np.arctan2(target_vector[1], target_vector[0])
            angle_diff = np.arctan2(np.sin(current_angle - desired_angle), np.cos(current_angle - desired_angle))

        w_ori = 0.3  # direction weight (Adjustable)
        angle_error_term = w_ori * angle_diff**2

        H = pos_error_term + angle_error_term

        # Optional: If the distance is very close, it can be considered partially completed and a small negative reward will be given.
        # dist_thresh_sq = vars_dict.get('explore_dist_thresh', 0.5)**2
        # if min_dist_sq < dist_thresh_sq:
        #     H -= 1.0 #  small bonus for reaching near a frontier
        return H

    @staticmethod
    def explore_gradient(x_i, t, i, vars_dict):
        """
        Compute the gradient of the exploration task with respect to robot state x_i (dH/dx)。
        The gradient points in the direction where H increases fastest, encouraging the controller to use negative gradients to drive the robot.
        Return:
            The gradient of the objective function with respect to the state [dH/dx, dH/dy, dH/dtheta]
        """
        exploration_frontiers = vars_dict.get('exploration_frontiers')
        gradient = np.zeros(3)

        if exploration_frontiers is None or exploration_frontiers.shape[0] == 0:
            return gradient

        current_pos = x_i[0:2]
        current_angle = x_i[2]

        # Find the nearest front point (The same logic as in the H function)
        distances_sq = np.sum((exploration_frontiers - current_pos)**2, axis=1)
        nearest_frontier_idx = np.argmin(distances_sq)
        nearest_frontier = exploration_frontiers[nearest_frontier_idx]

        # Calculate position gradient dH/d(pos) = 2 * w_pos * (current_pos - nearest_frontier)
        w_pos = 0.5
        pos_gradient = 2 * w_pos * (current_pos - nearest_frontier)
        gradient[0:2] = pos_gradient

        # Calculate the angular gradient dH/dtheta = 2 * w_ori * angle_diff
        target_vector = nearest_frontier - current_pos
        if np.linalg.norm(target_vector) < 1e-6:
            angle_diff = 0.0
        else:
            desired_angle = np.arctan2(target_vector[1], target_vector[0])
            angle_diff = np.arctan2(np.sin(current_angle - desired_angle), np.cos(current_angle - desired_angle))
        w_ori = 0.3
        angle_gradient = 2 * w_ori * angle_diff
        gradient[2] = angle_gradient

        return gradient

    @staticmethod
    def explore_time_derivative(x_i, t, i, vars_dict):
        """
        Return:
            The derivative of the objective function with respect to time (dH/dt), Here is 0
        """
        return 0.0

    @staticmethod
    def print_debug_info(x_i, t, i, vars_dict, iter_count):
        """
        Print the debugging information of the exploration task, including the distance to the nearest frontier point, the heading difference exclusive sum function value.
        
        Parameters:
            x_i: Robot status [x, y, theta]
            t: current time
            i: Robot Index
            vars_dict: dictionary containing 'exploration_frontiers'
            iter_count: Current iteration count
        """
        if iter_count % 10 != 0:  # Print every 10 iterations to avoid screen swiping
            return
            
        exploration_frontiers = vars_dict.get('exploration_frontiers')
        # Check if the exploration front is valid
        if exploration_frontiers is None or exploration_frontiers.shape[0] == 0:
            print(f"DEBUG: Robot {i} (Task 1) There is no frontier to explore. The exploration mission has been completed.")
            return

        current_pos = x_i[0:2]
        current_angle = x_i[2]
        
        # Calculate the distance to the nearest frontier point
        distances_sq = np.sum((exploration_frontiers - current_pos)**2, axis=1)
        nearest_frontier_idx = np.argmin(distances_sq)
        nearest_frontier = exploration_frontiers[nearest_frontier_idx]
        min_dist = np.sqrt(distances_sq[nearest_frontier_idx])
        
        # Calculate heading difference
        target_vector = nearest_frontier - current_pos
        if np.linalg.norm(target_vector) < 1e-6:
            angle_diff = 0.0
        else:
            desired_angle = np.arctan2(target_vector[1], target_vector[0])
            angle_diff = np.arctan2(np.sin(current_angle - desired_angle), np.cos(current_angle - desired_angle))
        
        # Calculate exploration task function value
        H_value = ExploreTask.explore_function(x_i, t, i, vars_dict)
        explore_dist_thresh = vars_dict.get('explore_dist_thresh', 0.5)
        print(f"DEBUG: Robot {i} (Task 1) Exploring. Nearest frontier: {min_dist:.2f}m (threshold: {explore_dist_thresh}), heading difference: {angle_diff:.2f}rad, function value: {H_value:.2f}")
        if exploration_frontiers.shape[0] > 1:
            print(f"       share {exploration_frontiers.shape[0]} Exploration frontier points to be explored")

    @staticmethod
    def apply_motion_control(x_i, t, i, vars_dict, dt):
        """
        Motion control for application exploration tasks.
        It is divided into two stages: Navigate to the nearest unexplored target point, and perform an exploration action (rotate in place) after arriving.
        Return:
            New Robot States
        """
        # --- Get the required global variables ---
        exploration_targets = vars_dict.get('exploration_targets', [])
        explore_dist_thresh = vars_dict.get('explore_dist_thresh', 0.2)
        exploration_action_timers = vars_dict.get('exploration_action_timers', {})
        exploring_action_info = vars_dict.get('exploring_action_info', {})
        # Note: exploration_action_duration needs to be available from vars_dict, or as a constant
        exploration_action_duration = vars_dict.get('exploration_action_duration', 2.0)
        u_max_list = vars_dict.get('u_max', [0.5, 2.5]) # Get speed limit
        max_v = u_max_list[0]
        max_omega = u_max_list[1]

        current_pos = x_i[0:2]
        current_theta = x_i[2]

        # --- Find the nearest unexplored target ---
        unexplored_targets = [target for target in exploration_targets if not target.get('explored', False)]

        if not unexplored_targets:
            # print(f"Robot {i}: No unexplored targets left.")
            # If there are no unexplored targets, you can choose to stop or perform other actions (such as movement to the Voronoi center)
            # Here we leave it motionless
            return x_i

        nearest_target = None
        min_dist_sq = float('inf')
        for target in unexplored_targets:
            dist_sq = np.sum((current_pos - target['position'])**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_target = target

        if nearest_target is None: # In theory it should not happen unless the list is empty
             return x_i

        target_pos = nearest_target['position']
        target_id = nearest_target['id'] # Get target ID
        dist_to_target = np.sqrt(min_dist_sq)

        # --- Check if the robot is performing exploration actions on this target ---
        is_performing_action = exploring_action_info.get(i) == target_id

        # --- State transition and motion control ---
        if is_performing_action:
            # Stage 2: perform exploratory actions (Spin in place)
            # Check if the timer has completed (This check is theoretically handled by GlobalVarsManager and then the status is updated.)
            # Only actions are performed here
            action_rotation_speed = 1.0 # Explore the rotation speed of an action (rad/s)
            new_theta = current_theta + action_rotation_speed * dt
            new_theta = np.mod(new_theta, 2 * np.pi)
            # print(f"Robot {i} performing explore action at target {target_id}")
            return np.array([current_pos[0], current_pos[1], new_theta])

        elif dist_to_target <= explore_dist_thresh:
            # Stage 1 -> 2 transition: Reach the target and start exploring
            # *important*: The logic to start the timer and update exploring_action_info should be external
            # (For example newTask.py or GlobalVarsManager) Triggered after detecting this state.
            # apply_motion_control It is usually only responsible for calculating the next state.
            # Here we simulate the first step of starting the action: a slight rotation
            print(f"Robot {i} reached target {target_id}, starting exploration action.")
            action_rotation_speed = 1.0
            new_theta = current_theta + action_rotation_speed * dt
            new_theta = np.mod(new_theta, 2 * np.pi)
            # Indicates the start of the action by returning an almost unchanged position and a new angle
            return np.array([current_pos[0], current_pos[1], new_theta])

        else:
            # Stage 1: Navigate to target
            # Use P controller
            Kp_linear = 0.6 # Navigation speed gain
            Kp_angular = 1.2 # Navigation angular velocity gain
            stop_dist = explore_dist_thresh # The stopping distance of navigation is the threshold

            vec_to_target = target_pos - current_pos
            # dist_to_target Calculated

            # Calculate target angle and heading difference
            desired_theta = np.arctan2(vec_to_target[1], vec_to_target[0])
            angle_diff = np.arctan2(np.sin(desired_theta - current_theta), np.cos(desired_theta - current_theta))

            # Pcontrollinear speed
            forward_speed = Kp_linear * dist_to_target
            angle_factor = max(0.0, 1.0 - abs(angle_diff) / (np.pi/2))
            forward_speed *= angle_factor
            forward_speed = np.clip(forward_speed, 0, max_v)

            # PcontrolAngular velocity
            angular_speed = Kp_angular * angle_diff
            angular_speed = np.clip(angular_speed, -max_omega, max_omega)

            # update status
            new_pos_x = current_pos[0] + forward_speed * np.cos(current_theta) * dt
            new_pos_y = current_pos[1] + forward_speed * np.sin(current_theta) * dt
            new_theta = current_theta + angular_speed * dt
            new_theta = np.mod(new_theta, 2 * np.pi)

            # print(f"Robot {i} navigating to explore target {target_id}, dist: {dist_to_target:.2f}")
            return np.array([new_pos_x, new_pos_y, new_theta])
