import numpy as np
from functools import partial

class AdaptiveTaskOptimizer:
    """
    Adaptive task optimizer
    Dynamically adjust task function params for robot task assignment
    """
    
    def __init__(self, scenario_params, global_vars_manager=None):
        self.scenario_params = scenario_params
        self.global_vars = global_vars_manager
        self.original_tasks = self._backup_original_tasks()
        self.optimization_history = []
        self.adaptation_frequency = 10  # update every 10 iterations
        self.current_iter = 0
        
    def _backup_original_tasks(self):
        """Backup original task functions for restore"""
        original = {}
        for i, task in enumerate(self.scenario_params['tasks']):
            original[i] = {
                'function': task['function'],
                'gradient': task['gradient'],
                'time_derivative': task['time_derivative']
            }
        return original
    
    def optimize_tasks(self, x, t, task_assignment, environment_state=None):
        """
        Optimize task functions from current state
        
        Args:
            x: robot state vector
            t: current time
            task_assignment: current task assignment result
            environment_state: optional environment state info
        """
        self.current_iter += 1
        
        # Optimize only at configured frequency
        if self.current_iter % self.adaptation_frequency != 0:
            return False
            
        # Extract robot positions
        robot_positions = x.copy()
        
        # # Optimize transport task
        if 1 in task_assignment:  # task 1 is transport
            transport_robots = [i for i, task in enumerate(task_assignment) if task == 1]
            self._optimize_transport(robot_positions, t, transport_robots)
            
        #Optimize coveragecontrolTask
        if 2 in task_assignment:  #Task 2 is to covercontrol
            coverage_robots = [i for i, task in enumerate(task_assignment) if task == 2]
            self._optimize_coverage(robot_positions, t, coverage_robots, environment_state)
        
        print(f"Task function params checked at t={t:.1f}s  check triggered (actual update depends on internal logic)") #Prompt check is triggered
        return True
        
    def _optimize_transport(self, robot_positions, t, transport_robots):
        """
        Optimize transport taskfunction(Revised version)

        Args:
            robot_positions (np.ndarray): state array for all robots (e.g., shape [3, num_robots])
            t (float): current time
            transport_robots (list):Index list of robots involved in transportation tasks
        """
        #Get relevant information in global variables
        p_transport = self.global_vars.p_transport_t # transport target at current time
        p_goal = self.global_vars.p_goal       # final goal
        p_start = self.global_vars.p_start     # start point

        # Compute progress factor for current scene
        #Notice: p_transportyesmovementof,It might be more appropriate to use it to calculate progress
        progress_factor = self._calculate_progress_factor(p_transport, p_start, p_goal)

        # Adjust weights by progress 
        distance_weight = 1.0
        # Progress weight increases near goal (affects reward)
        progress_weight = 0.5 + 0.5 * progress_factor
        # Coordination weight high early, decreases near goal
        coordination_weight = 0.3 * (1 - progress_factor)
        # ideal inter-robot spacing
        ideal_dist = 0.3

        # --- Optimized transport task function (squared distance) ---
        def optimized_transport_function(x_i, t, robot_idx, vars_dict=None):
            """
            Compute optimized transport utility value.
            Maximize this utility (minimize cost).
            Use squared distance to match original transport_function.
            """
            if vars_dict is None:
                # In production, ensure latest global vars are available
                vars_dict = self.global_vars.get_all_vars()
            #getcurrent timetransport point(may change over time)
            p_t = vars_dict.get('p_transport_t', p_transport)

            # 1. Core distance cost (negative squared distance, weighted)
            #    Minimize distance to transport point; utility is negative cost
            distance_utility = -distance_weight * np.linalg.norm(x_i[:2] - p_t)**2

            # 2. Coordination cost (penalize deviation from ideal spacing)
            #    Minimize coordination cost; utility is negative cost
            coordination_cost = 0
            if len(transport_robots) > 1:
                for j in transport_robots:
                    if j != robot_idx:
                        # Get current positions of other transport robots
                        # Note: pass latest robot_positions here
                        other_pos = robot_positions[:2, j]
                        dist = np.linalg.norm(x_i[:2] - other_pos)
                        # Use squared difference for smoother gradient
                        coordination_cost += (dist - ideal_dist)**2
            coordination_utility = -coordination_weight * coordination_cost

            # 3. Progress reward (encourage transport point toward final goal)
            #    Note: reward does not depend on x_i; no gradient contribution
            #    Maximize reward; positive utility
            #    Negative distance p_t to p_goal may represent remaining distance
            progress_reward_utility = -progress_weight * np.linalg.norm(p_t - p_goal)
            # Or keep original form as positive incentive:
            # progress_reward_utility = progress_weight * some_positive_measure

            # Total utility = distance + coordination + progress reward
            #We want to maximize this total utility value
            total_utility = distance_utility + coordination_utility #+ progress_reward_utility
            # Note: original returned -(cost + cost - reward). If progress_reward is positive,
            # and cost is positive, maximizing -cost - cost + reward is reasonable.
            # distance_utility and coordination_utility are already negative cost (utility).
            # If progress_reward_utility is positive utility, add directly.
            # If progress_reward is distance-to-goal cost, use negative sign.
            # progress_reward_utility commented out; no gradient effect; meaning depends on goal.

            return total_utility # Return total utility

        # --- Optimized gradient (squared distance and coordination) ---
        def optimized_transport_gradient(x_i, t, robot_idx, vars_dict=None):
            """
            Compute gradient of optimized transport function.
            Gradient of optimized_transport_function.
            """
            if vars_dict is None:
                # Ensure latest global vars available
                vars_dict = self.global_vars.get_all_vars()
            #getcurrent timetransport point
            p_t = vars_dict.get('p_transport_t', p_transport)

            # Initialize gradient (x, y, theta)
            gradient = np.zeros_like(x_i)

            # 1. Gradient of distance utility
            #    d/dx_i (-distance_weight * ||x_i[:2] - p_t||^2)
            #    = -distance_weight * 2 * (x_i[:2] - p_t)
            #    Matches original transport_gradient form
            vec_to_target = x_i[:2] - p_t
            gradient[:2] += -distance_weight * 2 * vec_to_target

            # 2. Gradient of coordination utility
            #    d/dx_i (-coordination_weight * sum((||x_i[:2]-other_j|| - ideal_dist)^2))
            #    = -coordination_weight * sum(2 * (||x_i[:2]-other_j|| - ideal_dist) * d(||x_i[:2]-other_j||)/dx_i)
            #    d(||x_i[:2]-other_j||)/dx_i = (x_i[:2] - other_j) / ||x_i[:2]-other_j||
            if len(transport_robots) > 1:
                for j in transport_robots:
                    if j != robot_idx:
                        other_pos = robot_positions[:2, j]
                        vec_to_other = x_i[:2] - other_pos
                        dist = np.linalg.norm(vec_to_other)
                        # Add epsilon to avoid division by zero
                        dist_safe = dist + 1e-6
                        # Compute gradient contribution
                        grad_coord_j = -coordination_weight * 2 * (dist - ideal_dist) * vec_to_other / dist_safe
                        gradient[:2] += grad_coord_j

            # 3. Gradient of progress reward utility
            #    progress_reward_utility independent of x_i; gradient is 0

            # gradient[2] Keep 0; tasks usually care about xy position only
            gradient[2] = 0

            # Return total utility gradient
            return gradient

        # Update task function and gradient
        # Ensure scenario_params['tasks'][0] is the transport task to update
        self.scenario_params['tasks'][0]['function'] = optimized_transport_function
        # self.scenario_params['tasks'][0]['gradient'] = optimized_transport_gradient

        # Record optimization history (remain unchanged)
        self.optimization_history.append({
            'time': t,
            'task_type': 'transport',
            'weights': {
                'distance': distance_weight,
                'progress': progress_weight, # progress_weight effect depends on progress_reward_utility definition
                'coordination': coordination_weight
            },
            'progress_factor': progress_factor # may record progress factor itself
        })

        print(f"Time {t}: Optimized transport task. Progress: {progress_factor:.2f}, Weights(d,p,c): ({distance_weight:.2f}, {progress_weight:.2f}, {coordination_weight:.2f})")

    
    def _optimize_coverage(self, robot_positions, t, coverage_robots, environment_state=None):
        """Optimize coveragecontrolTask function"""
        # Get environment info and POI
        poi = self.global_vars.poi
        mud_area = (self.global_vars.x_mud, self.global_vars.y_mud) if hasattr(self.global_vars, 'x_mud') else None
        
        # Compute coverage-related parameters
        coverage_factor = 1.0
        uniformity_factor = 0.7
        
        # Time factor - adjust weights over time
        if hasattr(self.global_vars, 't_start') and t > self.global_vars.t_start:
            elapsed_time = t - self.global_vars.t_start
            time_factor = min(1.0, elapsed_time / self.global_vars.delta_t if hasattr(self.global_vars, 'delta_t') else 0.5)
            coverage_factor = 1.0 + 0.5 * time_factor
            uniformity_factor = 0.7 * (1 - 0.3 * time_factor)
        
        # Environment factor - adjust coverage if mud regions exist
        mud_avoidance_factor = 0.0
        if mud_area is not None:
            mud_avoidance_factor = 0.3
        
        #Create optimized coveragecontrolTask function
        def optimized_coverage_function(x_i, t, robot_idx, vars_dict=None):
            """
            Calculate the optimized function value for the coverage control task.
            
            Parameters:
                x_i: Robot state
                t: Current time
                robot_idx: Robot index
                vars_dict: Optional global variable dictionary
            """
            if vars_dict is None:
                # Try to get from self.global_vars if it exists
                if hasattr(globals().get('self', None), 'global_vars') and globals().get('self').global_vars is not None:
                    vars_dict = globals().get('self').global_vars.get_all_vars()
                else:
                    # If global_vars is not available, use the static method from CoverageControl
                    from task_utils.coverage_module import CoverageControl
                    vars_dict = CoverageControl.get_global_vars_dict()
                    if vars_dict is None:
                        print("Warning: Unable to access global_vars to get vars_dict.")
                        vars_dict = {}  # Provide an empty dictionary to avoid subsequent errors

            # Get necessary information
            current_pos = x_i[:2]
            current_angle = x_i[2] if len(x_i) > 2 else 0
            
            # Get G (Voronoi Centroids) and poi
            G = vars_dict.get('G', None)
            poi = vars_dict.get('poi', np.array([0, 0]))
            
            # Coverage, uniformity, and mud avoidance factors
            coverage_factor = vars_dict.get('coverage_factor', 1.0)
            uniformity_factor = vars_dict.get('uniformity_factor', 0.5)
            mud_avoidance_factor = vars_dict.get('mud_avoidance_factor', 1.0)
            
            # 1. Centroid distance cost (Voronoi-based)
            centroid_distance_cost = 0.0
            if G is not None and isinstance(G, np.ndarray) and G.ndim == 2 and G.shape[0] == 2:
                if 0 <= robot_idx < G.shape[1]:
                    centroid = G[:, robot_idx]  # Get centroid [cx, cy] for robot_idx
                    dist_sq = np.sum((current_pos - centroid)**2)
                    centroid_distance_cost = dist_sq * coverage_factor
                else:
                    print(f"Warning (coverage cost): robot_idx {robot_idx} out of range for G (columns: {G.shape[1]}).")
                    centroid_distance_cost = coverage_factor * np.sum((current_pos - poi)**2)
            else:
                print(f"Warning (coverage cost): G not available or has unexpected format.")
                centroid_distance_cost = coverage_factor * np.sum((current_pos - poi)**2)
            
            # 2. Orientation towards POI cost
            angle_to_poi = np.arctan2(poi[1] - current_pos[1], poi[0] - current_pos[0])
            angle_difference_cost = (current_angle - angle_to_poi)**2
            
            # 3. Uniformity cost
            uniformity_cost = 0.0
            coverage_robots = vars_dict.get('coverage_robots', [robot_idx])  # Default to just this robot
            all_robot_states = vars_dict.get('x', None)
            
            if all_robot_states is not None and isinstance(all_robot_states, np.ndarray):
                if all_robot_states.ndim == 2 and all_robot_states.shape[0] >= 2:
                    for j in coverage_robots:
                        if j != robot_idx and 0 <= j < all_robot_states.shape[1]:
                            other_pos = all_robot_states[:2, j]
                            dist = np.linalg.norm(current_pos - other_pos)
                            ideal_dist = 0.5  # Ideal distribution distance
                            uniformity_cost += uniformity_factor * (dist - ideal_dist)**2
            
            # 4. Mud penalty
            mud_penalty = 0.0
            mud_map_info = vars_dict.get('mud_map', None)
            if mud_map_info is not None:
                try:
                    x_mud = mud_map_info['x_coords']
                    y_mud = mud_map_info['y_coords']
                    min_dist_to_mud = float('inf')
                    for i in range(len(x_mud)):
                        mud_point = np.array([x_mud[i], y_mud[i]])
                        dist = np.linalg.norm(current_pos - mud_point)
                        min_dist_to_mud = min(min_dist_to_mud, dist)
                    
                    mud_threshold = 0.3
                    if min_dist_to_mud < mud_threshold:
                        mud_penalty = mud_avoidance_factor * (mud_threshold - min_dist_to_mud)
                except (KeyError, TypeError, IndexError) as e:
                    print(f"Warning (mud penalty): Error processing mud information: {e}")
            
            # Total cost (negative for maximization)
            total_cost = -(centroid_distance_cost + angle_difference_cost + uniformity_cost + mud_penalty)
            
            return float(total_cost)
        #Create optimized coveragecontroltask gradient
        def optimized_coverage_gradient(x_i, t, robot_idx, vars_dict=None):
            """
            Calculate the gradient of the optimized coverage control task.
            
            Parameters:
                x_i: Robot state
                t: Current time
                robot_idx: Robot index
                vars_dict: Optional global variable dictionary
            """
            if vars_dict is None:
                # Similar initialization as in the main function
                if hasattr(globals().get('self', None), 'global_vars') and globals().get('self').global_vars is not None:
                    vars_dict = globals().get('self').global_vars.get_all_vars()
                else:
                    from task_utils.coverage_module import CoverageControl
                    vars_dict = CoverageControl.get_global_vars_dict()
                    if vars_dict is None:
                        print("Warning: Unable to access global_vars to get vars_dict.")
                        vars_dict = {}
            
            # Get necessary information
            current_pos = x_i[:2]
            current_angle = x_i[2] if len(x_i) > 2 else 0
            
            # Get G and poi
            G = vars_dict.get('G', None)
            poi = vars_dict.get('poi', np.array([0, 0]))
            
            # Factors
            coverage_factor = vars_dict.get('coverage_factor', 1.0)
            uniformity_factor = vars_dict.get('uniformity_factor', 0.5)
            mud_avoidance_factor = vars_dict.get('mud_avoidance_factor', 1.0)
            
            # Initialize gradient vector
            gradient = np.zeros_like(x_i)
            
            # 1. Gradient of centroid distance cost
            if G is not None and isinstance(G, np.ndarray) and G.ndim == 2 and G.shape[0] == 2:
                if 0 <= robot_idx < G.shape[1]:
                    centroid = G[:, robot_idx]
                    gradient[:2] = -2 * coverage_factor * (current_pos - centroid)
                else:
                    gradient[:2] = -2 * coverage_factor * (current_pos - poi)
            else:
                gradient[:2] = -2 * coverage_factor * (current_pos - poi)
            
            # 2. Gradient of orientation towards POI cost
            if len(x_i) > 2:
                angle_to_poi = np.arctan2(poi[1] - current_pos[1], poi[0] - current_pos[0])
                gradient[2] = -2 * (current_angle - angle_to_poi)
                
                # Add position contribution to angle gradient
                dx_to_poi = poi[0] - current_pos[0]
                dy_to_poi = poi[1] - current_pos[1]
                dist_sq = dx_to_poi**2 + dy_to_poi**2
                if dist_sq > 1e-6:  # Avoid division by zero
                    gradient[0] += -2 * (current_angle - angle_to_poi) * (dy_to_poi / dist_sq)
                    gradient[1] += 2 * (current_angle - angle_to_poi) * (dx_to_poi / dist_sq)
            
            # 3. Gradient of uniformity cost
            coverage_robots = vars_dict.get('coverage_robots', [robot_idx])
            all_robot_states = vars_dict.get('x', None)
            
            if all_robot_states is not None and isinstance(all_robot_states, np.ndarray):
                if all_robot_states.ndim == 2 and all_robot_states.shape[0] >= 2:
                    for j in coverage_robots:
                        if j != robot_idx and 0 <= j < all_robot_states.shape[1]:
                            other_pos = all_robot_states[:2, j]
                            diff = current_pos - other_pos
                            dist = np.linalg.norm(diff)
                            if dist > 1e-6:  # Avoid division by zero
                                ideal_dist = 0.5
                                gradient[:2] += -2 * uniformity_factor * (dist - ideal_dist) * diff / dist
            
            # 4. Gradient of mud penalty
            mud_map_info = vars_dict.get('mud_map', None)
            if mud_map_info is not None:
                try:
                    x_mud = mud_map_info['x_coords']
                    y_mud = mud_map_info['y_coords']
                    min_dist = float('inf')
                    nearest_mud_point = None
                    
                    for i in range(len(x_mud)):
                        mud_point = np.array([x_mud[i], y_mud[i]])
                        dist = np.linalg.norm(current_pos - mud_point)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_mud_point = mud_point
                    
                    mud_threshold = 0.3
                    if min_dist < mud_threshold and nearest_mud_point is not None:
                        diff = current_pos - nearest_mud_point
                        if np.linalg.norm(diff) > 1e-6:  # Avoid division by zero
                            norm_diff = diff / np.linalg.norm(diff)
                            gradient[:2] += -mud_avoidance_factor * norm_diff
                except (KeyError, TypeError, IndexError) as e:
                    print(f"Warning (mud gradient): Error processing mud information: {e}")
            
            # Negate gradient for maximization
            return -gradient
        
        # Update task function
        self.scenario_params['tasks'][1]['function'] = optimized_coverage_function
        # self.scenario_params['tasks'][1]['gradient'] = optimized_coverage_gradient
        
        # Record optimization history
        self.optimization_history.append({
            'time': t,
            'task_type': 'coverage',
            'weights': {
                'coverage': coverage_factor,
                'uniformity': uniformity_factor,
                'mud_avoidance': mud_avoidance_factor
            }
        })
    
    def _calculate_progress_factor(self, current_pos, start_pos, goal_pos):
        """Compute transport progress factor (0-1)"""
        total_distance = np.linalg.norm(goal_pos - start_pos)
        if total_distance < 1e-6:
            return 1.0
            
        current_to_start = np.linalg.norm(current_pos - start_pos)
        current_to_goal = np.linalg.norm(current_pos - goal_pos)
        
        # Compute progress via projection
        progress = current_to_start / total_distance
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, progress))
    
    def reset_to_original(self):
        """Reset to original task functions"""
        for i, task in self.original_tasks.items():
            self.scenario_params['tasks'][i]['function'] = task['function']
            self.scenario_params['tasks'][i]['gradient'] = task['gradient']
            self.scenario_params['tasks'][i]['time_derivative'] = task['time_derivative']
        
        return self.scenario_params
    
    def get_optimization_history(self):
        """Get optimization history"""
        return self.optimization_history
