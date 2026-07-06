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
        
        # Optimize transport task
        if 1 in task_assignment:  # task 1 is transport
            transport_robots = [i for i, task in enumerate(task_assignment) if task == 1]
            self._optimize_transport_task(robot_positions, t, transport_robots)
            
        # Optimize coverage control tasks
        if 2 in task_assignment:  # Task 2 is to override control
            coverage_robots = [i for i, task in enumerate(task_assignment) if task == 2]
            self._optimize_coverage_task(robot_positions, t, coverage_robots, environment_state)
        
        print(f"Task function params checked at t={t:.1f}s  check triggered (actual update depends on internal logic)") # Prompt check is triggered
        return True
    
    def _optimize_transport_task(self, robot_positions, t, transport_robots):
        """Optimize transport task"""
        # Get context info
        context = self._get_transport_context(t, transport_robots)
        
        # Compute weight factors
        weights = self._calculate_transport_weights(context)
        
        # Update task function and gradient
        optimized_function = self._create_optimized_transport_function(robot_positions, weights, context, transport_robots)
        optimized_gradient = self._create_optimized_transport_gradient(robot_positions, weights, context, transport_robots)
        
        # Apply optimized function and gradient
        self.scenario_params['tasks'][0]['function'] = optimized_function
        # self.scenario_params['tasks'][0]['gradient'] = optimized_gradient
        
        # Record optimization history
        self._record_transport_optimization(t, weights, context)
    
    def _get_transport_context(self, t, transport_robots):
        """Get contextual information for transportation tasks"""
        context = {
            'p_transport': self.global_vars.p_transport_t,  # transport target at current time
            'p_goal': self.global_vars.p_goal,              # final goal
            'p_start': self.global_vars.p_start,            # start point
            'transport_robots': transport_robots,           # Robots involved in transportation
            'time': t                                       # current time
        }
        
        # Calculate progress factor
        context['progress_factor'] = self._calculate_progress_factor(
            context['p_transport'], context['p_start'], context['p_goal'])
        
        return context
    
    def _calculate_transport_weights(self, context):
        """Calculate weighting factors for transportation tasks"""
        progress_factor = context['progress_factor']
        
        weights = {
            'distance': 1.0,                                    # distance weight
            'progress': 0.5 + 0.5 * progress_factor,            # Progress weight, increasing as the target is approached
            'coordination': 0.3 * (1 - progress_factor)         # Coordination weight, which decreases as the target is approached
        }
        
        return weights
    
    def _create_optimized_transport_function(self, robot_positions, weights, context, transport_robots):
        """Create an optimized transportation task function"""
        p_transport = context['p_transport']
        p_goal = context['p_goal']
        ideal_dist = 0.3  # ideal inter-robot spacing
        
        def optimized_function(x_i, t, robot_idx, vars_dict=None):
            """Optimized transportation task function"""
            if vars_dict is None:
                vars_dict = self.global_vars.get_all_vars()
            
            # Get the shipping point at current time
            p_t = vars_dict.get('p_transport_t', p_transport)
            
            # 1. Core Goal: Distance Utility
            distance_utility = -weights['distance'] * np.linalg.norm(x_i[:2] - p_t)**2
            
            # 2. Collaboration item: Coordination utility
            coordination_utility = 0
            if len(transport_robots) > 1:
                for j in transport_robots:
                    if j != robot_idx:
                        other_pos = robot_positions[:2, j]
                        dist = np.linalg.norm(x_i[:2] - other_pos)
                        coordination_utility += -weights['coordination'] * (dist - ideal_dist)**2
            
            # 3. Progress item: progress reward
            # progress_utility = -weights['progress'] * np.linalg.norm(p_t - p_goal)
            
            # total utility
            total_utility = distance_utility + coordination_utility  # + progress_utility
            
            return total_utility
        
        return optimized_function
    
    def _create_optimized_transport_gradient(self, robot_positions, weights, context, transport_robots):
        """Create an optimized gradient function for transportation tasks"""
        p_transport = context['p_transport']
        ideal_dist = 0.3
        
        def optimized_gradient(x_i, t, robot_idx, vars_dict=None):
            """Optimized transportation task gradient"""
            if vars_dict is None:
                vars_dict = self.global_vars.get_all_vars()
            
            # Get the shipping point at current time
            p_t = vars_dict.get('p_transport_t', p_transport)
            
            # Initialize gradient
            gradient = np.zeros_like(x_i)
            
            # 1. Gradient of distance utility
            vec_to_target = x_i[:2] - p_t
            gradient[:2] += -weights['distance'] * 2 * vec_to_target
            
            # 2. Gradient of coordination utility
            if len(transport_robots) > 1:
                for j in transport_robots:
                    if j != robot_idx:
                        other_pos = robot_positions[:2, j]
                        vec_to_other = x_i[:2] - other_pos
                        dist = np.linalg.norm(vec_to_other)
                        dist_safe = dist + 1e-6  # Prevent division by zero
                        grad_coord_j = -weights['coordination'] * 2 * (dist - ideal_dist) * vec_to_other / dist_safe
                        gradient[:2] += grad_coord_j
            
            # The third component of the gradient(rotation angle)remain at 0
            gradient[2] = 0
            
            return gradient
        
        return optimized_gradient
    
    def _record_transport_optimization(self, t, weights, context):
        """Record the optimization history of transportation tasks"""
        self.optimization_history.append({
            'time': t,
            'task_type': 'transport',
            'weights': weights,
            'progress_factor': context['progress_factor']
        })
        
        print(f"Time {t}: Optimized transport task. Progress: {context['progress_factor']:.2f}, "
              f"Weights(d,p,c): ({weights['distance']:.2f}, {weights['progress']:.2f}, {weights['coordination']:.2f})")
    
    def _optimize_coverage_task(self, robot_positions, t, coverage_robots, environment_state=None):
        """Optimize coverage control tasks"""
        # Get context info
        context = self._get_coverage_context(t, coverage_robots, environment_state)
        
        # Compute weight factors
        weights = self._calculate_coverage_weights(context)
        
        # Update task function and gradient
        optimized_function = self._create_optimized_coverage_function(weights, context)
        optimized_gradient = self._create_optimized_coverage_gradient(weights, context)
        
        # Apply optimized function and gradient
        self.scenario_params['tasks'][1]['function'] = optimized_function
        # self.scenario_params['tasks'][1]['gradient'] = optimized_gradient
        
        # Record optimization history
        self._record_coverage_optimization(t, weights)
    
    def _get_coverage_context(self, t, coverage_robots, environment_state=None):
        """Get context information covering the control task"""
        context = {
            'poi': self.global_vars.poi,                     # points of interest
            'coverage_robots': coverage_robots,              # Robots participating in overlay control
            'time': t                                        # current time
        }
        
        # Add mud area information
        if hasattr(self.global_vars, 'x_mud'):
            context['mud_area'] = {
                'x_mud': self.global_vars.x_mud,
                'y_mud': self.global_vars.y_mud
            }
        
        # Add time related information
        if hasattr(self.global_vars, 't_start'):
            context['t_start'] = self.global_vars.t_start
            if hasattr(self.global_vars, 'delta_t'):
                context['delta_t'] = self.global_vars.delta_t
                if t > context['t_start']:
                    context['elapsed_time'] = t - context['t_start']
                    context['time_factor'] = min(1.0, context['elapsed_time'] / context['delta_t'])
        
        return context
    
    def _calculate_coverage_weights(self, context):
        """Calculate the weight factor covering the control task"""
        weights = {
            'coverage': 1.0,       # Override weight
            'uniformity': 0.7,     # Uniformity weight
            'mud_avoidance': 0.0   # Mud floor avoidance weights
        }
        
        # Adjust weights based on time
        if 'time_factor' in context:
            time_factor = context['time_factor']
            weights['coverage'] = 1.0 + 0.5 * time_factor
            weights['uniformity'] = 0.7 * (1 - 0.3 * time_factor)
        
        # Adjust weights according to environment
        if 'mud_area' in context:
            weights['mud_avoidance'] = 0.3
        
        return weights
    
    def _create_optimized_coverage_function(self, weights, context):
        """Create an optimized coverage control task function"""
        def optimized_function(x_i, t, robot_idx, vars_dict=None):
            """Optimized coverage control function"""
            if vars_dict is None:
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
            
            # 1. Core Goal: Center Distance Cost
            centroid_distance_cost = 0.0
            if G is not None and isinstance(G, np.ndarray) and G.ndim == 2 and G.shape[0] == 2:
                if 0 <= robot_idx < G.shape[1]:
                    centroid = G[:, robot_idx]
                    centroid_distance_cost = np.sum((current_pos - centroid)**2) * weights['coverage']
                else:
                    centroid_distance_cost = weights['coverage'] * np.sum((current_pos - poi)**2)
            else:
                centroid_distance_cost = weights['coverage'] * np.sum((current_pos - poi)**2)
            
            # 2. Core Goal: Toward Cost
            angle_to_poi = np.arctan2(poi[1] - current_pos[1], poi[0] - current_pos[0])
            angle_difference_cost = (current_angle - angle_to_poi)**2
            
            # 3. Collaboration Item: Uniformity Cost
            uniformity_cost = 0.0
            coverage_robots = vars_dict.get('coverage_robots', [robot_idx])
            all_robot_states = vars_dict.get('x', None)
            
            if all_robot_states is not None and isinstance(all_robot_states, np.ndarray):
                if all_robot_states.ndim == 2 and all_robot_states.shape[0] >= 2:
                    for j in coverage_robots:
                        if j != robot_idx and 0 <= j < all_robot_states.shape[1]:
                            other_pos = all_robot_states[:2, j]
                            dist = np.linalg.norm(current_pos - other_pos)
                            ideal_dist = 0.5
                            uniformity_cost += weights['uniformity'] * (dist - ideal_dist)**2
            
            # 4. Environmental Adaptation: Mud Punishment
            mud_penalty = 0.0
            mud_map_info = vars_dict.get('mud_map', None)
            if mud_map_info is not None and weights['mud_avoidance'] > 0:
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
                        mud_penalty = weights['mud_avoidance'] * (mud_threshold - min_dist_to_mud)
                except (KeyError, TypeError, IndexError) as e:
                    print(f"Warning (mud penalty): Error processing mud information: {e}")
            
            # total cost(Take negative values ​​to facilitate maximization)
            total_cost = -(centroid_distance_cost + angle_difference_cost + uniformity_cost + mud_penalty)
            
            return float(total_cost)
        
        return optimized_function
    
    def _create_optimized_coverage_gradient(self, weights, context):
        """Create an optimized gradient function for the coverage control task"""
        def optimized_gradient(x_i, t, robot_idx, vars_dict=None):
            """Optimized coverage control gradient"""
            if vars_dict is None:
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
            
            # Initialize gradient vector
            gradient = np.zeros_like(x_i)
            
            # 1. Gradient of center distance cost
            if G is not None and isinstance(G, np.ndarray) and G.ndim == 2 and G.shape[0] == 2:
                if 0 <= robot_idx < G.shape[1]:
                    centroid = G[:, robot_idx]
                    gradient[:2] = -2 * weights['coverage'] * (current_pos - centroid)
                else:
                    gradient[:2] = -2 * weights['coverage'] * (current_pos - poi)
            else:
                gradient[:2] = -2 * weights['coverage'] * (current_pos - poi)
            
            # 2. Gradient towards cost
            iflen(x_i) > 2:
                angle_to_poi = np.arctan2(poi[1] - current_pos[1], poi[0] - current_pos[0])
                gradient[2] = -2 * (current_angle - angle_to_poi)
                
                # Contribution of position to angular gradient
                dx_to_poi = poi[0] - current_pos[0]
                dy_to_poi = poi[1] - current_pos[1]
                dist_sq = dx_to_poi**2 + dy_to_poi**2
                if dist_sq > 1e-6:
                    gradient[0] += -2 * (current_angle - angle_to_poi) * (dy_to_poi / dist_sq)
                    gradient[1] += 2 * (current_angle - angle_to_poi) * (dx_to_poi / dist_sq)
            
            # 3. Gradient of Uniformity Cost
            coverage_robots = vars_dict.get('coverage_robots', [robot_idx])
            all_robot_states = vars_dict.get('x', None)
            
            if all_robot_states is not None and isinstance(all_robot_states, np.ndarray):
                if all_robot_states.ndim == 2 and all_robot_states.shape[0] >= 2:
                    for j in coverage_robots:
                        if j != robot_idx and 0 <= j < all_robot_states.shape[1]:
                            other_pos = all_robot_states[:2, j]
                            diff = current_pos - other_pos
                            dist = np.linalg.norm(diff)
                            if dist > 1e-6:
                                ideal_dist = 0.5
                                gradient[:2] += -2 * weights['uniformity'] * (dist - ideal_dist) * diff / dist
            
            # 4. Mud Punishment Gradient
            if weights['mud_avoidance'] > 0:
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
                            if np.linalg.norm(diff) > 1e-6:
                                norm_diff = diff / np.linalg.norm(diff)
                                gradient[:2] += -weights['mud_avoidance'] * norm_diff
                    except (KeyError, TypeError, IndexError) as e:
                        print(f"Warning (mud gradient): Error processing mud information: {e}")
            
            # Take negative values and use them to maximize
            return -gradient
        
        return optimized_gradient
    
    def _record_coverage_optimization(self, t, weights):
        """Record the optimization history of the coverage control task"""
        self.optimization_history.append({
            'time': t,
            'task_type': 'coverage',
            'weights': weights
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