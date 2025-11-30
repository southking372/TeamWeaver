# task_utils/task_module/opt_params_task.py
import numpy as np
from typing import List, Dict, Any

class OptimizationConfigTask:
    
    def __init__(self, n_r=2, n_t=13):
        """
        Initializes optimization parameter configuration.
        The default task bounds are for the 13 base task types.
        """
        self.n_r = n_r
        self.n_t = n_t

        # A mapping from task type name to its default robot bounds
        self.TASK_TYPE_TO_BOUNDS = {
            "Navigate": [0, 2],
            "Explore": [0, 2],
            "Pick": [0, 2],
            "Place": [0, 2],
            "Open": [0, 2],
            "Close": [0, 2],
            "Clean": [0, 1],
            "Fill": [0, 1],
            "Pour": [0, 1],
            "PowerOn": [0, 1],
            "PowerOff": [0, 1],
            "Rearrange": [0, 2],
            "Wait": [0, 2]
        }
        self.opt_params = self._initialize_opt_params()
    
    def reset(self):
        """
        Resets the optimization parameters to their default initial state.
        """
        self.opt_params = self._initialize_opt_params()
        # print("[DEBUG] OptimizationConfigTask reset completed")

    def _initialize_opt_params(self):
        # The 'n_r_bounds' here is a template for the 13 base task types.
        # It will be used by get_instance_robot_bounds to create instance-specific bounds.
        partnr_task_bounds = [self.TASK_TYPE_TO_BOUNDS.get(name, [0, self.n_r]) for name in self.TASK_TYPE_TO_BOUNDS]

        n_r_bounds_template = np.array(partnr_task_bounds[:self.n_t])
        
        opt_params = {
            'l': 1e-6,
            'kappa': 1e6,
            'delta_max': 1e3,
            'n_r_bounds': n_r_bounds_template,
            'gamma': lambda x: 5*x
        }
        return opt_params
    
    def get_instance_robot_bounds(self, task_instances: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generates an n_r_bounds array specifically for the given task instances.

        Args:
            task_instances: A list of task instance dictionaries, where each dict
                            must have a 'task_type' key.

        Returns:
            A NumPy array of shape [num_instances, 2] with the min/max robot bounds
            for each task instance.
        """
        instance_bounds = []
        for instance in task_instances:
            task_type = instance.get('task_type')
            if task_type in self.TASK_TYPE_TO_BOUNDS:
                bounds = self.TASK_TYPE_TO_BOUNDS[task_type]
                # Ensure max bound does not exceed the number of available robots
                bounds[1] = min(bounds[1], self.n_r)
                instance_bounds.append(bounds)
            else:
                # Default for unknown task types
                print(f"[Warning] Unknown task type '{task_type}' in get_instance_robot_bounds. Using default [0, {self.n_r}] bounds.")
                instance_bounds.append([0, self.n_r])
        
        return np.array(instance_bounds, dtype=int)

    def update_robot_bounds(self, task_idx, min_robots, max_robots):
        if 0 <= task_idx < self.n_t and 0 <= min_robots <= max_robots <= self.n_r:
            self.opt_params['n_r_bounds'][task_idx] = [min_robots, max_robots]
        else:
            print(f"Error：update_robot_bounds invalid，task_idx: {task_idx}, min: {min_robots}, max: {max_robots}")
    
    def update_weight_lambda(self, l_value):
        if l_value > 0:
            self.opt_params['l'] = l_value
        else:
            print(f"Error：update_weight_lambda invalid，l_value: {l_value}")
    
    def update_kappa(self, kappa_value):
        if kappa_value > 0:
            self.opt_params['kappa'] = kappa_value
        else:
            print(f"Error：update_kappa invalid，kappa_value: {kappa_value}")
    
    def update_delta_max(self, delta_max_value):
        if delta_max_value > 0:
            self.opt_params['delta_max'] = delta_max_value
        else:
            print(f"Error：update_delta_max invalid，delta_max_value: {delta_max_value}")
    
    def update_gamma_function(self, gamma_function):
        if callable(gamma_function):
            self.opt_params['gamma'] = gamma_function
        else:
            print(f"Error：update_gamma_function invalid，gamma_function: {gamma_function}")
    
    def get_opt_params(self):
        return self.opt_params
    def get_robot_bounds(self):
        return self.opt_params['n_r_bounds']
    def get_weight_lambda(self):
        return self.opt_params['l']
    def get_kappa(self):
        return self.opt_params['kappa']
    def get_delta_max(self):
        return self.opt_params['delta_max']
    def get_gamma_function(self):
        return self.opt_params['gamma']