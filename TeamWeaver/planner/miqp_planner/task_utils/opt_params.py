import numpy as np

class OptimizationConfig:
    """
    Configuration class for optimization parameters.
    Manages parameters related to the RTA (Runtime Assurance) controller optimization.
    """
    
    def __init__(self, custom_params=None):
        """
        Initialize optimization parameters with default values or custom parameters
        
        Parameters:
            custom_params: Optional dictionary with custom parameter values
        """
        # Default optimization parameters
        self.opt_params = {
            'l': 1e-6,           # Relative weight delta/u in the cost
            'kappa': 1e6,        # Scale between tasks with different priorities
            'delta_max': 1e3,    # Maximum allowable delta
            'n_r_bounds': np.array([[1, 1], [3, 3]]),  # Row i is min and max number of robots for task i
            'gamma': lambda x: 5*x,  # Class K function for task execution
            'u_min': -100.0,     # Input lower bound
            'u_max': 100.0,      # Input upper bound
            'u_weight': 1.0      # Input penalty weight
        }
        
        # Update with custom parameters if provided
        if custom_params:
            self.update_params(custom_params)
    
    def update_params(self, params_dict):
        """
        Update optimization parameters with new values
        
        Parameters:
            params_dict: Dictionary containing parameter keys and new values
        """
        for key, value in params_dict.items():
            if key in self.opt_params:
                self.opt_params[key] = value
            else:
                print(f"Warning: Parameter '{key}' is not recognized and will be ignored.")
    
    def get_opt_params(self):
        """Get the complete optimization parameters dictionary"""
        return self.opt_params
    
    def set_input_bounds(self, u_min, u_max):
        """
        Set the minimum and maximum input bounds
        
        Parameters:
            u_min: Minimum input value
            u_max: Maximum input value
        """
        self.opt_params['u_min'] = u_min
        self.opt_params['u_max'] = u_max
    
    def set_robot_bounds(self, task_idx, min_robots, max_robots):
        """
        Set the bounds for the number of robots assigned to a specific task
        
        Parameters:
            task_idx: Index of the task
            min_robots: Minimum number of robots required
            max_robots: Maximum number of robots allowed
        """
        if task_idx < self.opt_params['n_r_bounds'].shape[0]:
            self.opt_params['n_r_bounds'][task_idx, 0] = min_robots
            self.opt_params['n_r_bounds'][task_idx, 1] = max_robots
        else:
            # Extend the array if needed
            new_bounds = np.array([[min_robots, max_robots]])
            self.opt_params['n_r_bounds'] = np.vstack((self.opt_params['n_r_bounds'], new_bounds))
    
    def set_task_scaling(self, kappa):
        """
        Set the scaling factor between tasks with different priorities
        
        Parameters:
            kappa: Scaling factor
        """
        self.opt_params['kappa'] = kappa
    
    def set_gamma_function(self, gamma_func):
        """
        Set the class K function for task execution
        
        Parameters:
            gamma_func: A callable function that takes one parameter and returns a value
        """
        if callable(gamma_func):
            self.opt_params['gamma'] = gamma_func
        else:
            print("Error: gamma_func must be a callable function")