# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

# task_utils/control_module.py
import numpy as np

class ControlConfig:
    
    def __init__(self, n_r=5, n_x=3, n_u=2):
        self.n_r = n_r
        self.n_x = n_x
        self.n_u = n_u
        self.control_params = self._initialize_control_params()
    
    def _initialize_control_params(self):
        #createcontrol parametersdictionary
        control_params = {
            'K': self._create_gain_matrix(),  #gain matrix
            'u_max': np.array([1.0, 1.0]),  #maximumcontrolenter
            'u_min': np.array([-1.0, -1.0]),  #smallestcontrolenter
            'alpha': 0.1,  # controlweight
            'beta': 0.9,  # task weights
            'gamma': 0.5,  #Obstacle weight
            'delta': 0.1,  #convergence threshold
            'max_iter': 100,  #Maximum number of iterations
            'control_inputs': np.zeros((self.n_u, self.n_r))  # controlinput storage
        }
        
        return control_params
    
    def _create_gain_matrix(self):
        K = np.zeros((self.n_u, self.n_x))
        K[0, 0] = 1.0  #position gain
        K[0, 1] = 1.0  #position gain
        K[1, 2] = 1.0  #Angle gain
        return K
    
    def get_control_params(self):
        return self.control_params
    
    def update_control_params(self, param_name, value):
        if param_name in self.control_params:
            self.control_params[param_name] = value
        else:
            print(f"Warning: parameter '{param_name}' does not exist incontrol parametersmiddle")
    
    def get_gain_matrix(self):
        return self.control_params['K']
    
    def get_control_limits(self):
        return self.control_params['u_min'], self.control_params['u_max']
    
    def get_weights(self):
        return {
            'alpha': self.control_params['alpha'],
            'beta': self.control_params['beta'],
            'gamma': self.control_params['gamma']
        }
    
    def get_convergence_params(self):
        return {
            'delta': self.control_params['delta'],
            'max_iter': self.control_params['max_iter']
        }
    
    def get_control_inputs(self):
        return self.control_params['control_inputs']
    
    def update_control_inputs(self, u):
        self.control_params['control_inputs'] = u
    
    def compute_control_input(self, x, x_d, alpha=None):
        K = self.get_gain_matrix()
        u_min, u_max = self.get_control_limits()
        
        e = x_d - x
        u = K @ e
        u = np.clip(u, u_min[:, np.newaxis], u_max[:, np.newaxis])
        if alpha is not None:
            u = u * alpha
        return u 
