# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

import numpy as np

class GlobalVarsManager:
    def __init__(self):
        #Initialize all global variables
        self.p_start = np.array([1, -0.6])
        self.p_goal = np.array([-1, 0.6])
        self.t_start = 2
        self.delta_t = 60
        self.p_transport_t = None
        self.poi = np.array([0, 1])
        self.G = None
        self.s = None
        self.x_mud = None
        self.y_mud = None
        self.robot_exo_dist = 4
        self.task_exo_dist = 2
        self.t_endogenous = 15
    
    def get_var(self, var_name):
        """Get the global variable with the specified name"""
        if hasattr(self, var_name):
            return getattr(self, var_name)
        return None
    
    def set_var(self, var_name, value):
        """Sets a global variable with the specified name"""
        setattr(self, var_name, value)
    
    def get_all_vars(self):
        """Get a dictionary of all global variables"""
        return {
            name: value for name, value in self.__dict__.items() 
            if not name.startswith('_') and not callable(value)
        }
