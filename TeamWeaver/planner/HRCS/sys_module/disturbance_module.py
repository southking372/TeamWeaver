# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

# task_utils/disturbance_module.py
import numpy as np
from matplotlib.path import Path

#Disturbance category
class DisturbanceConfig:
    def __init__(self, global_vars):
        self.global_vars = global_vars
        
    def get_global_vars_dict(self):
        if hasattr(self.global_vars, 'get_all_vars'):
            return self.global_vars.get_all_vars()
        
        vars_dict = {}
        if hasattr(self.global_vars, 'get_var'):
            vars_dict = {
                'robot_exo_dist': self.global_vars.get_var('robot_exo_dist', 4),
                'task_exo_dist': self.global_vars.get_var('task_exo_dist', 2),
                'x_mud': self.global_vars.get_var('x_mud'),
                'y_mud': self.global_vars.get_var('y_mud'),
                't_endogenous': self.global_vars.get_var('t_endogenous', 15)
            }
        else:
            vars_dict = {
                'robot_exo_dist': getattr(self.global_vars, 'robot_exo_dist', 4),
                'task_exo_dist': getattr(self.global_vars, 'task_exo_dist', 2),
                'x_mud': getattr(self.global_vars, 'x_mud', None),
                'y_mud': getattr(self.global_vars, 'y_mud', None),
                't_endogenous': getattr(self.global_vars, 't_endogenous', 15)
            }
        
        return vars_dict
    
    def check_exogenous_disturbance(self, x, alpha, i, vars_dict=None):
        if vars_dict is None:
            vars_dict = self.get_global_vars_dict()
            
        if vars_dict is not None:
            robot_exo_dist_val = vars_dict.get('robot_exo_dist')
            task_exo_dist_val = vars_dict.get('task_exo_dist')
            x_mud_val = vars_dict.get('x_mud')
            y_mud_val = vars_dict.get('y_mud')
        else:
            robot_exo_dist_val = getattr(self.global_vars, 'robot_exo_dist', 4)
            task_exo_dist_val = getattr(self.global_vars, 'task_exo_dist', 2)
            x_mud_val = getattr(self.global_vars, 'x_mud', None)
            y_mud_val = getattr(self.global_vars, 'y_mud', None)
        
        #Check if the point is within the mud area
        in_mud = False
        if x_mud_val is not None and y_mud_val is not None:
            path = Path(np.column_stack((x_mud_val, y_mud_val)))
            in_mud = path.contains_point((x[0, i], x[1, i]))
        
        return (i == robot_exo_dist_val - 1) or (alpha[task_exo_dist_val - 1, i] > 0) or in_mud 
        #I think the three conditions here should be independent of each other. Whether the robot is disturbed, the task it is performing is disturbed, or the robot gets stuck in the mud, it will all have an impact on the robot.
