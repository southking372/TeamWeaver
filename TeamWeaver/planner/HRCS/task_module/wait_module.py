# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

# task_utils/wait_module.py
import numpy as np
import time

class WaitTask:
    
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
                elif hasattr(global_vars_manager, 'get_var'):
                    global_vars_dict = {
                        'wait_step_threshold': global_vars_manager.get_var('wait_step_threshold', 5.0),
                        'sim_freq': global_vars_manager.get_var('sim_freq', 1.0),
                        'wait_elapsed_time': global_vars_manager.get_var('wait_elapsed_time', 0.0)
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
    def wait_function(x_i, t, i, vars_dict=None):
        if vars_dict is None:
            vars_dict = WaitTask.get_global_vars_dict()
            
        step_threshold = vars_dict.get('wait_step_threshold', 5.0)
        sim_freq = vars_dict.get('sim_freq', 1.0)
        elapsed_time = vars_dict.get('wait_elapsed_time', 0.0)
        if step_threshold > 0:
            elapsed_time = elapsed_time % step_threshold
        
        wait_complete = elapsed_time >= step_threshold
        return -1.0 if wait_complete else -10.0

    @staticmethod
    def wait_gradient(x_i, t, i, vars_dict=None):
        gradient = np.zeros(3)
        gradient[0:2] = -0.05 * x_i[0:2]
        return gradient

    @staticmethod
    def wait_time_derivative(x_i, t, i, vars_dict=None):
        return 0 

    @staticmethod
    def apply_motion_control(x_i, t, i, vars_dict, dt):
        # For wait tasks, the robot does not move; return current state
        return x_i 