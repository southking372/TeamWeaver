# task_utils/transport_module.py
import numpy as np
from habitat_llm.planner.miqp_planner.task_utils.tools_util import clamp

class TransportTask:
    """
    Implementation of the transport task functions.
    """
    
    @staticmethod
    def get_global_vars_dict():
        """
        Get global variables dictionary from GlobalVarsManager
        """
        global_vars_dict = None
        try:
            # Get GlobalVarsManager instance
            import sys
            global_vars_manager = None
            
            # Search for GlobalVarsManager instance in all modules
            for module_name, module in sys.modules.items():
                if hasattr(module, 'global_vars'):
                    global_vars_manager = getattr(module, 'global_vars')
                    break
            
            if global_vars_manager is not None:
                global_vars_dict = global_vars_manager.get_all_vars()
            else:
                print("Can not find global_vars_manager")
        
        except Exception as e:
            print(f"Error getting global variable manager: {e}")
            global_vars_dict = None
            
        return global_vars_dict
    
    @staticmethod
    def p_transport(t, vars_dict=None):
        """
        Calculate the position of the transport point at a specified time.
        
        Parameters:
            t: Current time
            vars_dict: Optional global variable dictionary; if None, use the global variable manager
        """
        if vars_dict is None:
            vars_dict = TransportTask.get_global_vars_dict()
            
        if vars_dict is not None:
            p_start_val = vars_dict.get('p_start')
            p_goal_val = vars_dict.get('p_goal')
            t_start_val = vars_dict.get('t_start')
            delta_t_val = vars_dict.get('delta_t')
        else:
            # Fallback to global variables if dictionary not available
            from central import global_vars
            p_start_val = global_vars.p_start
            p_goal_val = global_vars.p_goal
            t_start_val = global_vars.t_start
            delta_t_val = global_vars.delta_t
        
        if t < t_start_val:
            return p_start_val
        elif t > t_start_val + delta_t_val:
            return p_goal_val
        else:
            x = clamp(1 - (t - t_start_val) / delta_t_val, 0, 1) * p_start_val[0] + clamp((t - t_start_val) / delta_t_val, 0, 1) * p_goal_val[0]
            y = clamp(1 - (t - t_start_val)**8 / delta_t_val**8, 0, 1) * p_start_val[1] + clamp((t - t_start_val)**8 / delta_t_val**8, 0, 1) * p_goal_val[1]
            return np.array([x, y])

    @staticmethod
    def p_transport_time_derivative(t, vars_dict=None):
        """
        Calculate the time derivative of the transport point position.
        
        Parameters:
            t: Current time
            vars_dict: Optional global variable dictionary; if None, use the global variable manager
        """
        if vars_dict is None:
            vars_dict = TransportTask.get_global_vars_dict()
            
        if vars_dict is not None:
            p_start_val = vars_dict.get('p_start')
            p_goal_val = vars_dict.get('p_goal')
            t_start_val = vars_dict.get('t_start')
            delta_t_val = vars_dict.get('delta_t')
        else:
            # Fallback to global variables if dictionary not available
            from central import global_vars
            p_start_val = global_vars.p_start
            p_goal_val = global_vars.p_goal
            t_start_val = global_vars.t_start
            delta_t_val = global_vars.delta_t
        
        if t < t_start_val or t > t_start_val + delta_t_val:
            return np.array([0, 0])
        else:
            dx_dt = (p_goal_val[0] - p_start_val[0]) / delta_t_val
            dy_dt = 8 * (t - t_start_val)**7 / delta_t_val**8 * (p_goal_val[1] - p_start_val[1])
            return np.array([dx_dt, dy_dt])

    @staticmethod
    def transport_function(x_i, t, i, vars_dict=None):
        """
        Calculate the value of the transport task function.
        
        Parameters:
            x_i: Robot state
            t: Current time
            i: Robot index
            vars_dict: Optional global variable dictionary; if None, use the global variable manager
        """
        if vars_dict is None:
            vars_dict = TransportTask.get_global_vars_dict()
            
        p_transport_val = TransportTask.p_transport(t, vars_dict)
        return -np.linalg.norm(x_i[0:2] - p_transport_val)**2

    @staticmethod
    def transport_gradient(x_i, t, i, vars_dict=None):
        """
        Calculate the gradient of the transport task.
        
        Parameters:
            x_i: Robot state
            t: Current time
            i: Robot index
            vars_dict: Optional global variable dictionary; if None, use the global variable manager
        """
        if vars_dict is None:
            vars_dict = TransportTask.get_global_vars_dict()
            
        gradient = np.zeros(3)
        p_transport_val = TransportTask.p_transport(t, vars_dict)
        gradient[0:2] = -2 * (x_i[0:2] - p_transport_val)
        gradient[2] = 0
        return gradient

    @staticmethod
    def transport_time_derivative(x_i, t, i, vars_dict=None):
        """
        Calculate the time derivative of the transport task.
        
        Parameters:
            x_i: Robot state
            t: Current time
            i: Robot index
            vars_dict: Optional global variable dictionary; if None, use the global variable manager
        """
        if vars_dict is None:
            vars_dict = TransportTask.get_global_vars_dict()
            
        p_transport_val = TransportTask.p_transport(t, vars_dict)
        p_transport_deriv = TransportTask.p_transport_time_derivative(t, vars_dict)
        return -2 * np.dot(x_i[0:2] - p_transport_val, p_transport_deriv)