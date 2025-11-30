# task_utils/disturbance_module.py
import numpy as np
from matplotlib.path import Path

# 干扰项类
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
        
        # 检查点是否在泥地区域内
        in_mud = False
        if x_mud_val is not None and y_mud_val is not None:
            path = Path(np.column_stack((x_mud_val, y_mud_val)))
            in_mud = path.contains_point((x[0, i], x[1, i]))
        
        return (i == robot_exo_dist_val - 1) or (alpha[task_exo_dist_val - 1, i] > 0) or in_mud 
        #我觉得这里的三个条件应该是互为独立的，无论是该机器人受到干扰，还是其正在执行的任务受到干扰，抑或是该机器人陷入泥地，都会对该机器人造成影响。