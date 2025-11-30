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
                    print("全局变量管理器缺少必要的方法")
                    global_vars_dict = {}
            else:
                print("无法找到global_vars_manager")
        
        except Exception as e:
            print(f"获取全局变量管理器时出错: {e}")
            global_vars_dict = None
            
        return global_vars_dict
    
    @staticmethod
    def wait_function(x_i, t, i, vars_dict=None):
        if vars_dict is None:
            vars_dict = WaitTask.get_global_vars_dict()
            
        step_threshold = vars_dict.get('wait_step_threshold', 5.0)
        sim_freq = vars_dict.get('sim_freq', 1.0)
        elapsed_time = vars_dict.get('wait_elapsed_time', 0.0)
        
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
        # 对于等待任务，机器人不移动，直接返回当前状态
        return x_i 