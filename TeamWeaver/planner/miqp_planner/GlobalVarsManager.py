import numpy as np

class GlobalVarsManager:
    def __init__(self):
        # 初始化所有全局变量
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
        """获取指定名称的全局变量"""
        if hasattr(self, var_name):
            return getattr(self, var_name)
        return None
    
    def set_var(self, var_name, value):
        """设置指定名称的全局变量"""
        setattr(self, var_name, value)
    
    def get_all_vars(self):
        """获取所有全局变量的字典"""
        return {
            name: value for name, value in self.__dict__.items() 
            if not name.startswith('_') and not callable(value)
        }