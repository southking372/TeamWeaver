# task_utils/control_module.py
import numpy as np

class ControlConfig:
    
    def __init__(self, n_r=5, n_x=3, n_u=2):
        self.n_r = n_r
        self.n_x = n_x
        self.n_u = n_u
        self.control_params = self._initialize_control_params()
    
    def _initialize_control_params(self):
        # 创建控制参数字典
        control_params = {
            'K': self._create_gain_matrix(),  # 增益矩阵
            'u_max': np.array([1.0, 1.0]),  # 最大控制输入
            'u_min': np.array([-1.0, -1.0]),  # 最小控制输入
            'alpha': 0.1,  # 控制权重
            'beta': 0.9,  # 任务权重
            'gamma': 0.5,  # 障碍物权重
            'delta': 0.1,  # 收敛阈值
            'max_iter': 100,  # 最大迭代次数
            'control_inputs': np.zeros((self.n_u, self.n_r))  # 控制输入存储
        }
        
        return control_params
    
    def _create_gain_matrix(self):
        K = np.zeros((self.n_u, self.n_x))
        K[0, 0] = 1.0  # 位置增益
        K[0, 1] = 1.0  # 位置增益
        K[1, 2] = 1.0  # 角度增益
        return K
    
    def get_control_params(self):
        return self.control_params
    
    def update_control_params(self, param_name, value):
        if param_name in self.control_params:
            self.control_params[param_name] = value
        else:
            print(f"警告：参数 '{param_name}' 不存在于控制参数中")
    
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