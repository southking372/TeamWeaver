# task_utils/simulation_module.py
import numpy as np

class SimulationConfig:
    
    def __init__(self, n_r=5, n_x=3):
        self.n_r = n_r
        self.n_x = n_x
        self.sim_params = self._initialize_sim_params()
    
    def _initialize_sim_params(self):
        sim_params = {
            'DT': 0.1,  # 时间步长
            'max_iter': 800,  # 最大迭代次数
            'initial_states': self._create_initial_states(),
            'trajectory': self._create_trajectory_storage()
        }
        
        return sim_params
    
    def _create_initial_states(self):
        x = np.zeros((self.n_x, self.n_r))
        x[0, :] = 1.65 * np.ones(self.n_r)  # 初始x位置
        x[1, :] = np.linspace(-1, 1, self.n_r)  # 初始y位置
        x[2, :] = np.zeros(self.n_r)  # 初始角度
        
        return x
    
    def _create_trajectory_storage(self):
        max_iter = self.sim_params['max_iter'] if hasattr(self, 'sim_params') else 800
        x_traj = np.zeros((self.n_x, self.n_r, max_iter + 1))
        x_traj[:, :, 0] = self._create_initial_states()
        
        return x_traj
    
    def get_sim_params(self):
        return self.sim_params
    
    def update_sim_params(self, param_name, value):
        if param_name in self.sim_params:
            self.sim_params[param_name] = value
        else:
            print(f"警告：参数 '{param_name}' 不存在于模拟参数中")
    
    def get_initial_states(self):
        return self.sim_params['initial_states']
    
    def get_trajectory(self):
        return self.sim_params['trajectory']
    
    def update_trajectory(self, iter, x):
        self.sim_params['trajectory'][:, :, iter + 1] = x
    
    def get_time_step(self):
        return self.sim_params['DT']
    
    def get_max_iter(self):
        return self.sim_params['max_iter']
    
    def get_current_time(self, iter):
        return iter * self.sim_params['DT'] 