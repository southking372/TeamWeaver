# task_utils/task_module/task_config_module.py
import numpy as np

class TaskConfig:
    
    def __init__(self, n_t=4, n_c=8):
        """
        初始化任务配置
        
        参数:
            n_t: 任务类型数量，默认为4 (Navi/探索/操作/Wait)
            n_c: 能力类别数量，默认为8
        """
        self.n_t = n_t
        self.n_c = n_c
        self.task_params = self._initialize_task_params()
    
    def _initialize_task_params(self):
        # 任务能力需求矩阵
        T = np.zeros((self.n_t, self.n_c))
        T[0, 0:2] = [1, 1]      # 任务1：导航(Navi) - 需要行走和运输能力
                                # 能力1-2（对应Motor类的navi/pick）
        T[1, 2:4] = [1, 1]      # 任务2：探索 - 需要Manipulation和交流能力
                                # 能力3-4（对应find_action和交流计算）
        T[2, 1:3] = [1, 1]      # 任务3：操作 - 需要Manipulation和运输能力
                                # 能力2-3（对应Motor类的place和Manipulation）
        T[3, 0] = 0             # 任务4：等待 - 不需要能力
                                # 能力1（基础Motor能力）
        
        # 映射关系 Hs (8能力 → 4功能)
        Hs = [np.zeros((1, 4)) for _ in range(self.n_c)]
        
        # Motor类能力（前5个）映射到行走/运输功能
        for i in range(5):
            Hs[i][0, 0] = 1     # 行走功能支撑Motor能力
            Hs[i][0, 1] = 1     # 运输功能支撑Motor能力
        
        Hs[5][0, 2] = 1        # Manipulation功能支撑find_action
        Hs[6][0, 3] = 1        # 交流计算功能支撑交流能力
        Hs[7][0, 3] = 1        # 交流计算功能支撑计算能力
        
        # 权重矩阵 ws
        ws = [np.eye(1) for _ in range(self.n_c)]  # 默认单位权重
        ws[0] = 2 * np.eye(1)  # 加强行走能力权重
        ws[2] = 1.5 * np.eye(1) # 加强Manipulation能力权重
        
        return {
            'T': T,
            'Hs': Hs,
            'ws': ws,
            'n_t': self.n_t,
            'n_c': self.n_c
        }
    
    def get_task_params(self):
        return self.task_params
    
    def get_task_matrix(self):
        return self.task_params['T']
    
    def get_mapping_matrices(self):
        return self.task_params['Hs']
    
    def get_weight_matrices(self):
        return self.task_params['ws']
    
    def get_task_count(self):
        return self.n_t
    
    def get_capability_count(self):
        return self.n_c
    
    def update_task_requirements(self, task_idx, requirements):
        if 0 <= task_idx < self.n_t and len(requirements) == self.n_c:
            self.task_params['T'][task_idx, :] = requirements
        else:
            print(f"警告：索引超出范围或需求数量不匹配，task_idx: {task_idx}, requirements: {requirements}")
    
    def update_mapping_matrix(self, capability_idx, mapping):
        if 0 <= capability_idx < self.n_c and mapping.shape == (1, 4):
            self.task_params['Hs'][capability_idx] = mapping
        else:
            print(f"警告：索引超出范围或映射矩阵维度不匹配，capability_idx: {capability_idx}, mapping: {mapping}")
    
    def update_weight_matrix(self, capability_idx, weight):
        if 0 <= capability_idx < self.n_c and weight.shape == (1, 1):
            self.task_params['ws'][capability_idx] = weight
        else:
            print(f"警告：索引超出范围或权重矩阵维度不匹配，capability_idx: {capability_idx}, weight: {weight}")
    
    def get_task_requirements(self, task_idx):
        if 0 <= task_idx < self.n_t:
            return self.task_params['T'][task_idx, :]
        else:
            print(f"警告：索引超出范围，task_idx: {task_idx}")
            return None 