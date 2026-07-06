# task_utils/task_module/task_config_module.py
import numpy as np

class TaskConfig:
    
    def __init__(self, n_t=4, n_c=8):
        """
        Initialize task configuration
        
        Parameters:
            n_t: Number of task types, default is 4 (Navi/explore/manipulation/Wait)
            n_c: capability categoriesquantity, default is 8
        """
        self.n_t = n_t
        self.n_c = n_c
        self.task_params = self._initialize_task_params()
    
    def _initialize_task_params(self):
        # mission capability requirements matrix
        T = np.zeros((self.n_t, self.n_c))
        T[0, 0:2] = [1, 1]      # task 1: Navi - needs locomotion and transport
                                # capabilities 1-2 (Motor navi/pick)
        T[1, 2:4] = [1, 1]      # task 2: explore - needs manipulation and communication
                                # capabilities 3-4 (find_action and communication-compute)
        T[2, 1:3] = [1, 1]      # Mission 3: Manipulation - Requires Manipulation and transportation capabilities
                                # capabilities 2-3 (Motor place and manipulation)
        T[3, 0] = 0             # task 4: wait - no capability required
                                # capability 1 (basic Motor)
        
        # mapping Hs (8 capabilities → 4 features)
        Hs = [np.zeros((1, 4)) for _ in range(self.n_c)]
        
        # Motor capabilities (first 5) map to locomotion/transport features
        for i in range(5):
            Hs[i][0, 0] = 1     # locomotion feature supports Motor capability
            Hs[i][0, 1] = 1     # transport feature supports Motor capability
        
        Hs[5][0, 2] = 1        # ManipulationFunction support find_action
        Hs[6][0, 3] = 1        # Communication computing capabilities support communication capabilities
        Hs[7][0, 3] = 1        # Communication computing capabilities support computing capabilities
        
        # weight matrix ws
        ws = [np.eye(1) for _ in range(self.n_c)]  # Default unit weight
        ws[0] = 2 * np.eye(1)  # Strengthen the weight of walking ability
        ws[2] = 1.5 * np.eye(1) # Strengthen the weight of Manipulation capabilities
        
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
            print(f"Warning: Index out of range or required number mismatch, task_idx: {task_idx}, requirements: {requirements}")
    
    def update_mapping_matrix(self, capability_idx, mapping):
        if 0 <= capability_idx < self.n_c and mapping.shape == (1, 4):
            self.task_params['Hs'][capability_idx] = mapping
        else:
            print(f"Warning: Index out of range or mapping matrix dimensions mismatch, capability_idx: {capability_idx}, mapping: {mapping}")
    
    def update_weight_matrix(self, capability_idx, weight):
        if 0 <= capability_idx < self.n_c and weight.shape == (1, 1):
            self.task_params['ws'][capability_idx] = weight
        else:
            print(f"Warning: Index out of range or weight matrix dimensions mismatch, capability_idx: {capability_idx}, weight: {weight}")
    
    def get_task_requirements(self, task_idx):
        if 0 <= task_idx < self.n_t:
            return self.task_params['T'][task_idx, :]
        else:
            print(f"Warning: Index out of range, task_idx: {task_idx}")
            return None 