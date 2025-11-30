# task_utils/task_module/robot_features_module.py
import numpy as np

class RobotFeaturesConfig:
    
    def __init__(self, n_r=5, n_f=4):
        self.n_r = n_r
        self.n_f = n_f
        self.robot_features = self._initialize_robot_features()
    
    def _initialize_robot_features(self):
        # 初始化机器人的特征和能力
        A = np.zeros((self.n_f, self.n_r))
        
        # 机器人特征配置（行：特征，列：机器人）
        # 特征1：行走，特征2：运输，特征3：Manipulation，特征4：交流计算
        A[:, 0] = [1, 1, 1, 1]  # 机器人1：全功能
        A[:, 1] = [1, 1, 1, 1]  # 机器人2：全功能
        A[:, 2] = [1, 1, 0, 0]  # 机器人3：行走+运输专长
        A[:, 3] = [1, 0, 1, 0]  # 机器人4：行走+Manipulation专长
        A[:, 4] = [1, 0, 0, 1]  # 机器人5：行走+交流计算专长
        
        return {
            'A': A,
            'n_r': self.n_r,
            'n_f': self.n_f
        }
    
    def get_robot_features(self):
        return self.robot_features
    
    def get_feature_matrix(self):
        return self.robot_features['A']
    
    def get_robot_count(self):
        return self.n_r
    
    def get_feature_count(self):
        return self.n_f
    
    def update_robot_feature(self, robot_idx, feature_idx, value):
        if 0 <= robot_idx < self.n_r and 0 <= feature_idx < self.n_f:
            self.robot_features['A'][feature_idx, robot_idx] = value
        else:
            print(f"警告：索引超出范围，robot_idx: {robot_idx}, feature_idx: {feature_idx}")
    
    def update_robot_features(self, robot_idx, features):
        if 0 <= robot_idx < self.n_r and len(features) == self.n_f:
            self.robot_features['A'][:, robot_idx] = features
        else:
            print(f"警告：索引超出范围或特征数量不匹配，robot_idx: {robot_idx}, features: {features}")
    
    def get_robot_capabilities(self, robot_idx):
        if 0 <= robot_idx < self.n_r:
            return self.robot_features['A'][:, robot_idx]
        else:
            print(f"警告：索引超出范围，robot_idx: {robot_idx}")
            return None 