# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

# task_utils/task_module/robot_features_module.py
import numpy as np

class RobotFeaturesConfig:
    
    def __init__(self, n_r=5, n_f=4):
        self.n_r = n_r
        self.n_f = n_f
        self.robot_features = self._initialize_robot_features()
    
    def _initialize_robot_features(self):
        # Initialize the robot’s characteristics and capabilities
        A = np.zeros((self.n_f, self.n_r))
        
        # Robot feature configuration (row: feature, column: robot)
        # Feature 1: Walking, Feature 2: Transportation, Feature 3: Manipulation, Feature 4: Communication Computing
        A[:, 0] = [1, 1, 1, 1]  # robot 1: full capability
        A[:, 1] = [1, 1, 1, 1]  # robot 2: full capability
        A[:, 2] = [1, 1, 0, 0]  # robot 3: locomotion+transport specialist
        A[:, 3] = [1, 0, 1, 0]  # robot 4: locomotion+manipulation specialist
        A[:, 4] = [1, 0, 0, 1]  # robot 5: locomotion+communication-compute specialist
        
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
            print(f"Warning: Index out of range, robot_idx: {robot_idx}, feature_idx: {feature_idx}")
    
    def update_robot_features(self, robot_idx, features):
        if 0 <= robot_idx < self.n_r and len(features) == self.n_f:
            self.robot_features['A'][:, robot_idx] = features
        else:
            print(f"Warning: Index out of range or mismatch in number of features, robot_idx: {robot_idx}, features: {features}")
    
    def get_robot_capabilities(self, robot_idx):
        if 0 <= robot_idx < self.n_r:
            return self.robot_features['A'][:, robot_idx]
        else:
            print(f"Warning: Index out of range, robot_idx: {robot_idx}")
            return None 