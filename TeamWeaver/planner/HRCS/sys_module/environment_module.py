# task_utils/environment_module.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.path import Path

class EnvironmentConfig:
    
    def __init__(self):
        self.env_params = self._initialize_env_params()
    
    def _initialize_env_params(self):
        environment = np.array([[1.8, 1.2], [-1.8, 1.2], [-1.8, -1.2], [1.8, -1.2]]).T
        
        def simple_density(x, y):
            return np.exp(-0.5 * ((x/1.8)**2 + (y/1.2)**2))
        
        env_params = {
            'environment': environment,
            'density_function': simple_density
        }
        
        return env_params
    
    def get_env_params(self):
        return self.env_params
    
    def update_density_function(self, new_density_function):
        self.env_params['density_function'] = new_density_function
    
    def plot_environment(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        env = self.env_params['environment']
        border = Rectangle((-1.8, -1.2), 3.6, 2.4, edgecolor=[0.5, 0.5, 0.5], linewidth=5, facecolor='none')
        ax.add_patch(border)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True)
        
        return ax 