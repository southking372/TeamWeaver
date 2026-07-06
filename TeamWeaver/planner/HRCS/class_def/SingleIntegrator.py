# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class SingleIntegrator:
    """
    SingleIntegratorClass simulates a single integrator system that can be used for robot motion simulation
    """
    
    def __init__(self, **kwargs):
        """
Initializing a single integrator system
        
        Parameters:
            width (float):The width of the robot, default is 0.1
            initial_state (numpy.ndarray):initial state[x, y], the default is[0, 0]
            simulation_time_step (float):simulationtime step, default is 0.1
        """
        self.w = kwargs.get('width', 1e-1)
        initial_state = kwargs.get('initial_state', np.array([0, 0]))
        self.dt = kwargs.get('simulation_time_step', 1e-1)
        
        self.x = initial_state[0]
        self.y = initial_state[1]
        
        self.h_r = None
    
    def getPose(self):
        """
Get current pose
        
        Returns:
            numpy.ndarray:Current location[x, y]
        """
        return np.array([self.x, self.y])
    
    def setPose(self, q):
        """
Set pose
        
        Parameters:
            q (numpy.ndarray):newposition[x, y]
        """
        self.x = q[0]
        self.y = q[1]
    
    def moveSingleIntegrator(self, v):
        """
at specified speedmovementsingle integrator
        
        Parameters:
            v (numpy.ndarray):velocity vector[vx, vy]
        """
        self.x = self.x + v[0] * self.dt
        self.y = self.y + v[1] * self.dt
    
    def goToPoint(self, p, K=1):
        """
Make a single integratormovementto designated point
        
        Parameters:
            p (numpy.ndarray):target point[x, y]
            K (float): controlGain, default is 1
        """
        current_pos = np.array([self.x, self.y])
        v = K * (p - current_pos)
        self.moveSingleIntegrator(v)
    
    def plotRobot(self, *args):
        """
drawing robot
        
        Parameters:
            *args:Drawing parameters such as color and line style
        """
        if not args:
            args = ([0.5, 0.5, 0.5], {'edgecolor': 'none'})
        
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = self.x + self.w * np.cos(theta)
        circle_y = self.y + self.w * np.sin(theta)
        
        if self.h_r is None:
            if len(args) == 1:
                color = args[0]
                edge_params = {}
            else:
                color = args[0]
                edge_params = args[1]
            
            plt.figure(1)
            self.h_r = plt.fill(circle_x, circle_y, color=color, **edge_params)
        else:
            self.h_r[0].set_xy(np.column_stack([circle_x, circle_y]))
        
        plt.draw()
        plt.pause(0.001)
