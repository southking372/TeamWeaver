# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

# task_utils/task_module/robot_dynamics_module.py
import numpy as np

class RobotDynamicsConfig:
    """
    Robot dynamics configuration class, used to manage robot dynamics parameters
    Adopt differential drive model: dx/dt = f(x) + g(x)u
    status x = [x, y, θ]
    control u = [v, ω] (linear speed, Angular velocity)
    """

    def __init__(self, n_x=3, n_u=2):
        """
        Initialize robot dynamics configuration

        Parameters:
            n_x: Status dimension, default is 3 (x, y, θ)
            n_u: controlInput dimension, default is 2 (v, ω)
        """
        if n_x != 3 or n_u != 2:
            print(f"warn: The current RobotDynamicsConfig implementation mainly targets n_x=3, n_u=2。n_x received={n_x}, n_u={n_u}")
            n_x = 3
            n_u = 2

        self.n_x = n_x
        self.n_u = n_u
        self.robot_dynamics = self._initialize_robot_dynamics()

    def _initialize_robot_dynamics(self):
        """
        Initialize f of the differential drive model(x) and g(x)
        dx/dt = f(x) + g(x)u
        f(x) = [0, 0, 0]^T
        g(x) = [[cos(theta), 0], [sin(theta), 0], [0, 1]]
        """
        def f(x):
            # No drift term
            return np.zeros(self.n_x)

        def g(x):
            # controlinput matrix
            g_matrix = np.zeros((self.n_x, self.n_u))
            theta = x[2]
            g_matrix[0, 0] = np.cos(theta)
            g_matrix[1, 0] = np.sin(theta)
            g_matrix[2, 1] = 1.0
            return g_matrix

        def sys_dyn(x, u):
            """
            Compute the continuous-time state derivative dx/dt = f(x) + g(x)u
            """
            if len(u) != self.n_u:
                raise ValueError(f"controlThe dimensions of input u should be {self.n_u}, but received {len(u)}")
            if len(x) != self.n_x:
                 raise ValueError(f"The dimensions of the state input x should be {self.n_x}, but received {len(x)}")
            return f(x) + g(x) @ u

        return {
            'f': f,
            'g': g,
            'n_x': self.n_x,
            'n_u': self.n_u,
            'sys_dyn': sys_dyn # stands for dx/dt
        }

    def get_robot_dynamics(self):
        return self.robot_dynamics

    def get_state_dimension(self):
        return self.n_x

    def get_control_dimension(self):
        return self.n_u

    def get_system_dynamics_derivative(self):
        return self.robot_dynamics['sys_dyn']

    def update_dynamics(self, f_new=None, g_new=None):
        updated = False
        if f_new is not None:
            self.robot_dynamics['f'] = f_new
            updated = True
        if g_new is not None:
            self.robot_dynamics['g'] = g_new
            updated = True

        if updated:
            f_func = self.robot_dynamics['f']
            g_func = self.robot_dynamics['g']
            n_u_local = self.n_u
            n_x_local = self.n_x
            def updated_sys_dyn(x, u):
                if len(u) != n_u_local:
                     raise ValueError(f"controlThe dimensions of input u should be {n_u_local}, but received {len(u)}")
                if len(x) != n_x_local:
                    raise ValueError(f"The dimensions of the state input x should be {n_x_local}, but received {len(x)}")
                return f_func(x) + g_func(x) @ u
            self.robot_dynamics['sys_dyn'] = updated_sys_dyn
