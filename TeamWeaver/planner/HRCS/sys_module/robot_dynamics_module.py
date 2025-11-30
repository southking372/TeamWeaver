# task_utils/task_module/robot_dynamics_module.py
import numpy as np

class RobotDynamicsConfig:
    """
    机器人动力学配置类，用于管理机器人动力学参数
    采用差速驱动模型: dx/dt = f(x) + g(x)u
    状态 x = [x, y, θ]
    控制 u = [v, ω] (线速度, 角速度)
    """

    def __init__(self, n_x=3, n_u=2):
        """
        初始化机器人动力学配置

        参数:
            n_x: 状态维度，默认为3 (x, y, θ)
            n_u: 控制输入维度，默认为2 (v, ω)
        """
        if n_x != 3 or n_u != 2:
            print(f"警告: 当前 RobotDynamicsConfig 实现主要针对 n_x=3, n_u=2。收到的 n_x={n_x}, n_u={n_u}")
            n_x = 3
            n_u = 2

        self.n_x = n_x
        self.n_u = n_u
        self.robot_dynamics = self._initialize_robot_dynamics()

    def _initialize_robot_dynamics(self):
        """
        初始化差速驱动模型的 f(x) 和 g(x)
        dx/dt = f(x) + g(x)u
        f(x) = [0, 0, 0]^T
        g(x) = [[cos(theta), 0], [sin(theta), 0], [0, 1]]
        """
        def f(x):
            # 无漂移项
            return np.zeros(self.n_x)

        def g(x):
            # 控制输入矩阵
            g_matrix = np.zeros((self.n_x, self.n_u))
            theta = x[2]
            g_matrix[0, 0] = np.cos(theta)
            g_matrix[1, 0] = np.sin(theta)
            g_matrix[2, 1] = 1.0
            return g_matrix

        def sys_dyn(x, u):
            """
            计算连续时间状态导数 dx/dt = f(x) + g(x)u
            """
            if len(u) != self.n_u:
                raise ValueError(f"控制输入 u 的维度应为 {self.n_u}, 但收到了 {len(u)}")
            if len(x) != self.n_x:
                 raise ValueError(f"状态输入 x 的维度应为 {self.n_x}, 但收到了 {len(x)}")
            return f(x) + g(x) @ u

        return {
            'f': f,
            'g': g,
            'n_x': self.n_x,
            'n_u': self.n_u,
            'sys_dyn': sys_dyn # 代表 dx/dt
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
                     raise ValueError(f"控制输入 u 的维度应为 {n_u_local}, 但收到了 {len(u)}")
                if len(x) != n_x_local:
                    raise ValueError(f"状态输入 x 的维度应为 {n_x_local}, 但收到了 {len(x)}")
                return f_func(x) + g_func(x) @ u
            self.robot_dynamics['sys_dyn'] = updated_sys_dyn
