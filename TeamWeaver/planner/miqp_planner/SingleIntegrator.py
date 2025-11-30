import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class SingleIntegrator:
    """
    SingleIntegrator类模拟一个单积分器系统，可用于机器人运动仿真
    """
    
    def __init__(self, **kwargs):
        """
        初始化单积分器系统
        
        参数:
            width (float): 机器人的宽度，默认为0.1
            initial_state (numpy.ndarray): 初始状态[x, y]，默认为[0, 0]
            simulation_time_step (float): 仿真时间步长，默认为0.1
        """
        # 设置默认参数
        self.w = kwargs.get('width', 1e-1)
        initial_state = kwargs.get('initial_state', np.array([0, 0]))
        self.dt = kwargs.get('simulation_time_step', 1e-1)
        
        # 设置状态
        self.x = initial_state[0]
        self.y = initial_state[1]
        
        # 绘图句柄
        self.h_r = None
    
    def getPose(self):
        """
        获取当前位姿
        
        返回:
            numpy.ndarray: 当前位置[x, y]
        """
        return np.array([self.x, self.y])
    
    def setPose(self, q):
        """
        设置位姿
        
        参数:
            q (numpy.ndarray): 新的位置[x, y]
        """
        self.x = q[0]
        self.y = q[1]
    
    def moveSingleIntegrator(self, v):
        """
        按照指定的速度移动单积分器
        
        参数:
            v (numpy.ndarray): 速度向量[vx, vy]
        """
        self.x = self.x + v[0] * self.dt
        self.y = self.y + v[1] * self.dt
    
    def goToPoint(self, p, K=1):
        """
        使单积分器移动到指定点
        
        参数:
            p (numpy.ndarray): 目标点[x, y]
            K (float): 控制增益，默认为1
        """
        current_pos = np.array([self.x, self.y])
        v = K * (p - current_pos)
        self.move_single_integrator(v)
    
    def plotRobot(self, *args):
        """
        绘制机器人
        
        参数:
            *args: 绘图参数，例如颜色和线型
        """
        if not args:
            args = ([0.5, 0.5, 0.5], {'edgecolor': 'none'})
        
        # 创建机器人圆形表示
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = self.x + self.w * np.cos(theta)
        circle_y = self.y + self.w * np.sin(theta)
        
        # 创建或更新图形
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
            # 更新现有图形
            self.h_r[0].set_xy(np.column_stack([circle_x, circle_y]))
        
        plt.draw()
        plt.pause(0.001)  # 相当于drawnow limitrate