# task_utils/coverage_module.py
import numpy as np

class CoverageControl:
    """
    Implementation of the coverage control task functions.
    """
    
    def __init__(self, poi=None):
        """
        Initialize the CoverageControl class.
        
        Parameters:
            poi: Point of interest (default: None)
        """
        self.poi = poi
    
    @staticmethod
    def get_global_vars_dict():
        """
        Get global variables dictionary from GlobalVarsManager
        """
        # 简化实现，直接返回None
        # 现在我们通过函数参数vars_dict显式传递全局变量
        print("Warning: get_global_vars_dict() called but not implemented. Variables should be passed via vars_dict parameter.")
        return None
    
    @staticmethod
    def phi_perimeter(x, y, vars_dict=None):
        """
        Calculate the density function for perimeter defense.
        
        Parameters:
            x: x-coordinate
            y: y-coordinate
            vars_dict: Optional global variable dictionary; if None, use the global variable manager
        """
        if vars_dict is None:
            vars_dict = CoverageControl.get_global_vars_dict()
            
        if vars_dict is not None:
            p_transport_val = vars_dict.get('p_transport_t')
        else:
            # 如果vars_dict为None，无法获取p_transport_t，返回默认值
            print("Warning: vars_dict is None, cannot access p_transport_t")
            return 0
        
        if p_transport_val is None:
            return 0
        try:
            distance_squared = (x - p_transport_val[0])**2 + (y - p_transport_val[1])**2
            ring_difference = distance_squared - 0.4**2
            if abs(ring_difference) > 10:
                return 0
            return np.exp(-100 * ring_difference**2)
        except Exception as e:
            print(f"phi_perimeter calculation error: {e}")
            return 0
    
    @staticmethod
    def coverage_control_task_function(xi, t, i, poi_val=None, G_val=None, vars_dict=None):
        """
        Calculate the function value for the coverage control task.
        
        Parameters:
            xi: Robot state
            t: Current time
            i: Robot index
            poi_val: Optional point of interest value, passed directly
            G_val: Optional G matrix value, passed directly
            vars_dict: Optional global variable dictionary; if None, use the global variable manager
        """
        if vars_dict is None:
            vars_dict = CoverageControl.get_global_vars_dict()
            
        if poi_val is None and vars_dict is not None:
            poi_val = vars_dict.get('poi')
        if G_val is None and vars_dict is not None:
            G_val = vars_dict.get('G')
            
        if poi_val is None or G_val is None:
            # 如果无法从vars_dict获取，使用默认值
            print(f"Warning: Required values missing. poi_val={poi_val}, G_val={G_val}")
            if poi_val is None:
                poi_val = np.array([0, 1])  # 默认兴趣点位置
            if G_val is None:
                # 创建形状为(2, max(1,i+1))的零矩阵作为默认G值
                G_val = np.zeros((2, max(1, i+1)))
        
        if G_val is None or i >= G_val.shape[1]:
            return 0
        return -np.linalg.norm(xi[0:2] - G_val[:,i])**2 - (xi[2] - np.arctan2(poi_val[1] - xi[1], poi_val[0] - xi[0]))**2

    @staticmethod
    def coverage_control_task_gradient(xi, t, i, poi_val=None, G_val=None, vars_dict=None):
        """
        Calculate the gradient of the coverage control task.
        
        Parameters:
            xi: Robot state
            t: Current time
            i: Robot index
            poi_val: Optional point of interest value, passed directly
            G_val: Optional G matrix value, passed directly
            vars_dict: Optional global variable dictionary; if None, use the global variable manager
        """
        if vars_dict is None:
            vars_dict = CoverageControl.get_global_vars_dict()
            
        if poi_val is None and vars_dict is not None:
            poi_val = vars_dict.get('poi')
        if G_val is None and vars_dict is not None:
            G_val = vars_dict.get('G')
            
        if poi_val is None or G_val is None:
            # 如果无法从vars_dict获取，使用默认值
            print(f"Warning: Required values missing. poi_val={poi_val}, G_val={G_val}")
            if poi_val is None:
                poi_val = np.array([0, 1])  # 默认兴趣点位置
            if G_val is None:
                # 创建形状为(2, max(1,i+1))的零矩阵作为默认G值
                G_val = np.zeros((2, max(1, i+1)))
        
        if G_val is None or i >= G_val.shape[1]:
            return np.zeros(3)
        angle_diff = xi[2] - np.arctan2(poi_val[1] - xi[1], poi_val[0] - xi[0])
        return np.array([
            -2 * (xi[0] - G_val[0,i]),
            -2 * (xi[1] - G_val[1,i]),
            -2 * angle_diff
        ])

    @staticmethod
    def coverage_control_task_time_derivative(xi, t, i, poi_val=None, G_val=None, vars_dict=None):
        """
        Calculate the time derivative of the coverage control task.
        
        Parameters:
            xi: Robot state
            t: Current time
            i: Robot index
            poi_val: Optional point of interest value, passed directly
            G_val: Optional G matrix value, passed directly
            vars_dict: Optional global variable dictionary; if None, use the global variable manager
        """
        return 0