# task_utils/task_module/clean_module.py
import numpy as np

class CleanTask:
    """Clean任务模块 - 处理清洁对象的任务（Agent 1专有能力）"""
    
    @staticmethod
    def clean_function(x, t, robot_id, vars_dict=None):
        """
        Clean任务效用函数
        
        参数:
            x: 机器人状态 [x, y, theta]
            t: 时间
            robot_id: 机器人ID
            vars_dict: 全局变量字典
        
        返回:
            效用值，越小表示越接近目标
        """
        # Agent 0 不具备Clean能力
        if robot_id == 0:
            return 1000.0  # 返回很大的值，表示无法执行Clean
        
        if vars_dict is None:
            target_object_position = np.array([0.5, -0.3])  # 默认待清洁对象位置
            clean_dist_thresh = 0.15
            object_clean_state = False
        else:
            target_object_position = vars_dict.get('target_object_position', np.array([0.5, -0.3]))
            clean_dist_thresh = vars_dict.get('clean_dist_thresh', 0.15)
            object_clean_state = vars_dict.get('object_clean_state', False)
        
        # 如果对象已经清洁完成，任务完成
        if object_clean_state:
            return 0.1  # 返回小值表示任务已完成
        
        # 计算到目标对象的距离
        robot_pos = x[:2]
        distance_to_object = np.linalg.norm(robot_pos - target_object_position)
        
        # 使用二次效用函数
        utility = (distance_to_object - clean_dist_thresh) ** 2
        
        # 如果已经在清洁范围内，降低效用值
        if distance_to_object <= clean_dist_thresh:
            utility = 0.1  # 接近完成状态
        
        return max(0.0, utility)
    
    @staticmethod
    def clean_gradient(x, t, robot_id, vars_dict=None):
        """Clean任务效用函数的梯度"""
        # Agent 0 不具备Clean能力
        if robot_id == 0:
            return np.zeros(3)
        
        if vars_dict is None:
            target_object_position = np.array([0.5, -0.3])
            clean_dist_thresh = 0.15
            object_clean_state = False
        else:
            target_object_position = vars_dict.get('target_object_position', np.array([0.5, -0.3]))
            clean_dist_thresh = vars_dict.get('clean_dist_thresh', 0.15)
            object_clean_state = vars_dict.get('object_clean_state', False)
        
        # 如果对象已经清洁完成，零梯度
        if object_clean_state:
            return np.zeros(3)
        
        robot_pos = x[:2]
        distance_to_object = np.linalg.norm(robot_pos - target_object_position)
        
        # 梯度计算
        gradient = np.zeros(3)
        if distance_to_object > 1e-6:  # 避免除零
            direction = (robot_pos - target_object_position) / distance_to_object
            gradient[:2] = 2 * (distance_to_object - clean_dist_thresh) * direction
        
        return gradient
    
    @staticmethod
    def clean_time_derivative(x, t, robot_id, vars_dict=None):
        """Clean任务效用函数的时间导数"""
        # Clean任务通常不直接依赖时间
        return 0.0
    
    @staticmethod
    def apply_motion_control(x, t, robot_id, vars_dict, dt):
        """
        应用Clean任务的运动控制
        
        参数:
            x: 当前机器人状态 [x, y, theta]
            t: 当前时间
            robot_id: 机器人ID
            vars_dict: 全局变量字典
            dt: 时间步长
        
        返回:
            新的机器人状态
        """
        # Agent 0 不具备Clean能力
        if robot_id == 0:
            print(f"Warning: Agent {robot_id} attempted to perform Clean task but lacks this capability")
            return x
        
        if vars_dict is None:
            return x
        
        target_object_position = vars_dict.get('target_object_position', np.array([0.5, -0.3]))
        clean_dist_thresh = vars_dict.get('clean_dist_thresh', 0.15)
        object_clean_state = vars_dict.get('object_clean_state', False)
        
        # 如果对象已经清洁完成，停止移动
        if object_clean_state:
            return x
        
        robot_pos = x[:2]
        distance_to_object = np.linalg.norm(robot_pos - target_object_position)
        
        # 如果已经在清洁范围内，执行清洁动作
        if distance_to_object <= clean_dist_thresh:
            # 模拟清洁成功
            if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                # 使用全局变量管理器
                vars_dict.set_var('object_clean_state', True)
            else:
                # 普通字典
                vars_dict['object_clean_state'] = True
            
            print(f"Robot {robot_id} successfully cleaned object at position {target_object_position}")
            return x
        
        # 向目标对象移动
        direction = target_object_position - robot_pos
        distance = np.linalg.norm(direction)
        
        if distance > 1e-6:
            direction = direction / distance
            
            # 计算期望角度
            desired_theta = np.arctan2(direction[1], direction[0])
            
            # 角度差
            theta_diff = desired_theta - x[2]
            theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))  # 规范化到[-π, π]
            
            # 控制参数
            linear_speed = 0.3
            angular_speed = 1.0
            
            new_x = x.copy()
            
            # 如果角度差较大，先转向
            if abs(theta_diff) > 0.1:
                new_x[2] += np.sign(theta_diff) * angular_speed * dt
            else:
                # 角度对齐后前进
                new_x[0] += linear_speed * np.cos(x[2]) * dt
                new_x[1] += linear_speed * np.sin(x[2]) * dt
                # 同时微调角度
                new_x[2] += 0.5 * theta_diff * dt
            
            # 角度规范化
            new_x[2] = np.arctan2(np.sin(new_x[2]), np.cos(new_x[2]))
            
            return new_x
        
        return x 