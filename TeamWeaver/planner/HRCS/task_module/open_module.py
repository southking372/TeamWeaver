# task_utils/task_module/open_module.py
import numpy as np

class OpenTask:
    """Open任务模块 - 处理开启家具或对象的任务"""
    
    @staticmethod
    def open_function(x, t, robot_id, vars_dict=None):
        """
        Open任务效用函数
        
        参数:
            x: 机器人状态 [x, y, theta]
            t: 时间
            robot_id: 机器人ID
            vars_dict: 全局变量字典
        
        返回:
            效用值，越小表示越接近目标
        """
        if vars_dict is None:
            target_furniture_position = np.array([1.0, 0.5])  # 默认家具位置
            operation_dist_thresh = 0.2
            furniture_open_state = False
        else:
            target_furniture_position = vars_dict.get('target_furniture_position', np.array([1.0, 0.5]))
            operation_dist_thresh = vars_dict.get('operation_dist_thresh', 0.2)
            furniture_open_state = vars_dict.get('furniture_open_state', False)
        
        # 如果家具已经打开，任务完成
        if furniture_open_state:
            return 0.1  # 返回小值表示任务已完成
        
        # 计算到目标家具的距离
        robot_pos = x[:2]
        distance_to_furniture = np.linalg.norm(robot_pos - target_furniture_position)
        
        # 使用二次效用函数
        utility = (distance_to_furniture - operation_dist_thresh) ** 2
        
        # 如果已经在操作范围内，降低效用值
        if distance_to_furniture <= operation_dist_thresh:
            utility = 0.1  # 接近完成状态
        
        return max(0.0, utility)
    
    @staticmethod
    def open_gradient(x, t, robot_id, vars_dict=None):
        """Open任务效用函数的梯度"""
        if vars_dict is None:
            target_furniture_position = np.array([1.0, 0.5])
            operation_dist_thresh = 0.2
            furniture_open_state = False
        else:
            target_furniture_position = vars_dict.get('target_furniture_position', np.array([1.0, 0.5]))
            operation_dist_thresh = vars_dict.get('operation_dist_thresh', 0.2)
            furniture_open_state = vars_dict.get('furniture_open_state', False)
        
        # 如果家具已经打开，零梯度
        if furniture_open_state:
            return np.zeros(3)
        
        robot_pos = x[:2]
        distance_to_furniture = np.linalg.norm(robot_pos - target_furniture_position)
        
        # 梯度计算
        gradient = np.zeros(3)
        if distance_to_furniture > 1e-6:  # 避免除零
            direction = (robot_pos - target_furniture_position) / distance_to_furniture
            gradient[:2] = 2 * (distance_to_furniture - operation_dist_thresh) * direction
        
        return gradient
    
    @staticmethod
    def open_time_derivative(x, t, robot_id, vars_dict=None):
        """Open任务效用函数的时间导数"""
        # Open任务通常不直接依赖时间
        return 0.0
    
    @staticmethod
    def apply_motion_control(x, t, robot_id, vars_dict, dt):
        """
        应用Open任务的运动控制
        
        参数:
            x: 当前机器人状态 [x, y, theta]
            t: 当前时间
            robot_id: 机器人ID
            vars_dict: 全局变量字典
            dt: 时间步长
        
        返回:
            新的机器人状态
        """
        if vars_dict is None:
            return x
        
        target_furniture_position = vars_dict.get('target_furniture_position', np.array([1.0, 0.5]))
        operation_dist_thresh = vars_dict.get('operation_dist_thresh', 0.2)
        furniture_open_state = vars_dict.get('furniture_open_state', False)
        
        # 如果家具已经打开，停止移动
        if furniture_open_state:
            return x
        
        robot_pos = x[:2]
        distance_to_furniture = np.linalg.norm(robot_pos - target_furniture_position)
        
        # 如果已经在操作范围内，执行开启动作
        if distance_to_furniture <= operation_dist_thresh:
            # 模拟开启成功
            if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                # 使用全局变量管理器
                vars_dict.set_var('furniture_open_state', True)
            else:
                # 普通字典
                vars_dict['furniture_open_state'] = True
            
            print(f"Robot {robot_id} successfully opened furniture at position {target_furniture_position}")
            return x
        
        # 向目标家具移动
        direction = target_furniture_position - robot_pos
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