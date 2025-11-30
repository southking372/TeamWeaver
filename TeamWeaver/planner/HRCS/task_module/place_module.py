# task_utils/task_module/place_module.py
import numpy as np

class PlaceTask:
    """Place任务模块 - 处理对象放置任务"""
    
    @staticmethod
    def place_function(x, t, robot_id, vars_dict=None):
        """
        Place任务效用函数
        
        参数:
            x: 机器人状态 [x, y, theta]
            t: 时间
            robot_id: 机器人ID
            vars_dict: 全局变量字典
        
        返回:
            效用值，越小表示越接近目标
        """
        if vars_dict is None:
            target_receptacle_position = np.array([-0.8, 0.7])
            place_dist_thresh = 0.15
            is_holding = False
            holding_robot_id = None
        else:
            target_receptacle_position = vars_dict.get('target_receptacle_position', np.array([-0.8, 0.7]))
            place_dist_thresh = vars_dict.get('place_dist_thresh', 0.15)
            is_holding = vars_dict.get('is_holding', False)
            holding_robot_id = vars_dict.get('holding_robot_id', None)
        
        # 只有持有物体的机器人才能执行Place任务
        if not is_holding or holding_robot_id != robot_id:
            return 1000.0  # 返回很大的值，表示无法执行Place
        
        # 计算到目标放置位置的距离
        robot_pos = x[:2]
        distance_to_receptacle = np.linalg.norm(robot_pos - target_receptacle_position)
        
        # 使用二次效用函数
        utility = (distance_to_receptacle - place_dist_thresh) ** 2
        
        # 如果已经在放置范围内，降低效用值
        if distance_to_receptacle <= place_dist_thresh:
            utility = 0.1  # 接近完成状态
        
        return max(0.0, utility)
    
    @staticmethod
    def place_gradient(x, t, robot_id, vars_dict=None):
        """Place任务效用函数的梯度"""
        if vars_dict is None:
            target_receptacle_position = np.array([-0.8, 0.7])
            place_dist_thresh = 0.15
            is_holding = False
            holding_robot_id = None
        else:
            target_receptacle_position = vars_dict.get('target_receptacle_position', np.array([-0.8, 0.7]))
            place_dist_thresh = vars_dict.get('place_dist_thresh', 0.15)
            is_holding = vars_dict.get('is_holding', False)
            holding_robot_id = vars_dict.get('holding_robot_id', None)
        
        # 只有持有物体的机器人才有梯度
        if not is_holding or holding_robot_id != robot_id:
            return np.zeros(3)
        
        robot_pos = x[:2]
        distance_to_receptacle = np.linalg.norm(robot_pos - target_receptacle_position)
        
        # 梯度计算
        gradient = np.zeros(3)
        if distance_to_receptacle > 1e-6:  # 避免除零
            direction = (robot_pos - target_receptacle_position) / distance_to_receptacle
            gradient[:2] = 2 * (distance_to_receptacle - place_dist_thresh) * direction
        
        return gradient
    
    @staticmethod
    def place_time_derivative(x, t, robot_id, vars_dict=None):
        """Place任务效用函数的时间导数"""
        # Place任务通常不直接依赖时间
        return 0.0
    
    @staticmethod
    def apply_motion_control(x, t, robot_id, vars_dict, dt):
        """
        应用Place任务的运动控制
        
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
        
        target_receptacle_position = vars_dict.get('target_receptacle_position', np.array([-0.8, 0.7]))
        place_dist_thresh = vars_dict.get('place_dist_thresh', 0.15)
        is_holding = vars_dict.get('is_holding', False)
        holding_robot_id = vars_dict.get('holding_robot_id', None)
        
        # 只有持有物体的机器人才能执行Place
        if not is_holding or holding_robot_id != robot_id:
            return x
        
        robot_pos = x[:2]
        distance_to_receptacle = np.linalg.norm(robot_pos - target_receptacle_position)
        
        # 如果已经在放置范围内，执行放置动作
        if distance_to_receptacle <= place_dist_thresh:
            # 模拟放置成功
            if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                # 使用全局变量管理器
                vars_dict.set_var('is_holding', False)
                vars_dict.set_var('holding_robot_id', None)
            else:
                # 普通字典
                vars_dict['is_holding'] = False
                vars_dict['holding_robot_id'] = None
            
            print(f"Robot {robot_id} successfully placed object at position {target_receptacle_position}")
            return x
        
        # 向目标放置位置移动
        direction = target_receptacle_position - robot_pos
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