# task_utils/task_module/state_manipulation_modules.py
import numpy as np

class FillTask:
    """Fill任务模块 - 处理填充容器的任务（Agent 1专有能力）"""
    
    @staticmethod
    def fill_function(x, t, robot_id, vars_dict=None):
        """Fill任务效用函数"""
        # Agent 0 不具备Fill能力
        if robot_id == 0:
            return 1000.0
        
        if vars_dict is None:
            target_container_position = np.array([0.3, 0.8])
            fill_dist_thresh = 0.15
            container_filled_state = False
        else:
            target_container_position = vars_dict.get('target_container_position', np.array([0.3, 0.8]))
            fill_dist_thresh = vars_dict.get('fill_dist_thresh', 0.15)
            container_filled_state = vars_dict.get('container_filled_state', False)
        
        if container_filled_state:
            return 0.1
        
        robot_pos = x[:2]
        distance_to_container = np.linalg.norm(robot_pos - target_container_position)
        utility = (distance_to_container - fill_dist_thresh) ** 2
        
        if distance_to_container <= fill_dist_thresh:
            utility = 0.1
        
        return max(0.0, utility)
    
    @staticmethod
    def fill_gradient(x, t, robot_id, vars_dict=None):
        """Fill任务效用函数的梯度"""
        if robot_id == 0:
            return np.zeros(3)
        
        if vars_dict is None:
            target_container_position = np.array([0.3, 0.8])
            fill_dist_thresh = 0.15
            container_filled_state = False
        else:
            target_container_position = vars_dict.get('target_container_position', np.array([0.3, 0.8]))
            fill_dist_thresh = vars_dict.get('fill_dist_thresh', 0.15)
            container_filled_state = vars_dict.get('container_filled_state', False)
        
        if container_filled_state:
            return np.zeros(3)
        
        robot_pos = x[:2]
        distance_to_container = np.linalg.norm(robot_pos - target_container_position)
        
        gradient = np.zeros(3)
        if distance_to_container > 1e-6:
            direction = (robot_pos - target_container_position) / distance_to_container
            gradient[:2] = 2 * (distance_to_container - fill_dist_thresh) * direction
        
        return gradient
    
    @staticmethod
    def fill_time_derivative(x, t, robot_id, vars_dict=None):
        return 0.0
    
    @staticmethod
    def apply_motion_control(x, t, robot_id, vars_dict, dt):
        """Fill任务的运动控制"""
        if robot_id == 0:
            return x
        
        if vars_dict is None:
            return x
        
        target_container_position = vars_dict.get('target_container_position', np.array([0.3, 0.8]))
        fill_dist_thresh = vars_dict.get('fill_dist_thresh', 0.15)
        container_filled_state = vars_dict.get('container_filled_state', False)
        
        if container_filled_state:
            return x
        
        robot_pos = x[:2]
        distance_to_container = np.linalg.norm(robot_pos - target_container_position)
        
        if distance_to_container <= fill_dist_thresh:
            # 模拟填充成功
            if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                vars_dict.set_var('container_filled_state', True)
            else:
                vars_dict['container_filled_state'] = True
            
            print(f"Robot {robot_id} successfully filled container at position {target_container_position}")
            return x
        
        # 向目标容器移动
        direction = target_container_position - robot_pos
        distance = np.linalg.norm(direction)
        
        if distance > 1e-6:
            direction = direction / distance
            desired_theta = np.arctan2(direction[1], direction[0])
            theta_diff = desired_theta - x[2]
            theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
            
            linear_speed = 0.3
            angular_speed = 1.0
            new_x = x.copy()
            
            if abs(theta_diff) > 0.1:
                new_x[2] += np.sign(theta_diff) * angular_speed * dt
            else:
                new_x[0] += linear_speed * np.cos(x[2]) * dt
                new_x[1] += linear_speed * np.sin(x[2]) * dt
                new_x[2] += 0.5 * theta_diff * dt
            
            new_x[2] = np.arctan2(np.sin(new_x[2]), np.cos(new_x[2]))
            return new_x
        
        return x

class PourTask:
    """Pour任务模块 - 处理倾倒液体的任务（Agent 1专有能力）"""
    
    @staticmethod
    def pour_function(x, t, robot_id, vars_dict=None):
        """Pour任务效用函数"""
        if robot_id == 0:
            return 1000.0
        
        if vars_dict is None:
            target_receptacle_position = np.array([-0.3, 0.6])
            pour_dist_thresh = 0.15
            pour_completed_state = False
        else:
            target_receptacle_position = vars_dict.get('target_receptacle_position', np.array([-0.3, 0.6]))
            pour_dist_thresh = vars_dict.get('pour_dist_thresh', 0.15)
            pour_completed_state = vars_dict.get('pour_completed_state', False)
        
        if pour_completed_state:
            return 0.1
        
        robot_pos = x[:2]
        distance_to_receptacle = np.linalg.norm(robot_pos - target_receptacle_position)
        utility = (distance_to_receptacle - pour_dist_thresh) ** 2
        
        if distance_to_receptacle <= pour_dist_thresh:
            utility = 0.1
        
        return max(0.0, utility)
    
    @staticmethod
    def pour_gradient(x, t, robot_id, vars_dict=None):
        """Pour任务效用函数的梯度"""
        if robot_id == 0:
            return np.zeros(3)
        
        if vars_dict is None:
            target_receptacle_position = np.array([-0.3, 0.6])
            pour_dist_thresh = 0.15
            pour_completed_state = False
        else:
            target_receptacle_position = vars_dict.get('target_receptacle_position', np.array([-0.3, 0.6]))
            pour_dist_thresh = vars_dict.get('pour_dist_thresh', 0.15)
            pour_completed_state = vars_dict.get('pour_completed_state', False)
        
        if pour_completed_state:
            return np.zeros(3)
        
        robot_pos = x[:2]
        distance_to_receptacle = np.linalg.norm(robot_pos - target_receptacle_position)
        
        gradient = np.zeros(3)
        if distance_to_receptacle > 1e-6:
            direction = (robot_pos - target_receptacle_position) / distance_to_receptacle
            gradient[:2] = 2 * (distance_to_receptacle - pour_dist_thresh) * direction
        
        return gradient
    
    @staticmethod
    def pour_time_derivative(x, t, robot_id, vars_dict=None):
        return 0.0
    
    @staticmethod
    def apply_motion_control(x, t, robot_id, vars_dict, dt):
        """Pour任务的运动控制"""
        if robot_id == 0:
            return x
        
        if vars_dict is None:
            return x
        
        target_receptacle_position = vars_dict.get('target_receptacle_position', np.array([-0.3, 0.6]))
        pour_dist_thresh = vars_dict.get('pour_dist_thresh', 0.15)
        pour_completed_state = vars_dict.get('pour_completed_state', False)
        
        if pour_completed_state:
            return x
        
        robot_pos = x[:2]
        distance_to_receptacle = np.linalg.norm(robot_pos - target_receptacle_position)
        
        if distance_to_receptacle <= pour_dist_thresh:
            # 模拟倾倒成功
            if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                vars_dict.set_var('pour_completed_state', True)
            else:
                vars_dict['pour_completed_state'] = True
            
            print(f"Robot {robot_id} successfully poured into receptacle at position {target_receptacle_position}")
            return x
        
        # 向目标容器移动
        direction = target_receptacle_position - robot_pos
        distance = np.linalg.norm(direction)
        
        if distance > 1e-6:
            direction = direction / distance
            desired_theta = np.arctan2(direction[1], direction[0])
            theta_diff = desired_theta - x[2]
            theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
            
            linear_speed = 0.3
            angular_speed = 1.0
            new_x = x.copy()
            
            if abs(theta_diff) > 0.1:
                new_x[2] += np.sign(theta_diff) * angular_speed * dt
            else:
                new_x[0] += linear_speed * np.cos(x[2]) * dt
                new_x[1] += linear_speed * np.sin(x[2]) * dt
                new_x[2] += 0.5 * theta_diff * dt
            
            new_x[2] = np.arctan2(np.sin(new_x[2]), np.cos(new_x[2]))
            return new_x
        
        return x

class PowerOnTask:
    """PowerOn任务模块 - 处理开启设备电源的任务（Agent 1专有能力）"""
    
    @staticmethod
    def poweron_function(x, t, robot_id, vars_dict=None):
        """PowerOn任务效用函数"""
        if robot_id == 0:
            return 1000.0
        
        if vars_dict is None:
            target_device_position = np.array([0.7, 0.4])
            power_dist_thresh = 0.15
            device_power_state = False
        else:
            target_device_position = vars_dict.get('target_device_position', np.array([0.7, 0.4]))
            power_dist_thresh = vars_dict.get('power_dist_thresh', 0.15)
            device_power_state = vars_dict.get('device_power_state', False)
        
        if device_power_state:
            return 0.1
        
        robot_pos = x[:2]
        distance_to_device = np.linalg.norm(robot_pos - target_device_position)
        utility = (distance_to_device - power_dist_thresh) ** 2
        
        if distance_to_device <= power_dist_thresh:
            utility = 0.1
        
        return max(0.0, utility)
    
    @staticmethod
    def poweron_gradient(x, t, robot_id, vars_dict=None):
        """PowerOn任务效用函数的梯度"""
        if robot_id == 0:
            return np.zeros(3)
        
        if vars_dict is None:
            target_device_position = np.array([0.7, 0.4])
            power_dist_thresh = 0.15
            device_power_state = False
        else:
            target_device_position = vars_dict.get('target_device_position', np.array([0.7, 0.4]))
            power_dist_thresh = vars_dict.get('power_dist_thresh', 0.15)
            device_power_state = vars_dict.get('device_power_state', False)
        
        if device_power_state:
            return np.zeros(3)
        
        robot_pos = x[:2]
        distance_to_device = np.linalg.norm(robot_pos - target_device_position)
        
        gradient = np.zeros(3)
        if distance_to_device > 1e-6:
            direction = (robot_pos - target_device_position) / distance_to_device
            gradient[:2] = 2 * (distance_to_device - power_dist_thresh) * direction
        
        return gradient
    
    @staticmethod
    def poweron_time_derivative(x, t, robot_id, vars_dict=None):
        return 0.0
    
    @staticmethod
    def apply_motion_control(x, t, robot_id, vars_dict, dt):
        """PowerOn任务的运动控制"""
        if robot_id == 0:
            return x
        
        if vars_dict is None:
            return x
        
        target_device_position = vars_dict.get('target_device_position', np.array([0.7, 0.4]))
        power_dist_thresh = vars_dict.get('power_dist_thresh', 0.15)
        device_power_state = vars_dict.get('device_power_state', False)
        
        if device_power_state:
            return x
        
        robot_pos = x[:2]
        distance_to_device = np.linalg.norm(robot_pos - target_device_position)
        
        if distance_to_device <= power_dist_thresh:
            # 模拟开启设备成功
            if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                vars_dict.set_var('device_power_state', True)
            else:
                vars_dict['device_power_state'] = True
            
            print(f"Robot {robot_id} successfully powered on device at position {target_device_position}")
            return x
        
        # 向目标设备移动
        direction = target_device_position - robot_pos
        distance = np.linalg.norm(direction)
        
        if distance > 1e-6:
            direction = direction / distance
            desired_theta = np.arctan2(direction[1], direction[0])
            theta_diff = desired_theta - x[2]
            theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
            
            linear_speed = 0.3
            angular_speed = 1.0
            new_x = x.copy()
            
            if abs(theta_diff) > 0.1:
                new_x[2] += np.sign(theta_diff) * angular_speed * dt
            else:
                new_x[0] += linear_speed * np.cos(x[2]) * dt
                new_x[1] += linear_speed * np.sin(x[2]) * dt
                new_x[2] += 0.5 * theta_diff * dt
            
            new_x[2] = np.arctan2(np.sin(new_x[2]), np.cos(new_x[2]))
            return new_x
        
        return x

class PowerOffTask:
    """PowerOff任务模块 - 处理关闭设备电源的任务（Agent 1专有能力）"""
    
    @staticmethod
    def poweroff_function(x, t, robot_id, vars_dict=None):
        """PowerOff任务效用函数"""
        if robot_id == 0:
            return 1000.0
        
        if vars_dict is None:
            target_device_position = np.array([0.7, 0.4])
            power_dist_thresh = 0.15
            device_power_state = True  # 需要关闭，所以初始状态为开启
        else:
            target_device_position = vars_dict.get('target_device_position', np.array([0.7, 0.4]))
            power_dist_thresh = vars_dict.get('power_dist_thresh', 0.15)
            device_power_state = vars_dict.get('device_power_state', True)
        
        if not device_power_state:
            return 0.1  # 已关闭
        
        robot_pos = x[:2]
        distance_to_device = np.linalg.norm(robot_pos - target_device_position)
        utility = (distance_to_device - power_dist_thresh) ** 2
        
        if distance_to_device <= power_dist_thresh:
            utility = 0.1
        
        return max(0.0, utility)
    
    @staticmethod
    def poweroff_gradient(x, t, robot_id, vars_dict=None):
        """PowerOff任务效用函数的梯度"""
        if robot_id == 0:
            return np.zeros(3)
        
        if vars_dict is None:
            target_device_position = np.array([0.7, 0.4])
            power_dist_thresh = 0.15
            device_power_state = True
        else:
            target_device_position = vars_dict.get('target_device_position', np.array([0.7, 0.4]))
            power_dist_thresh = vars_dict.get('power_dist_thresh', 0.15)
            device_power_state = vars_dict.get('device_power_state', True)
        
        if not device_power_state:
            return np.zeros(3)
        
        robot_pos = x[:2]
        distance_to_device = np.linalg.norm(robot_pos - target_device_position)
        
        gradient = np.zeros(3)
        if distance_to_device > 1e-6:
            direction = (robot_pos - target_device_position) / distance_to_device
            gradient[:2] = 2 * (distance_to_device - power_dist_thresh) * direction
        
        return gradient
    
    @staticmethod
    def poweroff_time_derivative(x, t, robot_id, vars_dict=None):
        return 0.0
    
    @staticmethod
    def apply_motion_control(x, t, robot_id, vars_dict, dt):
        """PowerOff任务的运动控制"""
        if robot_id == 0:
            return x
        
        if vars_dict is None:
            return x
        
        target_device_position = vars_dict.get('target_device_position', np.array([0.7, 0.4]))
        power_dist_thresh = vars_dict.get('power_dist_thresh', 0.15)
        device_power_state = vars_dict.get('device_power_state', True)
        
        if not device_power_state:
            return x
        
        robot_pos = x[:2]
        distance_to_device = np.linalg.norm(robot_pos - target_device_position)
        
        if distance_to_device <= power_dist_thresh:
            # 模拟关闭设备成功
            if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                vars_dict.set_var('device_power_state', False)
            else:
                vars_dict['device_power_state'] = False
            
            print(f"Robot {robot_id} successfully powered off device at position {target_device_position}")
            return x
        
        # 向目标设备移动
        direction = target_device_position - robot_pos
        distance = np.linalg.norm(direction)
        
        if distance > 1e-6:
            direction = direction / distance
            desired_theta = np.arctan2(direction[1], direction[0])
            theta_diff = desired_theta - x[2]
            theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
            
            linear_speed = 0.3
            angular_speed = 1.0
            new_x = x.copy()
            
            if abs(theta_diff) > 0.1:
                new_x[2] += np.sign(theta_diff) * angular_speed * dt
            else:
                new_x[0] += linear_speed * np.cos(x[2]) * dt
                new_x[1] += linear_speed * np.sin(x[2]) * dt
                new_x[2] += 0.5 * theta_diff * dt
            
            new_x[2] = np.arctan2(np.sin(new_x[2]), np.cos(new_x[2]))
            return new_x
        
        return x

class RearrangeTask:
    """Rearrange任务模块 - 处理重新排列对象的任务（组合Pick+Place）"""
    
    @staticmethod
    def rearrange_function(x, t, robot_id, vars_dict=None):
        """Rearrange任务效用函数 - 组合Pick和Place的逻辑"""
        if vars_dict is None:
            target_object_position = np.array([0.8, -0.5])
            target_receptacle_position = np.array([-0.8, 0.7])
            pick_dist_thresh = 0.15
            place_dist_thresh = 0.15
            is_holding = False
            holding_robot_id = None
        else:
            target_object_position = vars_dict.get('target_object_position', np.array([0.8, -0.5]))
            target_receptacle_position = vars_dict.get('target_receptacle_position', np.array([-0.8, 0.7]))
            pick_dist_thresh = vars_dict.get('pick_dist_thresh', 0.15)
            place_dist_thresh = vars_dict.get('place_dist_thresh', 0.15)
            is_holding = vars_dict.get('is_holding', False)
            holding_robot_id = vars_dict.get('holding_robot_id', None)
        
        robot_pos = x[:2]
        
        # 如果其他机器人持有物体，当前机器人无法执行Rearrange
        if is_holding and holding_robot_id != robot_id:
            return 1000.0
        
        # 如果当前机器人持有物体，计算到放置位置的距离
        if is_holding and holding_robot_id == robot_id:
            distance_to_receptacle = np.linalg.norm(robot_pos - target_receptacle_position)
            utility = (distance_to_receptacle - place_dist_thresh) ** 2
            if distance_to_receptacle <= place_dist_thresh:
                utility = 0.1  # 接近完成
        else:
            # 如果没有持有物体，计算到对象的距离
            distance_to_object = np.linalg.norm(robot_pos - target_object_position)
            utility = (distance_to_object - pick_dist_thresh) ** 2
            if distance_to_object <= pick_dist_thresh:
                utility = 0.5  # 中间状态，需要转换到Place阶段
        
        return max(0.0, utility)
    
    @staticmethod
    def rearrange_gradient(x, t, robot_id, vars_dict=None):
        """Rearrange任务效用函数的梯度"""
        if vars_dict is None:
            target_object_position = np.array([0.8, -0.5])
            target_receptacle_position = np.array([-0.8, 0.7])
            pick_dist_thresh = 0.15
            place_dist_thresh = 0.15
            is_holding = False
            holding_robot_id = None
        else:
            target_object_position = vars_dict.get('target_object_position', np.array([0.8, -0.5]))
            target_receptacle_position = vars_dict.get('target_receptacle_position', np.array([-0.8, 0.7]))
            pick_dist_thresh = vars_dict.get('pick_dist_thresh', 0.15)
            place_dist_thresh = vars_dict.get('place_dist_thresh', 0.15)
            is_holding = vars_dict.get('is_holding', False)
            holding_robot_id = vars_dict.get('holding_robot_id', None)
        
        robot_pos = x[:2]
        gradient = np.zeros(3)
        
        # 如果其他机器人持有物体，零梯度
        if is_holding and holding_robot_id != robot_id:
            return gradient
        
        # 根据当前状态计算梯度
        if is_holding and holding_robot_id == robot_id:
            # Place阶段
            distance_to_receptacle = np.linalg.norm(robot_pos - target_receptacle_position)
            if distance_to_receptacle > 1e-6:
                direction = (robot_pos - target_receptacle_position) / distance_to_receptacle
                gradient[:2] = 2 * (distance_to_receptacle - place_dist_thresh) * direction
        else:
            # Pick阶段
            distance_to_object = np.linalg.norm(robot_pos - target_object_position)
            if distance_to_object > 1e-6:
                direction = (robot_pos - target_object_position) / distance_to_object
                gradient[:2] = 2 * (distance_to_object - pick_dist_thresh) * direction
        
        return gradient
    
    @staticmethod
    def rearrange_time_derivative(x, t, robot_id, vars_dict=None):
        return 0.0
    
    @staticmethod
    def apply_motion_control(x, t, robot_id, vars_dict, dt):
        """Rearrange任务的运动控制 - 自动状态机Pick→Place"""
        if vars_dict is None:
            return x
        
        target_object_position = vars_dict.get('target_object_position', np.array([0.8, -0.5]))
        target_receptacle_position = vars_dict.get('target_receptacle_position', np.array([-0.8, 0.7]))
        pick_dist_thresh = vars_dict.get('pick_dist_thresh', 0.15)
        place_dist_thresh = vars_dict.get('place_dist_thresh', 0.15)
        is_holding = vars_dict.get('is_holding', False)
        holding_robot_id = vars_dict.get('holding_robot_id', None)
        
        robot_pos = x[:2]
        
        # 如果其他机器人持有物体，停止移动
        if is_holding and holding_robot_id != robot_id:
            return x
        
        # Place阶段：机器人持有物体，需要放置
        if is_holding and holding_robot_id == robot_id:
            distance_to_receptacle = np.linalg.norm(robot_pos - target_receptacle_position)
            
            if distance_to_receptacle <= place_dist_thresh:
                # 执行放置
                if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                    vars_dict.set_var('is_holding', False)
                    vars_dict.set_var('holding_robot_id', None)
                else:
                    vars_dict['is_holding'] = False
                    vars_dict['holding_robot_id'] = None
                
                print(f"Robot {robot_id} completed Rearrange: placed object at {target_receptacle_position}")
                return x
            
            # 向放置位置移动
            target_position = target_receptacle_position
        else:
            # Pick阶段：需要抓取物体
            distance_to_object = np.linalg.norm(robot_pos - target_object_position)
            
            if distance_to_object <= pick_dist_thresh:
                # 执行抓取
                if hasattr(vars_dict, '__class__') and hasattr(vars_dict.__class__, '__name__'):
                    vars_dict.set_var('is_holding', True)
                    vars_dict.set_var('holding_robot_id', robot_id)
                else:
                    vars_dict['is_holding'] = True
                    vars_dict['holding_robot_id'] = robot_id
                
                print(f"Robot {robot_id} Rearrange: picked object, now moving to place")
                return x
            
            # 向抓取位置移动
            target_position = target_object_position
        
        # 通用移动逻辑
        direction = target_position - robot_pos
        distance = np.linalg.norm(direction)
        
        if distance > 1e-6:
            direction = direction / distance
            desired_theta = np.arctan2(direction[1], direction[0])
            theta_diff = desired_theta - x[2]
            theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
            
            linear_speed = 0.3
            angular_speed = 1.0
            new_x = x.copy()
            
            if abs(theta_diff) > 0.1:
                new_x[2] += np.sign(theta_diff) * angular_speed * dt
            else:
                new_x[0] += linear_speed * np.cos(x[2]) * dt
                new_x[1] += linear_speed * np.sin(x[2]) * dt
                new_x[2] += 0.5 * theta_diff * dt
            
            new_x[2] = np.arctan2(np.sin(new_x[2]), np.cos(new_x[2]))
            return new_x
        
        return x 