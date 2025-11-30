# task_utils/manipulation_module.py
import numpy as np
from enum import Enum

class ManipulationPhase(Enum):
    """
    操作任务的阶段枚举
    """
    NAV_OBJ = 0    # 导航到物体
    PICK = 1       # 抓取物体
    NAV_REC = 2    # 导航到目标位置
    PLACE = 3      # 放置物体

class ManipulationTask:
    """
    操作任务的实现类，对应 partnr-planner 中的 rearrange 组合技能。
    这是一个复合任务，包括导航到物体、抓取、导航到目标位置、放置等步骤。
    """
    
    @staticmethod
    def get_global_vars_dict():
        """
        从GlobalVarsManager获取全局变量字典
        """
        global_vars_dict = None
        try:
            # 获取GlobalVarsManager实例
            import sys
            global_vars_manager = None
            
            # 在所有模块中搜索GlobalVarsManager实例
            for module_name, module in sys.modules.items():
                if hasattr(module, 'global_vars'):
                    global_vars_manager = getattr(module, 'global_vars')
                    break
            
            if global_vars_manager is not None:
                if hasattr(global_vars_manager, 'get_all_vars'):
                    global_vars_dict = global_vars_manager.get_all_vars()
                elif hasattr(global_vars_manager, 'get_var'):
                    # 构建字典，包含所有操作任务需要的变量
                    global_vars_dict = {
                        'manipulation_phase': global_vars_manager.get_var('manipulation_phase', ManipulationPhase.NAV_OBJ),
                        'target_object_position': global_vars_manager.get_var('target_object_position', np.array([0.8, -0.5])),
                        'target_receptacle_position': global_vars_manager.get_var('target_receptacle_position', np.array([-0.8, 0.7])),
                        'pick_dist_thresh': global_vars_manager.get_var('pick_dist_thresh', 0.3),
                        'place_dist_thresh': global_vars_manager.get_var('place_dist_thresh', 0.3),
                        'pick_action_value': global_vars_manager.get_var('pick_action_value', 1.0),
                        'place_action_value': global_vars_manager.get_var('place_action_value', 1.0),
                        'is_holding': global_vars_manager.get_var('is_holding', False),
                        'holding_robot_id': global_vars_manager.get_var('holding_robot_id', None)
                    }
                else:
                    print("全局变量管理器缺少必要的方法")
                    global_vars_dict = {}
            else:
                print("无法找到global_vars_manager")
        
        except Exception as e:
            print(f"获取全局变量管理器时出错: {e}")
            global_vars_dict = None
            
        return global_vars_dict
    
    @staticmethod
    def manipulation_function(x_i, t, i, vars_dict=None):
        if vars_dict is None:
            vars_dict = ManipulationTask.get_global_vars_dict()
            if vars_dict is None: return -100

        current_phase = vars_dict.get('manipulation_phase', ManipulationPhase.NAV_OBJ)
        is_holding = vars_dict.get('is_holding', False)
        holding_robot_id = vars_dict.get('holding_robot_id', None) # 获取持有者ID
        nav_attraction_factor = 10.0 # 大幅增加导航吸引力
        
        # 阶段权重系数，后期阶段权重更高
        phase_weights = {
            ManipulationPhase.NAV_OBJ: 1.0,  # 基础权重
            ManipulationPhase.PICK: 1.2,     # 略高权重
            ManipulationPhase.NAV_REC: 1.5,  # 较高权重
            ManipulationPhase.PLACE: 2.0     # 最高权重
        }
        current_weight = phase_weights.get(current_phase, 1.0)

        if is_holding:
            if holding_robot_id == i:
                if current_phase == ManipulationPhase.NAV_REC:
                    target_pos = vars_dict.get('target_receptacle_position')
                    if target_pos is None: return -0.1 * np.linalg.norm(x_i[0:2])**2
                    pos_error = -nav_attraction_factor * 0.5 * np.linalg.norm(x_i[0:2] - target_pos)**2
                    target_vector = target_pos - x_i[0:2]
                    if np.linalg.norm(target_vector) < 1e-6: desired_angle = x_i[2]
                    else: desired_angle = np.arctan2(target_vector[1], target_vector[0])
                    angle_diff = np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
                    angle_error = -0.3 * angle_diff**2
                    return current_weight * (pos_error + angle_error) + 20.0
                elif current_phase == ManipulationPhase.PLACE:
                    return 50.0 * current_weight
            else:
                if current_phase == ManipulationPhase.NAV_OBJ or current_phase == ManipulationPhase.PICK:
                    return -100.0 
                if current_phase == ManipulationPhase.NAV_REC or current_phase == ManipulationPhase.PLACE:
                    return -100.0 
        
        if (current_phase == ManipulationPhase.NAV_REC or current_phase == ManipulationPhase.PLACE):
            if not is_holding or holding_robot_id != i:
                return -100.0

        if current_phase == ManipulationPhase.NAV_OBJ:
            target_obj = vars_dict.get('target_object_position')
            if target_obj is None: return -0.1 * np.linalg.norm(x_i[0:2])**2
            pos_error = -nav_attraction_factor * 0.5 * np.linalg.norm(x_i[0:2] - target_obj)**2
            target_vector = target_obj - x_i[0:2]
            if np.linalg.norm(target_vector) < 1e-6: desired_angle = x_i[2]
            else: desired_angle = np.arctan2(target_vector[1], target_vector[0])
            angle_diff = np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
            angle_error = -0.3 * angle_diff**2
            return current_weight * (pos_error + angle_error)

        elif current_phase == ManipulationPhase.PICK:
            target_obj = vars_dict.get('target_object_position')
            pick_dist_thresh = vars_dict.get('pick_dist_thresh', 0.3)
            if target_obj is not None:
                dist_to_obj = np.linalg.norm(x_i[0:2] - target_obj)
                if dist_to_obj < pick_dist_thresh:
                    pick_action_value = vars_dict.get('pick_action_value', 1.0)
                    proximity_bonus = 1.0 - (dist_to_obj / pick_dist_thresh)
                    return current_weight * (pick_action_value + proximity_bonus)
                else:
                    return -10.0 * dist_to_obj
            else:
                return -100.0

        elif current_phase == ManipulationPhase.NAV_REC:
            target_pos = vars_dict.get('target_receptacle_position')
            if target_pos is None: return -0.1 * np.linalg.norm(x_i[0:2])**2
            pos_error = -nav_attraction_factor * 0.5 * np.linalg.norm(x_i[0:2] - target_pos)**2
            target_vector = target_pos - x_i[0:2]
            if np.linalg.norm(target_vector) < 1e-6: desired_angle = x_i[2]
            else: desired_angle = np.arctan2(target_vector[1], target_vector[0])
            angle_diff = np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
            angle_error = -0.3 * angle_diff**2
            if is_holding and holding_robot_id == i:
                return current_weight * (pos_error + angle_error) + 20.0
            return current_weight * (pos_error + angle_error)

        elif current_phase == ManipulationPhase.PLACE:
            target_pos = vars_dict.get('target_receptacle_position')
            place_dist_thresh = vars_dict.get('place_dist_thresh', 0.3)
            if target_pos is not None:
                dist_to_rec = np.linalg.norm(x_i[0:2] - target_pos)
                if dist_to_rec < place_dist_thresh:
                    place_action_value = vars_dict.get('place_action_value', 1.0)
                    proximity_bonus = 1.0 - (dist_to_rec / place_dist_thresh)
                    if is_holding and holding_robot_id == i:
                        return current_weight * (place_action_value + proximity_bonus) + 30.0
                    return current_weight * (place_action_value + proximity_bonus)
                else:
                    # 不在PLACE范围内，给予负的任务函数值，引导机器人回到目标位置
                    return -10.0 * dist_to_rec
            else:
                return -100.0
        else:
            print(f"Warning: Unknown manipulation phase {current_phase}")
            return -0.1 * np.linalg.norm(x_i[0:2])**2

    @staticmethod
    def manipulation_gradient(x_i, t, i, vars_dict=None):
        """
        Return:
            目标函数相对于状态的梯度 [dx, dy, dtheta]
        """
        if vars_dict is None:
            vars_dict = ManipulationTask.get_global_vars_dict()
            if vars_dict is None: return np.zeros(3)

        current_phase = vars_dict.get('manipulation_phase', ManipulationPhase.NAV_OBJ)
        is_holding = vars_dict.get('is_holding', False)
        holding_robot_id = vars_dict.get('holding_robot_id', None)
        gradient = np.zeros(3)
        nav_attraction_factor = 10.0

        if is_holding and (current_phase == ManipulationPhase.NAV_OBJ or current_phase == ManipulationPhase.PICK):
            return gradient
        if (current_phase == ManipulationPhase.NAV_REC or current_phase == ManipulationPhase.PLACE):
            if not is_holding or holding_robot_id != i:
                return gradient

        if current_phase == ManipulationPhase.NAV_OBJ:
            target_obj = vars_dict.get('target_object_position')
            if target_obj is None:
                gradient[0:2] = -0.2 * x_i[0:2]
                return gradient
            gradient[0:2] = -nav_attraction_factor * (x_i[0:2] - target_obj)
            target_vector = target_obj - x_i[0:2]
            if np.linalg.norm(target_vector) > 1e-6:
                desired_angle = np.arctan2(target_vector[1], target_vector[0])
                angle_diff = np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
                gradient[2] = -0.6 * angle_diff # 角度梯度强度暂时不变

        elif current_phase == ManipulationPhase.PICK:
            target_obj = vars_dict.get('target_object_position')
            pick_dist_thresh = vars_dict.get('pick_dist_thresh', 0.3)
            if target_obj is not None:
                dist_to_obj = np.linalg.norm(x_i[0:2] - target_obj)
                if dist_to_obj < pick_dist_thresh:
                    gradient[0:2] = -0.1 * (x_i[0:2] - target_obj)
                    target_vector = target_obj - x_i[0:2]
                    if np.linalg.norm(target_vector) > 1e-6:
                        desired_angle = np.arctan2(target_vector[1], target_vector[0])
                        angle_diff = np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
                        gradient[2] = -0.3 * angle_diff
                else:
                    gradient[0:2] = -nav_attraction_factor * (x_i[0:2] - target_obj)
                    target_vector = target_obj - x_i[0:2]
                    if np.linalg.norm(target_vector) > 1e-6:
                        desired_angle = np.arctan2(target_vector[1], target_vector[0])
                        angle_diff = np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
                        gradient[2] = -0.6 * angle_diff

        elif current_phase == ManipulationPhase.NAV_REC:
            target_pos = vars_dict.get('target_receptacle_position')
            if target_pos is None:
                gradient[0:2] = -0.2 * x_i[0:2]
                return gradient
            gradient[0:2] = -nav_attraction_factor * (x_i[0:2] - target_pos)
            target_vector = target_pos - x_i[0:2]
            if np.linalg.norm(target_vector) > 1e-6:
                desired_angle = np.arctan2(target_vector[1], target_vector[0])
                angle_diff = np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
                gradient[2] = -0.6 * angle_diff

        elif current_phase == ManipulationPhase.PLACE:
            target_pos = vars_dict.get('target_receptacle_position')
            place_dist_thresh = vars_dict.get('place_dist_thresh', 0.3)
            if target_pos is not None:
                dist_to_rec = np.linalg.norm(x_i[0:2] - target_pos)
                if dist_to_rec < place_dist_thresh:
                    gradient[0:2] = -0.1 * (x_i[0:2] - target_pos)
                    target_vector = target_pos - x_i[0:2]
                    if np.linalg.norm(target_vector) > 1e-6:
                        desired_angle = np.arctan2(target_vector[1], target_vector[0])
                        angle_diff = np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
                        gradient[2] = -0.3 * angle_diff
                else:
                    gradient[0:2] = -nav_attraction_factor * (x_i[0:2] - target_pos)
                    target_vector = target_pos - x_i[0:2]
                    if np.linalg.norm(target_vector) > 1e-6:
                        desired_angle = np.arctan2(target_vector[1], target_vector[0])
                        angle_diff = np.arctan2(np.sin(x_i[2] - desired_angle), np.cos(x_i[2] - desired_angle))
                        gradient[2] = -0.6 * angle_diff

        return gradient

    @staticmethod
    def manipulation_time_derivative(x_i, t, i, vars_dict=None):
        return 0 

    @staticmethod
    def apply_motion_control(x_i, t, i, vars_dict, dt):
        if vars_dict is None:
            vars_dict = ManipulationTask.get_global_vars_dict()
            if vars_dict is None:
                return x_i
        current_phase = vars_dict.get('manipulation_phase', ManipulationPhase.NAV_OBJ)
        is_holding = vars_dict.get('is_holding', False)
        holding_robot_id = vars_dict.get('holding_robot_id', None)

        # 如果其他机器人在搬运且处于NAV_REC或PLACE，当前机器人不动
        if is_holding and holding_robot_id != i and (current_phase == ManipulationPhase.NAV_REC or current_phase == ManipulationPhase.PLACE):
            return x_i

        current_pos = x_i[0:2]
        current_theta = x_i[2]

        # --- 统一的导航控制逻辑 ---
        target_pos = None
        if current_phase == ManipulationPhase.NAV_OBJ:
            target_pos = vars_dict.get('target_object_position')
        elif current_phase == ManipulationPhase.NAV_REC:
            target_pos = vars_dict.get('target_receptacle_position')

        if target_pos is not None and (current_phase == ManipulationPhase.NAV_OBJ or current_phase == ManipulationPhase.NAV_REC):
            # 控制参数
            Kp_linear = 0.5  # 线性速度比例增益 (稍快一点)
            Kp_angular = 1.0 # 角速度比例增益 (稍快一点)
            u_max_list = vars_dict.get('u_max', [0.5, 2.5]) # 从全局获取速度限制
            max_v = u_max_list[0] # 最大线速度
            max_omega = u_max_list[1] # 最大角速度
            stop_dist = 0.05  # 停止距离阈值 (非常近)
            angle_stop_threshold = 0.05 # 角度停止阈值 (弧度)

            # 计算误差
            vec_to_target = target_pos - current_pos
            dist_to_target = np.linalg.norm(vec_to_target)

            # 计算目标角度和角度差 (确保vec_to_target非零)
            desired_theta = current_theta # 默认保持当前角度
            if dist_to_target > 1e-6:
                desired_theta = np.arctan2(vec_to_target[1], vec_to_target[0])
            angle_diff = np.arctan2(np.sin(desired_theta - current_theta), np.cos(desired_theta - current_theta)) # desired - current

            # 如果已经非常接近目标
            if dist_to_target < stop_dist:
                forward_speed = 0 # 停止前进
                # P控制角速度以对准
                angular_speed = Kp_angular * angle_diff
                # 如果角度误差也很小，则停止转动
                if abs(angle_diff) < angle_stop_threshold:
                     angular_speed = 0
                # 限制角速度
                angular_speed = np.clip(angular_speed, -max_omega, max_omega)

            # --- 如果距离较远，则同时控制线速度和角速度 ---
            else:
                # P控制线速度
                forward_speed = Kp_linear * dist_to_target
                # 根据角度误差减小线速度
                angle_factor = max(0.0, 1.0 - abs(angle_diff) / (np.pi/2)) # 角度差越大，速度越慢
                forward_speed *= angle_factor
                # 限制线速度
                forward_speed = np.clip(forward_speed, 0, max_v)

                # P控制角速度
                angular_speed = Kp_angular * angle_diff
                # 限制角速度
                angular_speed = np.clip(angular_speed, -max_omega, max_omega)

            # 更新状态
            new_pos_x = current_pos[0] + forward_speed * np.cos(current_theta) * dt
            new_pos_y = current_pos[1] + forward_speed * np.sin(current_theta) * dt
            new_theta = current_theta + angular_speed * dt
            # 限制角度在 [0, 2*pi)
            new_theta = np.mod(new_theta, 2 * np.pi)

            return np.array([new_pos_x, new_pos_y, new_theta])

        elif current_phase == ManipulationPhase.PICK:
            # PICK 阶段通常是瞬间动作或由外部控制，这里保持不动
            return x_i

        elif current_phase == ManipulationPhase.PLACE:
            # PLACE 阶段通常是瞬间动作或由外部控制，这里保持不动
            return x_i

        # 其他未知阶段或无目标时保持不动
        return x_i

    def __init__(self):
        self.manipulation_phase = ManipulationPhase.NAV_OBJ  # 初始化操作阶段
        self.is_holding = False  # 初始化is_holding状态
        self.manipulation_phase_timer = 0  # 初始化操作阶段计时器
        self.manipulation_action_fixed_duration = 1.0  # 假设固定持续时间为1秒

    def update_phase(self, new_phase):
        self.manipulation_phase = new_phase
        self.manipulation_phase_timer = 0  # 重置操作阶段计时器

    def update_timer(self, dt):
        self.manipulation_phase_timer += dt

    def update_is_holding(self):
        # 在PICK阶段结束时更新is_holding状态
        if self.manipulation_phase == ManipulationPhase.PICK:
            if self.manipulation_phase_timer >= self.manipulation_action_fixed_duration:
                self.is_holding = True  # 抓取成功后更新状态

        # 在PLACE阶段结束时更新is_holding状态
        if self.manipulation_phase == ManipulationPhase.PLACE:
            if self.manipulation_phase_timer >= self.manipulation_action_fixed_duration:
                self.is_holding = False  # 放置成功后更新状态 