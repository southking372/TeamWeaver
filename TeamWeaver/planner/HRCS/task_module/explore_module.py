# task_utils/explore_module.py
import numpy as np

class ExploreTask:
    """
    探索任务的实现类，目标是驱动机器人在环境中进行探索和扫描, 模拟 partnr-planner 中 ExploreSkill 的行为与衡量
    该行为基于语义探索代理(Sem_Exp_Env_Agent等) -> 这一部分要和PARTNR对齐
    Function是相关于: 导航到环境中的探索前沿(frontiers)
    """

    @staticmethod
    def get_global_vars_dict():
        global_vars_dict = None
        try:
            import sys
            global_vars_manager = None
            for module_name, module in sys.modules.items():
                if hasattr(module, 'global_vars'):
                    global_vars_manager = getattr(module, 'global_vars')
                    break

            if global_vars_manager is not None:
                if hasattr(global_vars_manager, 'get_all_vars'):
                    global_vars_dict = global_vars_manager.get_all_vars()
                    if 'exploration_frontiers' not in global_vars_dict:
                         global_vars_dict['exploration_frontiers'] = []
                elif hasattr(global_vars_manager, 'get_var'):
                     global_vars_dict = {
                         'exploration_frontiers': global_vars_manager.get_var('exploration_frontiers', []),
                         'explored_map': global_vars_manager.get_var('explored_map'),
                         'explore_dist_thresh': global_vars_manager.get_var('explore_dist_thresh', 0.5), # 可能不再直接使用，但保留
                         'G': global_vars_manager.get_var('G')
                     }
                else:
                    print("全局变量管理器缺少必要的方法")
                    global_vars_dict = {'exploration_frontiers': []}
            else:
                print("无法找到global_vars_manager")
                global_vars_dict = {'exploration_frontiers': []}
        except Exception as e:
            print(f"获取全局变量管理器时出错: {e}")
            global_vars_dict = {'exploration_frontiers': []}
            
        if 'exploration_frontiers' in global_vars_dict and isinstance(global_vars_dict['exploration_frontiers'], list):
             global_vars_dict['exploration_frontiers'] = np.array(global_vars_dict['exploration_frontiers'])
        if global_vars_dict.get('exploration_frontiers') is None:
            global_vars_dict['exploration_frontiers'] = np.array([])
        return global_vars_dict

    @staticmethod
    def explore_function(x_i, t, i, vars_dict):
        """
        Return:
            目标函数值 H
        """
        exploration_frontiers = vars_dict.get('exploration_frontiers')

        if exploration_frontiers is None or exploration_frontiers.shape[0] == 0:
            # print(f"Warning: No exploration frontiers found for ExploreTask.explore_function at t={t}, robot={i}. Assuming exploration complete.")
            # 没有前沿点，认为探索完成，H=0
            return 0.0

        current_pos = x_i[0:2]
        current_angle = x_i[2]

        # 计算所有前沿点到当前位置的距离平方
        distances_sq = np.sum((exploration_frontiers - current_pos)**2, axis=1)
        nearest_frontier_idx = np.argmin(distances_sq)
        nearest_frontier = exploration_frontiers[nearest_frontier_idx]
        min_dist_sq = distances_sq[nearest_frontier_idx]

        # 计算位置误差项 (使用距离平方)
        w_pos = 0.5
        pos_error_term = w_pos * min_dist_sq

        target_vector = nearest_frontier - current_pos
        if np.linalg.norm(target_vector) < 1e-6:
            angle_diff = 0.0
        else:
            desired_angle = np.arctan2(target_vector[1], target_vector[0])
            angle_diff = np.arctan2(np.sin(current_angle - desired_angle), np.cos(current_angle - desired_angle))

        w_ori = 0.3  # 方向权重 (可调)
        angle_error_term = w_ori * angle_diff**2

        H = pos_error_term + angle_error_term

        # 可选：如果距离非常近，可以认为局部完成，给予一个小的负值奖励
        # dist_thresh_sq = vars_dict.get('explore_dist_thresh', 0.5)**2
        # if min_dist_sq < dist_thresh_sq:
        #     H -= 1.0 #  small bonus for reaching near a frontier
        return H

    @staticmethod
    def explore_gradient(x_i, t, i, vars_dict):
        """
        计算探索任务关于机器人状态 x_i 的梯度 (dH/dx)。
        梯度指向 H 增加最快的方向，鼓励控制器使用负梯度来驱动机器人。
        Return:
            目标函数相对于状态的梯度 [dH/dx, dH/dy, dH/dtheta]
        """
        exploration_frontiers = vars_dict.get('exploration_frontiers')
        gradient = np.zeros(3)

        if exploration_frontiers is None or exploration_frontiers.shape[0] == 0:
            return gradient

        current_pos = x_i[0:2]
        current_angle = x_i[2]

        # 找到最近的前沿点 (与 H 函数中相同的逻辑)
        distances_sq = np.sum((exploration_frontiers - current_pos)**2, axis=1)
        nearest_frontier_idx = np.argmin(distances_sq)
        nearest_frontier = exploration_frontiers[nearest_frontier_idx]

        # 计算位置梯度 dH/d(pos) = 2 * w_pos * (current_pos - nearest_frontier)
        w_pos = 0.5
        pos_gradient = 2 * w_pos * (current_pos - nearest_frontier)
        gradient[0:2] = pos_gradient

        # 计算角度梯度 dH/dtheta = 2 * w_ori * angle_diff
        target_vector = nearest_frontier - current_pos
        if np.linalg.norm(target_vector) < 1e-6:
            angle_diff = 0.0
        else:
            desired_angle = np.arctan2(target_vector[1], target_vector[0])
            angle_diff = np.arctan2(np.sin(current_angle - desired_angle), np.cos(current_angle - desired_angle))
        w_ori = 0.3
        angle_gradient = 2 * w_ori * angle_diff
        gradient[2] = angle_gradient

        return gradient

    @staticmethod
    def explore_time_derivative(x_i, t, i, vars_dict):
        """
        Return:
            目标函数相对于时间的导数 (dH/dt), 此处为 0
        """
        return 0.0

    @staticmethod
    def print_debug_info(x_i, t, i, vars_dict, iter_count):
        """
        打印探索任务的调试信息，包括距离最近前沿点的距离、角度差异和函数值。
        
        Parameters:
            x_i: 机器人状态 [x, y, theta]
            t: 当前时间
            i: 机器人索引
            vars_dict: 包含 'exploration_frontiers' 的字典
            iter_count: 当前迭代计数
        """
        if iter_count % 10 != 0:  # 每10次迭代打印一次，避免刷屏
            return
            
        exploration_frontiers = vars_dict.get('exploration_frontiers')
        # 检查探索前沿是否有效
        if exploration_frontiers is None or exploration_frontiers.shape[0] == 0:
            print(f"DEBUG: Robot {i} (Task 1) 无可探索前沿点，探索任务已完成")
            return

        current_pos = x_i[0:2]
        current_angle = x_i[2]
        
        # 计算到最近前沿点的距离
        distances_sq = np.sum((exploration_frontiers - current_pos)**2, axis=1)
        nearest_frontier_idx = np.argmin(distances_sq)
        nearest_frontier = exploration_frontiers[nearest_frontier_idx]
        min_dist = np.sqrt(distances_sq[nearest_frontier_idx])
        
        # 计算角度差异
        target_vector = nearest_frontier - current_pos
        if np.linalg.norm(target_vector) < 1e-6:
            angle_diff = 0.0
        else:
            desired_angle = np.arctan2(target_vector[1], target_vector[0])
            angle_diff = np.arctan2(np.sin(current_angle - desired_angle), np.cos(current_angle - desired_angle))
        
        # 计算探索任务函数值
        H_value = ExploreTask.explore_function(x_i, t, i, vars_dict)
        explore_dist_thresh = vars_dict.get('explore_dist_thresh', 0.5)
        print(f"DEBUG: Robot {i} (Task 1) 探索中. 距最近前沿: {min_dist:.2f}m (阈值: {explore_dist_thresh}), 角度差: {angle_diff:.2f}rad, 函数值: {H_value:.2f}")
        if exploration_frontiers.shape[0] > 1:
            print(f"       共有 {exploration_frontiers.shape[0]} 个探索前沿点待探索")

    @staticmethod
    def apply_motion_control(x_i, t, i, vars_dict, dt):
        """
        应用探索任务的运动控制。
        分为两个阶段：导航到最近的未探索目标点，到达后执行探索动作（原地旋转）。
        Return:
            New Robot States
        """
        # --- 获取所需全局变量 ---
        exploration_targets = vars_dict.get('exploration_targets', [])
        explore_dist_thresh = vars_dict.get('explore_dist_thresh', 0.2)
        exploration_action_timers = vars_dict.get('exploration_action_timers', {})
        exploring_action_info = vars_dict.get('exploring_action_info', {})
        # 注意： exploration_action_duration 需要能从 vars_dict 获取，或者作为常量
        exploration_action_duration = vars_dict.get('exploration_action_duration', 2.0)
        u_max_list = vars_dict.get('u_max', [0.5, 2.5]) # 获取速度限制
        max_v = u_max_list[0]
        max_omega = u_max_list[1]

        current_pos = x_i[0:2]
        current_theta = x_i[2]

        # --- 查找最近的未探索目标 ---
        unexplored_targets = [target for target in exploration_targets if not target.get('explored', False)]

        if not unexplored_targets:
            # print(f"Robot {i}: No unexplored targets left.")
            # 如果没有未探索目标，可以选择停止或执行其他行为（如移动到Voronoi中心）
            # 这里我们让它保持不动
            return x_i

        nearest_target = None
        min_dist_sq = float('inf')
        for target in unexplored_targets:
            dist_sq = np.sum((current_pos - target['position'])**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_target = target

        if nearest_target is None: # 理论上不应发生，除非列表为空
             return x_i

        target_pos = nearest_target['position']
        target_id = nearest_target['id'] # 获取目标ID
        dist_to_target = np.sqrt(min_dist_sq)

        # --- 检查机器人是否正在对此目标执行探索动作 ---
        is_performing_action = exploring_action_info.get(i) == target_id

        # --- 状态转换和运动控制 ---
        if is_performing_action:
            # 阶段 2: 执行探索动作 (原地旋转)
            # 检查计时器是否已完成 (这个检查理论上由 GlobalVarsManager 处理后更新状态)
            # 这里只执行动作
            action_rotation_speed = 1.0 # 探索动作的旋转速度 (rad/s)
            new_theta = current_theta + action_rotation_speed * dt
            new_theta = np.mod(new_theta, 2 * np.pi)
            # print(f"Robot {i} performing explore action at target {target_id}")
            return np.array([current_pos[0], current_pos[1], new_theta])

        elif dist_to_target <= explore_dist_thresh:
            # 阶段 1 -> 2 过渡: 到达目标，开始执行探索动作
            # *重要*: 启动计时器和更新 exploring_action_info 的逻辑应该在外部
            # (例如 newTask.py 或 GlobalVarsManager) 检测到这个状态后触发。
            # apply_motion_control 本身通常只负责计算下一步状态。
            # 这里我们模拟开始动作的第一步：稍微旋转一下
            print(f"Robot {i} reached target {target_id}, starting exploration action.")
            action_rotation_speed = 1.0
            new_theta = current_theta + action_rotation_speed * dt
            new_theta = np.mod(new_theta, 2 * np.pi)
            # 通过返回几乎不变的位置和新角度来表示动作开始
            return np.array([current_pos[0], current_pos[1], new_theta])

        else:
            # 阶段 1: 导航到目标
            # 使用 P 控制器
            Kp_linear = 0.6 # 导航速度增益
            Kp_angular = 1.2 # 导航角速度增益
            stop_dist = explore_dist_thresh # 导航的停止距离就是阈值

            vec_to_target = target_pos - current_pos
            # dist_to_target 已计算

            # 计算目标角度和角度差
            desired_theta = np.arctan2(vec_to_target[1], vec_to_target[0])
            angle_diff = np.arctan2(np.sin(desired_theta - current_theta), np.cos(desired_theta - current_theta))

            # P控制线速度
            forward_speed = Kp_linear * dist_to_target
            angle_factor = max(0.0, 1.0 - abs(angle_diff) / (np.pi/2))
            forward_speed *= angle_factor
            forward_speed = np.clip(forward_speed, 0, max_v)

            # P控制角速度
            angular_speed = Kp_angular * angle_diff
            angular_speed = np.clip(angular_speed, -max_omega, max_omega)

            # 更新状态
            new_pos_x = current_pos[0] + forward_speed * np.cos(current_theta) * dt
            new_pos_y = current_pos[1] + forward_speed * np.sin(current_theta) * dt
            new_theta = current_theta + angular_speed * dt
            new_theta = np.mod(new_theta, 2 * np.pi)

            # print(f"Robot {i} navigating to explore target {target_id}, dist: {dist_to_target:.2f}")
            return np.array([new_pos_x, new_pos_y, new_theta])
