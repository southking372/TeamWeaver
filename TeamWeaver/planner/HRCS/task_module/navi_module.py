# HRCS/task_module/navi_module.py
import numpy as np
import warnings
from habitat_llm.planner.HRCS.sys_module.tools_util import clamp

class NaviTask:

    @staticmethod
    def get_navi_params(vars_dict):
        if vars_dict is None:
            raise ValueError("NaviTask.get_navi_params 需要一个有效的 vars_dict。")

        required_keys = ['p_goal', 'theta_goal', 'dist_thresh', 'orientation_weight']
        missing_keys = [key for key in required_keys if key not in vars_dict]
        if missing_keys:
            raise KeyError(f"NaviTask 需要的导航参数在 vars_dict 中缺失: {missing_keys}")

        p_goal = vars_dict['p_goal']
        theta_goal = vars_dict['theta_goal']
        dist_thresh = vars_dict['dist_thresh']
        orientation_weight = vars_dict['orientation_weight']

        if not isinstance(p_goal, np.ndarray):
            p_goal = np.array(p_goal)
        if p_goal.shape != (2,):
            warnings.warn(f"'p_goal' 缺失，使用回退值 0")
            p_goal = np.zeros(2, dtype=np.float32)
            #raise ValueError(f"NaviTask 收到的 'p_goal' 形状应为 (2,), 但得到 {p_goal.shape}")

        return p_goal, theta_goal, dist_thresh, orientation_weight

    @staticmethod
    def navi_function(x_i, t, i, vars_dict):
        """
        计算导航任务函数值 H(x_i)。
        定义为误差的负值，目标是最大化这个值。
        H(x_i) = -0.5 * ||pos(x_i) - p_goal||^2 - w_orient * angle_diff(theta(x_i), theta_goal)^2 + bonus

        参数:
            x_i: 机器人状态 [x, y, theta]
            t: 当前时间 (在这个静态目标版本中未使用)
            i: 机器人索引 (未使用)
            vars_dict: 包含导航参数的字典 ('p_goal', 'theta_goal', 'dist_thresh', 'orientation_weight')

        返回:
            目标函数值
        """
        p_goal, theta_goal, dist_thresh, orientation_weight = NaviTask.get_navi_params(vars_dict)

        robot_pos = x_i[0:2]
        robot_theta = x_i[2]
        pos_error_term = -0.5 * np.linalg.norm(robot_pos - p_goal)**2

        angle_diff = np.arctan2(np.sin(robot_theta - theta_goal), np.cos(robot_theta - theta_goal))
        angle_error_term = -orientation_weight * angle_diff**2
        dist_to_goal = np.linalg.norm(robot_pos - p_goal)
        arrival_bonus = 2.0 if dist_to_goal < dist_thresh else 0.0

        return pos_error_term + angle_error_term + arrival_bonus

    @staticmethod
    def navi_gradient(x_i, t, i, vars_dict):
        """
        计算导航任务函数 H(x_i) 关于机器人状态 x_i 的梯度 (dH/dx_i)。

        返回:
            目标函数相对于状态的梯度 [dH/dx, dH/dy, dH/dtheta]
        """
        p_goal, theta_goal, dist_thresh, orientation_weight = NaviTask.get_navi_params(vars_dict)

        robot_pos = x_i[0:2]
        robot_theta = x_i[2]

        gradient = np.zeros(3)

        # 位置梯度: d/dx (-0.5 * ||x - p_goal||^2) = -(x - p_goal)
        gradient[0:2] = -(robot_pos - p_goal)

        # 角度梯度: d/dtheta (-w * angle_diff^2) = -2 * w * angle_diff
        # angle_diff = atan2(sin(theta - theta_goal), cos(theta - theta_goal)) ≈ theta - theta_goal (when close)
        # d(angle_diff)/dtheta = 1
        angle_diff = np.arctan2(np.sin(robot_theta - theta_goal), np.cos(robot_theta - theta_goal))
        gradient[2] = -2.0 * orientation_weight * angle_diff

        return gradient

    @staticmethod
    def navi_time_derivative(x_i, t, i, vars_dict):
        # 因为 p_goal 和 theta_goal 不随时间变化，所以 dH/dt = 0
        return 0.0

    @staticmethod
    def print_debug_info(x_i, t, i, vars_dict, iter_count):
        if iter_count % 10 != 0:  # 每10次迭代打印一次，避免刷屏
            return
        try:
            p_goal, theta_goal, dist_thresh, orientation_weight = NaviTask.get_navi_params(vars_dict)
        except (ValueError, KeyError) as e:
            print(f"DEBUG: Robot {i} (Task 0) 导航参数无效: {e}")
            return
            
        robot_pos = x_i[0:2]
        robot_theta = x_i[2]
        
        dist_to_goal = np.linalg.norm(robot_pos - p_goal)
        angle_diff = np.arctan2(np.sin(robot_theta - theta_goal), np.cos(robot_theta - theta_goal))
        pos_error_term = -0.5 * dist_to_goal**2
        angle_error_term = -orientation_weight * angle_diff**2
        arrival_bonus = 2.0 if dist_to_goal < dist_thresh else 0.0
        H_value = pos_error_term + angle_error_term + arrival_bonus
        
        arrived_status = "已到达" if dist_to_goal < dist_thresh else "未到达"
        print(f"DEBUG: Robot {i} (Task 0) 导航中. 距离目标: {dist_to_goal:.2f}m (阈值: {dist_thresh}), 状态: {arrived_status}")
        print(f"       角度差: {angle_diff:.2f}rad, 位置误差项: {pos_error_term:.2f}, 角度误差项: {angle_error_term:.2f}")
        print(f"       到达奖励: {arrival_bonus:.1f}, 总函数值: {H_value:.2f}")

    @staticmethod
    def apply_motion_control(x_i, t, i, vars_dict, dt):
        try:
            p_goal, theta_goal, dist_thresh, orientation_weight = NaviTask.get_navi_params(vars_dict)
        except (ValueError, KeyError) as e:
            print(f"警告: 导航任务的参数缺失或无效: {e}")
            return x_i  # 返回原始状态，不做改变
            
        # 获取当前位置和朝向
        curr_pos = x_i[0:2]
        curr_theta = x_i[2]
        
        # 计算到目标的距离和方向
        vec_to_goal = p_goal - curr_pos
        dist_to_goal = np.linalg.norm(vec_to_goal)
        
        if dist_to_goal < dist_thresh:
            angle_diff = np.arctan2(np.sin(curr_theta - theta_goal), np.cos(curr_theta - theta_goal))
            theta_adjust_rate = 0.5
            new_theta = curr_theta - theta_adjust_rate * angle_diff * dt
            return np.array([curr_pos[0], curr_pos[1], new_theta])
            
        desired_theta = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        angle_diff = np.arctan2(np.sin(curr_theta - desired_theta), np.cos(curr_theta - desired_theta))
        
        max_speed = 0.4
        speed_factor = min(1.0, dist_to_goal / 2.0)
        angle_factor = max(0.2, np.cos(angle_diff)**2)
        forward_speed = max_speed * speed_factor * angle_factor
        
        max_angular_speed = 0.8
        angular_speed = -max_angular_speed * np.sin(angle_diff)
        new_pos_x = curr_pos[0] + forward_speed * np.cos(curr_theta) * dt
        new_pos_y = curr_pos[1] + forward_speed * np.sin(curr_theta) * dt
        new_theta = curr_theta + angular_speed * dt
        
        return np.array([new_pos_x, new_pos_y, new_theta])

