import numpy as np
from functools import partial

class AdaptiveTaskOptimizer:
    """
    自适应任务优化器
    根据场景动态调整任务函数参数，优化机器人任务分配
    """
    
    def __init__(self, scenario_params, global_vars_manager=None):
        self.scenario_params = scenario_params
        self.global_vars = global_vars_manager
        self.original_tasks = self._backup_original_tasks()
        self.optimization_history = []
        self.adaptation_frequency = 10  # 每10次迭代更新一次
        self.current_iter = 0
        
    def _backup_original_tasks(self):
        """备份原始任务函数，以便需要时恢复"""
        original = {}
        for i, task in enumerate(self.scenario_params['tasks']):
            original[i] = {
                'function': task['function'],
                'gradient': task['gradient'],
                'time_derivative': task['time_derivative']
            }
        return original
    
    def optimize_tasks(self, x, t, task_assignment, environment_state=None):
        """
        根据当前状态优化任务函数
        
        Args:
            x: 机器人状态向量
            t: 当前时间
            task_assignment: 当前任务分配结果
            environment_state: 可选的环境状态信息
        """
        self.current_iter += 1
        
        # 仅在特定频率下执行优化
        if self.current_iter % self.adaptation_frequency != 0:
            return False
            
        # 提取机器人位置
        robot_positions = x.copy()
        
        # # 优化运输任务
        if 1 in task_assignment:  # 任务1是运输任务
            transport_robots = [i for i, task in enumerate(task_assignment) if task == 1]
            self._optimize_transport(robot_positions, t, transport_robots)
            
        # 优化覆盖控制任务
        if 2 in task_assignment:  # 任务2是覆盖控制
            coverage_robots = [i for i, task in enumerate(task_assignment) if task == 2]
            self._optimize_coverage(robot_positions, t, coverage_robots, environment_state)
        
        print(f"任务函数参数在 t={t:.1f}s 触发检查 (实际更新依赖内部逻辑)") # 提示检查被触发
        return True
        
    def _optimize_transport(self, robot_positions, t, transport_robots):
        """
        优化运输任务函数 (修正版)

        Args:
            robot_positions (np.ndarray): 所有机器人的状态数组 (e.g., shape [3, num_robots])
            t (float): 当前时间
            transport_robots (list): 参与运输任务的机器人索引列表
        """
        # 获取全局变量中的相关信息
        p_transport = self.global_vars.p_transport_t # 当前时刻的运输目标点
        p_goal = self.global_vars.p_goal       # 最终目标点
        p_start = self.global_vars.p_start     # 起始点

        # 计算当前场景的进度因子
        # 注意: p_transport 是移动的, 用它来计算进度可能更合适
        progress_factor = self._calculate_progress_factor(p_transport, p_start, p_goal)

        # 根据进度调整权重 
        distance_weight = 1.0
        # 进度权重，随着接近目标而增加 (影响奖励项)
        progress_weight = 0.5 + 0.5 * progress_factor
        # 协调权重，开始时需要更多协调，随着接近目标而减少
        coordination_weight = 0.3 * (1 - progress_factor)
        # 理想的机器人间距
        ideal_dist = 0.3

        # --- 优化后的运输任务函数 (使用平方距离) ---
        def optimized_transport_function(x_i, t, robot_idx, vars_dict=None):
            """
            计算优化后的运输任务函数值 (效用函数)。
            目标是最大化此函数值 (最小化成本)。
            使用平方距离以接近原始 transport_function。
            """
            if vars_dict is None:
                # 在实际应用中，确保能正确获取最新的全局变量
                vars_dict = self.global_vars.get_all_vars()
            # 获取当前时间的运输点 (可能随时间变化)
            p_t = vars_dict.get('p_transport_t', p_transport)

            # 1. 核心距离成本 (负平方距离，乘以权重)
            #    目标是最小化到运输点的距离，所以效用是负的成本
            distance_utility = -distance_weight * np.linalg.norm(x_i[:2] - p_t)**2

            # 2. 协调成本 (惩罚偏离理想间距)
            #    目标是最小化协调成本，所以效用是负的成本
            coordination_cost = 0
            if len(transport_robots) > 1:
                for j in transport_robots:
                    if j != robot_idx:
                        # 获取其他参与运输的机器人的当前位置
                        # 注意：这里需要传入最新的 robot_positions
                        other_pos = robot_positions[:2, j]
                        dist = np.linalg.norm(x_i[:2] - other_pos)
                        # 使用平方差以获得更平滑的梯度
                        coordination_cost += (dist - ideal_dist)**2
            coordination_utility = -coordination_weight * coordination_cost

            # 3. 进度奖励 (鼓励运输点接近最终目标)
            #    注意: 这个奖励不直接依赖于 x_i，因此不影响梯度
            #    我们希望最大化奖励，所以它是正的效用
            #    使用 p_t 到 p_goal 的负距离可能更直观，表示“剩余距离”的负值
            progress_reward_utility = -progress_weight * np.linalg.norm(p_t - p_goal)
            # 或者保持原来的形式，假设它代表某种正向激励:
            # progress_reward_utility = progress_weight * some_positive_measure

            # 总效用 = 距离效用 + 协调效用 + 进度奖励效用
            # 我们希望最大化这个总效用值
            total_utility = distance_utility + coordination_utility #+ progress_reward_utility
            # 注意: 原始代码返回 -(cost + cost - reward)。如果 progress_reward 是正向奖励，
            # 且 cost 是正值成本，那么最大化 -cost - cost + reward 是合理的。
            # 我们的 distance_utility 和 coordination_utility 已经是负成本（效用）了。
            # 如果 progress_reward_utility 代表正效用，直接相加即可。
            # 如果你原来的 progress_reward 是指距离目标点的距离（成本），那应该用负号。
            # 这里暂时注释掉 progress_reward_utility，因为它不影响梯度，且其含义需根据具体目标确定。

            return total_utility # 返回总效用值

        # --- 优化后的梯度函数 (对应平方距离和平方协调成本) ---
        def optimized_transport_gradient(x_i, t, robot_idx, vars_dict=None):
            """
            计算优化后的运输任务函数的梯度。
            对应于 optimized_transport_function 的梯度。
            """
            if vars_dict is None:
                # 确保能正确获取最新的全局变量
                vars_dict = self.global_vars.get_all_vars()
            # 获取当前时间的运输点
            p_t = vars_dict.get('p_transport_t', p_transport)

            # 初始化梯度 (x, y, theta)
            gradient = np.zeros_like(x_i)

            # 1. 距离效用的梯度
            #    d/dx_i (-distance_weight * ||x_i[:2] - p_t||^2)
            #    = -distance_weight * 2 * (x_i[:2] - p_t)
            #    这与原始 transport_gradient 的形式一致
            vec_to_target = x_i[:2] - p_t
            gradient[:2] += -distance_weight * 2 * vec_to_target

            # 2. 协调效用的梯度
            #    d/dx_i (-coordination_weight * sum((||x_i[:2]-other_j|| - ideal_dist)^2))
            #    = -coordination_weight * sum(2 * (||x_i[:2]-other_j|| - ideal_dist) * d(||x_i[:2]-other_j||)/dx_i)
            #    d(||x_i[:2]-other_j||)/dx_i = (x_i[:2] - other_j) / ||x_i[:2]-other_j||
            if len(transport_robots) > 1:
                for j in transport_robots:
                    if j != robot_idx:
                        other_pos = robot_positions[:2, j]
                        vec_to_other = x_i[:2] - other_pos
                        dist = np.linalg.norm(vec_to_other)
                        # 添加 epsilon 防止除以零
                        dist_safe = dist + 1e-6
                        # 计算梯度贡献
                        grad_coord_j = -coordination_weight * 2 * (dist - ideal_dist) * vec_to_other / dist_safe
                        gradient[:2] += grad_coord_j

            # 3. 进度奖励效用的梯度
            #    由于 progress_reward_utility 不依赖于 x_i，其梯度为 0

            # gradient[2] 保持为 0，因为任务通常只关心 xy 平面位置
            gradient[2] = 0

            # 返回总效用的梯度
            return gradient

        # 更新任务函数和梯度
        # 确保 self.scenario_params['tasks'][0] 确实是你要更新的运输任务
        self.scenario_params['tasks'][0]['function'] = optimized_transport_function
        # self.scenario_params['tasks'][0]['gradient'] = optimized_transport_gradient

        # 记录优化历史 (保持不变)
        self.optimization_history.append({
            'time': t,
            'task_type': 'transport',
            'weights': {
                'distance': distance_weight,
                'progress': progress_weight, # 注意 progress_weight 的实际影响取决于 progress_reward_utility 的定义
                'coordination': coordination_weight
            },
            'progress_factor': progress_factor # 可以记录进度因子本身
        })

        print(f"Time {t}: Optimized transport task. Progress: {progress_factor:.2f}, Weights(d,p,c): ({distance_weight:.2f}, {progress_weight:.2f}, {coordination_weight:.2f})")

    
    def _optimize_coverage(self, robot_positions, t, coverage_robots, environment_state=None):
        """优化覆盖控制任务函数"""
        # 获取环境信息和POI点
        poi = self.global_vars.poi
        mud_area = (self.global_vars.x_mud, self.global_vars.y_mud) if hasattr(self.global_vars, 'x_mud') else None
        
        # 计算覆盖任务的相关参数
        coverage_factor = 1.0
        uniformity_factor = 0.7
        
        # 考虑时间因素 - 随着时间推移调整权重
        if hasattr(self.global_vars, 't_start') and t > self.global_vars.t_start:
            elapsed_time = t - self.global_vars.t_start
            time_factor = min(1.0, elapsed_time / self.global_vars.delta_t if hasattr(self.global_vars, 'delta_t') else 0.5)
            coverage_factor = 1.0 + 0.5 * time_factor
            uniformity_factor = 0.7 * (1 - 0.3 * time_factor)
        
        # 考虑环境因素 - 如果有泥地区域，调整覆盖策略
        mud_avoidance_factor = 0.0
        if mud_area is not None:
            mud_avoidance_factor = 0.3
        
        # 创建优化后的覆盖控制任务函数
        def optimized_coverage_function(x_i, t, robot_idx, vars_dict=None):
            """
            Calculate the optimized function value for the coverage control task.
            
            Parameters:
                x_i: Robot state
                t: Current time
                robot_idx: Robot index
                vars_dict: Optional global variable dictionary
            """
            if vars_dict is None:
                # Try to get from self.global_vars if it exists
                if hasattr(globals().get('self', None), 'global_vars') and globals().get('self').global_vars is not None:
                    vars_dict = globals().get('self').global_vars.get_all_vars()
                else:
                    # If global_vars is not available, use the static method from CoverageControl
                    from task_utils.coverage_module import CoverageControl
                    vars_dict = CoverageControl.get_global_vars_dict()
                    if vars_dict is None:
                        print("Warning: Unable to access global_vars to get vars_dict.")
                        vars_dict = {}  # Provide an empty dictionary to avoid subsequent errors

            # Get necessary information
            current_pos = x_i[:2]
            current_angle = x_i[2] if len(x_i) > 2 else 0
            
            # Get G (Voronoi Centroids) and poi
            G = vars_dict.get('G', None)
            poi = vars_dict.get('poi', np.array([0, 0]))
            
            # Coverage, uniformity, and mud avoidance factors
            coverage_factor = vars_dict.get('coverage_factor', 1.0)
            uniformity_factor = vars_dict.get('uniformity_factor', 0.5)
            mud_avoidance_factor = vars_dict.get('mud_avoidance_factor', 1.0)
            
            # 1. Centroid distance cost (Voronoi-based)
            centroid_distance_cost = 0.0
            if G is not None and isinstance(G, np.ndarray) and G.ndim == 2 and G.shape[0] == 2:
                if 0 <= robot_idx < G.shape[1]:
                    centroid = G[:, robot_idx]  # Get centroid [cx, cy] for robot_idx
                    dist_sq = np.sum((current_pos - centroid)**2)
                    centroid_distance_cost = dist_sq * coverage_factor
                else:
                    print(f"Warning (coverage cost): robot_idx {robot_idx} out of range for G (columns: {G.shape[1]}).")
                    centroid_distance_cost = coverage_factor * np.sum((current_pos - poi)**2)
            else:
                print(f"Warning (coverage cost): G not available or has unexpected format.")
                centroid_distance_cost = coverage_factor * np.sum((current_pos - poi)**2)
            
            # 2. Orientation towards POI cost
            angle_to_poi = np.arctan2(poi[1] - current_pos[1], poi[0] - current_pos[0])
            angle_difference_cost = (current_angle - angle_to_poi)**2
            
            # 3. Uniformity cost
            uniformity_cost = 0.0
            coverage_robots = vars_dict.get('coverage_robots', [robot_idx])  # Default to just this robot
            all_robot_states = vars_dict.get('x', None)
            
            if all_robot_states is not None and isinstance(all_robot_states, np.ndarray):
                if all_robot_states.ndim == 2 and all_robot_states.shape[0] >= 2:
                    for j in coverage_robots:
                        if j != robot_idx and 0 <= j < all_robot_states.shape[1]:
                            other_pos = all_robot_states[:2, j]
                            dist = np.linalg.norm(current_pos - other_pos)
                            ideal_dist = 0.5  # Ideal distribution distance
                            uniformity_cost += uniformity_factor * (dist - ideal_dist)**2
            
            # 4. Mud penalty
            mud_penalty = 0.0
            mud_map_info = vars_dict.get('mud_map', None)
            if mud_map_info is not None:
                try:
                    x_mud = mud_map_info['x_coords']
                    y_mud = mud_map_info['y_coords']
                    min_dist_to_mud = float('inf')
                    for i in range(len(x_mud)):
                        mud_point = np.array([x_mud[i], y_mud[i]])
                        dist = np.linalg.norm(current_pos - mud_point)
                        min_dist_to_mud = min(min_dist_to_mud, dist)
                    
                    mud_threshold = 0.3
                    if min_dist_to_mud < mud_threshold:
                        mud_penalty = mud_avoidance_factor * (mud_threshold - min_dist_to_mud)
                except (KeyError, TypeError, IndexError) as e:
                    print(f"Warning (mud penalty): Error processing mud information: {e}")
            
            # Total cost (negative for maximization)
            total_cost = -(centroid_distance_cost + angle_difference_cost + uniformity_cost + mud_penalty)
            
            return float(total_cost)
        # 创建优化后的覆盖控制任务梯度
        def optimized_coverage_gradient(x_i, t, robot_idx, vars_dict=None):
            """
            Calculate the gradient of the optimized coverage control task.
            
            Parameters:
                x_i: Robot state
                t: Current time
                robot_idx: Robot index
                vars_dict: Optional global variable dictionary
            """
            if vars_dict is None:
                # Similar initialization as in the main function
                if hasattr(globals().get('self', None), 'global_vars') and globals().get('self').global_vars is not None:
                    vars_dict = globals().get('self').global_vars.get_all_vars()
                else:
                    from task_utils.coverage_module import CoverageControl
                    vars_dict = CoverageControl.get_global_vars_dict()
                    if vars_dict is None:
                        print("Warning: Unable to access global_vars to get vars_dict.")
                        vars_dict = {}
            
            # Get necessary information
            current_pos = x_i[:2]
            current_angle = x_i[2] if len(x_i) > 2 else 0
            
            # Get G and poi
            G = vars_dict.get('G', None)
            poi = vars_dict.get('poi', np.array([0, 0]))
            
            # Factors
            coverage_factor = vars_dict.get('coverage_factor', 1.0)
            uniformity_factor = vars_dict.get('uniformity_factor', 0.5)
            mud_avoidance_factor = vars_dict.get('mud_avoidance_factor', 1.0)
            
            # Initialize gradient vector
            gradient = np.zeros_like(x_i)
            
            # 1. Gradient of centroid distance cost
            if G is not None and isinstance(G, np.ndarray) and G.ndim == 2 and G.shape[0] == 2:
                if 0 <= robot_idx < G.shape[1]:
                    centroid = G[:, robot_idx]
                    gradient[:2] = -2 * coverage_factor * (current_pos - centroid)
                else:
                    gradient[:2] = -2 * coverage_factor * (current_pos - poi)
            else:
                gradient[:2] = -2 * coverage_factor * (current_pos - poi)
            
            # 2. Gradient of orientation towards POI cost
            if len(x_i) > 2:
                angle_to_poi = np.arctan2(poi[1] - current_pos[1], poi[0] - current_pos[0])
                gradient[2] = -2 * (current_angle - angle_to_poi)
                
                # Add position contribution to angle gradient
                dx_to_poi = poi[0] - current_pos[0]
                dy_to_poi = poi[1] - current_pos[1]
                dist_sq = dx_to_poi**2 + dy_to_poi**2
                if dist_sq > 1e-6:  # Avoid division by zero
                    gradient[0] += -2 * (current_angle - angle_to_poi) * (dy_to_poi / dist_sq)
                    gradient[1] += 2 * (current_angle - angle_to_poi) * (dx_to_poi / dist_sq)
            
            # 3. Gradient of uniformity cost
            coverage_robots = vars_dict.get('coverage_robots', [robot_idx])
            all_robot_states = vars_dict.get('x', None)
            
            if all_robot_states is not None and isinstance(all_robot_states, np.ndarray):
                if all_robot_states.ndim == 2 and all_robot_states.shape[0] >= 2:
                    for j in coverage_robots:
                        if j != robot_idx and 0 <= j < all_robot_states.shape[1]:
                            other_pos = all_robot_states[:2, j]
                            diff = current_pos - other_pos
                            dist = np.linalg.norm(diff)
                            if dist > 1e-6:  # Avoid division by zero
                                ideal_dist = 0.5
                                gradient[:2] += -2 * uniformity_factor * (dist - ideal_dist) * diff / dist
            
            # 4. Gradient of mud penalty
            mud_map_info = vars_dict.get('mud_map', None)
            if mud_map_info is not None:
                try:
                    x_mud = mud_map_info['x_coords']
                    y_mud = mud_map_info['y_coords']
                    min_dist = float('inf')
                    nearest_mud_point = None
                    
                    for i in range(len(x_mud)):
                        mud_point = np.array([x_mud[i], y_mud[i]])
                        dist = np.linalg.norm(current_pos - mud_point)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_mud_point = mud_point
                    
                    mud_threshold = 0.3
                    if min_dist < mud_threshold and nearest_mud_point is not None:
                        diff = current_pos - nearest_mud_point
                        if np.linalg.norm(diff) > 1e-6:  # Avoid division by zero
                            norm_diff = diff / np.linalg.norm(diff)
                            gradient[:2] += -mud_avoidance_factor * norm_diff
                except (KeyError, TypeError, IndexError) as e:
                    print(f"Warning (mud gradient): Error processing mud information: {e}")
            
            # Negate gradient for maximization
            return -gradient
        
        # 更新任务函数
        self.scenario_params['tasks'][1]['function'] = optimized_coverage_function
        # self.scenario_params['tasks'][1]['gradient'] = optimized_coverage_gradient
        
        # 记录优化历史
        self.optimization_history.append({
            'time': t,
            'task_type': 'coverage',
            'weights': {
                'coverage': coverage_factor,
                'uniformity': uniformity_factor,
                'mud_avoidance': mud_avoidance_factor
            }
        })
    
    def _calculate_progress_factor(self, current_pos, start_pos, goal_pos):
        """计算运输任务的进度因子 (0-1)"""
        total_distance = np.linalg.norm(goal_pos - start_pos)
        if total_distance < 1e-6:
            return 1.0
            
        current_to_start = np.linalg.norm(current_pos - start_pos)
        current_to_goal = np.linalg.norm(current_pos - goal_pos)
        
        # 使用投影方法计算进度
        progress = current_to_start / total_distance
        
        # 确保值在0-1范围内
        return max(0.0, min(1.0, progress))
    
    def reset_to_original(self):
        """重置为原始任务函数"""
        for i, task in self.original_tasks.items():
            self.scenario_params['tasks'][i]['function'] = task['function']
            self.scenario_params['tasks'][i]['gradient'] = task['gradient']
            self.scenario_params['tasks'][i]['time_derivative'] = task['time_derivative']
        
        return self.scenario_params
    
    def get_optimization_history(self):
        """获取优化历史记录"""
        return self.optimization_history
