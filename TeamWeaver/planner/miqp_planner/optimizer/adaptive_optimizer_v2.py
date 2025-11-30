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
        
        # 优化运输任务
        if 1 in task_assignment:  # 任务1是运输任务
            transport_robots = [i for i, task in enumerate(task_assignment) if task == 1]
            self._optimize_transport_task(robot_positions, t, transport_robots)
            
        # 优化覆盖控制任务
        if 2 in task_assignment:  # 任务2是覆盖控制
            coverage_robots = [i for i, task in enumerate(task_assignment) if task == 2]
            self._optimize_coverage_task(robot_positions, t, coverage_robots, environment_state)
        
        print(f"任务函数参数在 t={t:.1f}s 触发检查 (实际更新依赖内部逻辑)") # 提示检查被触发
        return True
    
    def _optimize_transport_task(self, robot_positions, t, transport_robots):
        """优化运输任务"""
        # 获取上下文信息
        context = self._get_transport_context(t, transport_robots)
        
        # 计算权重因子
        weights = self._calculate_transport_weights(context)
        
        # 更新任务函数和梯度
        optimized_function = self._create_optimized_transport_function(robot_positions, weights, context, transport_robots)
        optimized_gradient = self._create_optimized_transport_gradient(robot_positions, weights, context, transport_robots)
        
        # 应用优化后的函数和梯度
        self.scenario_params['tasks'][0]['function'] = optimized_function
        # self.scenario_params['tasks'][0]['gradient'] = optimized_gradient
        
        # 记录优化历史
        self._record_transport_optimization(t, weights, context)
    
    def _get_transport_context(self, t, transport_robots):
        """获取运输任务的上下文信息"""
        context = {
            'p_transport': self.global_vars.p_transport_t,  # 当前时刻的运输目标点
            'p_goal': self.global_vars.p_goal,              # 最终目标点
            'p_start': self.global_vars.p_start,            # 起始点
            'transport_robots': transport_robots,           # 参与运输的机器人
            'time': t                                       # 当前时间
        }
        
        # 计算进度因子
        context['progress_factor'] = self._calculate_progress_factor(
            context['p_transport'], context['p_start'], context['p_goal'])
        
        return context
    
    def _calculate_transport_weights(self, context):
        """计算运输任务的权重因子"""
        progress_factor = context['progress_factor']
        
        weights = {
            'distance': 1.0,                                    # 距离权重
            'progress': 0.5 + 0.5 * progress_factor,            # 进度权重，随接近目标增加
            'coordination': 0.3 * (1 - progress_factor)         # 协调权重，随接近目标减少
        }
        
        return weights
    
    def _create_optimized_transport_function(self, robot_positions, weights, context, transport_robots):
        """创建优化后的运输任务函数"""
        p_transport = context['p_transport']
        p_goal = context['p_goal']
        ideal_dist = 0.3  # 理想的机器人间距
        
        def optimized_function(x_i, t, robot_idx, vars_dict=None):
            """优化后的运输任务函数"""
            if vars_dict is None:
                vars_dict = self.global_vars.get_all_vars()
            
            # 获取当前时间的运输点
            p_t = vars_dict.get('p_transport_t', p_transport)
            
            # 1. 核心目标：距离效用
            distance_utility = -weights['distance'] * np.linalg.norm(x_i[:2] - p_t)**2
            
            # 2. 协作项：协调效用
            coordination_utility = 0
            if len(transport_robots) > 1:
                for j in transport_robots:
                    if j != robot_idx:
                        other_pos = robot_positions[:2, j]
                        dist = np.linalg.norm(x_i[:2] - other_pos)
                        coordination_utility += -weights['coordination'] * (dist - ideal_dist)**2
            
            # 3. 进度项：进度奖励
            # progress_utility = -weights['progress'] * np.linalg.norm(p_t - p_goal)
            
            # 总效用
            total_utility = distance_utility + coordination_utility  # + progress_utility
            
            return total_utility
        
        return optimized_function
    
    def _create_optimized_transport_gradient(self, robot_positions, weights, context, transport_robots):
        """创建优化后的运输任务梯度函数"""
        p_transport = context['p_transport']
        ideal_dist = 0.3
        
        def optimized_gradient(x_i, t, robot_idx, vars_dict=None):
            """优化后的运输任务梯度"""
            if vars_dict is None:
                vars_dict = self.global_vars.get_all_vars()
            
            # 获取当前时间的运输点
            p_t = vars_dict.get('p_transport_t', p_transport)
            
            # 初始化梯度
            gradient = np.zeros_like(x_i)
            
            # 1. 距离效用的梯度
            vec_to_target = x_i[:2] - p_t
            gradient[:2] += -weights['distance'] * 2 * vec_to_target
            
            # 2. 协调效用的梯度
            if len(transport_robots) > 1:
                for j in transport_robots:
                    if j != robot_idx:
                        other_pos = robot_positions[:2, j]
                        vec_to_other = x_i[:2] - other_pos
                        dist = np.linalg.norm(vec_to_other)
                        dist_safe = dist + 1e-6  # 防止除以零
                        grad_coord_j = -weights['coordination'] * 2 * (dist - ideal_dist) * vec_to_other / dist_safe
                        gradient[:2] += grad_coord_j
            
            # 梯度的第三个分量(旋转角)保持为0
            gradient[2] = 0
            
            return gradient
        
        return optimized_gradient
    
    def _record_transport_optimization(self, t, weights, context):
        """记录运输任务的优化历史"""
        self.optimization_history.append({
            'time': t,
            'task_type': 'transport',
            'weights': weights,
            'progress_factor': context['progress_factor']
        })
        
        print(f"Time {t}: Optimized transport task. Progress: {context['progress_factor']:.2f}, "
              f"Weights(d,p,c): ({weights['distance']:.2f}, {weights['progress']:.2f}, {weights['coordination']:.2f})")
    
    def _optimize_coverage_task(self, robot_positions, t, coverage_robots, environment_state=None):
        """优化覆盖控制任务"""
        # 获取上下文信息
        context = self._get_coverage_context(t, coverage_robots, environment_state)
        
        # 计算权重因子
        weights = self._calculate_coverage_weights(context)
        
        # 更新任务函数和梯度
        optimized_function = self._create_optimized_coverage_function(weights, context)
        optimized_gradient = self._create_optimized_coverage_gradient(weights, context)
        
        # 应用优化后的函数和梯度
        self.scenario_params['tasks'][1]['function'] = optimized_function
        # self.scenario_params['tasks'][1]['gradient'] = optimized_gradient
        
        # 记录优化历史
        self._record_coverage_optimization(t, weights)
    
    def _get_coverage_context(self, t, coverage_robots, environment_state=None):
        """获取覆盖控制任务的上下文信息"""
        context = {
            'poi': self.global_vars.poi,                     # 兴趣点
            'coverage_robots': coverage_robots,              # 参与覆盖控制的机器人
            'time': t                                        # 当前时间
        }
        
        # 添加泥地区域信息
        if hasattr(self.global_vars, 'x_mud'):
            context['mud_area'] = {
                'x_mud': self.global_vars.x_mud,
                'y_mud': self.global_vars.y_mud
            }
        
        # 添加时间相关信息
        if hasattr(self.global_vars, 't_start'):
            context['t_start'] = self.global_vars.t_start
            if hasattr(self.global_vars, 'delta_t'):
                context['delta_t'] = self.global_vars.delta_t
                if t > context['t_start']:
                    context['elapsed_time'] = t - context['t_start']
                    context['time_factor'] = min(1.0, context['elapsed_time'] / context['delta_t'])
        
        return context
    
    def _calculate_coverage_weights(self, context):
        """计算覆盖控制任务的权重因子"""
        weights = {
            'coverage': 1.0,       # 覆盖权重
            'uniformity': 0.7,     # 均匀性权重
            'mud_avoidance': 0.0   # 泥地避免权重
        }
        
        # 根据时间调整权重
        if 'time_factor' in context:
            time_factor = context['time_factor']
            weights['coverage'] = 1.0 + 0.5 * time_factor
            weights['uniformity'] = 0.7 * (1 - 0.3 * time_factor)
        
        # 根据环境调整权重
        if 'mud_area' in context:
            weights['mud_avoidance'] = 0.3
        
        return weights
    
    def _create_optimized_coverage_function(self, weights, context):
        """创建优化后的覆盖控制任务函数"""
        def optimized_function(x_i, t, robot_idx, vars_dict=None):
            """优化后的覆盖控制函数"""
            if vars_dict is None:
                if hasattr(globals().get('self', None), 'global_vars') and globals().get('self').global_vars is not None:
                    vars_dict = globals().get('self').global_vars.get_all_vars()
                else:
                    from task_utils.coverage_module import CoverageControl
                    vars_dict = CoverageControl.get_global_vars_dict()
                    if vars_dict is None:
                        print("Warning: Unable to access global_vars to get vars_dict.")
                        vars_dict = {}
            
            # 获取必要信息
            current_pos = x_i[:2]
            current_angle = x_i[2] if len(x_i) > 2 else 0
            
            # 获取G和poi
            G = vars_dict.get('G', None)
            poi = vars_dict.get('poi', np.array([0, 0]))
            
            # 1. 核心目标：中心距离成本
            centroid_distance_cost = 0.0
            if G is not None and isinstance(G, np.ndarray) and G.ndim == 2 and G.shape[0] == 2:
                if 0 <= robot_idx < G.shape[1]:
                    centroid = G[:, robot_idx]
                    centroid_distance_cost = np.sum((current_pos - centroid)**2) * weights['coverage']
                else:
                    centroid_distance_cost = weights['coverage'] * np.sum((current_pos - poi)**2)
            else:
                centroid_distance_cost = weights['coverage'] * np.sum((current_pos - poi)**2)
            
            # 2. 核心目标：朝向成本
            angle_to_poi = np.arctan2(poi[1] - current_pos[1], poi[0] - current_pos[0])
            angle_difference_cost = (current_angle - angle_to_poi)**2
            
            # 3. 协作项：均匀性成本
            uniformity_cost = 0.0
            coverage_robots = vars_dict.get('coverage_robots', [robot_idx])
            all_robot_states = vars_dict.get('x', None)
            
            if all_robot_states is not None and isinstance(all_robot_states, np.ndarray):
                if all_robot_states.ndim == 2 and all_robot_states.shape[0] >= 2:
                    for j in coverage_robots:
                        if j != robot_idx and 0 <= j < all_robot_states.shape[1]:
                            other_pos = all_robot_states[:2, j]
                            dist = np.linalg.norm(current_pos - other_pos)
                            ideal_dist = 0.5
                            uniformity_cost += weights['uniformity'] * (dist - ideal_dist)**2
            
            # 4. 环境适应项：泥地惩罚
            mud_penalty = 0.0
            mud_map_info = vars_dict.get('mud_map', None)
            if mud_map_info is not None and weights['mud_avoidance'] > 0:
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
                        mud_penalty = weights['mud_avoidance'] * (mud_threshold - min_dist_to_mud)
                except (KeyError, TypeError, IndexError) as e:
                    print(f"Warning (mud penalty): Error processing mud information: {e}")
            
            # 总成本(取负值，便于最大化)
            total_cost = -(centroid_distance_cost + angle_difference_cost + uniformity_cost + mud_penalty)
            
            return float(total_cost)
        
        return optimized_function
    
    def _create_optimized_coverage_gradient(self, weights, context):
        """创建优化后的覆盖控制任务梯度函数"""
        def optimized_gradient(x_i, t, robot_idx, vars_dict=None):
            """优化后的覆盖控制梯度"""
            if vars_dict is None:
                if hasattr(globals().get('self', None), 'global_vars') and globals().get('self').global_vars is not None:
                    vars_dict = globals().get('self').global_vars.get_all_vars()
                else:
                    from task_utils.coverage_module import CoverageControl
                    vars_dict = CoverageControl.get_global_vars_dict()
                    if vars_dict is None:
                        print("Warning: Unable to access global_vars to get vars_dict.")
                        vars_dict = {}
            
            # 获取必要信息
            current_pos = x_i[:2]
            current_angle = x_i[2] if len(x_i) > 2 else 0
            
            # 获取G和poi
            G = vars_dict.get('G', None)
            poi = vars_dict.get('poi', np.array([0, 0]))
            
            # 初始化梯度向量
            gradient = np.zeros_like(x_i)
            
            # 1. 中心距离成本的梯度
            if G is not None and isinstance(G, np.ndarray) and G.ndim == 2 and G.shape[0] == 2:
                if 0 <= robot_idx < G.shape[1]:
                    centroid = G[:, robot_idx]
                    gradient[:2] = -2 * weights['coverage'] * (current_pos - centroid)
                else:
                    gradient[:2] = -2 * weights['coverage'] * (current_pos - poi)
            else:
                gradient[:2] = -2 * weights['coverage'] * (current_pos - poi)
            
            # 2. 朝向成本的梯度
            if len(x_i) > 2:
                angle_to_poi = np.arctan2(poi[1] - current_pos[1], poi[0] - current_pos[0])
                gradient[2] = -2 * (current_angle - angle_to_poi)
                
                # 位置对角度梯度的贡献
                dx_to_poi = poi[0] - current_pos[0]
                dy_to_poi = poi[1] - current_pos[1]
                dist_sq = dx_to_poi**2 + dy_to_poi**2
                if dist_sq > 1e-6:
                    gradient[0] += -2 * (current_angle - angle_to_poi) * (dy_to_poi / dist_sq)
                    gradient[1] += 2 * (current_angle - angle_to_poi) * (dx_to_poi / dist_sq)
            
            # 3. 均匀性成本的梯度
            coverage_robots = vars_dict.get('coverage_robots', [robot_idx])
            all_robot_states = vars_dict.get('x', None)
            
            if all_robot_states is not None and isinstance(all_robot_states, np.ndarray):
                if all_robot_states.ndim == 2 and all_robot_states.shape[0] >= 2:
                    for j in coverage_robots:
                        if j != robot_idx and 0 <= j < all_robot_states.shape[1]:
                            other_pos = all_robot_states[:2, j]
                            diff = current_pos - other_pos
                            dist = np.linalg.norm(diff)
                            if dist > 1e-6:
                                ideal_dist = 0.5
                                gradient[:2] += -2 * weights['uniformity'] * (dist - ideal_dist) * diff / dist
            
            # 4. 泥地惩罚的梯度
            if weights['mud_avoidance'] > 0:
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
                            if np.linalg.norm(diff) > 1e-6:
                                norm_diff = diff / np.linalg.norm(diff)
                                gradient[:2] += -weights['mud_avoidance'] * norm_diff
                    except (KeyError, TypeError, IndexError) as e:
                        print(f"Warning (mud gradient): Error processing mud information: {e}")
            
            # 取负值，用于最大化
            return -gradient
        
        return optimized_gradient
    
    def _record_coverage_optimization(self, t, weights):
        """记录覆盖控制任务的优化历史"""
        self.optimization_history.append({
            'time': t,
            'task_type': 'coverage',
            'weights': weights
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