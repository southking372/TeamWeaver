# task_utils/task_module/scenario_params_task.py
import numpy as np
from habitat_llm.planner.HRCS.task_module.navi_module import NaviTask
from habitat_llm.planner.HRCS.task_module.explore_module import ExploreTask
from habitat_llm.planner.HRCS.task_module.manipulation_module import ManipulationTask, ManipulationPhase
from habitat_llm.planner.HRCS.task_module.wait_module import WaitTask
from habitat_llm.planner.HRCS.task_module.pick_module import PickTask
from habitat_llm.planner.HRCS.task_module.place_module import PlaceTask
from habitat_llm.planner.HRCS.task_module.open_module import OpenTask
from habitat_llm.planner.HRCS.task_module.close_module import CloseTask
from habitat_llm.planner.HRCS.task_module.clean_module import CleanTask
from habitat_llm.planner.HRCS.task_module.state_manipulation_modules import (
    FillTask, PourTask, PowerOnTask, PowerOffTask, RearrangeTask
)
from habitat_llm.planner.HRCS.sys_module.robot_dynamics_module import RobotDynamicsConfig

class ScenarioConfigTask:
    
    def __init__(self, n_r=2, n_t=13, n_c=5, n_f=5, n_x=3, n_u=2):
        """
        初始化场景配置任务 - 简化的PARTNR Agent能力模型
        
        参数:
            n_r: 机器人数量，默认为2 (Agent 0, Agent 1)
            n_t: 任务类型数量，默认为13 (Navigate, Explore, Pick, Place, Open, Close, Clean, Fill, Pour, PowerOn, PowerOff, Rearrange, Wait)
            n_c: 能力类别数量，默认为5 (简化的特征分类：基础移动, 物体操作, 基本控制, 液体处理, 电源控制)
            n_f: 功能维度，默认为5 (基础移动/物体操作/基本控制/液体处理/电源控制)
            n_x: 状态维度，默认为3 (x, y, θ)
            n_u: 控制输入维度，默认为2 (v, ω)
        """
        self.n_r = n_r
        self.n_t = n_t
        self.n_c = n_c
        self.n_f = n_f
        self.n_x = n_x
        self.n_u = n_u
        
        # 确保传入的 n_x, n_u 与 RobotDynamicsConfig 期望的一致
        if n_x != 3 or n_u != 2:
             print(f"警告: ScenarioConfigTask 收到 n_x={n_x}, n_u={n_u}, 但通常期望 n_x=3, n_u=2 用于差速驱动模型。")

        self.scenario_params = self._initialize_scenario_params()
        
        # 初始化任务相关的全局变量
        self._initialize_global_task_vars()
    
    def _initialize_global_task_vars(self):
        self.global_task_vars = {
            # NaviTask 相关变量
            'p_goal': np.array([1.5, 1.0]),
            'theta_goal': 0.0,
            'dist_thresh': 0.2,
            'orientation_weight': 0.3,
            
            # ExploreTask相关变量
            'exploration_targets': [
                {'position': np.array([1.0, 0.8]), 'explored': False, 'id': 0},
                {'position': np.array([-1.0, 0.5]), 'explored': False, 'id': 1},
                {'position': np.array([0.5, -0.8]), 'explored': False, 'id': 2}
            ],
            'explored_map': None,
            'explore_dist_thresh': 0.2,
            'exploration_action_duration': 2.0,
            'exploring_action_info': {},
            'exploration_action_timers': {},
            
            # Pick/Place/Manipulation相关变量
            'target_object_position': np.array([0.8, -0.5]),
            'target_receptacle_position': np.array([-0.8, 0.7]),
            'pick_dist_thresh': 0.15,
            'place_dist_thresh': 0.15,
            'manipulation_progress': 0.0,
            'is_holding': False,
            'holding_robot_id': None,
            'manipulation_phase': ManipulationPhase.NAV_OBJ,
            
            # Open/Close相关变量
            'target_furniture_position': np.array([1.0, 0.5]),
            'operation_dist_thresh': 0.2,
            'furniture_open_state': False,
            
            # Clean相关变量
            'clean_dist_thresh': 0.15,
            'object_clean_state': False,
            
            # Fill/Pour相关变量
            'target_container_position': np.array([0.3, 0.8]),
            'fill_dist_thresh': 0.15,
            'container_filled_state': False,
            'pour_dist_thresh': 0.15,
            'pour_completed_state': False,
            
            # PowerOn/PowerOff相关变量
            'target_device_position': np.array([0.7, 0.4]),
            'power_dist_thresh': 0.15,
            'device_power_state': False,
            
            # WaitTask相关变量
            'wait_step_threshold': 5.0,
            'sim_freq': 1.0,
            'wait_elapsed_time': 0.0
        }
    
    def get_global_task_vars(self):
        return self.global_task_vars
    
    def update_global_task_var(self, var_name, value):
        if var_name in self.global_task_vars:
            self.global_task_vars[var_name] = value
        else:
            print(f"警告: 尝试更新不存在的全局任务变量 '{var_name}'")
            # self.global_task_vars[var_name] = value # 可选：动态添加
    
    def _initialize_scenario_params(self):
        # 初始化机器人的特征和能力 - 简化的PARTNR Agent配置
        # 特征维度：[基础移动, 物体操作, 基本控制, 液体处理, 电源控制]
        # Agent 0: Close, Explore, Navigate, Open, Pick, Place, Rearrange, Wait
        # Agent 1: Clean, Close, Explore, Fill, Navigate, Open, Pick, Place, Pour, PowerOff, PowerOn, Rearrange, Wait
        
        partnr_agent_features = [
            # Agent 0: 基础移动、物体操作和基本控制能力
            [1, 1, 1, 0, 0],  # 基础移动+物体操作+基本控制，无液体处理/电源控制
            # Agent 1: 全功能Agent
            [1, 1, 1, 1, 1]   # 所有功能都具备
        ]

        # 根据实际 n_r 初始化 A 矩阵
        A = np.zeros((self.n_f, self.n_r))

        # 分配PARTNR特征
        for i in range(self.n_r):
            if i < len(partnr_agent_features):
                feature_vector = partnr_agent_features[i]
                if len(feature_vector) == self.n_f:
                    A[:, i] = feature_vector
                else:
                    print(f"警告: Agent {i} 的特征长度 ({len(feature_vector)}) 与 n_f ({self.n_f}) 不匹配")
                    A[:, i] = np.zeros(self.n_f)
            else:
                # 额外的机器人使用Agent 1的配置
                A[:, i] = partnr_agent_features[1] if len(partnr_agent_features) > 1 else np.ones(self.n_f)

        # 任务能力需求矩阵 - 13种PARTNR工具映射到5种简化能力
        # 任务顺序: [Navigate, Explore, Pick, Place, Open, Close, Clean, Fill, Pour, PowerOn, PowerOff, Rearrange, Wait]
        # 能力分类: [基础移动, 物体操作, 基本控制, 液体处理, 电源控制]
        T = np.zeros((self.n_t, self.n_c))
        
        # 任务到能力的映射关系
        if self.n_t > 0 and self.n_c > 0: T[0, 0] = 1   # Navigate → 基础移动
        if self.n_t > 1 and self.n_c > 0: T[1, 0] = 1   # Explore → 基础移动
        if self.n_t > 2 and self.n_c > 1: T[2, 1] = 1   # Pick → 物体操作
        if self.n_t > 3 and self.n_c > 1: T[3, 1] = 1   # Place → 物体操作
        if self.n_t > 4 and self.n_c > 2: T[4, 2] = 1   # Open → 基本控制
        if self.n_t > 5 and self.n_c > 2: T[5, 2] = 1   # Close → 基本控制
        if self.n_t > 6 and self.n_c > 3: T[6, 3] = 1   # Clean → 液体处理
        if self.n_t > 7 and self.n_c > 3: T[7, 3] = 1   # Fill → 液体处理
        if self.n_t > 8 and self.n_c > 3: T[8, 3] = 1   # Pour → 液体处理
        if self.n_t > 9 and self.n_c > 4: T[9, 4] = 1   # PowerOn → 电源控制
        if self.n_t > 10 and self.n_c > 4: T[10, 4] = 1 # PowerOff → 电源控制
        if self.n_t > 11 and self.n_c > 1: T[11, 1] = 1 # Rearrange → 物体操作
        if self.n_t > 12 and self.n_c > 0: T[12, 0] = 1 # Wait → 基础移动
        
        # 映射关系 Hs (5能力 → 5功能) - 简化的PARTNR工具
        # 功能维度：[基础移动, 物体操作, 基本控制, 液体处理, 电源控制]
        Hs = [np.zeros((1, self.n_f)) for _ in range(self.n_c)]
        
        # 能力0：基础移动 → 基础移动功能
        if self.n_c > 0: Hs[0][0, 0] = 1
        # 能力1：物体操作 → 基础移动+物体操作功能
        if self.n_c > 1 and self.n_f >= 2: 
            Hs[1][0, 0] = 1  # 需要基础移动
            Hs[1][0, 1] = 1  # 需要物体操作
        # 能力2：基本控制 → 基础移动+基本控制功能
        if self.n_c > 2 and self.n_f >= 3:
            Hs[2][0, 0] = 1  # 需要基础移动
            Hs[2][0, 2] = 1  # 需要基本控制
        # 能力3：液体处理 → 基础移动+液体处理功能
        if self.n_c > 3 and self.n_f >= 4:
            Hs[3][0, 0] = 1  # 需要基础移动
            Hs[3][0, 3] = 1  # 需要液体处理
        # 能力4：电源控制 → 基础移动+电源控制功能
        if self.n_c > 4 and self.n_f >= 5:
            Hs[4][0, 0] = 1  # 需要基础移动
            Hs[4][0, 4] = 1  # 需要电源控制
        
        # 权重矩阵 ws - 简化的能力重要性权重
        ws = [np.eye(1) for _ in range(self.n_c)]  # 默认单位权重
        # 调整关键能力的权重
        if self.n_c > 0: ws[0] = 2.0 * np.eye(1)    # 基础移动 - 基础且重要
        if self.n_c > 1: ws[1] = 2.5 * np.eye(1)    # 物体操作 - 核心功能
        if self.n_c > 2: ws[2] = 2.0 * np.eye(1)    # 基本控制 - 重要控制
        if self.n_c > 3: ws[3] = 1.8 * np.eye(1)    # 液体处理 - 中等重要
        if self.n_c > 4: ws[4] = 1.5 * np.eye(1)    # 电源控制 - 相对次要
        
        # 初始化任务函数 - 对应13种PARTNR工具
        tasks = [None] * self.n_t
        
        # 任务0：Navigate
        if self.n_t > 0:
            tasks[0] = {
                'function': NaviTask.navi_function,
                'gradient': NaviTask.navi_gradient,
                'time_derivative': NaviTask.navi_time_derivative,
                'name': 'Navigate'
            }
        
        # 任务1：Explore
        if self.n_t > 1:
            tasks[1] = {
                'function': ExploreTask.explore_function,
                'gradient': ExploreTask.explore_gradient,
                'time_derivative': ExploreTask.explore_time_derivative,
                'name': 'Explore'
            }
        
        # 任务2：Pick
        if self.n_t > 2:
            tasks[2] = {
                'function': PickTask.pick_function,
                'gradient': PickTask.pick_gradient,
                'time_derivative': PickTask.pick_time_derivative,
                'name': 'Pick'
            }
        
        # 任务3：Place
        if self.n_t > 3:
            tasks[3] = {
                'function': PlaceTask.place_function,
                'gradient': PlaceTask.place_gradient,
                'time_derivative': PlaceTask.place_time_derivative,
                'name': 'Place'
            }
        
        # 任务4：Open
        if self.n_t > 4:
            tasks[4] = {
                'function': OpenTask.open_function,
                'gradient': OpenTask.open_gradient,
                'time_derivative': OpenTask.open_time_derivative,
                'name': 'Open'
            }
        
        # 任务5：Close
        if self.n_t > 5:
            tasks[5] = {
                'function': CloseTask.close_function,
                'gradient': CloseTask.close_gradient,
                'time_derivative': CloseTask.close_time_derivative,
                'name': 'Close'
            }
        
        # 任务6：Clean
        if self.n_t > 6:
            tasks[6] = {
                'function': CleanTask.clean_function,
                'gradient': CleanTask.clean_gradient,
                'time_derivative': CleanTask.clean_time_derivative,
                'name': 'Clean'
            }
        
        # 任务7：Fill
        if self.n_t > 7:
            tasks[7] = {
                'function': FillTask.fill_function,
                'gradient': FillTask.fill_gradient,
                'time_derivative': FillTask.fill_time_derivative,
                'name': 'Fill'
            }
        
        # 任务8：Pour
        if self.n_t > 8:
            tasks[8] = {
                'function': PourTask.pour_function,
                'gradient': PourTask.pour_gradient,
                'time_derivative': PourTask.pour_time_derivative,
                'name': 'Pour'
            }
        
        # 任务9：PowerOn
        if self.n_t > 9:
            tasks[9] = {
                'function': PowerOnTask.poweron_function,
                'gradient': PowerOnTask.poweron_gradient,
                'time_derivative': PowerOnTask.poweron_time_derivative,
                'name': 'PowerOn'
            }
        
        # 任务10：PowerOff
        if self.n_t > 10:
            tasks[10] = {
                'function': PowerOffTask.poweroff_function,
                'gradient': PowerOffTask.poweroff_gradient,
                'time_derivative': PowerOffTask.poweroff_time_derivative,
                'name': 'PowerOff'
            }
        
        # 任务11：Rearrange
        if self.n_t > 11:
            tasks[11] = {
                'function': RearrangeTask.rearrange_function,
                'gradient': RearrangeTask.rearrange_gradient,
                'time_derivative': RearrangeTask.rearrange_time_derivative,
                'name': 'Rearrange'
            }
        
        # 任务12：Wait
        if self.n_t > 12:
            tasks[12] = {
                'function': WaitTask.wait_function,
                'gradient': WaitTask.wait_gradient,
                'time_derivative': WaitTask.wait_time_derivative,
                'name': 'Wait'
            }
        
        # 机器人动力学模型
        robot_dyn_config = RobotDynamicsConfig(n_x=self.n_x, n_u=self.n_u)
        robot_dyn = robot_dyn_config.get_robot_dynamics()
        
        return {
            'A': A, 'Hs': Hs, 'T': T, 'ws': ws,
            'tasks': tasks,
            'robot_dyn': robot_dyn
        }
    
    def update_scenario_from_world_state(self, world_state):
        """
        根据最新的世界状态更新场景参数，特别是与任务相关的目标。
        """
        if not world_state:
            return

        # 更新导航目标 (p_goal) - 示例：使用第一个找到的物体位置
        if 'object_positions' in world_state and world_state['object_positions']:
            # 选择一个目标，例如第一个物体
            first_object_name = next(iter(world_state['object_positions']))
            self.update_global_task_var('p_goal', world_state['object_positions'][first_object_name])
        
        # 更新探索目标
        if 'furniture_positions' in world_state and world_state['furniture_positions']:
            new_explore_targets = []
            for i, (name, pos) in enumerate(world_state['furniture_positions'].items()):
                new_explore_targets.append({'position': pos, 'explored': False, 'id': i})
            self.update_global_task_var('exploration_targets', new_explore_targets)

    def get_scenario_params(self):
        return self.scenario_params
    
    def get_updated_scenario_params(self):
        return {'A': self.scenario_params['A'].copy()}
    
    def update_robot_features(self, robot_idx, features):
        if 0 <= robot_idx < self.n_r and len(features) == self.n_f:
            self.scenario_params['A'][:, robot_idx] = features
        else:
            print(f"警告：索引超出范围或特征数量不匹配，robot_idx: {robot_idx}, features: {features}")
    
    def get_robot_features(self):
        return self.scenario_params['A']
    
    def get_task_matrix(self):
        return self.scenario_params['T']
    
    def get_mapping_matrices(self):
        return self.scenario_params['Hs']
    
    def get_weight_matrices(self):
        return self.scenario_params['ws']
    
    def get_tasks(self):
        return self.scenario_params['tasks']
    
    def get_robot_dynamics(self):
        return self.scenario_params['robot_dyn']
    
    def get_robot_count(self):
        return self.n_r
    
    def get_task_count(self):
        return self.n_t
    
    def get_capability_count(self):
        return self.n_c
    
    def get_feature_count(self):
        return self.n_f
    
    def get_state_dimension(self):
        return self.scenario_params['robot_dyn']['n_x']
    
    def get_control_dimension(self):
        return self.scenario_params['robot_dyn']['n_u'] 