import numpy as np
from habitat_llm.planner.HRCS.task_module.manipulation_module import ManipulationPhase

class GlobalVarsManager_task:
    def __init__(self):
        """
        初始化全局变量管理器 - 对齐scenario_params_task.py中的13种PARTNR任务
        """
        # 初始化NaviTask相关变量
        self.dp_dt = np.array([0.0, 0.0])         # 导航目标点的速度
        self.dist_thresh = 0.5                    # 导航距离阈值
        
        # 初始化ExploreTask相关变量
        self.exploration_targets = [              # 探索目标列表 (基础目标点)
            {'position': np.array([1.0, 0.8]), 'explored': False},
            {'position': np.array([-1.0, 0.5]), 'explored': False},
            {'position': np.array([0.5, -0.8]), 'explored': False}
        ]
        self.explored_map = None                  # 已探索地图（暂未使用）
        self.explore_dist_thresh = 0.4            # 用于标记 target explored 的阈值
        self.exploration_frontiers = np.array([]) # 当前的探索前沿点，由未探索的 exploration_targets 生成
        
        # 初始化ManipulationTask相关变量
        self.manipulation_phase = ManipulationPhase.NAV_OBJ  # 操作阶段
        self.target_object_position = np.array([0.8, -0.5])  # 目标物体位置
        self.target_receptacle_position = np.array([-0.8, 0.7])  # 目标放置位置
        self.pick_dist_thresh = 0.2               # 抓取距离阈值
        self.place_dist_thresh = 0.2              # 放置距离阈值
        self.pick_action_value = 1.5              # 抓取动作价值
        self.place_action_value = 1.5             # 放置动作价值
        
        # 初始化WaitTask相关变量
        self.wait_step_threshold = 5.0            # 等待时间阈值（秒）
        self.sim_freq = 1.0                       # 仿真频率
        self.wait_elapsed_time = 0.0              # 已等待时间

        self.manipulation_phase_timer = 0.0         # 当前操作子阶段（PICK/PLACE）的计时器
        self.manipulation_action_fixed_duration = 5.0 # PICK/PLACE 阶段固定持续时间（秒）

        self._update_exploration_frontiers()
    
    def _update_exploration_frontiers(self):
        if self.exploration_targets is None:
            self.exploration_frontiers = np.array([])
            return

        unexplored_frontiers = [
            target['position'] for target in self.exploration_targets
            if not target.get('explored', False)
        ]

        if unexplored_frontiers:
            self.exploration_frontiers = np.array(unexplored_frontiers)
        else:
            self.exploration_frontiers = np.array([])
    
    def get_var(self, var_name, default_value=None):
        if hasattr(self, var_name):
            return getattr(self, var_name)
        return default_value
    
    def set_var(self, var_name, value):
        setattr(self, var_name, value)
    
    def register_var(self, var_name, value):
        setattr(self, var_name, value)
    
    def get_all_vars(self):
        return {
            name: value for name, value in self.__dict__.items() 
            if not name.startswith('_') and not callable(value)
        }
    
    def register_vars_from_dict(self, vars_dict):
        for var_name, value in vars_dict.items():
            self.register_var(var_name, value)
    
    def update_task_timer(self, dt):
        self.wait_elapsed_time += dt

        if self.manipulation_phase == ManipulationPhase.PICK or self.manipulation_phase == ManipulationPhase.PLACE:
            self.manipulation_phase_timer += dt
        else:
            self.manipulation_phase_timer = 0.0

    def reset_wait_timer(self):
        self.wait_elapsed_time = 0.0
    
    def update_exploration_status(self, robot_positions):
        if self.exploration_targets is None:
            return 0
            
        updated_count = 0
        needs_frontier_update = False
        for robot_pos in robot_positions:
            for target in self.exploration_targets:
                if not target.get('explored', False):
                    dist = np.linalg.norm(robot_pos - target['position'])
                    if dist < self.explore_dist_thresh:
                        target['explored'] = True
                        updated_count += 1
                        needs_frontier_update = True

        if needs_frontier_update:
            self._update_exploration_frontiers()

        return updated_count 

    def check_and_advance_manipulation_phase(self, robot_idx):
        if self.x is None or robot_idx >= self.x.shape[1]:
            return False
        robot_pos = self.x[0:2, robot_idx]
        advanced = False

        if self.manipulation_phase == ManipulationPhase.NAV_OBJ:
            dist_to_obj = np.linalg.norm(robot_pos - self.target_object_position)
            if dist_to_obj < self.pick_dist_thresh:
                self.manipulation_phase = ManipulationPhase.PICK
                self.manipulation_phase_timer = 0.0 # 开始计时
                print(f"Robot {robot_idx}: Reached object, advancing to PICK phase (fixed duration).")
                advanced = True
        elif self.manipulation_phase == ManipulationPhase.PICK:
            # 检查是否在PICK范围内
            dist_to_obj = np.linalg.norm(robot_pos - self.target_object_position)
            if dist_to_obj >= self.pick_dist_thresh:
                # 如果不在范围内，返回到导航阶段
                self.manipulation_phase = ManipulationPhase.NAV_OBJ
                self.manipulation_phase_timer = 0.0
                print(f"Robot {robot_idx}: Out of PICK range, returning to NAV_OBJ phase.")
                advanced = True
            # 检查是否达到固定时长
            elif self.manipulation_phase_timer >= self.manipulation_action_fixed_duration:
                self.manipulation_phase = ManipulationPhase.NAV_REC
                self.is_holding = True  # 更新为持有状态
                self.holding_robot_id = robot_idx # 记录持有者ID
                # self.manipulation_phase_timer = 0.0 # 会在 update_task_timer 中重置
                print(f"Robot {robot_idx}: PICK finished, holding object. Advancing to NAV_REC.")
                advanced = True
        elif self.manipulation_phase == ManipulationPhase.NAV_REC:
            dist_to_rec = np.linalg.norm(robot_pos - self.target_receptacle_position)
            if dist_to_rec < self.place_dist_thresh:
                self.manipulation_phase = ManipulationPhase.PLACE
                self.manipulation_phase_timer = 0.0 # 开始计时
                print(f"Robot {robot_idx}: Reached receptacle, advancing to PLACE phase (fixed duration).")
                advanced = True
            # 检查是否超出范围
            # elif dist_to_rec >= self.place_dist_thresh * 1.5:  # 允许一定的缓冲距离
            #     # 如果超出范围太多，返回到导航阶段
            #     self.manipulation_phase = ManipulationPhase.NAV_REC
            #     self.manipulation_phase_timer = 0.0
            #     print(f"Robot {robot_idx}: Out of NAV_REC range, returning to NAV_REC phase.")
            #     advanced = True
        elif self.manipulation_phase == ManipulationPhase.PLACE:
            # 检查是否在PLACE范围内
            dist_to_rec = np.linalg.norm(robot_pos - self.target_receptacle_position)
            if dist_to_rec >= self.place_dist_thresh:
                # 如果不在范围内，返回到导航阶段
                self.manipulation_phase = ManipulationPhase.NAV_REC
                self.manipulation_phase_timer = 0.0
                print(f"Robot {robot_idx}: Out of PLACE range, returning to NAV_REC phase.")
                advanced = True
            # 检查是否达到固定时长
            elif self.manipulation_phase_timer >= self.manipulation_action_fixed_duration:
                # 操作循环完成，回到初始状态
                self.manipulation_phase = ManipulationPhase.NAV_OBJ
                self.is_holding = False # 更新为非持有状态
                self.holding_robot_id = None # 清除持有者ID
                # self.manipulation_phase_timer = 0.0 # 会在 update_task_timer 中重置
                print(f"Robot {robot_idx}: PLACE finished, released object. Manipulation cycle completed.")
                advanced = True
                # 可能需要更新目标物体/位置或标记任务完成

        return advanced 