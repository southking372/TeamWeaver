# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

import numpy as np
from task_module.manipulation_module import ManipulationPhase

class GlobalVarsManager_task:
    def __init__(self):
        """
Initialize Global Variable Manager - Alignmentscenario_params_task.py13 kinds ofPARTNRTask
        """
        #initializationNaviTaskrelated variables
        self.dp_dt = np.array([0.0, 0.0])         #Navigation target speed
        self.dist_thresh = 0.5                    #Navigation distance threshold
        
        #initializationExploreTaskrelated variables
        self.exploration_targets = [              #Explore target list(basic target point)
            {'position': np.array([1.0, 0.8]), 'explored': False},
            {'position': np.array([-1.0, 0.5]), 'explored': False},
            {'position': np.array([0.5, -0.8]), 'explored': False}
        ]
        self.explored_map = None                  #Explored map (not used yet)
        self.explore_dist_thresh = 0.4            #for markingtarget exploredthreshold
        self.exploration_frontiers = np.array([]) #Current exploration frontiers, consisting of unexploredexploration_targetsgenerate
        
        #initializationManipulationTaskrelated variables
        self.manipulation_phase = ManipulationPhase.NAV_OBJ  # manipulationstage
        self.target_object_position = np.array([0.8, -0.5])  #Target object position
        self.target_receptacle_position = np.array([-0.8, 0.7])  #target placement location
        self.pick_dist_thresh = 0.2               #Crawl distance threshold
        self.place_dist_thresh = 0.2              #Place distance threshold
        self.pick_action_value = 1.5              #Grab action value
        self.place_action_value = 1.5             #Place action value
        self.is_holding = False
        self.holding_robot_id = None
        self.active_manipulation_robot_id = None  # only one robot drives pick/place FSM
        self.pick_succeeded = False               # True after Pick completes with object acquired
        self.place_completed = False              # True after Place completes successfully
        
        # Demo task completion flags (Navigate / Explore / Manipulation)
        self.navi_completed = False
        self.explore_completed = False
        self.manipulation_completed = False
        
        #initializationWaitTaskrelated variables
        self.wait_step_threshold = 5.0            #Wait time threshold (seconds)
        self.sim_freq = 1.0                       #Simulation frequency
        self.wait_elapsed_time = 0.0              #Waiting time

        self.manipulation_phase_timer = 0.0         #currentmanipulationsubstage (PICK/PLACE) timer
        self.manipulation_action_fixed_duration = 5.0 # PICK/PLACEPhase fixed duration (seconds)

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

    def get_effective_wait_time(self):
        """Elapsed wait time within the current replan cycle (mod 5s threshold)."""
        if self.wait_step_threshold <= 0:
            return self.wait_elapsed_time
        return self.wait_elapsed_time % self.wait_step_threshold
    
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

        if self.exploration_targets:
            self.explore_completed = all(
                target.get('explored', False) for target in self.exploration_targets
            )

        return updated_count 

    def update_navi_completion(self, x, task_assignment, navi_task_idx=1):
        """Mark Navigate complete once any assigned robot reaches the goal."""
        if self.navi_completed or x is None or self.p_goal is None:
            return
        dist_thresh = getattr(self, 'dist_thresh', 0.2)
        for i, task in enumerate(task_assignment):
            if task == navi_task_idx:
                if np.linalg.norm(x[0:2, i] - self.p_goal) < dist_thresh:
                    self.navi_completed = True
                    print(f"Navigate task completed: Robot {i} reached goal.")
                    break

    def all_demo_tasks_completed(self):
        """Navigate, Explore, and Manipulation (Pick+Place) are all done."""
        return (
            self.navi_completed
            and self.explore_completed
            and self.manipulation_completed
        )

    def get_demo_completion_summary(self):
        return {
            'navi': self.navi_completed,
            'explore': self.explore_completed,
            'manipulation': self.manipulation_completed,
            'pick_succeeded': self.pick_succeeded,
            'place_completed': self.place_completed,
        }

    def get_manipulation_executor(self):
        """Robot authorized to advance the shared manipulation state machine."""
        if self.holding_robot_id is not None:
            return self.holding_robot_id
        return self.active_manipulation_robot_id

    def select_manipulation_executor(self, candidate_robot_indices):
        """Pick the single robot that may advance manipulation this step."""
        if not candidate_robot_indices:
            return None

        executor = self.get_manipulation_executor()
        if executor is not None:
            return executor if executor in candidate_robot_indices else None

        if self.manipulation_phase in (ManipulationPhase.NAV_OBJ, ManipulationPhase.PICK):
            target = self.target_object_position
        else:
            target = self.target_receptacle_position

        if self.x is None or target is None:
            return candidate_robot_indices[0]

        return min(
            candidate_robot_indices,
            key=lambda idx: np.linalg.norm(self.x[0:2, idx] - target),
        )

    def check_and_advance_manipulation_phase(self, robot_idx):
        if self.x is None or robot_idx >= self.x.shape[1]:
            return False

        executor = self.get_manipulation_executor()
        if executor is not None and executor != robot_idx:
            return False

        robot_pos = self.x[0:2, robot_idx]
        advanced = False

        if self.manipulation_phase == ManipulationPhase.NAV_OBJ:
            if self.is_holding:
                return False
            dist_to_obj = np.linalg.norm(robot_pos - self.target_object_position)
            if dist_to_obj < self.pick_dist_thresh:
                self.active_manipulation_robot_id = robot_idx
                self.manipulation_phase = ManipulationPhase.PICK
                self.manipulation_phase_timer = 0.0
                print(f"Robot {robot_idx}: Reached object, advancing to PICK phase (fixed duration).")
                advanced = True
        elif self.manipulation_phase == ManipulationPhase.PICK:
            dist_to_obj = np.linalg.norm(robot_pos - self.target_object_position)
            if dist_to_obj >= self.pick_dist_thresh:
                self.manipulation_phase = ManipulationPhase.NAV_OBJ
                self.manipulation_phase_timer = 0.0
                self.active_manipulation_robot_id = None
                self.pick_succeeded = False
                print(f"Robot {robot_idx}: Out of PICK range, returning to NAV_OBJ phase.")
                advanced = True
            elif self.manipulation_phase_timer >= self.manipulation_action_fixed_duration:
                self.manipulation_phase = ManipulationPhase.NAV_REC
                self.is_holding = True
                self.holding_robot_id = robot_idx
                self.active_manipulation_robot_id = robot_idx
                self.pick_succeeded = True
                print(f"Robot {robot_idx}: PICK finished, holding object. Advancing to NAV_REC.")
                advanced = True
        elif self.manipulation_phase == ManipulationPhase.NAV_REC:
            if not (self.pick_succeeded and self.is_holding):
                return False
            dist_to_rec = np.linalg.norm(robot_pos - self.target_receptacle_position)
            if dist_to_rec < self.place_dist_thresh:
                self.manipulation_phase = ManipulationPhase.PLACE
                self.manipulation_phase_timer = 0.0
                print(f"Robot {robot_idx}: Reached receptacle, advancing to PLACE phase (Pick succeeded).")
                advanced = True
        elif self.manipulation_phase == ManipulationPhase.PLACE:
            if not (self.pick_succeeded and self.is_holding):
                self.manipulation_phase = ManipulationPhase.NAV_OBJ
                self.manipulation_phase_timer = 0.0
                print(f"Robot {robot_idx}: PLACE blocked — Pick not completed or object not held.")
                return True
            dist_to_rec = np.linalg.norm(robot_pos - self.target_receptacle_position)
            if dist_to_rec >= self.place_dist_thresh:
                self.manipulation_phase = ManipulationPhase.NAV_REC
                self.manipulation_phase_timer = 0.0
                print(f"Robot {robot_idx}: Out of PLACE range, returning to NAV_REC phase.")
                advanced = True
            elif self.manipulation_phase_timer >= self.manipulation_action_fixed_duration:
                self.manipulation_phase = ManipulationPhase.NAV_OBJ
                self.is_holding = False
                self.holding_robot_id = None
                self.active_manipulation_robot_id = None
                self.place_completed = True
                self.manipulation_completed = True
                print(f"Robot {robot_idx}: PLACE finished, released object. Manipulation task completed.")
                advanced = True

        return advanced
