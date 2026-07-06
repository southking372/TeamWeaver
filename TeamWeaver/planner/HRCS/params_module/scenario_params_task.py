# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

# task_utils/task_module/scenario_params_task.py
import numpy as np
from task_module.navi_module import NaviTask
from task_module.explore_module import ExploreTask
from task_module.manipulation_module import ManipulationTask, ManipulationPhase
from task_module.wait_module import WaitTask
from task_module.pick_module import PickTask
from task_module.place_module import PlaceTask
from task_module.open_module import OpenTask
from task_module.close_module import CloseTask
from task_module.clean_module import CleanTask
from task_module.state_manipulation_modules import (
    FillTask, PourTask, PowerOnTask, PowerOffTask, RearrangeTask
)
from sys_module.robot_dynamics_module import RobotDynamicsConfig

class ScenarioConfigTask:
    
    def __init__(self, n_r=2, n_t=13, n_c=5, n_f=5, n_x=3, n_u=2):
        """
        Initialization scenario configuration task - simplified PARTNR Agent capability model
        
        Parameters:
            n_r: number of robots，Default is 2 (Agent 0, Agent 1)
            n_t: Number of task types, default is 13 (Navigate, Explore, Pick, Place, Open, Close, Clean, Fill, Pour, PowerOn, PowerOff, Rearrange, Wait)
            n_c: capability categoriesquantity, default is 5 (Simplified feature classification: basic movement, object manipulation, Basic control, liquiddeal with, powercontrol)
            n_f: Functional dimension, default is 5 (Basic movement/object manipulation/Basic control/liquiddeal with/powercontrol)
            n_x: Status dimension, default is 3 (x, y, θ)
            n_u: controlInput dimension, default is 2 (v, ω)
        """
        self.n_r = n_r
        self.n_t = n_t
        self.n_c = n_c
        self.n_f = n_f
        self.n_x = n_x
        self.n_u = n_u
        
        # Make sure the n_x passed in, n_u Consistent with what RobotDynamicsConfig expects
        if n_x != 3 or n_u != 2:
             print(f"warn: ScenarioConfigTask received n_x={n_x}, n_u={n_u}, But usually expect n_x=3, n_u=2 For use on differential drive models.")

        self.scenario_params = self._initialize_scenario_params()
        
        # Initialize task-related global variables
        self._initialize_global_task_vars()
    
    def _initialize_global_task_vars(self):
        self.global_task_vars = {
            # NaviTask related variables
            'p_goal': np.array([1.5, 1.0]),
            'theta_goal': 0.0,
            'dist_thresh': 0.2,
            'orientation_weight': 0.3,
            
            # ExploreTaskrelated variables
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
            
            # Pick/Place/Manipulationrelated variables
            'target_object_position': np.array([0.8, -0.5]),
            'target_receptacle_position': np.array([-0.8, 0.7]),
            'pick_dist_thresh': 0.15,
            'place_dist_thresh': 0.15,
            'manipulation_progress': 0.0,
            'is_holding': False,
            'holding_robot_id': None,
            'active_manipulation_robot_id': None,
            'pick_succeeded': False,
            'place_completed': False,
            'navi_completed': False,
            'explore_completed': False,
            'manipulation_completed': False,
            'manipulation_phase': ManipulationPhase.NAV_OBJ,
            
            # Open/Closerelated variables
            'target_furniture_position': np.array([1.0, 0.5]),
            'operation_dist_thresh': 0.2,
            'furniture_open_state': False,
            
            # Cleanrelated variables
            'clean_dist_thresh': 0.15,
            'object_clean_state': False,
            
            # Fill/Pourrelated variables
            'target_container_position': np.array([0.3, 0.8]),
            'fill_dist_thresh': 0.15,
            'container_filled_state': False,
            'pour_dist_thresh': 0.15,
            'pour_completed_state': False,
            
            # PowerOn/PowerOffrelated variables
            'target_device_position': np.array([0.7, 0.4]),
            'power_dist_thresh': 0.15,
            'device_power_state': False,
            
            # WaitTaskrelated variables
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
            print(f"warn: Attempt to update global task variable that does not exist '{var_name}'")
            # self.global_task_vars[var_name] = value # Optional: add dynamically
    
    def _initialize_scenario_params(self):
        # Initializing Robot Characteristics and Capabilities - Simplified PARTNR Agent Configuration
        # Feature dimensions:[Basic movement, object manipulation, Basic control, liquiddeal with, powercontrol]
        # Agent 0: Close, Explore, Navigate, Open, Pick, Place, Rearrange, Wait
        # Agent 1: Clean, Close, Explore, Fill, Navigate, Open, Pick, Place, Pour, PowerOff, PowerOn, Rearrange, Wait
        
        partnr_agent_features = [
            # Agent 0: Basic movement, object manipulation and basic control capabilities
            [1, 1, 1, 0, 0],  # Basic movement+object manipulation+basic control, no liquid processing/powercontrol
            # Agent 1: full capabilityAgent
            [1, 1, 1, 1, 1]   # All functions are available
        ]

        # Initialize matrix A based on actual n_r
        A = np.zeros((self.n_f, self.n_r))

        # Assign PARTNR characteristics
        for i in range(self.n_r):
            if i < len(partnr_agent_features):
                feature_vector = partnr_agent_features[i]
                if len(feature_vector) == self.n_f:
                    A[:, i] = feature_vector
                else:
                    print(f"warn: Agent {i} The characteristic length of ({len(feature_vector)}) with n_f ({self.n_f}) no match")
                    A[:, i] = np.zeros(self.n_f)
            else:
                # Additional robots use Agent 1's configuration (truncate/pad to n_f)
                fallback = partnr_agent_features[1] if len(partnr_agent_features) > 1 else np.ones(self.n_f)
                vec = np.asarray(fallback, dtype=float).reshape(-1)
                if vec.size >= self.n_f:
                    A[:, i] = vec[:self.n_f]
                else:
                    A[:, i] = np.pad(vec, (0, self.n_f - vec.size))

        # Mission Capability Requirements Matrix - 13 PARTNR tools mapped to 5 simplified capabilities
        # Task sequence: [Navigate, Explore, Pick, Place, Open, Close, Clean, Fill, Pour, PowerOn, PowerOff, Rearrange, Wait]
        # Ability classification: [Basic movement, object manipulation, Basic control, liquiddeal with, powercontrol]
        T = np.zeros((self.n_t, self.n_c))
        
        # Mapping relationship from tasks to abilities
        if self.n_t > 0 and self.n_c > 0: T[0, 0] = 1   # Navigate → Basic movement
        if self.n_t > 1 and self.n_c > 0: T[1, 0] = 1   # Explore → Basic movement
        if self.n_t > 2 and self.n_c > 1: T[2, 1] = 1   # Pick → object manipulation
        if self.n_t > 3 and self.n_c > 1: T[3, 1] = 1   # Place → object manipulation
        if self.n_t > 4 and self.n_c > 2: T[4, 2] = 1   # Open → Basic control
        if self.n_t > 5 and self.n_c > 2: T[5, 2] = 1   # Close → Basic control
        if self.n_t > 6 and self.n_c > 3: T[6, 3] = 1   # Clean → liquidProcess
        if self.n_t > 7 and self.n_c > 3: T[7, 3] = 1   # Fill → liquidProcess
        if self.n_t > 8 and self.n_c > 3: T[8, 3] = 1   # Pour → liquidProcess
        if self.n_t > 9 and self.n_c > 4: T[9, 4] = 1   # PowerOn → powercontrol
        if self.n_t > 10 and self.n_c > 4: T[10, 4] = 1 # PowerOff → powercontrol
        if self.n_t > 11 and self.n_c > 1: T[11, 1] = 1 # Rearrange → object manipulation
        if self.n_t > 12 and self.n_c > 0: T[12, 0] = 1 # Wait → Basic movement
        
        # Mapping relationship Hs (5Capabilities → 5 functions) - Simplified PARTNR tool
        # Functional dimensions:[Basic movement, object manipulation, Basic control, liquiddeal with, powercontrol]
        Hs = [np.zeros((1, self.n_f)) for _ in range(self.n_c)]
        
        # Capability 0: Basic movement → Basic movement function
        if self.n_c > 0: Hs[0][0, 0] = 1
        # Ability 1: Object manipulation → basic movement+object manipulation function
        if self.n_c > 1 and self.n_f >= 2: 
            Hs[1][0, 0] = 1  # Requires basic movement
            Hs[1][0, 1] = 1  # Requires object manipulation
        # Ability 2: Basic control → basic movement+basic control function
        if self.n_c > 2 and self.n_f >= 3:
            Hs[2][0, 0] = 1  # Requires basic movement
            Hs[2][0, 2] = 1  # Need basic control
        # Capability 3: Liquid processing → Basic movement+liquid processing function
        if self.n_c > 3 and self.n_f >= 4:
            Hs[3][0, 0] = 1  # Requires basic movement
            Hs[3][0, 3] = 1  # Need liquid processing
        # Capability 4: powercontrol → basic movement+powercontrol function
        if self.n_c > 4 and self.n_f >= 5:
            Hs[4][0, 0] = 1  # Requires basic movement
            Hs[4][0, 4] = 1  # Requires powercontrol
        
        # Weight matrix ws - Simplified capability importance weights
        ws = [np.eye(1) for _ in range(self.n_c)]  # Default unit weight
        # Adjust the weight of key capabilities
        if self.n_c > 0: ws[0] = 2.0 * np.eye(1)    # Basic movement - basic and important
        if self.n_c > 1: ws[1] = 2.5 * np.eye(1)    # Object manipulation - core functionality
        if self.n_c > 2: ws[2] = 2.0 * np.eye(1)    # Basic controls - important controls
        if self.n_c > 3: ws[3] = 1.8 * np.eye(1)    # liquidProcessing - Moderately Important
        if self.n_c > 4: ws[4] = 1.5 * np.eye(1)    # powercontrol - relatively minor
        
        # Initialization task function - corresponding to 13 PARTNR tools
        tasks = [None] * self.n_t
        
        # Mission 0: Navigate
        if self.n_t > 0:
            tasks[0] = {
                'function': NaviTask.navi_function,
                'gradient': NaviTask.navi_gradient,
                'time_derivative': NaviTask.navi_time_derivative,
                'name': 'Navigate'
            }
        
        # Task 1: Explore
        if self.n_t > 1:
            tasks[1] = {
                'function': ExploreTask.explore_function,
                'gradient': ExploreTask.explore_gradient,
                'time_derivative': ExploreTask.explore_time_derivative,
                'name': 'Explore'
            }
        
        # Task 2: Pick
        if self.n_t > 2:
            tasks[2] = {
                'function': PickTask.pick_function,
                'gradient': PickTask.pick_gradient,
                'time_derivative': PickTask.pick_time_derivative,
                'name': 'Pick'
            }
        
        # Task 3: Place
        if self.n_t > 3:
            tasks[3] = {
                'function': PlaceTask.place_function,
                'gradient': PlaceTask.place_gradient,
                'time_derivative': PlaceTask.place_time_derivative,
                'name': 'Place'
            }
        
        # Task 4: Open
        if self.n_t > 4:
            tasks[4] = {
                'function': OpenTask.open_function,
                'gradient': OpenTask.open_gradient,
                'time_derivative': OpenTask.open_time_derivative,
                'name': 'Open'
            }
        
        # Task 5: Close
        if self.n_t > 5:
            tasks[5] = {
                'function': CloseTask.close_function,
                'gradient': CloseTask.close_gradient,
                'time_derivative': CloseTask.close_time_derivative,
                'name': 'Close'
            }
        
        # Task 6: Clean
        if self.n_t > 6:
            tasks[6] = {
                'function': CleanTask.clean_function,
                'gradient': CleanTask.clean_gradient,
                'time_derivative': CleanTask.clean_time_derivative,
                'name': 'Clean'
            }
        
        # Task 7: Fill
        if self.n_t > 7:
            tasks[7] = {
                'function': FillTask.fill_function,
                'gradient': FillTask.fill_gradient,
                'time_derivative': FillTask.fill_time_derivative,
                'name': 'Fill'
            }
        
        # Mission 8: Pour
        if self.n_t > 8:
            tasks[8] = {
                'function': PourTask.pour_function,
                'gradient': PourTask.pour_gradient,
                'time_derivative': PourTask.pour_time_derivative,
                'name': 'Pour'
            }
        
        # Task 9: PowerOn
        if self.n_t > 9:
            tasks[9] = {
                'function': PowerOnTask.poweron_function,
                'gradient': PowerOnTask.poweron_gradient,
                'time_derivative': PowerOnTask.poweron_time_derivative,
                'name': 'PowerOn'
            }
        
        # Task 10: PowerOff
        if self.n_t > 10:
            tasks[10] = {
                'function': PowerOffTask.poweroff_function,
                'gradient': PowerOffTask.poweroff_gradient,
                'time_derivative': PowerOffTask.poweroff_time_derivative,
                'name': 'PowerOff'
            }
        
        # Mission 11: Rearrange
        if self.n_t > 11:
            tasks[11] = {
                'function': RearrangeTask.rearrange_function,
                'gradient': RearrangeTask.rearrange_gradient,
                'time_derivative': RearrangeTask.rearrange_time_derivative,
                'name': 'Rearrange'
            }
        
        # Task 12: Wait
        if self.n_t > 12:
            tasks[12] = {
                'function': WaitTask.wait_function,
                'gradient': WaitTask.wait_gradient,
                'time_derivative': WaitTask.wait_time_derivative,
                'name': 'Wait'
            }
        
        # Robot dynamics model
        robot_dyn_config = RobotDynamicsConfig(n_x=self.n_x, n_u=self.n_u)
        robot_dyn = robot_dyn_config.get_robot_dynamics()
        
        return {
            'A': A, 'Hs': Hs, 'T': T, 'ws': ws,
            'tasks': tasks,
            'robot_dyn': robot_dyn
        }
    
    def update_scenario_from_world_state(self, world_state):
        """
        Update scene parameters, especially mission-related objectives, based on the latest world state.
        """
        if not world_state:
            return

        # Update navigation target (p_goal) - Example: Use the first found object position
        if 'object_positions' in world_state and world_state['object_positions']:
            # Select a target, such as the first object
            first_object_name = next(iter(world_state['object_positions']))
            self.update_global_task_var('p_goal', world_state['object_positions'][first_object_name])
        
        # Update exploration goals
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
            print(f"Warning: Index out of range or mismatch in number of features, robot_idx: {robot_idx}, features: {features}")
    
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