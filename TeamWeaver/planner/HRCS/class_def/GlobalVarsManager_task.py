import numpy as np
from habitat_llm.planner.HRCS.task_module.manipulation_module import ManipulationPhase

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
                self.manipulation_phase_timer = 0.0 #Start timing
                print(f"Robot {robot_idx}: Reached object, advancing to PICK phase (fixed duration).")
                advanced = True
        elif self.manipulation_phase == ManipulationPhase.PICK:
            #Check if there isPICKwithin range
            dist_to_obj = np.linalg.norm(robot_pos - self.target_object_position)
            if dist_to_obj >= self.pick_dist_thresh:
                #If not in range, return to navigation phase
                self.manipulation_phase = ManipulationPhase.NAV_OBJ
                self.manipulation_phase_timer = 0.0
                print(f"Robot {robot_idx}: Out of PICK range, returning to NAV_OBJ phase.")
                advanced = True
            #Check if a fixed duration has been reached
            elif self.manipulation_phase_timer >= self.manipulation_action_fixed_duration:
                self.manipulation_phase = ManipulationPhase.NAV_REC
                self.is_holding = True  #Update to holding status
                self.holding_robot_id = robot_idx #record holderID
                # self.manipulation_phase_timer = 0.0 #will be inupdate_task_timermedium reset
                print(f"Robot {robot_idx}: PICK finished, holding object. Advancing to NAV_REC.")
                advanced = True
        elif self.manipulation_phase == ManipulationPhase.NAV_REC:
            dist_to_rec = np.linalg.norm(robot_pos - self.target_receptacle_position)
            if dist_to_rec < self.place_dist_thresh:
                self.manipulation_phase = ManipulationPhase.PLACE
                self.manipulation_phase_timer = 0.0 #Start timing
                print(f"Robot {robot_idx}: Reached receptacle, advancing to PLACE phase (fixed duration).")
                advanced = True
            #Check if out of range
            # elif dist_to_rec >= self.place_dist_thresh * 1.5:  #Allow a certain buffer distance
            #     #If it's too far out of scope, return to the navigation phase
            #     self.manipulation_phase = ManipulationPhase.NAV_REC
            #     self.manipulation_phase_timer = 0.0
            #     print(f"Robot {robot_idx}: Out of NAV_REC range, returning to NAV_REC phase.")
            #     advanced = True
        elif self.manipulation_phase == ManipulationPhase.PLACE:
            #Check if there isPLACEwithin range
            dist_to_rec = np.linalg.norm(robot_pos - self.target_receptacle_position)
            if dist_to_rec >= self.place_dist_thresh:
                #If not in range, return to navigation phase
                self.manipulation_phase = ManipulationPhase.NAV_REC
                self.manipulation_phase_timer = 0.0
                print(f"Robot {robot_idx}: Out of PLACE range, returning to NAV_REC phase.")
                advanced = True
            #Check if a fixed duration has been reached
            elif self.manipulation_phase_timer >= self.manipulation_action_fixed_duration:
                # manipulationThe cycle is completed and returns to the initial state
                self.manipulation_phase = ManipulationPhase.NAV_OBJ
                self.is_holding = False #Update to non-held status
                self.holding_robot_id = None #clear holderID
                # self.manipulation_phase_timer = 0.0 #will be inupdate_task_timermedium reset
                print(f"Robot {robot_idx}: PLACE finished, released object. Manipulation cycle completed.")
                advanced = True
                #Target objects may need to be updated/Location or marking task completed

        return advanced 
