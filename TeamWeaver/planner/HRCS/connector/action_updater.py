from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from habitat_llm.planner.HRCS.connector.planner_utils import find_target_position

class ActionUpdater:
    """
Handle high-level actions and generate scene parameter updates.
This class combines action processing logic withPerceptionConnectorDecoupled.
    """

    def process_and_get_updates(
        self,
        high_level_actions: Dict[int, Tuple[str, str, Optional[str]]], 
        world_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
Processes all actions and returns a dictionary containing all parameter updates.
        """
        all_updates = {}
        motor_skill_updates = self._process_motor_skill_actions(high_level_actions, world_state)
        state_manipulation_updates = self._process_state_manipulation_actions(high_level_actions, world_state)
        all_updates.update(motor_skill_updates)
        all_updates.update(state_manipulation_updates)
        return all_updates

    def _process_motor_skill_actions(
        self, 
        high_level_actions: Dict[int, Tuple[str, str, Optional[str]]], 
        world_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
deal withMotor SkillsRelated action updates
based onAgentTool configuration: Navigate, Pick, Place, Rearrange, Explore, Wait, Open, Close
        """
        updates = {}
        
        for agent_id, action_tuple in high_level_actions.items():
            if not action_tuple or action_tuple[2] is None:  #Skip invalid actions
                continue
                
            tool_name, args_str, target_name = action_tuple
            
            # NavigationRelated(exact matchNavigate)
            if tool_name == 'Navigate':
                nav_updates = self._update_navigation_params(target_name, world_state)
                updates.update(nav_updates)
                print(f"DEBUG: Agent {agent_id} using Navigate to {target_name}")
            
            # ManipulationRelated(exact matchPick, Place, Rearrange)
            elif tool_name in ['Pick', 'Place', 'Rearrange']:
                manip_updates = self._update_manipulation_params(tool_name, target_name, world_state)
                updates.update(manip_updates)
                print(f"DEBUG: Agent {agent_id} using {tool_name} on {target_name}")
            
            # ExplorationRelated(exact matchExplore)
            elif tool_name == 'Explore':
                explore_updates = self._update_exploration_params(target_name, world_state)
                updates.update(explore_updates)
                print(f"DEBUG: Agent {agent_id} using Explore in {target_name}")
            
            #Articulatedcontrol (exact matchOpen, Close)
            elif tool_name in ['Open', 'Close']:
                # Open/CloseMainly affects the state of the environment and usually does not need to be updatedMIQPparameter
                print(f"DEBUG: Agent {agent_id} using {tool_name} on {target_name}")
                
            # Waitaction(exact matchWait)
            elif tool_name == 'Wait':
                print(f"DEBUG: Agent {agent_id} waiting - no parameter updates needed")
            
            #Unrecognized action
            else:
                print(f"WARNING: Unrecognized motor skill action: {tool_name} for Agent {agent_id}")
        
        return updates

    def _update_navigation_params(self, target_name: str, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Update navigation related parameters"""
        updates = {}
        
        #Find target location
        target_pos = find_target_position(target_name, world_state)
        if target_pos:
            # MIQPuseXZPlane coordinates
            nav_goal = np.array([target_pos[0], target_pos[2]])
            updates['p_goal'] = nav_goal
            updates['theta_goal'] = 0.0  #Default orientation
            print(f"DEBUG: Updated navigation goal to {nav_goal} for target '{target_name}'")
        else:
            print(f"WARNING: Could not find position for navigation target '{target_name}'")
        
        return updates

    def _update_manipulation_params(
        self, 
        tool_name: str, 
        target_name: str, 
        world_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """renewmanipulationRelated parameters, based on exact tool name matching"""
        updates = {}
        
        # Pickaction(exact match)
        if tool_name == 'Pick':
            target_pos = find_target_position(target_name, world_state)
            if target_pos:
                obj_pos = np.array([target_pos[0], target_pos[2]])
                updates['target_object_position'] = obj_pos
                print(f"DEBUG: Updated pick target to {obj_pos} for object '{target_name}'")
        
        # Placeaction(exact match)
        elif tool_name == 'Place':
            target_pos = find_target_position(target_name, world_state)
            if target_pos:
                place_pos = np.array([target_pos[0], target_pos[2]])
                updates['target_receptacle_position'] = place_pos
                print(f"DEBUG: Updated place target to {place_pos} for receptacle '{target_name}'")
        
        # Rearrangeaction(exact match)
        elif tool_name == 'Rearrange':
            # Rearrange[object, spatial_relation, furniture, spatial_constraint, reference_object]
            #Mainly focus on target furniture location
            target_pos = find_target_position(target_name, world_state)
            if target_pos:
                rearrange_pos = np.array([target_pos[0], target_pos[2]])
                updates['target_receptacle_position'] = rearrange_pos
                print(f"DEBUG: Updated rearrange target to {rearrange_pos} for '{target_name}'")
        
        return updates

    def _update_exploration_params(self, target_name: str, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Update exploration related parameters"""
        updates = {}
        
        #Generate exploration points based on goals
        exploration_targets = []
        
        if target_name and target_name != 'environment':
            #Find furniture related to your goals/area
            for furn_name, furn_info in world_state.get('furniture_positions', {}).items():
                if (target_name.lower() in furn_name.lower() and 
                    furn_info and 'position' in furn_info):
                    
                    exploration_targets.append({
                        'position': np.array([furn_info['position'][0], furn_info['position'][2]]),
                        'explored': False,
                        'id': hash(furn_name)
                    })
        
        if exploration_targets:
            updates['exploration_targets'] = exploration_targets
            print(f"DEBUG: Generated {len(exploration_targets)} exploration targets for '{target_name}'")
        else:
            print(f"DEBUG: Using default exploration targets for '{target_name}'")
        
        return updates

    def _process_state_manipulation_actions(
        self, 
        high_level_actions: Dict[int, Tuple[str, str, Optional[str]]], 
        world_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Processing statusmanipulationTool related actions"""
        updates = {}
        
        for agent_id, action_tuple in high_level_actions.items():
            if not action_tuple or action_tuple[2] is None:
                continue
                
            tool_name, args_str, target_name = action_tuple
            
            #statemanipulationtool(Agent1 Proprietary capabilities)
            if tool_name in ['Clean', 'Fill', 'Pour', 'PowerOn', 'PowerOff']:
                state_updates = self._update_state_manipulation_params(tool_name, target_name, world_state, agent_id)
                updates.update(state_updates)
                print(f"DEBUG: Agent {agent_id} using state manipulation tool '{tool_name}' on '{target_name}'")
        
        return updates

    def _update_state_manipulation_params(
        self, 
        tool_name: str, 
        target_name: str, 
        world_state: Dict[str, Any],
        agent_id: int
    ) -> Dict[str, Any]:
        """update statusmanipulationRelated parameters"""
        updates = {}
        
        #examineAgentDo you have permission for this tool?
        if agent_id == 0 and tool_name in ['Clean', 'Fill', 'Pour', 'PowerOn', 'PowerOff']:
            print(f"WARNING: Agent {agent_id} attempting to use {tool_name} but lacks this capability")
            return updates
        
        #Get target object location
        target_pos = find_target_position(target_name, world_state)
        
        if tool_name == 'Clean':
            if target_pos:
                clean_pos = np.array([target_pos[0], target_pos[2]])
                updates['target_object_position'] = clean_pos
                updates['operation_type'] = 'clean'
                print(f"DEBUG: Updated clean target to {clean_pos} for object '{target_name}'")
                
        elif tool_name in ['Fill', 'Pour']:
            if target_pos:
                fluid_pos = np.array([target_pos[0], target_pos[2]])
                updates['target_receptacle_position'] = fluid_pos
                updates['operation_type'] = 'fluid_manipulation'
                print(f"DEBUG: Updated {tool_name.lower()} target to {fluid_pos} for '{target_name}'")
                
        elif tool_name in ['PowerOn', 'PowerOff']:
            if target_pos:
                power_pos = np.array([target_pos[0], target_pos[2]])
                updates['target_object_position'] = power_pos
                updates['operation_type'] = 'power_control'
                updates['power_state'] = tool_name == 'PowerOn'
                print(f"DEBUG: Updated power control target to {power_pos} for '{target_name}' (state: {tool_name})")
        
        return updates 
