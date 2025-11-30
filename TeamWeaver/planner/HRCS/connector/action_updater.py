from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from habitat_llm.planner.HRCS.connector.planner_utils import find_target_position

class ActionUpdater:
    """
    处理高级动作并生成场景参数更新。
    此类将动作处理逻辑与PerceptionConnector解耦。
    """

    def process_and_get_updates(
        self,
        high_level_actions: Dict[int, Tuple[str, str, Optional[str]]], 
        world_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理所有动作并返回一个包含所有参数更新的字典。
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
        处理Motor Skills相关的动作更新
        基于Agent工具配置: Navigate, Pick, Place, Rearrange, Explore, Wait, Open, Close
        """
        updates = {}
        
        for agent_id, action_tuple in high_level_actions.items():
            if not action_tuple or action_tuple[2] is None:  # 跳过无效动作
                continue
                
            tool_name, args_str, target_name = action_tuple
            
            # Navigation相关 (精确匹配 Navigate)
            if tool_name == 'Navigate':
                nav_updates = self._update_navigation_params(target_name, world_state)
                updates.update(nav_updates)
                print(f"DEBUG: Agent {agent_id} using Navigate to {target_name}")
            
            # Manipulation相关 (精确匹配 Pick, Place, Rearrange)
            elif tool_name in ['Pick', 'Place', 'Rearrange']:
                manip_updates = self._update_manipulation_params(tool_name, target_name, world_state)
                updates.update(manip_updates)
                print(f"DEBUG: Agent {agent_id} using {tool_name} on {target_name}")
            
            # Exploration相关 (精确匹配 Explore)
            elif tool_name == 'Explore':
                explore_updates = self._update_exploration_params(target_name, world_state)
                updates.update(explore_updates)
                print(f"DEBUG: Agent {agent_id} using Explore in {target_name}")
            
            # 铰接控制 (精确匹配 Open, Close)
            elif tool_name in ['Open', 'Close']:
                # Open/Close 主要影响环境状态，通常不需要更新MIQP参数
                print(f"DEBUG: Agent {agent_id} using {tool_name} on {target_name}")
                
            # Wait动作 (精确匹配 Wait)
            elif tool_name == 'Wait':
                print(f"DEBUG: Agent {agent_id} waiting - no parameter updates needed")
            
            # 未识别的动作
            else:
                print(f"WARNING: Unrecognized motor skill action: {tool_name} for Agent {agent_id}")
        
        return updates

    def _update_navigation_params(self, target_name: str, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """更新导航相关参数"""
        updates = {}
        
        # 查找目标位置
        target_pos = find_target_position(target_name, world_state)
        if target_pos:
            # MIQP使用XZ平面坐标
            nav_goal = np.array([target_pos[0], target_pos[2]])
            updates['p_goal'] = nav_goal
            updates['theta_goal'] = 0.0  # 默认朝向
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
        """更新操作相关参数，基于精确的工具名称匹配"""
        updates = {}
        
        # Pick动作 (精确匹配)
        if tool_name == 'Pick':
            target_pos = find_target_position(target_name, world_state)
            if target_pos:
                obj_pos = np.array([target_pos[0], target_pos[2]])
                updates['target_object_position'] = obj_pos
                print(f"DEBUG: Updated pick target to {obj_pos} for object '{target_name}'")
        
        # Place动作 (精确匹配)
        elif tool_name == 'Place':
            target_pos = find_target_position(target_name, world_state)
            if target_pos:
                place_pos = np.array([target_pos[0], target_pos[2]])
                updates['target_receptacle_position'] = place_pos
                print(f"DEBUG: Updated place target to {place_pos} for receptacle '{target_name}'")
        
        # Rearrange动作 (精确匹配)
        elif tool_name == 'Rearrange':
            # Rearrange[object, spatial_relation, furniture, spatial_constraint, reference_object]
            # 主要关注目标家具位置
            target_pos = find_target_position(target_name, world_state)
            if target_pos:
                rearrange_pos = np.array([target_pos[0], target_pos[2]])
                updates['target_receptacle_position'] = rearrange_pos
                print(f"DEBUG: Updated rearrange target to {rearrange_pos} for '{target_name}'")
        
        return updates

    def _update_exploration_params(self, target_name: str, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """更新探索相关参数"""
        updates = {}
        
        # 基于目标生成探索点
        exploration_targets = []
        
        if target_name and target_name != 'environment':
            # 查找与目标相关的家具/区域
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
        """处理状态操作工具相关的动作"""
        updates = {}
        
        for agent_id, action_tuple in high_level_actions.items():
            if not action_tuple or action_tuple[2] is None:
                continue
                
            tool_name, args_str, target_name = action_tuple
            
            # 状态操作工具 (Agent 1 专有能力)
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
        """更新状态操作相关参数"""
        updates = {}
        
        # 检查Agent是否有该工具的权限 
        if agent_id == 0 and tool_name in ['Clean', 'Fill', 'Pour', 'PowerOn', 'PowerOff']:
            print(f"WARNING: Agent {agent_id} attempting to use {tool_name} but lacks this capability")
            return updates
        
        # 获取目标对象位置
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