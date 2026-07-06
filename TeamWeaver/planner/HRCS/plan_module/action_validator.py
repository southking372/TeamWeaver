# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import math

class ActionValidator:
    """
    Position verification and intelligent correction module to verify the rationality of actions, especially the spatial position preconditions
    """
    
    def __init__(self, position_threshold: float = 2.0):
        """
        Args:
            position_threshold: Distance threshold, the agent needs to be within this distance to perform the Pick action
        """
        self.position_threshold = position_threshold
        self.validation_history = []

    def reset(self):
        self.validation_history = []
        # print("[DEBUG] ActionValidator reset completed")

    def validate_actions_with_context(
        self,
        high_level_actions: Dict[int, Tuple[str, str, Optional[str]]],
        world_graph: Dict[int, Any],
        execution_manager,
        agents: List[Any]
    ) -> Dict[int, Tuple[str, str, Optional[str]]]:
        """
        Validate and correct actions based on context
        Returns:
            Action dictionary corrected after verification
        """
        validated_actions = {}
        validation_log = []
        
        print(f"[ActionValidator] Validating {len(high_level_actions)} actions with context...")
        
        for agent_id, action_tuple in high_level_actions.items():
            if not action_tuple or len(action_tuple) < 2:
                validated_actions[agent_id] = ("Wait", "", None)
                validation_log.append(f"Agent {agent_id}: Invalid action format → Wait")
                continue
                
            action_name, action_target, action_error = action_tuple
            
            # Get the historical context of the agent
            recent_actions = execution_manager.get_recent_actions(agent_id, lookback_steps=3)
            latest_observation = execution_manager.get_latest_observation(agent_id)
            
            # Perform specific verification
            validated_action = self._validate_single_action(
                agent_id, action_name, action_target, action_error,
                world_graph, recent_actions, latest_observation
            )
            
            validated_actions[agent_id] = validated_action
            
            # Record verification log
            if validated_action != action_tuple:
                validation_log.append(
                    f"Agent {agent_id}: {action_name}[{action_target}] → "
                    f"{validated_action[0]}[{validated_action[1]}] (Correction)"
                )
            else:
                validation_log.append(
                    f"Agent {agent_id}: {action_name}[{action_target}] ✓ (pass)"
                )
        
        # Output verification results
        if validation_log:
            print("[ActionValidator] Validation results:")
            for log_entry in validation_log:
                print(f"  {log_entry}")
        
        return validated_actions

    def _validate_single_action(
        self,
        agent_id: int,
        action_name: str,
        action_target: str,
        action_error: Optional[str],
        world_graph: Dict[int, Any],
        recent_actions: List[Tuple[str, str, Optional[str]]],
        latest_observation: Optional[str]
    ) -> Tuple[str, str, Optional[str]]:
        """
        Verify the action of a single agent
        Returns:
            Verified action tuple
        """
        
        # 1. If the action itself has an error, return Wait directly.
        if action_error:
            return ("Wait", "", f"Original action error: {action_error}")
        
        # 2. PickSpecial validation of actions
        if action_name == "Pick":
            return self._validate_pick_action(
                agent_id, action_target, world_graph, recent_actions, latest_observation
            )
        
        # 3. PlaceVerification of actions
        elif action_name == "Place":
            return self._validate_place_action(
                agent_id, action_target, world_graph, recent_actions, latest_observation
            )
        
        # 4. Basic verification of other actions
        else:
            return self._validate_general_action(
                agent_id, action_name, action_target, recent_actions, latest_observation
            )

    def _validate_pick_action(
        self,
        agent_id: int,
        target_object: str,
        world_graph: Dict[int, Any],
        recent_actions: List[Tuple[str, str, Optional[str]]],
        latest_observation: Optional[str]
    ) -> Tuple[str, str, Optional[str]]:
        """
        Verify the rationality of the Pick action
        
        Key points to check:
        1. Is there a recent Navigate to target object?
        2. Or whether the current position is close enough to the target object
        3. If you are not satisfied with it, we recommend Navigate
        """
        
        # Check if there is a recent Navigate to the target object
        has_recent_navigation = False
        for action_name, action_target, _ in recent_actions:
            if action_name == "Navigate" and action_target == target_object:
                has_recent_navigation = True
                break
        
        if has_recent_navigation:
            # There is recent navigation, and the Pick action is reasonable
            return ("Pick", target_object, None)
        
        # Check if the current position is close enough to the target object
        if self._is_agent_close_to_object(agent_id, target_object, world_graph):
            # The location is close enough and the Pick action is reasonable
            return ("Pick", target_object, None)
        
        # Check if there are any failed observations, if so, recommend Navigate
        if latest_observation and ("not close enough" in latest_observation.lower() or 
                                  "failed to pick" in latest_observation.lower()):
            return ("Navigate", target_object, None)
        
        # Other situations: There is no recent navigation and the location is not close enough. It is recommended to navigate first.
        return ("Navigate", target_object, None)

    def _validate_place_action(
        self,
        agent_id: int,
        target_location: str,
        world_graph: Dict[int, Any],
        recent_actions: List[Tuple[str, str, Optional[str]]],
        latest_observation: Optional[str]
    ) -> Tuple[str, str, Optional[str]]:
        """
        Verify the rationality of the Place action
        Check:
        1. Whether holding object
        2. Is it close to the target position?
        """
        
        # Check for signs of objects being held
        has_recent_pick = any(action_name == "Pick" for action_name, _, _ in recent_actions)
        
        # Check if object is held from observation
        is_holding_object = (latest_observation and 
                           ("held by" in latest_observation.lower() or 
                            "successful execution" in latest_observation.lower()))
        
        if not (has_recent_pick or is_holding_object):
            # There is no object being held and the Place action is unreasonable. It is recommended to wait or explore.
            return ("Wait", "", None)
        
        # Check if it is close to the target location
        if self._is_agent_close_to_location(agent_id, target_location, world_graph):
            return ("Place", target_location, None)
        
        # Need to navigate to the target location first
        return ("Navigate", target_location, None)

    def _validate_general_action(
        self,
        agent_id: int,
        action_name: str,
        action_target: str,
        recent_actions: List[Tuple[str, str, Optional[str]]],
        latest_observation: Optional[str]
    ) -> Tuple[str, str, Optional[str]]:
        """
        Verify the rationality of general actions
        """
        
        # Check if there are repeated actions (possibly stuck in a loop)
        recent_same_actions = [
            (name, target) for name, target, _ in recent_actions 
            if name == action_name and target == action_target
        ]
        
        if len(recent_same_actions) >= 2:
            # If you repeat the same action continuously, there may be a problem. It is recommended to wait.
            return ("Wait", "", f"Avoiding repeated action: {action_name}[{action_target}]")
        
        # In other cases, keep the original action
        return (action_name, action_target, None)

    def _is_agent_close_to_object(
        self, 
        agent_id: int, 
        object_name: str, 
        world_graph: Dict[int, Any]
    ) -> bool:
        """
        Determine whether the agent is close enough to the target object
        
        Args:
            agent_id: agent ID
            object_name: target object name
            world_graph: world map information
            
        Returns:
            is it close enough
        """
        try:
            if agent_id not in world_graph:
                return False
                
            agent_graph = world_graph[agent_id]
            
            # Get agent location
            agent_pos = self._get_agent_position_from_graph(agent_graph, agent_id)
            if agent_pos is None:
                return False
                
            # Get object position
            object_pos = self._get_object_position_from_graph(agent_graph, object_name)
            if object_pos is None:
                return False
                
            # Calculate distance
            distance = self._calculate_distance(agent_pos, object_pos)
            
            return distance <= self.position_threshold
            
        except Exception as e:
            print(f"[ActionValidator] Error checking agent-object distance: {e}")
            return False

    def _is_agent_close_to_location(
        self, 
        agent_id: int, 
        location_name: str, 
        world_graph: Dict[int, Any]
    ) -> bool:
        """
        Determine whether the agent is close enough to the target location
        """
        try:
            if agent_id not in world_graph:
                return False
            agent_graph = world_graph[agent_id]
            
            # agentlocation
            agent_pos = self._get_agent_position_from_graph(agent_graph, agent_id)
            if agent_pos is None:
                return False
                
            # location coordinates
            location_pos = self._get_location_position_from_graph(agent_graph, location_name)
            if location_pos is None:
                return False
                
            distance = self._calculate_distance(agent_pos, location_pos)
            
            return distance <= self.position_threshold
            
        except Exception as e:
            print(f"[ActionValidator] Error checking agent-location distance: {e}")
            return False

    def _get_agent_position_from_graph(self, graph, agent_id: int) -> Optional[List[float]]:
        """Get agent location from world graph"""
        try:
            # Try to get the agent location information from the graph
            # This needs to be implemented based on the actual world graph structure
            if hasattr(graph, 'get_agent_position'):
                return graph.get_agent_position(agent_id)
            
            # Alternative: Find agent from node
            for node in graph.nodes:
                if hasattr(node, 'agent_id') and node.agent_id == agent_id:
                    return [node.position.x, node.position.y, node.position.z]
                    
            return None
        except:
            return None

    def _get_object_position_from_graph(self, graph, object_name: str) -> Optional[List[float]]:
        """Get object position from world graph"""
        try:
            # Find the object node with the specified name
            for node in graph.nodes:
                if hasattr(node, 'name') and node.name == object_name:
                    return [node.position.x, node.position.y, node.position.z]
                    
            return None
        except:
            return None

    def _get_location_position_from_graph(self, graph, location_name: str) -> Optional[List[float]]:
        """Get location coordinates from world graph"""
        try:
            for node in graph.nodes:
                if hasattr(node, 'name') and node.name == location_name:
                    return [node.position.x, node.position.y, node.position.z]
                    
            return None
        except:
            return None

    def _calculate_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """Calculate the Euclidean distance between two points"""
        try:
            # Calculate 2D distance using coordinates (ignoring height)
            dx = pos1[0] - pos2[0]
            dz = pos1[2] - pos2[2] if len(pos1) > 2 and len(pos2) > 2 else 0
            
            return math.sqrt(dx * dx + dz * dz)
        except:
            return float('inf') 