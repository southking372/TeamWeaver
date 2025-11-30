from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import math

class ActionValidator:
    """
    位置验证和智能修正模块，验证action的合理性，特别是空间位置前置条件
    """
    
    def __init__(self, position_threshold: float = 2.0):
        """
        Args:
            position_threshold: 距离阈值，agent需要在此距离内才能执行Pick动作
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
        基于上下文验证和修正actions
        Returns:
            验证后修正的动作字典
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
            
            # 获取agent的历史上下文
            recent_actions = execution_manager.get_recent_actions(agent_id, lookback_steps=3)
            latest_observation = execution_manager.get_latest_observation(agent_id)
            
            # 执行具体验证
            validated_action = self._validate_single_action(
                agent_id, action_name, action_target, action_error,
                world_graph, recent_actions, latest_observation
            )
            
            validated_actions[agent_id] = validated_action
            
            # 记录验证日志
            if validated_action != action_tuple:
                validation_log.append(
                    f"Agent {agent_id}: {action_name}[{action_target}] → "
                    f"{validated_action[0]}[{validated_action[1]}] (修正)"
                )
            else:
                validation_log.append(
                    f"Agent {agent_id}: {action_name}[{action_target}] ✓ (通过)"
                )
        
        # 输出验证结果
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
        验证单个agent的action
        Returns:
            验证后的action tuple
        """
        
        # 1. 如果action本身就有错误，直接返回Wait
        if action_error:
            return ("Wait", "", f"Original action error: {action_error}")
        
        # 2. Pick动作的特殊验证
        if action_name == "Pick":
            return self._validate_pick_action(
                agent_id, action_target, world_graph, recent_actions, latest_observation
            )
        
        # 3. Place动作的验证
        elif action_name == "Place":
            return self._validate_place_action(
                agent_id, action_target, world_graph, recent_actions, latest_observation
            )
        
        # 4. 其他动作的基础验证
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
        验证Pick动作的合理性
        
        重点检查：
        1. 是否有recent Navigate to target object
        2. 或者当前位置是否足够接近目标物体
        3. 如果都不满足，建议Navigate
        """
        
        # 检查最近是否有Navigate到目标物体
        has_recent_navigation = False
        for action_name, action_target, _ in recent_actions:
            if action_name == "Navigate" and action_target == target_object:
                has_recent_navigation = True
                break
        
        if has_recent_navigation:
            # 有recent navigation，Pick动作合理
            return ("Pick", target_object, None)
        
        # 检查当前位置是否足够接近目标物体
        if self._is_agent_close_to_object(agent_id, target_object, world_graph):
            # 位置足够接近，Pick动作合理
            return ("Pick", target_object, None)
        
        # 检查是否有失败观察，如果有，建议Navigate
        if latest_observation and ("not close enough" in latest_observation.lower() or 
                                  "failed to pick" in latest_observation.lower()):
            return ("Navigate", target_object, None)
        
        # 其他情况：没有recent navigation且位置不够接近，建议先Navigate
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
        验证Place动作的合理性
        检查：
        1. 是否holding object
        2. 是否接近目标位置
        """
        
        # 检查是否有持有物体的迹象
        has_recent_pick = any(action_name == "Pick" for action_name, _, _ in recent_actions)
        
        # 从观察中检查是否持有物体
        is_holding_object = (latest_observation and 
                           ("held by" in latest_observation.lower() or 
                            "successful execution" in latest_observation.lower()))
        
        if not (has_recent_pick or is_holding_object):
            # 没有持有物体，Place动作不合理，建议等待或探索
            return ("Wait", "", None)
        
        # 检查是否接近目标位置
        if self._is_agent_close_to_location(agent_id, target_location, world_graph):
            return ("Place", target_location, None)
        
        # 需要先导航到目标位置
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
        验证一般动作的合理性
        """
        
        # 检查是否有重复的动作（可能陷入循环）
        recent_same_actions = [
            (name, target) for name, target, _ in recent_actions 
            if name == action_name and target == action_target
        ]
        
        if len(recent_same_actions) >= 2:
            # 连续重复相同动作，可能有问题，建议等待
            return ("Wait", "", f"Avoiding repeated action: {action_name}[{action_target}]")
        
        # 其他情况保持原动作
        return (action_name, action_target, None)

    def _is_agent_close_to_object(
        self, 
        agent_id: int, 
        object_name: str, 
        world_graph: Dict[int, Any]
    ) -> bool:
        """
        判断agent是否足够接近目标物体
        
        Args:
            agent_id: agent ID
            object_name: 目标物体名称
            world_graph: 世界图信息
            
        Returns:
            是否足够接近
        """
        try:
            if agent_id not in world_graph:
                return False
                
            agent_graph = world_graph[agent_id]
            
            # 获取agent位置
            agent_pos = self._get_agent_position_from_graph(agent_graph, agent_id)
            if agent_pos is None:
                return False
                
            # 获取物体位置
            object_pos = self._get_object_position_from_graph(agent_graph, object_name)
            if object_pos is None:
                return False
                
            # 计算距离
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
        判断agent是否足够接近目标位置
        """
        try:
            if agent_id not in world_graph:
                return False
            agent_graph = world_graph[agent_id]
            
            # agent位置
            agent_pos = self._get_agent_position_from_graph(agent_graph, agent_id)
            if agent_pos is None:
                return False
                
            # 位置坐标
            location_pos = self._get_location_position_from_graph(agent_graph, location_name)
            if location_pos is None:
                return False
                
            distance = self._calculate_distance(agent_pos, location_pos)
            
            return distance <= self.position_threshold
            
        except Exception as e:
            print(f"[ActionValidator] Error checking agent-location distance: {e}")
            return False

    def _get_agent_position_from_graph(self, graph, agent_id: int) -> Optional[List[float]]:
        """从world graph获取agent位置"""
        try:
            # 尝试从graph中获取agent位置信息
            # 这里需要根据实际的world graph结构来实现
            if hasattr(graph, 'get_agent_position'):
                return graph.get_agent_position(agent_id)
            
            # 替代方案：从节点中查找agent
            for node in graph.nodes:
                if hasattr(node, 'agent_id') and node.agent_id == agent_id:
                    return [node.position.x, node.position.y, node.position.z]
                    
            return None
        except:
            return None

    def _get_object_position_from_graph(self, graph, object_name: str) -> Optional[List[float]]:
        """从world graph获取物体位置"""
        try:
            # 查找指定名称的物体节点
            for node in graph.nodes:
                if hasattr(node, 'name') and node.name == object_name:
                    return [node.position.x, node.position.y, node.position.z]
                    
            return None
        except:
            return None

    def _get_location_position_from_graph(self, graph, location_name: str) -> Optional[List[float]]:
        """从world graph获取位置坐标"""
        try:
            for node in graph.nodes:
                if hasattr(node, 'name') and node.name == location_name:
                    return [node.position.x, node.position.y, node.position.z]
                    
            return None
        except:
            return None

    def _calculate_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """计算两点间的欧几里得距离"""
        try:
            # 使用坐标计算2D距离（忽略高度）
            dx = pos1[0] - pos2[0]
            dz = pos1[2] - pos2[2] if len(pos1) > 2 and len(pos2) > 2 else 0
            
            return math.sqrt(dx * dx + dz * dz)
        except:
            return float('inf') 