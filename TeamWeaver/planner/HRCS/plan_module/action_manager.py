from typing import Dict, List, Any, Optional, Tuple
from .action_validator import ActionValidator

class ActionManager:
    def __init__(self, actions_parser):
        self.actions_parser = actions_parser
        self.params = {}
        self.last_parsed_actions = {}
        self.action_history = []
        
        # 集成位置验证器
        self.action_validator = ActionValidator(position_threshold=2.0)

    def reset(self):
        """
        Reset the ActionManager state.
        Clear action history and cached parsed actions.
        """
        self.last_parsed_actions = {}
        self.action_history = []
        self.action_validator.reset()
        print("[DEBUG] ActionManager reset completed")

    def parse_and_validate_actions(
        self, 
        llm_response: str, 
        agents, 
        world_graph: Dict[int, Any],
        execution_manager
    ) -> Dict[int, Tuple[str, str, Optional[str]]]:
        """
        解析并验证LLM响应中的高级动作
        Returns:
            验证后的高级动作字典
        """
        
        # Step 1: 解析LLM响应
        print(f"[ActionManager] Step 1: Parsing LLM response...")
        parsed_actions = self.parse_high_level_actions(llm_response, agents)
        
        # Step 2: 位置验证和智能修正
        print(f"[ActionManager] Step 2: Validating with context...")
        validated_actions = self.action_validator.validate_actions_with_context(
            parsed_actions, world_graph, execution_manager, agents
        )
        
        # Step 3: 记录到历史
        self.action_history.append({
            'original': parsed_actions,
            'validated': validated_actions,
            'timestamp': self._get_timestamp()
        })
        
        self.last_parsed_actions = validated_actions
        
        print(f"[ActionManager] Action processing complete: "
              f"{len(parsed_actions)} parsed → {len(validated_actions)} validated")
        
        return validated_actions

    def parse_high_level_actions(self, llm_response: str, agents) -> Dict[int, Tuple[str, str, Optional[str]]]:
        """
        解析LLM响应中的高级动作

        :param llm_response: LLM生成的响应
        :return: 解析出的高级动作字典
        """
        try:
            return self.actions_parser(agents, llm_response, self.params)
        except Exception as e:
            print(f"[ERROR] 高级动作解析失败: {e}")
            return {
                agent.uid: ("Explore", "environment", None) 
                for agent in agents
            }

    def adjust_actions_with_phase_awareness(
        self,
        high_level_actions: Dict[int, Tuple[str, str, Optional[str]]],
        agent_task_assignments: Dict[int, List[Dict[str, Any]]],
        current_phase: Dict[str, Any],
        agents
    ) -> Dict[int, Tuple[str, str, Optional[str]]]:
        """
        **重构为目标导向验证**：验证LLM动作是否有助于实现阶段目标，
        而不是强制替换为子任务类型。保持LLM的智能决策能力。
        """
        adjusted_actions = {}
        
        print(f"[DEBUG] Validating actions for phase {current_phase['phase_id']} objectives:")
        
        for agent_id, action_tuple in high_level_actions.items():
            assigned_tasks = agent_task_assignments.get(agent_id, [])
            
            if not assigned_tasks:
                adjusted_actions[agent_id] = action_tuple if action_tuple else ("Wait", "", None)
                print(f"    Agent {agent_id}: No assigned objectives, keeping LLM decision: {adjusted_actions[agent_id][0]}")
                continue

            # 获取主要目标
            primary_objective = assigned_tasks[0]
            objective_type = primary_objective.get('task_type', 'Unknown')
            objective_target = primary_objective.get('target', 'Unknown')
            
            if not action_tuple or not action_tuple[0]:
                # LLM没有生成有效动作，提供目标导向的建议
                suggested_action = self._suggest_action_for_objective(primary_objective, action_tuple)
                adjusted_actions[agent_id] = suggested_action
                print(f"    Agent {agent_id}: LLM provided no action for objective '{objective_type}→{objective_target}', suggesting: {suggested_action[0]}")
            else:
                # LLM生成了动作，验证是否有助于目标实现
                llm_action, llm_target, llm_error = action_tuple
                
                if self._action_advances_objective(llm_action, llm_target, primary_objective):
                    # LLM的动作有助于目标实现，保持不变
                    adjusted_actions[agent_id] = action_tuple
                    print(f"    Agent {agent_id}: LLM action '{llm_action}[{llm_target}]' advances objective '{objective_type}→{objective_target}' ✓")
                else:
                    # LLM的动作可能偏离目标，提供目标导向的建议但保持一定灵活性
                    if self._is_exploration_reasonable(llm_action, primary_objective):
                        # 如果是合理的探索行为，允许执行
                        adjusted_actions[agent_id] = action_tuple
                        print(f"    Agent {agent_id}: LLM action '{llm_action}[{llm_target}]' is reasonable exploration for objective '{objective_type}→{objective_target}' ✓")
                    else:
                        # 提供更好的建议
                        suggested_action = self._suggest_action_for_objective(primary_objective, action_tuple)
                        adjusted_actions[agent_id] = suggested_action
                        print(f"    Agent {agent_id}: LLM action '{llm_action}[{llm_target}]' may not advance objective '{objective_type}→{objective_target}', suggesting: {suggested_action[0]}")

        # 确保所有智能体都有动作
        for agent_id in range(len(agents)):
            if agent_id not in adjusted_actions:
                adjusted_actions[agent_id] = ("Wait", "", None)
                print(f"    Agent {agent_id}: No action assigned, defaulting to Wait.")

        return adjusted_actions

    def get_action_validation_summary(self) -> Dict[str, Any]:
        """
        获取action验证的统计摘要
        Returns:
            包含验证统计信息的字典
        """
        if not self.action_history:
            return {"total_actions": 0, "corrections_made": 0, "correction_rate": 0.0}
        total_actions = 0
        corrections_made = 0
        
        for history_entry in self.action_history:
            original = history_entry['original']
            validated = history_entry['validated']
            for agent_id in original:
                total_actions += 1
                if original.get(agent_id) != validated.get(agent_id):
                    corrections_made += 1
        
        correction_rate = corrections_made / total_actions if total_actions > 0 else 0.0
        
        return {
            "total_actions": total_actions,
            "corrections_made": corrections_made,
            "correction_rate": correction_rate,
            "recent_history_length": len(self.action_history)
        }

    def _get_timestamp(self) -> float:
        import time
        return time.time()

    def _action_advances_objective(self, action_name: str, action_target: str, objective: Dict[str, Any]) -> bool:
        """
        判断给定的动作是否有助于实现目标
        """
        objective_type = objective.get('task_type', '')
        objective_target = objective.get('target', '')
        
        # 直接匹配：动作类型与目标类型相同，且目标对象相同
        if action_name == objective_type and action_target == objective_target:
            return True
        
        # 逻辑链匹配：判断动作是否是实现目标的必要步骤
        if objective_type == 'Pick':
            # Pick目标：Navigate到目标物体是有帮助的
            if action_name == 'Navigate' and action_target == objective_target:
                return True
            # Explore找到目标物体也是有帮助的
            if action_name == 'Explore':
                return True  # 探索通常是有帮助的
                
        elif objective_type == 'Place':
            # Place目标：Navigate到放置位置是有帮助的
            if action_name == 'Navigate' and action_target == objective_target:
                return True
            # Pick已经持有的物体到目标位置也是有帮助的
            if action_name == 'Pick':
                return True
                
        elif objective_type == 'Navigate':
            # Navigate目标：直接Navigate到目标是有帮助的
            if action_name == 'Navigate' and action_target == objective_target:
                return True
            # Explore相关区域也是有帮助的
            if action_name == 'Explore':
                return True
                
        elif objective_type == 'Explore':
            # Explore目标：直接Explore目标区域
            if action_name == 'Explore' and action_target == objective_target:
                return True
            # Navigate到目标区域也有帮助
            if action_name == 'Navigate':
                return True
        
        return False

    def _is_exploration_reasonable(self, action_name: str, objective: Dict[str, Any]) -> bool:
        """
        判断探索行为是否在目标实现的合理范围内
        """
        # Explore动作通常是合理的，特别是当目标需要寻找对象时
        if action_name == 'Explore':
            return True
        
        # Wait动作在某些情况下也是合理的（等待其他智能体完成前置条件）
        if action_name == 'Wait':
            return True
            
        return False

    def _suggest_action_for_objective(self, objective: Dict[str, Any], original_action: Tuple[str, str, Optional[str]]) -> Tuple[str, str, Optional[str]]:
        """
        基于目标建议合适的动作，但尽量保持LLM的智能判断
        """
        objective_type = objective.get('task_type', '')
        objective_target = objective.get('target', '')
        
        # 如果有原始动作且是Wait，可能表示智能体在等待，保持这个决策
        if original_action and original_action[0] == 'Wait':
            return original_action
        
        # 根据目标类型提供建议
        if objective_type == 'Pick':
            # Pick目标：建议先探索找到对象（除非已知对象位置）
            return ('Explore', 'environment', None)  # 让LLM通过探索找到对象
            
        elif objective_type == 'Place':
            # Place目标：建议导航到放置位置
            return ('Navigate', objective_target, None)
            
        elif objective_type == 'Navigate':
            # Navigate目标：直接导航
            return ('Navigate', objective_target, None)
            
        elif objective_type == 'Explore':
            # Explore目标：直接探索
            return ('Explore', objective_target, None)
            
        else:
            # 其他目标：转换为相应动作
            return (objective_type, objective_target, None)

    def generate_action_from_subtask(self, subtask: Dict[str, Any]) -> Optional[Tuple[str, str, Optional[str]]]:
        """从子任务生成动作"""
        task_type = subtask.get('task_type', '')
        target = subtask.get('target', '')
        
        if not task_type or not target:
            return None
            
        return (task_type, target, None)

    def action_matches_task_type(self, action_name: str, task_type: str) -> bool:
        return action_name == task_type or (
            action_name in ['Pick', 'Place'] and task_type == 'Rearrange'
        ) 