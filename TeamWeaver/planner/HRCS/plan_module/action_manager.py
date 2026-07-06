# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

from typing import Dict, List, Any, Optional, Tuple
from .action_validator import ActionValidator

class ActionManager:
    def __init__(self, actions_parser):
        self.actions_parser = actions_parser
        self.params = {}
        self.last_parsed_actions = {}
        self.action_history = []
        
        #Integrated location validator
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
parse and verifyLLMAdvanced actions in response
        Returns:
Verified advanced action dictionary
        """
        
        # Step 1:parseLLMresponse
        print(f"[ActionManager] Step 1: Parsing LLM response...")
        parsed_actions = self.parse_high_level_actions(llm_response, agents)
        
        # Step 2:Location verification and smart corrections
        print(f"[ActionManager] Step 2: Validating with context...")
        validated_actions = self.action_validator.validate_actions_with_context(
            parsed_actions, world_graph, execution_manager, agents
        )
        
        # Step 3:recorded in history
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
        Parse high-level actions from LLM response

        :param llm_response: LLM-generated response
        :return: parsed high-level action dict
        """
        try:
            return self.actions_parser(agents, llm_response, self.params)
        except Exception as e:
            print(f"[ERROR] High-level action parsing failed: {e}")
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
        **Refactoring into goal-directed verification**:verifyLLMWhether the action helps achieve the stage goal,
Instead of forcing replacement with subtask type. KeepLLMintelligent decision-making capabilities.
        """
        adjusted_actions = {}
        
        print(f"[DEBUG] Validating actions for phase {current_phase['phase_id']} objectives:")
        
        for agent_id, action_tuple in high_level_actions.items():
            assigned_tasks = agent_task_assignments.get(agent_id, [])
            
            if not assigned_tasks:
                adjusted_actions[agent_id] = action_tuple if action_tuple else ("Wait", "", None)
                print(f"    Agent {agent_id}: No assigned objectives, keeping LLM decision: {adjusted_actions[agent_id][0]}")
                continue

            #Get main target
            primary_objective = assigned_tasks[0]
            objective_type = primary_objective.get('task_type', 'Unknown')
            objective_target = primary_objective.get('target', 'Unknown')
            
            if not action_tuple or not action_tuple[0]:
                # LLMNo effective actions are generated, providing goal-oriented suggestions
                suggested_action = self._suggest_action_for_objective(primary_objective, action_tuple)
                adjusted_actions[agent_id] = suggested_action
                print(f"    Agent {agent_id}: LLM provided no action for objective '{objective_type}→{objective_target}', suggesting: {suggested_action[0]}")
            else:
                # LLMGenerate actions to verify whether they help achieve the goal
                llm_action, llm_target, llm_error = action_tuple
                
                if self._action_advances_objective(llm_action, llm_target, primary_objective):
                    # LLMactions that contribute to goal achievement and remain unchanged
                    adjusted_actions[agent_id] = action_tuple
                    print(f"    Agent {agent_id}: LLM action '{llm_action}[{llm_target}]' advances objective '{objective_type}→{objective_target}' ✓")
                else:
                    # LLMThe actions may deviate from the target, providing goal-oriented suggestions but maintaining a certain degree of flexibility
                    if self._is_exploration_reasonable(llm_action, primary_objective):
                        #If it is reasonable exploration behavior, execution is allowed
                        adjusted_actions[agent_id] = action_tuple
                        print(f"    Agent {agent_id}: LLM action '{llm_action}[{llm_target}]' is reasonable exploration for objective '{objective_type}→{objective_target}' ✓")
                    else:
                        #Provide better advice
                        suggested_action = self._suggest_action_for_objective(primary_objective, action_tuple)
                        adjusted_actions[agent_id] = suggested_action
                        print(f"    Agent {agent_id}: LLM action '{llm_action}[{llm_target}]' may not advance objective '{objective_type}→{objective_target}', suggesting: {suggested_action[0]}")

        #Make sure all agents have actions
        for agent_id in range(len(agents)):
            if agent_id not in adjusted_actions:
                adjusted_actions[agent_id] = ("Wait", "", None)
                print(f"    Agent {agent_id}: No action assigned, defaulting to Wait.")

        return adjusted_actions

    def get_action_validation_summary(self) -> Dict[str, Any]:
        """
getactionValidated statistical summary
        Returns:
A dictionary containing validation statistics
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
Determine whether a given action will help achieve a goal
        """
        objective_type = objective.get('task_type', '')
        objective_target = objective.get('target', '')
        
        #Direct match: The action type is the same as the target type, and the target object is the same
        if action_name == objective_type and action_target == objective_target:
            return True
        
        #Logical chain matching: Determine whether the action is a necessary step to achieve the goal
        if objective_type == 'Pick':
            # PickTarget:NavigateIt is helpful to reach the target object
            if action_name == 'Navigate' and action_target == objective_target:
                return True
            # ExploreIt is also helpful to find the target object
            if action_name == 'Explore':
                return True  #Exploration is often helpful
                
        elif objective_type == 'Place':
            # PlaceTarget:Navigateto placement location is helpful
            if action_name == 'Navigate' and action_target == objective_target:
                return True
            # PickAlready holding the object to the target location is also helpful
            if action_name == 'Pick':
                return True
                
        elif objective_type == 'Navigate':
            # NavigateGoal: DirectNavigateIt is helpful to reach the goal
            if action_name == 'Navigate' and action_target == objective_target:
                return True
            # ExploreRelated areas are also helpful
            if action_name == 'Explore':
                return True
                
        elif objective_type == 'Explore':
            # ExploreGoal: DirectExploretarget area
            if action_name == 'Explore' and action_target == objective_target:
                return True
            # NavigateIt also helps to get to the target area
            if action_name == 'Navigate':
                return True
        
        return False

    def _is_exploration_reasonable(self, action_name: str, objective: Dict[str, Any]) -> bool:
        """
Determine whether exploration behavior is within a reasonable range for goal achievement
        """
        # ExploreActions are often justified, especially when the goal requires finding an object
        if action_name == 'Explore':
            return True
        
        # WaitActions are also justified in certain situations (waiting for other agents to complete preconditions)
        if action_name == 'Wait':
            return True
            
        return False

    def _suggest_action_for_objective(self, objective: Dict[str, Any], original_action: Tuple[str, str, Optional[str]]) -> Tuple[str, str, Optional[str]]:
        """
Suggest appropriate actions based on goals, but try to keepLLMintelligent judgment
        """
        objective_type = objective.get('task_type', '')
        objective_target = objective.get('target', '')
        
        #If there is a primitive action and isWait, may indicate that the agent is waiting to keep this decision
        if original_action and original_action[0] == 'Wait':
            return original_action
        
        #Provide recommendations based on target type
        if objective_type == 'Pick':
            # PickGoal: It is recommended to explore and find the object first (unless the object location is known)
            return ('Explore', 'environment', None)  #letLLMFind objects through exploration
            
        elif objective_type == 'Place':
            # PlaceTarget: Suggested navigation to drop location
            return ('Navigate', objective_target, None)
            
        elif objective_type == 'Navigate':
            # NavigateGoal: direct navigation
            return ('Navigate', objective_target, None)
            
        elif objective_type == 'Explore':
            # ExploreGoal: Direct exploration
            return ('Explore', objective_target, None)
            
        else:
            #Other goals: Convert to corresponding actions
            return (objective_type, objective_target, None)

    def generate_action_from_subtask(self, subtask: Dict[str, Any]) -> Optional[Tuple[str, str, Optional[str]]]:
        """Generate actions from subtasks"""
        task_type = subtask.get('task_type', '')
        target = subtask.get('target', '')
        
        if not task_type or not target:
            return None
            
        return (task_type, target, None)

    def action_matches_task_type(self, action_name: str, task_type: str) -> bool:
        return action_name == task_type or (
            action_name in ['Pick', 'Place'] and task_type == 'Rearrange'
        ) 
