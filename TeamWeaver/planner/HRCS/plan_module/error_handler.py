from typing import Dict, List, Any, Optional, Tuple
import time

class ErrorHandler:
    """
    Handles error detection and recovery during the planning process.
    """
    def __init__(self):
        self.error_history = []
        self.recovery_attempts = {}
        self.persistent_failures = set()

    def reset(self):
        """
        Reset the ErrorHandler state.
        Clear error history and recovery tracking.
        """
        self.error_history = []
        self.recovery_attempts = {}
        self.persistent_failures = set()
        # print("[DEBUG] ErrorHandler reset completed")

    def recover_and_log_assignments(
        self,
        agent_task_assignments: Dict[int, List[Dict[str, Any]]],
        last_high_level_actions: Dict[int, Tuple[str, str, str]],
        latest_agent_response: Dict[int, str]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Applies intelligent error recovery and logs the outcome.
        This method checks for action failures and may add recovery tasks.
        It prints the changes if any recovery actions were taken.
        """
        try:
            recovered_assignments = self.apply_intelligent_error_recovery(
                agent_task_assignments,
                last_high_level_actions,
                latest_agent_response
            )

            # Log the results if changes were made
            if recovered_assignments != agent_task_assignments:
                print(f"[DEBUG] **RECOVERY** Updated task assignments after error recovery:")
                for agent_id, tasks in recovered_assignments.items():
                    if tasks:
                        task_summaries = []
                        for t in tasks:
                            recovery_marker = " [RECOVERY]" if t.get('is_recovery', False) else ""
                            task_summaries.append(f"{t['task_type']}→{t['target']}{recovery_marker}")
                        print(f"  Agent {agent_id}: {task_summaries}")
                    else:
                        print(f"  Agent {agent_id}: []")
            else:
                print(f"[DEBUG] **RECOVERY** No error recovery needed")
            
            return recovered_assignments

        except Exception as e:
            print(f"[ERROR] Error recovery failed: {e}")
            return agent_task_assignments

    def apply_intelligent_error_recovery(
        self,
        agent_task_assignments: Dict[int, List[Dict[str, Any]]],
        last_high_level_actions: Dict[int, Tuple[str, str, str]],
        latest_agent_response: Dict[int, str]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Apply intelligent error recovery to handle failed operations."""
        try:
            if not latest_agent_response or not last_high_level_actions:
                return agent_task_assignments

            updated_assignments = agent_task_assignments.copy()

            for agent_id, response in latest_agent_response.items():
                if not response:
                    continue

                current_action = last_high_level_actions.get(agent_id)
                if not current_action:
                    continue

                recovery_action = self.analyze_failure_and_suggest_recovery(
                    agent_id, response, current_action
                )

                if recovery_action:
                    action_name, args, target = recovery_action

                    recovery_task = {
                        'task_id': f'recovery_{agent_id}_{action_name}',
                        'task_type': action_name,
                        'target': target if target else args,
                        'description': f'Recovery action for Agent {agent_id}',
                        'priority': 5,
                        'estimated_duration': 10.0,
                        'preferred_agent': agent_id,
                        'prerequisites': [],
                        'can_parallel': False,
                        'phase_group': 'recovery',
                        'is_recovery': True,
                        'original_task': current_action[0],
                        'recovery_reason': response[:100]
                    }

                    if agent_id not in updated_assignments:
                        updated_assignments[agent_id] = []

                    updated_assignments[agent_id].insert(0, recovery_task)

            return updated_assignments

        except Exception:
            return agent_task_assignments

    def analyze_failure_and_suggest_recovery(
        self,
        agent_id: int,
        status: str,
        current_action: Tuple[str, str, Optional[str]]
    ) -> Optional[Tuple[str, str, Optional[str]]]:
        """Analyze agent failure status and suggest recovery actions."""
        try:
            if not status or not current_action:
                return None

            action_name, args, target = current_action
            status_lower = status.lower()

            if action_name == "Pick" and "not close enough" in status_lower:
                return ("Navigate", target, target)
            elif action_name == "Pick" and ("object not found" in status_lower or "not present in the graph" in status_lower):
                return ("Explore", "environment", "environment")
            elif action_name == "Place" and "not close enough" in status_lower:
                return ("Navigate", target, target)
            elif "navigation failed" in status_lower or "path not found" in status_lower:
                return ("Explore", "environment", "environment")
            elif "collision" in status_lower or "blocked" in status_lower:
                return ("Wait", "", "")

            return None

        except Exception:
            return None 