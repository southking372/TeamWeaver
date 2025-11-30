from typing import Dict, Any

class FeedbackManager:
    def __init__(self):
        self.feedback_history = []
        self.phase_completion_tracking = {}
        self.agent_performance_metrics = {}

    def reset(self):
        """
        Reset the FeedbackManager state.
        Clear feedback history and performance tracking.
        """
        self.feedback_history = []
        self.phase_completion_tracking = {}
        self.agent_performance_metrics = {}
        print("[DEBUG] FeedbackManager reset completed")

    def get_agent_completion_statuses(self, agents: list, latest_agent_response: Dict[int, str]) -> Dict[int, str]:
        """Get agent completion statuses, prioritizing latest responses."""
        agent_statuses = {}
        
        try:
            if latest_agent_response:
                for agent in agents:
                    response = latest_agent_response.get(agent.uid, "")
                    if response and response.strip():
                        agent_statuses[agent.uid] = response.strip()
                        continue
                    
                    state_desc = agent.get_last_state_description()
                    agent_statuses[agent.uid] = state_desc if state_desc else "No status available"
            else:
                for agent in agents:
                    state_desc = agent.get_last_state_description()
                    agent_statuses[agent.uid] = state_desc if state_desc else "No status available"
            
            if not agent_statuses and agents:
                agent_statuses = {agent.uid: "Status unknown" for agent in agents}
            
            return agent_statuses
            
        except Exception:
            return {agent.uid: "Error extracting status" for agent in agents}

    def check_and_advance_phase(self, perception_connector, agents: list, latest_agent_response: Dict[int, str], current_phase: Dict[str, Any]) -> bool:
        """
        Checks if the current phase is complete and advances to the next one if possible.
        Returns whether a phase transition is pending.
        """
        phase_transition_pending = False
        print(f"[Step 15/15] Checking phase completion...")
        try:
            # 智能获取agent状态 - 优先使用最新的responses
            agent_statuses = self.get_agent_completion_statuses(agents, latest_agent_response)
            
            print(f"[DEBUG] Phase completion check for Phase {current_phase['phase_id']}:")
            print(f"  Agent completion statuses: {agent_statuses}")
            
            # Heuristic check: if all agents are waiting/idle, the phase is likely complete.
            all_agents_idle = False
            if agent_statuses:
                idle_keywords = ["wait", "done", "success", "complete", "no status available", "status unknown"]
                num_idle_agents = sum(1 for status in agent_statuses.values() if any(keyword in status.lower() for keyword in idle_keywords))
                if num_idle_agents == len(agents):
                    all_agents_idle = True
                    print("[DEBUG] Heuristic check: All agents appear to be idle.")

            # 检查当前阶段是否完成
            is_complete = perception_connector.is_current_phase_complete(agent_statuses)

            if not is_complete and all_agents_idle and current_phase.get('tasks'):
                print("[INFO] Heuristic override: Forcing phase completion because all agents are idle.")
                is_complete = True

            if is_complete:
                print(f"[SUCCESS] Phase {current_phase['phase_id']} completed!")
                
                # 尝试推进到下一阶段
                if perception_connector.advance_to_next_phase():
                    next_phase = perception_connector.get_current_phase_tasks()
                    if next_phase:
                        print(f"[INFO] Advanced to phase {next_phase['phase_id']}:")
                        for task in next_phase['tasks']:
                            print(f"    Next: {task['task_type']} → {task['target']}")
                        
                        # 设置阶段转换标志
                        phase_transition_pending = True
                        print(f"[DEBUG] Phase transition pending, will force plan on next iteration")
                else:
                    print(f"[SUCCESS] All phases completed - Task sequence finished!")
        except Exception as e:
            print(f"[ERROR] Phase completion check failed: {e}")
            import traceback
            traceback.print_exc()
        
        return phase_transition_pending 