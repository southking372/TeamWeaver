from typing import Dict, List, Any, Optional

class PromptBuilder:
    def __init__(self, planner_config, env_interface):
        self.planner_config = planner_config
        self.env_interface = env_interface
        self.world_graph = self.env_interface.full_world_graph
        # Add tags for convenience
        self.assistant_tag = self.planner_config.llm.assistant_tag
        self.eot_tag = self.planner_config.llm.eot_tag
        self.user_tag = self.planner_config.llm.user_tag
        self.cached_prompts = {}
        self.prompt_history = []

    def reset(self):
        """
        Reset the PromptBuilder state.
        Clear cached prompts and prompt history.
        """
        self.cached_prompts = {}
        self.prompt_history = []
        print("[DEBUG] PromptBuilder reset completed")

    def prepare_llm_prompt(self, current_prompt: str, miqp_guidance: Optional[str]) -> str:
        """
        Prepares the final prompt for the LLM by injecting MIQP guidance as a distinct
        system message. This maintains the integrity of the dialogue history.
        """
        assistant_turn_start = f"{self.assistant_tag}Thought:"
        
        # Ensure the base prompt is clean and ready for additions
        prompt_for_llm = current_prompt.strip()

        if miqp_guidance:
            # Inject guidance as a new, distinct "User" turn with a system-like persona
            # This is more robust and logically sound than altering past turns.
            # The persona "System (MIQP Guidance)" clarifies the source of the information.
            guidance_injection = (
                f"\n{self.user_tag}System (MIQP Guidance):\n{miqp_guidance}\n"
                f"{self.eot_tag}"
            )
            prompt_for_llm += guidance_injection

        # Ensure the prompt ends correctly for the assistant to start its turn
        if not prompt_for_llm.endswith(assistant_turn_start):
            prompt_for_llm += f"\n{assistant_turn_start}"
            
        return prompt_for_llm
    
    def _format_task_description(self, task: Dict[str, Any]) -> str:
        """Formats a single task dictionary into a human-readable string."""
        task_type = task.get('task_type', 'Unknown')
        target = task.get('target', 'None')
        
        if target and target.lower() != 'none':
            if task_type == 'Pick':
                return f"pick up the {target}"
            elif task_type == 'Place':
                return f"place the item at {target}"
            elif task_type == 'Navigate':
                return f"go to the {target}"
            elif task_type == 'Explore':
                return f"explore the {target}"
            else:
                return f"execute {task_type} on {target}"
        else:
            # Generic actions for tasks without a specific target
            if task_type == 'Navigate':
                return "move to a strategic location"
            elif task_type == 'Explore':
                return "explore the area"
            else:
                return f"perform {task_type}"

    def build_miqp_guidance_addition(
        self,
        current_phase_tasks: List[Dict[str, Any]],
        agent_task_assignments: Dict[int, List[Dict[str, Any]]],
        current_phase: Dict[str, Any],
        perception_connector,
        alpha_result: Optional[Any] = None,
        world_state: Optional[Dict[str, Any]] = None,
        validation_feedback: Optional[str] = None
    ) -> str:
        """
        Builds concise and effective MIQP guidance in a dialogue-friendly format,
        offering suggestions for task planning.
        """
        guidance_parts = []

        # 1. Header with Phase Goal
        try:
            phase_id = current_phase.get('phase_id', 0)
            total_phases = len(perception_connector.phase_manager.task_execution_phases)
            phase_desc = current_phase.get('phase_description', 'current set of tasks')
            header = f"Suggestion for Coordination (Phase {phase_id + 1}/{total_phases}: {phase_desc}):"
            guidance_parts.append(header)
        except (KeyError, AttributeError, TypeError):
            guidance_parts.append("Suggestion for Coordination:")

        # 2. Agent Task Assignments
        if agent_task_assignments:
            assignment_lines = ["Based on optimization, the suggested assignments are:"]
            sorted_agent_ids = sorted(agent_task_assignments.keys())
            
            for agent_id in sorted_agent_ids:
                tasks = agent_task_assignments[agent_id]
                agent_line = f"- Agent {agent_id}: "
                if tasks:
                    task_descriptions = [self._format_task_description(t) for t in tasks]
                    agent_line += "Should focus on -> " + ", then ".join(task_descriptions) + "."
                else:
                    agent_line += "Should wait or standby."
                assignment_lines.append(agent_line)
            
            guidance_parts.append("\n".join(assignment_lines))

        # 3. Key Contextual Information (Concise)
        if world_state:
            context_lines = ["\nRelevant context:"]
            object_positions = world_state.get('object_positions')
            if object_positions:
                key_objects = []
                # Limit to 5 key objects to keep the prompt concise
                for name, info in list(object_positions.items())[:5]:
                    if info and 'parent' in info and info['parent']:
                        key_objects.append(f"{name} is on {info['parent']}")
                    else:
                        key_objects.append(f"{name} location is known")
                if key_objects:
                    context_lines.append(f"- Key items: {'; '.join(key_objects)}.")
            
            if len(context_lines) > 1:
                 guidance_parts.append("\n".join(context_lines))

        # 4. Validation Feedback
        if validation_feedback:
            feedback_section = [
                "\nFeedback on previous action:",
                validation_feedback,
                "Please adjust your plan accordingly."
            ]
            guidance_parts.append("\n".join(feedback_section))
        
        # 5. Closing Note
        guidance_parts.append("\nThis is a suggestion. You can adjust the plan based on your reasoning.")

        return "\n".join(guidance_parts) 