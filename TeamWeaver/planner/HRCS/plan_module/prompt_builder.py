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
        # try:
        #     phase_id = current_phase.get('phase_id', 0)
        #     total_phases = len(perception_connector.phase_manager.task_execution_phases)
        #     phase_desc = current_phase.get('phase_description', 'current set of tasks')
        #     header = f"Suggestion for Coordination (Phase {phase_id + 1}/{total_phases}: {phase_desc}):"
        #     guidance_parts.append(header)
        # except (KeyError, AttributeError, TypeError):
        #     guidance_parts.append("Suggestion for Coordination:")

        # 2. Agent Task Assignments
        if agent_task_assignments:
            # assignment_lines = ["Based on optimization, the suggested assignments are:"]
            assignment_lines = ["The suggested assignments are:"]
            sorted_agent_ids = sorted(agent_task_assignments.keys())
            
            for agent_id in sorted_agent_ids:
                tasks = agent_task_assignments[agent_id]
                agent_line = f"- Agent {agent_id}: "
                if tasks:
                    task_descriptions = [self._format_task_description(t) for t in tasks]
                    agent_line += "Can focus on -> " + ", then ".join(task_descriptions) + "."
                else:
                    agent_line += "You can reason this agent task."
                assignment_lines.append(agent_line)
            
            guidance_parts.append("\n".join(assignment_lines))

        # 3. Key Contextual Information (Concise)
        if world_state:
            context_lines = ["\nRelevant context:"]
            object_positions = world_state.get('object_positions')
            if object_positions:
                key_objects = []
                for name, info in list(object_positions.items())[:12]:
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
        # Break any loop after 5 iterations—move on. 
        # guidance_parts.append("\nWhen last sentence in 'still in progress', you should keep the last plan till it finished.")
        guidance_parts.append("\nThis is a suggestion. You can adjust the plan based on your reasoning.")
        # guidance_parts.append("\nSpecify the exactly objects' name from the task instruction.")
        # guidance_parts.append("\nThis is a suggestion. You can explore the target place for more information, and adjust the plan based on your reasoning.")
        return "\n".join(guidance_parts)
    
    def build_lp_guidance_addition(
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
            # assignment_lines = ["The suggested assignments are:"]
            sorted_agent_ids = sorted(agent_task_assignments.keys())
            for agent_id in sorted_agent_ids:
                tasks = agent_task_assignments[agent_id]
                agent_line = f"- Agent {agent_id}: "
                if tasks:
                    task_descriptions = [self._format_task_description(t) for t in tasks]
                    agent_line += "Can focus on -> " + ", then ".join(task_descriptions) + "."
                else:
                    agent_line += "You can reason this agent task."
                assignment_lines.append(agent_line)
            guidance_parts.append("\n".join(assignment_lines))

        # 3. Key Contextual Information (Concise)
        if world_state:
            context_lines = ["\nPart of the Relevant context:"]
            object_positions = world_state.get('object_positions')
            if object_positions:
                key_objects = []
                for name, info in list(object_positions.items())[:5]:
                    if info and 'parent' in info and info['parent']:
                        key_objects.append(f"{name} is on {info['parent']}")
                    else:
                        key_objects.append(f"{name} location is known")
                if key_objects:
                    context_lines.append(f"- Key items: {'; '.join(key_objects)}.")
            if len(context_lines) > 1:
                 guidance_parts.append("\n".join(context_lines))
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

    def build_global_guidance_overview(
        self,
        structured_subtasks: List[Dict[str, Any]],
        execution_phases: List[Dict[str, Any]],
        max_tasks_per_phase: int = 6,
    ) -> str:
        """
        基于“任务分解后的所有实例任务 + 阶段划分”，构建一个全局性的精炼提示，
        用于指导整体协同与阶段路线图。输出风格与 build_miqp_guidance_addition/miqp_guidance 保持一致。
        """
        guidance_parts: List[str] = []

        # 1) Header
        try:
            total_phases = len(execution_phases) if execution_phases else 0
            header = f"Suggestion for Coordination (Global Overview: {total_phases} phases):"
            guidance_parts.append(header)
        except Exception:
            guidance_parts.append("Suggestion for Coordination (Global Overview):")

        # 2) Global summary
        try:
            total_tasks = len(structured_subtasks) if structured_subtasks else 0
            type_count: Dict[str, int] = {}
            if structured_subtasks:
                for t in structured_subtasks:
                    tt = t.get('task_type', 'Unknown')
                    type_count[tt] = type_count.get(tt, 0) + 1
            if total_tasks > 0:
                summary_lines = [
                    "Overall plan summary:",
                    f"- Total subtasks: {total_tasks}",
                    f"- Phase count: {len(execution_phases) if execution_phases else 0}",
                ]
                if type_count:
                    type_s = ", ".join([f"{k}: {v}" for k, v in sorted(type_count.items())])
                    summary_lines.append(f"- By type: {type_s}")
                guidance_parts.append("\n".join(summary_lines))
        except Exception:
            pass

        # 3) Phase roadmap
        if execution_phases:
            roadmap_lines: List[str] = ["\nPhase roadmap (sequence and parallelism):"]
            for idx, phase in enumerate(execution_phases):
                tasks = phase.get('tasks', []) or []
                max_parallel = phase.get('max_parallel_tasks', None)
                phase_head = f"- Phase {idx+1}/{len(execution_phases)}"
                if isinstance(max_parallel, int):
                    phase_head += f" (max_parallel: {max_parallel})"
                roadmap_lines.append(phase_head + ":")

                # 展示任务概述（限长）
                shown = 0
                for task in tasks:
                    if shown >= max_tasks_per_phase:
                        roadmap_lines.append("  · ...")
                        break
                    tt = task.get('task_type', 'Unknown')
                    target = task.get('target', 'None')
                    if target and str(target).lower() != 'none':
                        roadmap_lines.append(f"  · {tt} → {target}")
                    else:
                        roadmap_lines.append(f"  · {tt}")
                    shown += 1
            guidance_parts.append("\n".join(roadmap_lines))

        # 4) Strategic guidance (generic, consistent with miqp_guidance tone)
        # strategy_lines = [
        #     "\nStrategic Guidance:",
        #     "- Ensure preconditions: navigate before pick/place; hold object before place.",
        #     "- Respect phase sequencing and use parallelism when allowed to improve efficiency.",
        #     "- Coordinate to avoid conflicts and duplicate work across agents.",
        #     "- Prefer clear subgoals and maintain progress toward the phase objectives.",
        # ]
        # guidance_parts.append("\n".join(strategy_lines))

        # 5) Closing note (保持与 build_miqp_guidance_addition 一致的口吻)
        guidance_parts.append(
            "\nThis is a suggestion. You can explore the target place for more information, and adjust the plan based on your reasoning."
        )

        return "\n".join(guidance_parts)