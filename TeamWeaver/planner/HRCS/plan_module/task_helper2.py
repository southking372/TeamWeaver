from typing import Dict, List, Any, TYPE_CHECKING, Optional, Tuple
import numpy as np
import re
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from habitat_llm.agent.agent import Agent
    from habitat_llm.planner.perception_connector import PerceptionConnector
    from habitat_llm.planner.HRCS.plan_module.miqp_solver_wrapper import MIQPSolverWrapper


@dataclass
class AgentResume:
    """
    Stores Agent details“CV”，for intelligent task allocation.
    Contains static capabilities, performance traits and dynamic states.
    """
    agent_id: int
    name: str
    # core competencies: AgentWhat types of tasks can be performed (hard constraints)
    capabilities: List[str] = field(default_factory=list)
    # Performance characteristics: AgentEfficiency and preference for performing tasks (soft constraints / Rating basis)
    performance_traits: Dict[str, float] = field(default_factory=dict)
    # Dynamic state: Contextual information for real-time decision-making
    current_task_load: int = 0  # Number of tasks currently assigned
    position: Optional[np.ndarray] = None # Update from world_state
    history: List[Dict[str, Any]] = field(default_factory=list) # To store action history
    
    def get_core_description(self, format_style="compact") -> str:
        specialty = self._identify_specialty()
        status = self._get_status_summary()
        if format_style == "prompt":
            return self._generate_prompt_description(specialty, status)
        else:
            description = f"{self.name} | {specialty} | {status}"
            return description
    
    def _generate_prompt_description(self, specialty: str, status: str) -> str:
        base_info = f"{self.name} (ID:{self.agent_id})"
        capability_desc = self._get_capability_description()
        strength_desc = self._get_strength_description(specialty)
        status_desc = self._get_current_status_description(status)
        description = f"{base_info}: {capability_desc} {strength_desc} {status_desc}"
        return description
    
    def _get_capability_description(self) -> str:
        if not self.capabilities:
            return "Multi-purpose robot"
        special_caps = []
        basic_caps = []
        for cap in self.capabilities:
            if cap in ['Clean', 'Fill', 'Pour', 'PowerOn', 'PowerOff']:
                special_caps.append(cap)
            elif cap not in ['Navigate', 'Wait', 'Explore']:
                basic_caps.append(cap)
        if special_caps:
            return f"Specialized robot capable of {', '.join(special_caps).lower()}, plus standard operations"
        elif len(basic_caps) > 3:
            return f"Multi-skilled robot with {len(self.capabilities)} capabilities including {', '.join(basic_caps[:3]).lower()}"
        else:
            return f"Robot capable of {', '.join(basic_caps).lower()}"
    
    def _get_strength_description(self, specialty: str) -> str:
        trait_descriptions = {
            'speed': f"excels in speed (score: {self.performance_traits.get('speed', 0):.1f})",
            'precision': f"highly precise (score: {self.performance_traits.get('precision', 0):.1f})",
            'energy_efficiency': f"energy efficient (score: {self.performance_traits.get('energy_efficiency', 0):.1f})",
            'exploration_bias': f"strong explorer (score: {self.performance_traits.get('exploration_bias', 0):.1f})",
            'liquid_handling_skill': f"liquid handling expert (score: {self.performance_traits.get('liquid_handling_skill', 0):.1f})",
            'power_control_skill': f"power systems specialist (score: {self.performance_traits.get('power_control_skill', 0):.1f})"
        }
        top_traits = sorted(self.performance_traits.items(), key=lambda x: x[1], reverse=True)[:2]
        descriptions = []
        for trait_name, score in top_traits:
            if score > 0.7 and trait_name in trait_descriptions:
                descriptions.append(trait_descriptions[trait_name])
        if descriptions:
            return f"- {'; '.join(descriptions)}."
        else:
            return f"- balanced performance across all areas."
    
    def _get_current_status_description(self, status: str) -> str:
        parts = status.split(" | ")
        load_desc = ""
        if "Idle" in status:
            load_desc = "Currently available"
        elif "Task" in status:
            task_count = [p for p in parts if "Task" in p][0]
            load_desc = f"Currently assigned {task_count.lower()}"
        pos_desc = ""
        position_parts = [p for p in parts if p.startswith("@")]
        if position_parts:
            pos_desc = f"located at {position_parts[0]}"
        performance_desc = ""
        if "Hot Streak" in status:
            performance_desc = "on a successful streak"
        elif "Struggling" in status:
            performance_desc = "recently experiencing difficulties"
        elif "Performing Well" in status:
            performance_desc = "performing reliably"
        status_parts = [desc for desc in [load_desc, pos_desc, performance_desc] if desc]
        if status_parts:
            return f"Status: {', '.join(status_parts)}."
        else:
            return "Status: ready for assignment."
    
    def _identify_specialty(self) -> str:
        if not self.performance_traits:
            return "General-purpose"
        top_trait_name = max(self.performance_traits, key=self.performance_traits.get)
        top_trait_value = self.performance_traits[top_trait_name]
        special_skills = []
        if self.performance_traits.get('liquid_handling_skill', 0) > 0.8:
            special_skills.append("Liquid Expert")
        if self.performance_traits.get('power_control_skill', 0) > 0.8:
            special_skills.append("Power Control")
        if special_skills:
            return " & ".join(special_skills)
        specialty_map = {
            'speed': "Speed-focused",
            'precision': "Precision Specialist", 
            'energy_efficiency': "Efficiency Expert",
            'exploration_bias': "Explorer"
        }
        specialty = specialty_map.get(top_trait_name, "Generalist")
        if top_trait_value > 0.8:
            specialty = f"High-{specialty}"
        return specialty

    def _get_status_summary(self) -> str:
        status_parts = []
        if self.current_task_load == 0:
            status_parts.append("Idle")
        elif self.current_task_load == 1:
            status_parts.append("1 Task")
        else:
            status_parts.append(f"{self.current_task_load} Tasks")
        if self.history:
            recent_performance = self._analyze_recent_performance()
            if recent_performance:
                status_parts.append(recent_performance)
        if self.position is not None:
            pos_str = f"@({self.position[0]:.1f},{self.position[1]:.1f})"
            status_parts.append(pos_str)
        return " | ".join(status_parts) if status_parts else "Ready"

    def _analyze_recent_performance(self) -> str:
        if len(self.history) < 2:
            return ""
        recent_actions = self.history[-3:]
        success_count = sum(1 for entry in recent_actions 
                          if "successful execution" in entry.get('feedback', '').lower())
        fail_count = sum(1 for entry in recent_actions 
                        if "fail" in entry.get('feedback', '').lower())
        if success_count >= 2 and fail_count == 0:
            return "↗ Hot Streak"
        elif fail_count >= 2:
            return "↘ Struggling"
        elif success_count > fail_count:
            return "✓ Performing Well"
        return ""
    
    def get_detailed_status(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'core_description': self.get_core_description(),
            'capabilities_count': len(self.capabilities),
            'top_capabilities': self.capabilities[:5],  # Top 5 abilities
            'performance_summary': {
                trait: f"{value:.2f}" for trait, value in 
                sorted(self.performance_traits.items(), 
                       key=lambda x: x[1], reverse=True)[:3]  # Top 3 performance traits
            },
            'current_load': self.current_task_load,
            'position': self.position.tolist() if self.position is not None else None,
            'recent_actions': len(self.history),
            'last_action': self.history[-1].get('action') if self.history else None
        }


class TaskHelper:
    def __init__(self, agents: List['Agent']):
        self._agents = agents
        # Agent "Resumes" as the central knowledge base for assignment
        self.agent_resumes: Dict[int, AgentResume] = self._initialize_agent_resumes()
        self.cached_agent_capabilities = {}
        self.last_assignments = {}
        self.assignment_history = []
        print("[DEBUG] TaskHelper reset completed")

    def reset(self):
        """
        Reset the TaskHelper state.
        Clear all cached data and assignment history.
        """
        self.cached_agent_capabilities = {}
        self.last_assignments = {}
        self.assignment_history = []
        print("[DEBUG] TaskHelper reset completed")

    def get_agents_summary(self, format_style: str = "compact") -> str:
        if format_style == "prompt" or format_style == "planning":
            return self._generate_prompt_summary()
        else:
            summaries = []
            for agent_id, resume in self.agent_resumes.items():
                summaries.append(f"  {resume.get_core_description('compact')}")
            header = f"=== Agent Team Status ({len(self.agent_resumes)} agents) ==="
            return header + "\n" + "\n".join(summaries)
    
    def _generate_prompt_summary(self) -> str:
        """Generate detailed team summaries suitable for LLM planning"""
        team_size = len(self.agent_resumes)
        header = f"Available Agents for Task Planning ({team_size} robots):"
        agent_descriptions = []
        for agent_id, resume in self.agent_resumes.items():
            desc = resume.get_core_description("prompt")
            agent_descriptions.append(f"• {desc}")
        
        team_overview = self._generate_team_capabilities_overview()
        planning_hints = self._generate_planning_hints()
        sections = [
            header,
            "",  # blank line
            "\n".join(agent_descriptions),
            "",  # blank line
            team_overview,
            "",  # blank line
            planning_hints
        ]
        
        return "\n".join(sections)
    
    def _generate_team_capabilities_overview(self) -> str:
        all_capabilities = set()
        special_skills = []
        for resume in self.agent_resumes.values():
            all_capabilities.update(resume.capabilities)
            if resume.performance_traits.get('liquid_handling_skill', 0) > 0.8:
                special_skills.append("liquid handling")
            if resume.performance_traits.get('power_control_skill', 0) > 0.8:
                special_skills.append("power control")
        basic_caps = [cap for cap in all_capabilities if cap not in ['Navigate', 'Wait', 'Explore']]
        overview = f"Team Capabilities: {len(all_capabilities)} total abilities including {', '.join(sorted(basic_caps)).lower()}"
        if special_skills:
            overview += f". Special expertise in {', '.join(special_skills)}"
        return overview + "."
    
    def _generate_planning_hints(self) -> str:
        hints = []
        speed_focused = [r for r in self.agent_resumes.values() if r.performance_traits.get('speed', 0) > 0.8]
        precision_focused = [r for r in self.agent_resumes.values() if r.performance_traits.get('precision', 0) > 0.8]
        if speed_focused and precision_focused:
            hints.append("Consider assigning exploration/navigation tasks to speed-focused agents and delicate operations to precision-focused agents")
        current_loads = [r.current_task_load for r in self.agent_resumes.values()]
        if max(current_loads) - min(current_loads) > 1:
            hints.append("Current workload is unbalanced - consider redistributing tasks")
        special_agents = [r for r in self.agent_resumes.values() 
                         if any(r.performance_traits.get(skill, 0) > 0.8 
                               for skill in ['liquid_handling_skill', 'power_control_skill'])]
        if special_agents:
            hints.append("Reserve specialized agents for tasks requiring their unique skills")
        if hints:
            return "Planning Recommendations: " + "; ".join(hints) + "."
        else:
            return "Planning Recommendations: Team is well-balanced and ready for diverse task assignments."

    def update_resumes_from_context(
        self, 
        world_state: Dict[str, Any], 
        agent_feedback: Dict[int, str],
        last_actions: Dict[int, Tuple[str, str, str]]
    ):
        """
        Update the resumes of all Agents based on the latest world state, feedback, and actions performed in the previous step.
        This method should be called at the beginning of each replan cycle to enable feedback-based learning.
        """
        for agent_id, resume in self.agent_resumes.items():
            # Update location information from world_state
            agent_name_for_pose = f"agent_{agent_id}"
            if agent_name_for_pose in world_state.get('agent_poses', {}):
                 resume.position = np.array(world_state['agent_poses'][agent_name_for_pose].get('position'))

            # Update performance traits based on feedback and specific actions
            feedback = agent_feedback.get(agent_id, "")
            last_action_tuple = last_actions.get(agent_id)
            
            if feedback and last_action_tuple:
                action_name = last_action_tuple[0]
                self._update_traits_from_feedback(resume, feedback, action_name)
                
                # Record history for more complex analysis
                history_entry = {
                    'action': action_name,
                    'params': last_action_tuple[1],
                    'feedback': feedback, 
                    'world_state_snapshot': world_state
                }
                resume.history.append(history_entry)
                if len(resume.history) > 20:
                    resume.history.pop(0)

    def _update_traits_from_feedback(self, resume: AgentResume, feedback: str, action_name: str):
        """
        Based on the execution feedback of specific actions, performance characteristics are updated in a refined manner to implement a simple learning mechanism.
        """
        feedback_lower = feedback.lower()
        action_lower = action_name.lower()

        # Successful Feedback -> Slightly improve related skill scores
        if "successful execution" in feedback_lower:
            if action_lower in ['pick', 'place', 'rearrange']:
                current_precision = resume.performance_traits.get('precision', 0.5)
                resume.performance_traits['precision'] = min(1.0, current_precision + 0.02)
                print(f"[Feedback] Agent {resume.agent_id} succeeded at '{action_name}'. Precision increased to {resume.performance_traits['precision']:.3f}")
            elif action_lower in ['navigate', 'explore']:
                current_speed = resume.performance_traits.get('speed', 0.5)
                resume.performance_traits['speed'] = min(1.0, current_speed + 0.02)
                print(f"[Feedback] Agent {resume.agent_id} succeeded at '{action_name}'. Speed increased to {resume.performance_traits['speed']:.3f}")

        # Failed feedback -> Significantly lowers relevant skill scores
        elif "fail" in feedback_lower or "unreachable" in feedback_lower:
            if action_lower in ['pick', 'place', 'rearrange']:
                current_precision = resume.performance_traits.get('precision', 0.5)
                resume.performance_traits['precision'] = max(0.1, current_precision - 0.1)
                print(f"[Feedback] Agent {resume.agent_id} failed at '{action_name}'. Precision decreased to {resume.performance_traits['precision']:.3f}")
            elif action_lower in ['navigate', 'explore']:
                current_speed = resume.performance_traits.get('speed', 0.5)
                resume.performance_traits['speed'] = max(0.1, current_speed - 0.1)
                print(f"[Feedback] Agent {resume.agent_id} failed at '{action_name}'. Speed decreased to {resume.performance_traits['speed']:.3f}")
        
        # "in progress" Make no adjustments and wait for the final result
        elif "in progress" in feedback_lower:
            pass
        
        else:
            # For other unknown types of feedback, consider making a small negative adjustment
            pass

    def _initialize_agent_resumes(self) -> Dict[int, AgentResume]:
        """
        Initialize all Agents“CV”。In practical applications, this information can be loaded from configuration files.
        """
        resumes = {}
        # Agent 0: "Workhorse" - Strong basic ability and fast speed
        resumes[0] = AgentResume(
            agent_id=0,
            name="Robot_0_Workhorse",
            capabilities=['Navigate', 'Explore', 'Pick', 'Place', 'Open', 'Close', 'Rearrange', 'Wait'],
            performance_traits={
                'speed': 0.9,
                'precision': 0.6,
                'energy_efficiency': 0.7,
                'exploration_bias': 0.5,
            }
        )
        
        # Agent 1: "Specialist" - More sophisticated, with special skills
        resumes[1] = AgentResume(
            agent_id=1,
            name="Robot_1_Specialist",
            capabilities=['Navigate', 'Explore', 'Pick', 'Place', 'Open', 'Close', 'Clean', 'Fill', 'Pour', 'PowerOn', 'PowerOff', 'Rearrange', 'Wait'],
            performance_traits={
                'speed': 0.6,
                'precision': 0.9,
                'energy_efficiency': 0.9,
                'exploration_bias': 0.8,
                'liquid_handling_skill': 0.95, # Special skill rating
                'power_control_skill': 0.9,   # Special skill rating
            }
        )
        return resumes

    def _calculate_assignment_score(self, task: Dict[str, Any], resume: AgentResume) -> float:
        """
        Core scoring function: Calculates the performance of an Agent for a specific task“adaptability score”。
        The higher the score, the better the fit. Returning 0 means it cannot be executed.
        """
        task_type = task.get('task_type', 'Wait')

        # 1. ability matching (hard constraints)
        if task_type not in resume.capabilities:
            return 0.0

        score = 1.0  # Basic points: 1 point if you can do it

        # 2. Performance trait matching (soft constraints/Bonus points)
        if task_type in ['Navigate', 'Explore']:
            score += resume.performance_traits.get('speed', 0.5) * 0.5
            score += resume.performance_traits.get('exploration_bias', 0.5) * 0.3

        if task_type in ['Place', 'Rearrange']:
            score += resume.performance_traits.get('precision', 0.5) * 0.7

        if task_type in ['Clean', 'Fill', 'Pour']:
            score += resume.performance_traits.get('liquid_handling_skill', 0.0) * 1.0

        if task_type in ['PowerOn', 'PowerOff']:
            score += resume.performance_traits.get('power_control_skill', 0.0) * 1.0
        
        # 3. Dynamic state effects (penalty item)，Implement load balancing
        score -= resume.current_task_load * 0.25

        # Punishment based on spatial distance
        # Objective facts as planning factors and attempts as Agent preferences
        target_pos = task.get('target_position')
        if target_pos is not None and resume.position is not None:
            distance = np.linalg.norm(target_pos - resume.position)
            score -= distance * 0.1 # The further the distance, the lower the score

        return max(0.1, score) # Make sure the agent that can do it has at least 0.1 points

    def _distribute_tasks_with_scoring(
        self, 
        tasks: List[Dict[str, Any]], 
        agent_ids: List[int]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Use a scoring-based system to intelligently assign a set of task instances to a set of Agents.
        """
        assignments: Dict[int, List[Dict[str, Any]]] = {i: [] for i in agent_ids}
        
        # Update Agent’s dynamic information in real time (task load)
        for agent_id in agent_ids:
            self.agent_resumes[agent_id].current_task_load = 0
            # Agent_resumes can also be updated from world_state here[agent_id].position

        # Traverse each task and find the most suitable Agent for it
        for task in tasks:
            scores = {
                agent_id: self._calculate_assignment_score(task, self.agent_resumes[agent_id])
                for agent_id in agent_ids
            }
            
            # Find the agent with the highest score
            if any(s > 0 for s in scores.values()):
                best_agent_id = max(scores, key=scores.get)
            else:
                # Fallback: If all agents cannot execute, the first one is assigned
                best_agent_id = agent_ids[0]
                print(f"[WARNING] No capable agent found for task {task.get('task_type')}, assigning to Agent {best_agent_id} as fallback.")

            # Assign tasks
            assignments[best_agent_id].append(task)
            # Update the agent's task load to provide the correct status for the next task assignment
            self.agent_resumes[best_agent_id].current_task_load += 1
            
        return assignments

    def get_aptitude_matrix(self, tasks: List[Dict[str, Any]], agents: List["Agent"]) -> np.ndarray:
        """
        Generate an adaptability matrix (Aptitude Matrix)，Indicates the adaptability score of each agent for each task.
        The matrix dimensions are (n_agents, n_tasks)。
        The higher the score, the more suitable the agent is for performing the task.
        """
        n_agents = len(agents)
        n_tasks = len(tasks)
        aptitude_matrix = np.zeros((n_agents, n_tasks))

        for i, agent in enumerate(agents):
            agent_id = agent.uid
            resume = self.agent_resumes.get(agent_id)
            if not resume:
                print(f"[WARNING] TaskHelper: No resume found for agent {agent_id}. Aptitude will be zero.")
                continue
            
            for j, task in enumerate(tasks):
                # Calculate scores using existing scoring functions
                score = self._calculate_assignment_score(task, resume)
                aptitude_matrix[i, j] = score
        
        # Normalization processing: ensure that the scores are within a reasonable range to avoid affecting the optimization solution
        # For example, you can normalize by column (task) so that the total score for each task tends to a fixed value
        # Here we first do a simple scaling to avoid 0 points
        if np.any(aptitude_matrix):
             aptitude_matrix = np.clip(aptitude_matrix, 0.1, 10.0) # Avoid 0 and overly large values
        
        print("[DEBUG] Generated Aptitude Matrix (Agents x Tasks):")
        print(aptitude_matrix)
        
        return aptitude_matrix

    def assign_tasks_for_phase(
        self,
        current_phase_tasks: List[Dict[str, Any]],
        alpha: Optional[np.ndarray],
        phase_task_info: Dict[str, Any],
        agents: List["Agent"],
        perception_connector: "PerceptionConnector",
        miqp_solver_wrapper: "MIQPSolverWrapper"
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        According to the macro allocation results of MIQP, combined with“CV”The scoring system is used for final assignment of task instances.
        """
        # MIQPSuggestions for assignments at the task type level are given (alphamatrix)
        if alpha is None or not np.any(alpha > 0.5):
            print("[DEBUG] MIQP solution not available or empty. Using resume-based heuristic assignment for all phase tasks.")
            # When there is no solution to the MIQP, all tasks in the current stage will be assigned through the scoring system
            agent_ids = [agent.uid for agent in agents]
            return self._distribute_tasks_with_scoring(current_phase_tasks, agent_ids)

        # Normal process: MIQP has solution
        print("[DEBUG] Using resume-based scoring system for instance-level assignment.")
        agent_ids = [agent.uid for agent in agents]
        agent_task_assignments = self._distribute_tasks_with_scoring(current_phase_tasks, agent_ids)

        print(f"[DEBUG] Resume-based phase task assignments:")
        for agent_id, tasks in agent_task_assignments.items():
            task_summaries = [f"{t['task_type']}→{t['target']}" for t in tasks] if tasks else "[]"
            print(f"  Agent {agent_id}: {task_summaries}")
        
        return agent_task_assignments

    def create_fallback_tasks(self, instruction: str, agents) -> List[Dict[str, Any]]:
        """Create a more robust, phased backup mission plan."""
        
        # Phase 1: Explore
        explore_phase = {
            'phase_id': 0,
            'tasks': [{
                'task_id': 'fallback_explore',
                'task_type': 'Explore',
                'target': 'environment',
                'description': 'Fallback: Explore the environment to find relevant objects.',
                'priority': 5,
                'estimated_duration': 20.0,
                'preferred_agent': None,
                'prerequisites': [],
                'can_parallel': True,
                'phase_group': 'fallback_preparation'
            }],
            'max_parallel_tasks': len(agents),
            'estimated_duration': 20.0,
            'required_agents': len(agents)
        }
        
        # Phase 2: Wait (allows for replanning after exploration)
        wait_phase = {
            'phase_id': 1,
            'tasks': [{
                'task_id': 'fallback_wait',
                'task_type': 'Wait',
                'target': '',
                'description': 'Fallback: Wait for the next planning cycle after exploration.',
                'priority': 1,
                'estimated_duration': 5.0,
                'preferred_agent': None,
                'prerequisites': ['fallback_explore'],
                'can_parallel': True,
                'phase_group': 'fallback_coordination'
            }],
            'max_parallel_tasks': len(agents),
            'estimated_duration': 5.0,
            'required_agents': len(agents)
        }

        print(f"[DEBUG] Created a robust 2-phase fallback plan: Explore -> Wait")
        return [explore_phase, wait_phase] 
    
    def _correct_llm_response(self, response: str) -> str:
        """
        Corrects the raw response from the LLM.
        - Fixes action formatting from "Action Arg" to "Action[Arg]".
        - Ensures only one action is assigned per agent, keeping the first one.
        """
        lines = response.split('\n')
        thought_lines = []
        action_lines_map = {}

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            # Check if the line is an action assignment
            action_match = re.match(r"^\s*Agent_(\d+)_Action:.*", stripped_line)
            if action_match:
                agent_id = int(action_match.group(1))

                # Correct format "Tool Arg" to "Tool[Arg]"
                format_match = re.match(r"^\s*(Agent_\d+_Action:\s*)(\w+)\s+([\w\d_]+)\s*$", stripped_line)
                if format_match:
                    stripped_line = f"{format_match.group(1)}{format_match.group(2)}[{format_match.group(3)}]"
                
                # Keep only the first action for each agent
                if agent_id not in action_lines_map:
                    action_lines_map[agent_id] = stripped_line
                else:
                    print(f"INFO: Ignoring duplicate action for agent {agent_id}: {stripped_line}")
            else:
                # It's a thought line
                thought_lines.append(line)

        # Reconstruct the response with thoughts first, then sorted actions
        final_lines = thought_lines + sorted(action_lines_map.values())
        return "\n".join(final_lines)