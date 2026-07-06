# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

"""
PerceptionConnector - connects perception system with MIQP optimization
Provides world state extraction, task decomposition, matrix updates, etc.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import json
import abc
import requests
import os
import sys
from pathlib import Path
from habitat_llm.planner.miqp_prompts import get_miqp_prompt, MIQPAnalysisPrompt, TaskDecompositionPrompt
from habitat_llm.planner.HRCS.connector.dependency_enhancer import TaskDependencyEnhancer
from habitat_llm.planner.HRCS.connector.planner_utils import (
    extract_json_from_text,
    get_llm_config,
    update_param_value,
)
from habitat_llm.planner.HRCS.connector.action_updater import ActionUpdater
from habitat_llm.planner.HRCS.connector.matrix_updater import MatrixUpdater
from habitat_llm.planner.HRCS.connector.phase_manager import PhaseManager
from habitat_llm.llm.instruct.utils import get_world_descr
import openai

if TYPE_CHECKING:
    from habitat_llm.planner.HRCS.params_module.scenario_params_task import ScenarioConfigTask
    from habitat_llm.agent.env import EnvironmentInterface


def quaternion_to_yaw(quaternion: List[float]) -> float:
    """Convert quaternion to Z-axis rotation (yaw)."""
    quat = np.array(quaternion)
    x, y, z, w = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)

class PerceptionConnector:
    """
    Connects perception (WorldGraph/env) with task planning params (ScenarioConfigTask).
    
    Main features:
    1. Task decomposition: LLM splits instruction into structured subtasks
    2. World state extraction: agent, object, furniture positions from environment
    3. MIQP matrix update: update optimization params from agent capability dimensions
    4. Scenario param update: update task targets/constraints from high-level actions
    5. Task assignment: assign subtasks to agents from MIQP optimization
    6.  Task sequencing: organize subtasks into dependency-ordered execution phases
    
    Capability dimensions:
    - Motor Skills: nav, pick, place, open, close, rearrange, explore, wait
    - Object States: power_on, power_off, clean, fill, pour  
    - Perception: find_receptacle, find_object, find_agent_action, find_room
    """
    
    # --- MIQP matrix initialization constants ---
    BASE_TASK_CAPABILITY_REQUIREMENTS = np.array([
        #Task: [Navigate, Explore, Pick, Place, Open, Close, Clean, Fill, Pour, PowerOn, PowerOff, Rearrange, Wait]
        # capabilities: [movement, manipulation, control, liquid, power]
        [1, 0, 0, 0, 0],  # Navigate
        [1, 0, 0, 0, 0],  # Explore
        [0, 1, 0, 0, 0],  # Pick
        [0, 1, 0, 0, 0],  # Place
        [0, 0, 1, 0, 0],  # Open
        [0, 0, 1, 0, 0],  # Close
        [0, 0, 0, 1, 0],  # Clean
        [0, 0, 0, 1, 0],  # Fill
        [0, 0, 0, 1, 0],  # Pour
        [0, 0, 0, 0, 1],  # PowerOn
        [0, 0, 0, 0, 1],  # PowerOff
        [0, 1, 0, 0, 0],  # Rearrange
        [1, 0, 0, 0, 0],  # Wait
    ], dtype=float)

    BASE_ROBOT_CAPABILITIES = np.array([
        [2.0, 1.8],  # movement
        [2.0, 1.8],  # manipulation
        [2.0, 1.8],  # control
        [0.0, 1.3],  # liquid
        [0.0, 1.3]   # power
    ], dtype=float)
    
    BASE_CAPABILITY_WEIGHTS = [
        2.0 * np.eye(1),  # movement
        2.5 * np.eye(1),  # manipulation
        2.0 * np.eye(1),  # control
        1.8 * np.eye(1),  # liquid
        1.5 * np.eye(1)   # power
    ]
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        api_key_filename: Optional[str] = None,
        llm_base_url: Optional[str] = "https://api.moonshot.cn/v1"
    ):
        self.last_world_state: Dict[str, Any] = {}
        self.llm_client = llm_client
        if api_key_filename is None:
            api_key_filename = "api_key"

        #  Task sequence management (now handled by PhaseManager)
        # self.task_execution_phases: List[Dict[str, Any]] = []
        # self.current_phase_index: int = 0
        self.task_dependency_graph: Dict[str, List[str]] = {}
        self.completed_tasks: List[str] = []
        self.active_tasks: List[str] = []
        self.phase_t_matrices: Dict[int, np.ndarray] = {} # per-phase T matrix cache
        
        self._init_llm_client(llm_client, api_key_filename, llm_base_url)

        self.matrix_updater = MatrixUpdater(self.llm_client)
        self.phase_manager = PhaseManager(self.llm_client)

    def reset(self):
        """
        Reset the state of the PerceptionConnector for a new episode.
        """
        self.last_world_state = {}
        self.task_dependency_graph = {}
        self.completed_tasks = []
        self.active_tasks = []
        self.phase_t_matrices = {}
        
        # Re-initialize the phase manager to clear its state
        if hasattr(self, 'phase_manager') and self.phase_manager.llm_client:
            self.phase_manager = PhaseManager(self.phase_manager.llm_client)
        else:
            self.phase_manager = PhaseManager(self.llm_client)

    # --- main public API ---
    def _get_world_description_for_prompt(self, env_interface: "EnvironmentInterface") -> str:
        """Generates a detailed world description string for the LLM prompt."""
        full_graph = env_interface.full_world_graph
        # We can use agent_uid=0 because it's a centralized planner and graph is full.
        world_description = get_world_descr(
            full_graph,
            agent_uid=0,
            include_room_name=True,
            add_state_info=True,
            centralized=True,
        )
        return world_description
        
    def extract_world_state(self, env_interface: "EnvironmentInterface") -> Dict[str, Any]:
        """
        Extract current world state from environment interface.
        """
        world_state: Dict[str, Any] = {
            'agent_poses': {},
            'object_positions': {},
            'furniture_positions': {},
        }
        full_graph = env_interface.full_world_graph

        # 1. extract agent poses
        agents = full_graph.get_agents()
        for agent_node in agents:
            agent_id = agent_node.name
            try:
                pos = agent_node.get_property("translation")
                try:
                    rot_quat = agent_node.get_property("rotation")
                    yaw = quaternion_to_yaw(rot_quat)
                except (KeyError, AttributeError, ValueError):
                    rot_quat = [0.0, 0.0, 0.0, 1.0]
                    yaw = 0.0
                world_state['agent_poses'][agent_id] = {'position': pos, 'rotation': rot_quat, 'yaw': yaw}
            except (KeyError, AttributeError, ValueError) as e:
                print(f"Error: Could not get agent '{agent_id}'  pose: {e}")
                world_state['agent_poses'][agent_id] = {'position': [0.0, 0.0, 0.0], 'rotation': [0.0, 0.0, 0.0, 1.0], 'yaw': 0.0}

        # 2. extract object positions and parents
        all_objects = full_graph.get_all_objects()
        for obj_node in all_objects:
            obj_name = obj_node.name
            try:
                pos = obj_node.get_property("translation")
                parent_node = full_graph.find_furniture_for_object(obj_node)
                parent_name = parent_node.name if parent_node else None
                world_state['object_positions'][obj_name] = {'position': pos, 'parent': parent_name}
            except (KeyError, AttributeError) as e:
                print(f"Error: Could not get object '{obj_name}'  position: {e}")
                world_state['object_positions'][obj_name] = None

        # 3. extract furniture positions
        all_furniture = full_graph.get_all_furnitures()
        for furn_node in all_furniture:
            furn_name = furn_node.name
            try:
                pos = furn_node.get_property("translation")
                world_state['furniture_positions'][furn_name] = {'position': pos}
            except (KeyError, AttributeError) as e:
                print(f"Error: Could not get furniture '{furn_name}'  position: {e}")
                world_state['furniture_positions'][furn_name] = None # or a default value

        self.last_world_state = world_state
        # print(f"[DEBUG-LYP-v2]: World state extracted: {world_state}")
        return world_state

    def structured_decompose_task_with_sequencing(
        self,
        instruction: str,
        env_interface: "EnvironmentInterface",
        llm_config: Dict[str, Any],
        max_agents: int = 2
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Use LLM to decompose instruction into sequenced subtasks and execution phases.
            
        Returns:
            (structured_subtasks, execution_phases)
            structured_subtasks: full subtask list
            execution_phases: phase-organized task execution plan
        """
        if not self.phase_manager or not self.phase_manager.llm_client:
            raise ValueError("PhaseManager or its LLM client not initialized. Cannot decompose task.")

        # 1. call PhaseManager for initial decomposition
        world_desc_string = self._get_world_description_for_prompt(env_interface)
        current_world_state = self.last_world_state or self.extract_world_state(env_interface)
        agent_info_string = self._get_agent_status_for_prompt(current_world_state)

        structured_subtasks = self.phase_manager.decompose_and_initialize_phases(
            instruction,
            world_desc_string,
            agent_info_string,
            llm_config
        )
        
        print(f"DEBUG: Initial LLM decomposition: {len(structured_subtasks)} tasks")
        
        # 2. semantic enhancement and phase organization (TaskDependencyEnhancer)
        # also organize tasks into execution phases
        enhancer = TaskDependencyEnhancer()
        enhanced_subtasks, execution_phases, dependency_graph = enhancer.structure_and_phase(
            structured_subtasks, max_agents
        )
        
        # 3. save task dependency graph
        self.task_dependency_graph = dependency_graph
        
        # 4. cache phase info and pass to PhaseManager
        # self.task_execution_phases = execution_phases # No longer stored locally
        # self.current_phase_index = 0 # No longer stored locally
        self.phase_manager.set_execution_phases(execution_phases)
        
        print(f"DEBUG: Task decomposed into {len(enhanced_subtasks)} subtasks across {len(execution_phases)} phases")
        for i, phase in enumerate(execution_phases):
            task_summaries = [f"{t['task_type']}→{t['target']}" for t in phase['tasks']]
            print(f"  Phase {i+1}: {task_summaries} (max_parallel: {phase['max_parallel_tasks']})")
        
        return enhanced_subtasks, execution_phases

    def _clean_task_dependencies(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean task dependencies; remove self-loops and invalid references"""
        cleaned_tasks = []
        task_ids = {task['task_id'] for task in tasks}
        
        for task in tasks:
            cleaned_task = task.copy()
            prereqs = cleaned_task.get('prerequisites', [])
            
            valid_prereqs = [p for p in prereqs if p != cleaned_task['task_id'] and p in task_ids]
            cleaned_task['prerequisites'] = valid_prereqs
            cleaned_tasks.append(cleaned_task)
            
        return cleaned_tasks

    def get_current_phase_tasks(self) -> Optional[Dict[str, Any]]:
        """Get tasks for current phase"""
        return self.phase_manager.get_current_phase_tasks()

    def get_enriched_current_phase(self, world_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get tasks for current phase, and augmented with world state information such as target location.
        """
        current_phase = self.phase_manager.get_current_phase_tasks()
        if not current_phase:
            return None

        current_phase_tasks = current_phase.get('tasks', [])
        enriched_tasks = []

        for task in current_phase_tasks:
            new_task = task.copy()
            target_name = new_task.get('target')
            if target_name:
                target_pos_data = None
                # Check for object position
                if world_state and 'object_positions' in world_state and target_name in world_state['object_positions']:
                    pos_info = world_state['object_positions'][target_name]
                    if pos_info and 'position' in pos_info:
                         target_pos_data = pos_info['position']
                # Check for furniture position
                elif world_state and 'furniture_positions' in world_state and target_name in world_state['furniture_positions']:
                    pos_info = world_state['furniture_positions'][target_name]
                    if pos_info and 'position' in pos_info:
                        target_pos_data = pos_info['position']

                if target_pos_data is not None:
                    new_task['target_pos'] = target_pos_data
                else:
                    print(f"  [Enrichment] WARNING: Could not find position for target '{target_name}'")
            enriched_tasks.append(new_task)
        
        enriched_phase = current_phase.copy()
        enriched_phase['tasks'] = enriched_tasks
        
        return enriched_phase

    def advance_to_next_phase(self) -> bool:
        """Advance to next execution phase"""
        return self.phase_manager.advance_to_next_phase()

    def is_current_phase_complete(self, agent_statuses: Dict[int, str]) -> bool:
        """Check whether current phase is complete"""
        return self.phase_manager.is_current_phase_complete(agent_statuses)

    def assign_tasks(
        self,
        subtasks: List[Dict[str, Any]],
        alpha_matrix: np.ndarray,
        agent_capabilities: Dict[int, List[str]]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Assigns subtasks to agents based on the MIQP optimization result (alpha_matrix).

        If the alpha_matrix is invalid or not provided, it falls back to a
        heuristic-based assignment.

        Args:
            subtasks: The list of subtasks for the current phase.
            alpha_matrix: The result matrix from the MIQP solver.
            agent_capabilities: A dictionary of capabilities for each agent.

        Returns:
            A dictionary mapping agent IDs to their assigned tasks.
        """
        num_agents = len(agent_capabilities)
        num_tasks_in_phase = len(subtasks)

        if alpha_matrix is None or alpha_matrix.shape != (num_agents, num_tasks_in_phase):
            print(f"[PerceptionConnector] Invalid alpha matrix (shape: {getattr(alpha_matrix, 'shape', 'None')}). "
                  f"Expected ({num_agents}, {num_tasks_in_phase}). Using heuristic assignment.")
            return self._heuristic_task_assignment(subtasks, agent_capabilities, num_agents)

        return self.map_subtasks_to_agents(subtasks, alpha_matrix, agent_capabilities)

    def map_subtasks_to_agents(
        self,
        subtasks: List[Dict[str, Any]],
        alpha_matrix: np.ndarray,
        agent_capabilities: Dict[int, List[str]]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Assign subtasks to agents from MIQP optimization result (alpha_matrix).
        """
        num_agents = alpha_matrix.shape[0]
        assignments = {i: [] for i in range(num_agents)}

        if not subtasks:
            return assignments

        num_tasks_in_phase = len(subtasks)
        
        # validate alpha_matrix dimensions
        if alpha_matrix.shape[1] != num_tasks_in_phase:
            print(f"[ERROR] Alpha matrix dimension mismatch in map_subtasks_to_agents. "
                  f"Expected {num_tasks_in_phase} tasks, but matrix has {alpha_matrix.shape[1]} columns. "
                  f"Falling back to heuristic assignment.")
            return self._heuristic_task_assignment(subtasks, agent_capabilities, num_agents)

        # each column is a task; assign to agent with max alpha
        for task_idx in range(num_tasks_in_phase):
            task = subtasks[task_idx]
            # find agent responsible for task (max alpha)
            assigned_agent_idx = np.argmax(alpha_matrix[:, task_idx])
            
            task_with_assignment_info = task.copy()
            task_with_assignment_info['assigned_agent'] = assigned_agent_idx
            task_with_assignment_info['assignment_confidence'] = np.max(alpha_matrix[:, task_idx])
            assignments[assigned_agent_idx].append(task_with_assignment_info)

        return assignments

    def _heuristic_task_assignment(
        self,
        tasks: List[Dict[str, Any]],
        agent_capabilities: Dict[int, List[str]],
        num_agents: int
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Heuristic task assignment as fallback when MIQP fails"""
        assignments = {i: [] for i in range(num_agents)}
        if not tasks:
            return assignments

        for task in tasks:
            task_type = task.get('task_type', 'Wait')
            
            capable_agents = [
                agent_id for agent_id, caps in agent_capabilities.items() if task_type in caps
            ]
            
            if capable_agents:
                # assign to least-loaded capable agent
                chosen_agent = min(capable_agents, key=lambda aid: len(assignments[aid]))
                assignments[chosen_agent].append(task)
            else:
                # if no capable agent, assign to least-loaded agent
                chosen_agent = min(assignments.keys(), key=lambda aid: len(assignments[aid]))
                assignments[chosen_agent].append(task)
        
        return assignments
        
    def build_phase_specific_t_matrix(
        self, 
        phase_info: Dict[str, Any],
        base_t_matrix: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[int], List[str]]:
        """
        Builds a T matrix for the specific task *instances* in the current phase.
        Each row in the returned matrix corresponds to a task instance.
        The matrix dimensions will be (num_task_instances, num_capabilities).

        Returns:
            Tuple[instance_t_matrix, active_task_indices, active_task_types]
            - instance_t_matrix: The instance-specific T matrix [num_instances x num_capabilities].
            - active_task_indices: A list of indices for the instances, i.e., range(num_instances).
            - active_task_types: A list of task types corresponding to each row/instance.
        """
        if base_t_matrix is None:
            base_t_matrix = self.BASE_TASK_CAPABILITY_REQUIREMENTS.copy()
        
        if not isinstance(base_t_matrix, np.ndarray):
            base_t_matrix = np.array(base_t_matrix)

        phase_tasks = phase_info.get('tasks', [])
        
        task_type_to_index = {
            'Navigate': 0, 'Explore': 1, 'Pick': 2, 'Place': 3,
            'Open': 4, 'Close': 5, 'Clean': 6, 'Fill': 7,
            'Pour': 8, 'PowerOn': 9, 'PowerOff': 10, 'Rearrange': 11, 'Wait': 12
        }
        
        num_instances = len(phase_tasks)
        num_capabilities = base_t_matrix.shape[1]
        
        # If no tasks, create a fallback for a single "Wait" task.
        if num_instances == 0:
            wait_idx = task_type_to_index['Wait']
            instance_t_matrix = base_t_matrix[wait_idx, :].reshape(1, -1)
            return instance_t_matrix, [0], ['Wait']

        instance_t_matrix = np.zeros((num_instances, num_capabilities))
        instance_task_types = []

        for i, task in enumerate(phase_tasks):
            task_type = task.get('task_type', 'Wait')
            instance_task_types.append(task_type)

            if task_type in task_type_to_index:
                type_idx = task_type_to_index[task_type]
                base_row = base_t_matrix[type_idx, :].copy()

                # Adjust requirement based on task priority
                priority = task.get('priority', 3)
                priority_multiplier = 1.0 + (priority - 3) * 0.1
                priority_multiplier = np.clip(priority_multiplier, 0.5, 2.0)
                
                instance_t_matrix[i, :] = base_row * priority_multiplier
            else:
                # Fallback for unknown task type, treat as 'Wait'
                wait_idx = task_type_to_index['Wait']
                instance_t_matrix[i, :] = base_t_matrix[wait_idx, :]

        active_task_indices = list(range(num_instances))
        
        # print(f"[DEBUG] Built instance-specific T matrix.")
        # print(f"  Matrix shape: {instance_t_matrix.shape}")
        # print(f"  Task instances: {len(phase_tasks)}")
        # print(f"  Instance task types: {instance_task_types}")
        # print(f"###################################Instance task matrix: {instance_t_matrix}")

        return instance_t_matrix, active_task_indices, instance_task_types

    def structured_decompose_task(
        self,
        instruction: str,
        env_interface: "EnvironmentInterface",
        llm_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to decompose instruction into structured subtask list.
        """
        if not self.llm_client:
            raise ValueError("LLM client not initialized. Cannot decompose task.")

        # 1. prepare object and agent state info
        objects_prompt_string = self._get_objects_string_for_prompt(env_interface)
        current_world_state = self.last_world_state or self.extract_world_state(env_interface)
        agent_info_string = self._get_agent_status_for_prompt(current_world_state)

        # 2. build structured prompt
        world_state_desc = f"Environment State:\n{objects_prompt_string}\n\nAgent Status:\n{agent_info_string}"
        prompt_template = TaskDecompositionPrompt("task_decomposition", get_llm_config())
        structured_prompt = prompt_template(instruction, world_state_desc)

        # 3. call LLM
        messages = [
            {"role": "system", "content": "You are a precise task decomposition assistant. Return only valid JSON."},
            {"role": "user", "content": structured_prompt}
        ]
        api_params = {
            "model": llm_config.get("gpt_version", "moonshot-v1-32k"),
            "messages": messages,
            "max_tokens": llm_config.get("max_tokens", 1500),
            "temperature": 0.1
        }

        try:
            response = self.llm_client.chat.completions.create(**api_params)
            decomposed_text = response.choices[0].message.content.strip()
            
            structured_tasks = self._parse_decomposition_response(decomposed_text)
            
            if structured_tasks:
                print(f"DEBUG: Successfully parsed {len(structured_tasks)} structured subtasks")
                return structured_tasks
            else:
                print("Warning: Failed to parse structured subtasks, using simple decomposition fallback.")
                return self._simple_task_decomposition(instruction)
                
        except Exception as e:
            print(f"Error calling LLM for structured task decomposition: {e}")
            raise

    def update_miqp_matrices(
        self,
        scenario_config: Union[Dict[str, Any], "ScenarioConfigTask"],
        phase_t_matrix: np.ndarray,
        structured_subtasks: List[Dict[str, Any]],
        world_state: Dict[str, Any]
    ) -> None:
        """
        Update MIQP parameter matrices (A, T, ws) from task analysis.
        Builds T matrix by rules directly; removed unstable LLM matrix generation.
        """
        try:
            print("=== [DEBUG-LYP-v3]: perception_connector.update_miqp_matrices called TODO Update ===")
            # 1. build T matrix directly
            # current_phase_info = {'tasks': structured_subtasks}
            # phase_t_matrix, _, _ = self.build_phase_specific_t_matrix(current_phase_info)
            update_param_value(scenario_config, 'T', phase_t_matrix)
            # print("DEBUG: MIQP matrix 'T' updated successfully via rule-based builder.")
            
            # 2. update A and ws matrices (default or simple heuristics)
            # A (capability matrix) - usually fixed unless specially adjusted
            # assumed not to need frequent updates; use standard values
            base_a_matrix = self.BASE_ROBOT_CAPABILITIES.copy()
            update_param_value(scenario_config, 'A', base_a_matrix)
            # print("DEBUG: MIQP matrix 'A' set to default value.")

            # ws (task weights) - may be set from task priority
            num_total_tasks = 13  # number of rows in T matrix
            task_weights = np.ones(num_total_tasks)
            task_type_to_index = {
                'Navigate': 0, 'Explore': 1, 'Pick': 2, 'Place': 3,
                'Open': 4, 'Close': 5, 'Clean': 6, 'Fill': 7,
                'Pour': 8, 'PowerOn': 9, 'PowerOff': 10, 'Rearrange': 11, 'Wait': 12
            }
            for task in structured_subtasks:
                task_type = task.get('task_type')
                if task_type in task_type_to_index:
                    idx = task_type_to_index[task_type]
                    # give higher weight to higher-priority tasks
                    priority = task.get('priority', 3)
                    task_weights[idx] = 1.0 + (priority - 3) * 0.2
            
            # ensure ws is a list
            update_param_value(scenario_config, 'ws', task_weights.tolist())
            # print("DEBUG: MIQP weights 'ws' updated based on task priorities.")

        except Exception as e:
            print(f"[ERROR] Failed to update MIQP matrices: {e}")
            # Fallback to defaults
            update_param_value(scenario_config, 'T', self.BASE_TASK_CAPABILITY_REQUIREMENTS.copy())
            update_param_value(scenario_config, 'A', self.BASE_ROBOT_CAPABILITIES.copy())
            update_param_value(scenario_config, 'ws', np.ones(13).tolist())


    def pre_update_scenario_params(
        self,
        scenario_config: Union[Dict[str, Any], "ScenarioConfigTask"],
        world_state: Dict[str, Any]
    ) -> None:
        """
        Update scenario params from world state (e.g. whether an agent holds an object).
thismanipulationPrior to task decomposition and planning.

        Args:
            scenario_config: ScenarioConfigTask instance or param dict.
            world_state: world state dict from extract_world_state.
        """
        agent_names = list(world_state.get('agent_poses', {}).keys())
        
        for obj_name, obj_info in world_state.get('object_positions', {}).items():
            if obj_info and obj_info.get('parent') in agent_names:
                holding_robot_id = obj_info['parent']
                update_param_value(scenario_config, 'is_holding', True)
                update_param_value(scenario_config, 'holding_robot_id', holding_robot_id)
                print(f"DEBUG: Pre-update: Agent '{holding_robot_id}' is holding '{obj_name}'.")
                return

            update_param_value(scenario_config, 'is_holding', False)
            update_param_value(scenario_config, 'holding_robot_id', None)

    def update_scenario_from_actions(
        self,
        scenario_config: Union[Dict[str, Any], "ScenarioConfigTask"],
        world_state: Dict[str, Any],
        high_level_actions: Dict[int, Tuple[str, str, Optional[str]]]
    ) -> None:
        """
        Delegates the update of scenario parameters to the ExecutionManager.
        This method is now a wrapper for backward compatibility and future flexibility.
        """
        # In a fully refactored system, the planner would call ExecutionManager directly.
        # This acts as a bridge during refactoring.
        from habitat_llm.planner.HRCS.plan_module.execution_manager import ExecutionManager
        
        execution_manager = ExecutionManager()
        execution_manager.update_scenario_for_execution(
            scenario_config,
            world_state,
            high_level_actions
        )

    def update_scenario_for_phase_execution(
        self,
        scenario_config: Union[Dict[str, Any], "ScenarioConfigTask"],
        current_phase_tasks: List[Dict[str, Any]],
        world_state: Dict[str, Any]
    ) -> None:
        """
        Update global task vars in scenario_config for current planning phase task instances.
        Ensure MIQP solver uses correct target position per task instance.
        """
        if not hasattr(scenario_config, 'update_global_task_var'):
            print("[WARNING] scenario_config object does not support update_global_task_var. Skipping phase execution update.")
            return

        print("[DEBUG] Updating global task variables for the current phase execution...")

        # Define global vars to update per task type and source keys in task dict
        var_map = {
            'Navigate':  {'p_goal': 'target'},
            'Pick':      {'target_object_position': 'target'},
            'Place':     {'target_receptacle_position': 'target'},
            'Open':      {'target_furniture_position': 'target'},
            'Close':     {'target_furniture_position': 'target'},
            'Clean':     {'target_object_position': 'target'},
            'Fill':      {'target_container_position': 'target'},
            'Pour':      {'target_receptacle_position': 'target'},
            'PowerOn':   {'target_device_position': 'target'},
            'PowerOff':  {'target_device_position': 'target'},
            'Rearrange': {
                'target_object_position': 'target',
                'target_receptacle_position': 'target_location'  # assume LLM decomposition provides target_location
            }
        }
        
        # iterate each task instance in current phase
        for task in current_phase_tasks:
            task_type = task.get('task_type')
            if task_type not in var_map:
                continue

            # get all var mappings for current task type
            mappings = var_map[task_type]
            for var_name, task_key in mappings.items():
                target_name = task.get(task_key)
                if not target_name:
                    continue  # skip if task dict lacks corresponding key (e.g. target_location)

                # look up target position in world state
                target_pos_data = None
                if world_state:
                    if 'object_positions' in world_state and target_name in world_state['object_positions']:
                        pos_info = world_state['object_positions'][target_name]
                        if pos_info and 'position' in pos_info:
                            target_pos_data = pos_info['position']
                    elif 'furniture_positions' in world_state and target_name in world_state['furniture_positions']:
                        pos_info = world_state['furniture_positions'][target_name]
                        if pos_info and 'position' in pos_info:
                            target_pos_data = pos_info['position']

                if target_pos_data:
                    # most task functions expect 2D [x, z] coordinates
                    pos_2d = np.array(target_pos_data)[[0, 2]]
                    scenario_config.update_global_task_var(var_name, pos_2d)
                    print(f"  - Updated global var '{var_name}' for task '{task_type}' with pos {pos_2d} from target '{target_name}'")
                else:
                    print(f"  [Updater] WARNING: Could not find position for target '{target_name}' (from key '{task_key}') in task '{task_type}'")


     # --- LLM task decomposition and analysis ---
    def _get_objects_string_for_prompt(self, env_interface: "EnvironmentInterface") -> str:
        """Format object list from WorldGraph for LLM prompt."""
        full_graph = env_interface.full_world_graph
        objects_for_prompt = []

        for obj_node in full_graph.get_all_objects():
            try:
                category = obj_node.category
            except AttributeError:
                category = "Unknown"
            objects_for_prompt.append({'name': obj_node.name, 'category': category or "Unknown"})
        
        for furn_node in full_graph.get_all_furnitures():
            try:
                category = furn_node.category
            except AttributeError:
                category = "Furniture"
            objects_for_prompt.append({'name': furn_node.name, 'category': category or "Furniture"})
        
        return f"objects = {json.dumps(objects_for_prompt)}"

    def _get_agent_status_for_prompt(self, world_state: Dict[str, Any]) -> str:
        """Format agent state from world state for LLM prompt."""
        if not world_state:
            return "No agent status available"
            
        agent_status_lines = []
        agent_poses = world_state.get('agent_poses', {})
        held_objects_by_agent = {agent: [] for agent in agent_poses}
        
        for obj_name, obj_info in world_state.get('object_positions', {}).items():
            if obj_info and obj_info.get('parent') in agent_poses:
                held_objects_by_agent[obj_info['parent']].append(obj_name)
        
        for agent_name, pose_info in agent_poses.items():
            pos_str = "Position unknown"
            if pose_info and 'position' in pose_info:
                pos = pose_info['position']
                pos_str = f"Position [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
            
            held_str = ", holding: " + ", ".join(held_objects_by_agent[agent_name]) if held_objects_by_agent.get(agent_name) else ", hands free"
            agent_status_lines.append(f"- {agent_name}: {pos_str}{held_str}")
        
        return "\n".join(agent_status_lines) if agent_status_lines else "No agents found."

    def _parse_decomposition_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse structured task decomposition JSON from LLM."""
        try:
            json_text = extract_json_from_text(response_text, list)
            if not json_text:
                print(f"Warning: No valid JSON array found in LLM response.")
                return []

            structured_tasks = json.loads(json_text)
            validated_tasks = []
            required_fields = ['task_type', 'target', 'description']
            for i, task in enumerate(structured_tasks):
                if not isinstance(task, dict) or not all(field in task for field in required_fields):
                    print(f"Warning: Task {i} is malformed: {task}")
                    continue
                
                validated_task = {
                    'task_type': task['task_type'],
                    'target': task['target'],
                    'description': task['description'],
                    'priority': task.get('priority', 3),
                    'estimated_duration': task.get('estimated_duration', 5.0),
                    'preferred_agent': task.get('preferred_agent'),
                    'prerequisites': task.get('prerequisites', [])
                }
                validated_tasks.append(validated_task)
            
            return validated_tasks
            
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: JSON parsing for decomposition failed: {e}")
            return []

    def _simple_task_decomposition(self, instruction: str) -> List[Dict[str, Any]]:
        """Rule-based simple decomposition when LLM decomposition fails."""
        tasks = []
        if "pick" in instruction.lower() and "place" in instruction.lower():
            tasks.extend([
                {'task_type': 'Explore', 'target': 'environment', 'description': 'Explore to find targets', 'priority': 2},
                {'task_type': 'Navigate', 'target': 'object_location', 'description': 'Navigate to object', 'priority': 3},
                {'task_type': 'Pick', 'target': 'target_object', 'description': 'Pick up object', 'priority': 4},
                {'task_type': 'Navigate', 'target': 'receptacle_location', 'description': 'Navigate to receptacle', 'priority': 3},
                {'task_type': 'Place', 'target': 'target_receptacle', 'description': 'Place object', 'priority': 4},
            ])
        else:
            tasks.append({'task_type': 'Explore', 'target': 'environment', 'description': instruction, 'priority': 3})

        # add defaults for all simple tasks
        for task in tasks:
            task.setdefault('estimated_duration', 10.0)
            task.setdefault('preferred_agent', None)
            task.setdefault('prerequisites', [])
        return tasks
    
    def _initialize_base_T_matrix(self) -> np.ndarray:
        """Return base task-capability requirement matrix T"""
        return self.BASE_TASK_CAPABILITY_REQUIREMENTS.copy()
    
    def _init_llm_client(self,
                        llm_client: Optional[Any] = None,
                        api_key_filename: Optional[str] = None,
                        llm_base_url: Optional[str] = "https://api.moonshot.cn/v1"):
        """Initialize LLM client"""
        if self.llm_client is not None:
            return
        try:
            api_key = os.environ.get("MOONSHOT_API_KEY")
            key_source = "MOONSHOT_API_KEY environment variable"
            if not api_key and api_key_filename:
                api_key_path = Path(api_key_filename + '.txt')
                if not api_key_path.exists():
                    api_key_path = Path(api_key_filename)
                if api_key_path.exists():
                    api_key = api_key_path.read_text().strip()
                    key_source = f"API key file {api_key_path}"
            if not api_key:
                print(
                    "PerceptionConnector: No API key found. Set MOONSHOT_API_KEY or provide "
                    f"{api_key_filename}.txt. Task decomposition will not be available."
                )
                return
            self.llm_client = openai.OpenAI(
                api_key=api_key,
                base_url=llm_base_url,
            )
            print(f"PerceptionConnector: LLM client initialized using {key_source} and base URL {llm_base_url}")
        except Exception as e:
            print(f"Error initializing LLM client in PerceptionConnector: {e}")
        return
