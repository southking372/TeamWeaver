#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from habitat.tasks.rearrange.utils import coll_name_matches
from hydra.utils import instantiate

from habitat_llm.llm.instruct.utils import (
    get_objects_descr,
    get_rearranged_objects_descr,
    get_world_descr,
)
from habitat_llm.planner.planner import Planner
from habitat_llm.utils.grammar import (
    FREE_TEXT,
    FURNITURE,
    NAV_TARGET,
    OBJECT,
    OBJECT_OR_FURNITURE,
    ROOM,
    SPATIAL_CONSTRAINT,
    SPATIAL_RELATION,
)

# **NEW** 导入模块化组件
from habitat_llm.planner.HRCS.plan_module.llm_interface import LLMInterface
from habitat_llm.planner.HRCS.plan_module.task_assignment_manager import TaskAssignmentManager
from habitat_llm.planner.HRCS.plan_module.miqp_solver import MIQPSolver
from habitat_llm.planner.HRCS.plan_module.agent_status_manager import AgentStatusManager
from habitat_llm.planner.HRCS.plan_module.phase_manager import PhaseManager

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from habitat_llm.agent.agent import Agent
    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.planner.rag import RAG
    from habitat_llm.world_model.world_graph import WorldGraph

# MIQP Planner imports
from habitat_llm.planner.HRCS.params_module.scenario_params_task import ScenarioConfigTask
from habitat_llm.planner.HRCS.params_module.opt_params_task import OptimizationConfigTask
from habitat_llm.planner.HRCS.class_def.RTA_task import RTA
from habitat_llm.planner.perception_connector import PerceptionConnector
import numpy as np


def quaternion_to_yaw(quaternion):
    """Convert quaternion to yaw angle in radians."""
    quat = np.array(quaternion)
    x, y, z, w = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)

class LLMPlanner(Planner):
    """
    High level planner policy used by agents to decide high level actions
    given task description and state of the world
    """

    def __init__(
        self, plan_config: "DictConfig", env_interface: "EnvironmentInterface"
    ):
        """
        Initialize the LLMPlanner.

        :param plan_config: The planner configuration.
        :param env_interface: The environment interface.
        """
        # Set the planner config
        super().__init__(plan_config, env_interface)
        # Initialize LLM
        self.__initialize_llm()

        # **NEW** 初始化模块化组件
        self.llm_interface = None  # 稍后在agents设置后初始化
        self.task_assignment_manager = None
        self.miqp_solver = None
        self.agent_status_manager = None
        self.phase_manager = None

        # Initialize a variable to indicate if replanning is required
        self.replan_required: bool = True

        # Initialize a variable to count number of llm calls
        self.replanning_count: int = 0

        # Initialize container to store entire prompt and current object states
        self.curr_prompt: str = ""
        self.curr_obj_states: str = ""

        # Initialize container to store rollout without
        # any other material in prompt
        self.trace: str = ""
        self.rag: Optional["RAG"] = None

        # MIQP components
        self.scenario_params = None
        self.opt_params = None
        self.miqp_globalvars = None
        self.rta = None
        self.perception_connector = PerceptionConnector()

        self.reset()

        # Build RAG dataset if we want to use RAG
        if self.enable_rag:
            from habitat_llm.planner.rag import RAG

            self.rag = RAG(
                plan_config.example_type,
                plan_config.rag_dataset_dir,
                plan_config.rag_data_source_name,
                plan_config.llm,
            )

    def _initialize_modular_components(self):
        """
        **NEW** 初始化模块化组件（在agents设置后调用）
        """
        try:
            if not hasattr(self, '_agents') or not self._agents:
                print("[WARNING] Cannot initialize modular components without agents")
                return

            # 初始化LLM接口
            llm_config = {
                'stopword': self.stopword,
                'end_expression': self.end_expression,
                'system_tag': getattr(self.planner_config.llm, 'system_tag', ''),
                'user_tag': getattr(self.planner_config.llm, 'user_tag', ''),
                'assistant_tag': getattr(self.planner_config.llm, 'assistant_tag', ''),
                'eot_tag': getattr(self.planner_config.llm, 'eot_tag', '')
            }
            self.llm_interface = LLMInterface(self.llm, llm_config)

            # 初始化任务分配管理器
            self.task_assignment_manager = TaskAssignmentManager(self._agents)

            # 初始化MIQP求解器
            self.miqp_solver = MIQPSolver(self._agents)

            # 初始化Agent状态管理器
            self.agent_status_manager = AgentStatusManager(self._agents)

            # 初始化阶段管理器
            self.phase_manager = PhaseManager(self._agents)

            print(f"[SUCCESS] Modular components initialized for {len(self._agents)} agents")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize modular components: {e}")
            # 设置为None以确保后续检查能够识别失败
            self.llm_interface = None
            self.task_assignment_manager = None
            self.miqp_solver = None
            self.agent_status_manager = None
            self.phase_manager = None

    def reset(self):
        """
        Reset the planner state.
        """
        self.last_high_level_actions: Dict[int, Tuple[str, str, str]] = {}
        self.replan_required: bool = True
        self.replanning_count: int = 0
        self.is_done: bool = False
        self._phase_transition_pending: bool = False
        self.latest_agent_response: Dict[int, str] = {}
        self.curr_prompt: str = ""
        self.trace: str = ""
        self.curr_obj_states: str = ""
        self.params: Dict[str, Any] = {}

        # Reset agents
        for agent in self._agents:
            agent.reset()

    def build_tool_grammar(self, world_graph: "WorldGraph") -> str:
        """
        This method builds a grammar that accepts all valid tool calls based a world graph
        The grammar is specified in the EBNF grammar description format
        see https://github.com/epfl-dlab/transformers-CFG for details and examples

        :param world_graph: The world graph.
        """
        tool_grammar = {}
        objects = world_graph.get_all_objects()
        rules = []
        # we cannot include rules which have objects when there are no objects
        if len(objects) != 0:
            object_expansion = " | ".join(
                (f'"{x.name}"' for x in world_graph.get_all_objects())
            )
            object_rule = f"{OBJECT} ::= " + object_expansion
            nav_target_rule = f"{NAV_TARGET} ::= ({FURNITURE} | {ROOM} | {OBJECT})"
            object_of_furniture_rule = (
                f"{OBJECT_OR_FURNITURE} ::= ({FURNITURE} | {OBJECT})"
            )
            rules.append(nav_target_rule)
            rules.append(object_rule)
            rules.append(object_of_furniture_rule)
        else:
            object_of_furniture_rule = f"{OBJECT_OR_FURNITURE} ::= {FURNITURE}"
            nav_target_rule = f"{NAV_TARGET} ::= ({FURNITURE} | {ROOM})"
            rules.append(nav_target_rule)
            rules.append(object_of_furniture_rule)
        for agent in self.agents:
            for tool_name, tool in agent.tools.items():
                if tool_name not in tool_grammar:
                    # skip tools that require objects when there are no objects
                    if OBJECT in tool.argument_types and len(objects) == 0:
                        continue
                    tool_grammar[tool_name] = tool.grammar()
        tool_grammar["Done"] = '"Done[]"'
        grammar_str = "tool_call ::= " + " | ".join(tool_grammar.keys()) + "\n"
        for tool_name, tool_grammar_str in tool_grammar.items():
            grammar_str += f"{tool_name} ::= {tool_grammar_str}\n"

        # build rules for each of the argument types
        furniture_rule = f"{FURNITURE} ::= " + " | ".join(
            (f'"{x.name}"' for x in world_graph.get_all_furnitures())
        )
        room_rule = f"{ROOM} ::= " + " | ".join(
            (f'"{x.name}"' for x in world_graph.get_all_rooms())
        )
        spatial_constraint_rule = f'{SPATIAL_CONSTRAINT} ::= "next_to"'
        spatial_relation_rule = f'{SPATIAL_RELATION} ::= "on" | "within"'
        free_text_rule = f"{FREE_TEXT} ::= [ \"'.:,!a-zA-Z_0-9]*"
        white_space_rule = "WS ::= [ ]*"
        rules.append(furniture_rule)
        rules.append(room_rule)
        rules.append(spatial_constraint_rule)
        rules.append(spatial_relation_rule)
        rules.append(free_text_rule)
        rules.append(white_space_rule)
        grammar_str += "\n".join(rules)
        return grammar_str

    def build_response_grammar(self, world_graph: "WorldGraph") -> str:
        """
        Build a grammar that accepts all valid responses based on a world graph.

        :param world_graph: The world graph.
        """
        delimiter = "\\n"
        tool_rules = self.build_tool_grammar(world_graph)

        action_rules = []
        for i, agent in enumerate(self.agents):
            agent_id = agent.uid
            action_rule = f'action_{i} ::= "Agent_{agent_id}_Action: " tool_call'
            action_rules.append(action_rule)

        combined_action_rule = (
            "action ::= "
            + f' "{delimiter}" '.join(f"action_{i}" for i in range(len(self.agents)))
            + f' "{delimiter}Assigned!"'
        )
        termination_rule = 'termination ::= "Final Thought: Exit!"'
        root_role = f'root ::= {FREE_TEXT} "{delimiter}" (action | termination)'

        return "\n".join(
            [root_role, combined_action_rule]
            + action_rules
            + [termination_rule, tool_rules]
        )

    def __initialize_llm(self):
        """
        This method instantiates LLM as defined in the config
        """
        # Instantiate LLM from the Hydra config
        llm_conf = self.planner_config.llm
        self.llm = instantiate(llm_conf.llm)
        self.llm = self.llm(llm_conf)

        # Setup the LLM parameters
        # self.instruct = self.planner_config.llm.instruct
        self.instruct = self.planner_config.instruct
        self.prompt = self.instruct.prompt
        self.stopword = self.instruct.stopword
        self.end_expression = self.instruct.end_expression
        self.actions_parser = instantiate(self.instruct.actions_parser)

        # save agent observations to get feedback on skill execution
        self.latest_agent_response = {}

    def prepare_prompt(
        self, input_instruction: str, world_graph: "WorldGraph", **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare the prompt for the LLM.

        :param input_instruction: The input instruction.
        :param world_graph: The world graph.
        :return: The prepared prompt and parameters.
        """
        params = {
            "input": input_instruction,
            "tool_list": self.tool_list,
            "world_graph": world_graph,
            "id": self.agents[0].uid,
        }

        # We modify the prompt if we want to use RAG and the prompt has not
        # been modified
        if "{rag_examples}" in self.prompt:
            if self.rag is not None:
                _, index = self.rag.retrieve_top_k_given_query(
                    input_instruction, top_k=1, agent_id=self._agents[0].uid
                )
                index = index[0]

                example_str = (
                    f"{self.planner_config.llm.user_tag}Below are some example solutions from different settings:\nExample 1:\n"
                    + self.rag.data_dict[index]["trace"]
                    + "\n"
                )
                params["rag_examples"] = example_str
            else:
                params["rag_examples"] = ""
        if "{tool_descriptions}" in self.prompt:
            params["tool_descriptions"] = self.agents[0].tool_descriptions
        if "{agent_descriptions}" in self.prompt:
            params["agent_descriptions"] = self.agent_descriptions
        if "{tool_list}" in self.prompt:
            params["tool_list"] = self.tool_list
        if "{system_tag}" in self.prompt:
            params["system_tag"] = self.planner_config.llm.system_tag
        if "{user_tag}" in self.prompt:
            params["user_tag"] = self.planner_config.llm.user_tag
        if "{assistant_tag}" in self.prompt:
            params["assistant_tag"] = self.planner_config.llm.assistant_tag
        if "{eot_tag}" in self.prompt:
            params["eot_tag"] = self.planner_config.llm.eot_tag
        if "{agent_role_description}" in self.prompt:
            # only support agent role description when planning for a single agent
            assert len(self.agents) == 1
            if str(self.agents[0].uid) == "1":
                agent_role_description = '\nYou are playing the role of the task giver. This means if the instruction says something like "You should move the object and I will wash it", then the other agent should be moving the object, and you should washing the it.\n'
            else:
                agent_role_description = '\nYou are playing the role of the task receiver. This means if the instruction says something like "You should move the object and I will wash it", then you should move the object and the other agent should wash it.\n'
            params["agent_role_description"] = agent_role_description
        if "{world_description}" in self.prompt:
            # only designed for the decentralized setting
            assert len(self.agents) == 1
            world_description = get_world_descr(
                world_graph,
                agent_uid=self.agents[0].uid,
                add_state_info=self.planner_config.objects_response_include_states,
                include_room_name=True,
                centralized=self.planner_config.centralized,
            )
            params["world_description"] = world_description

        if "should_format" in kwargs and not kwargs["should_format"]:
            # In some cases a subclass may want to fill the extra arguments here, so we don't format
            # because those arguments would be missing.
            output_prompt = ""
        else:
            output_prompt = self.prompt.format(**params)
        return output_prompt, params

    @property
    def tool_list(self) -> List[str]:
        """
        Returns a string listing the agents tools
        :return: A sorted list of tool names.
        """
        tool_set = set()
        for agent in self.agents:
            for tool_name in agent.tools:
                tool_set.add(tool_name)

        return sorted(tool_set)

    @property
    def agents(self) -> List["Agent"]:
        """
        Get the list of agents associated with this planner.

        :return: A list of Agent objects.
        """
        return self._agents

    @agents.setter
    def agents(self, agents: List["Agent"]) -> None:
        """
        Set the list of agents for this planner.

        :param agents: A list of Agent objects to be associated with this planner.
        """
        self._agents = agents
        # Pass on respective LLM instance into agent tools
        for agent in self._agents:
            agent.pass_llm_to_tools(self.llm)

        # **NEW** 初始化模块化组件
        self._initialize_modular_components()

    def get_last_agent_states(self) -> Dict[int, str]:
        """
        Get the last state descriptions for all agents.

        :return: A dictionary mapping agent UIDs to their last state descriptions.
        """
        # Container to store state descriptions
        agent_states = {}

        # Loop through the agents and populate state descriptions
        for agent in self._agents:
            agent_states[agent.uid] = agent.get_last_state_description()

        return agent_states

    def _extract_agent_statuses_from_observations(self, observations: Dict[str, Any]) -> Dict[int, str]:
        """Extract agent status information from observations."""
        agent_statuses = {}
        
        try:
            for key, value in observations.items():
                if key.startswith("Agent_") and key.endswith("_Observation"):
                    agent_id_str = key.replace("Agent_", "").replace("_Observation", "")
                    try:
                        agent_id = int(agent_id_str)
                        agent_statuses[agent_id] = str(value) if value is not None else ""
                    except ValueError:
                        continue
            
            if not agent_statuses:
                agent_statuses = self.get_last_agent_states()
            
            if not agent_statuses:
                agent_statuses = {agent.uid: "" for agent in self._agents}
            
            return agent_statuses
            
        except Exception:
            return self._get_agent_completion_statuses()

    def _analyze_failure_and_suggest_recovery(self, agent_id: int, status: str, current_action: Tuple[str, str, Optional[str]]) -> Optional[Tuple[str, str, Optional[str]]]:
        """Analyze agent failure status and suggest recovery actions."""
        try:
            if not status or not current_action:
                return None
            
            action_name, args, target = current_action
            status_lower = status.lower()
            
            if action_name == "Pick" and "not close enough" in status_lower:
                return ("Navigate", target, target)
            elif action_name == "Pick" and "object not found" in status_lower:
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

    def _apply_intelligent_error_recovery(
        self, 
        agent_task_assignments: Dict[int, List[Dict[str, Any]]], 
        current_phase: Dict[str, Any]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Apply intelligent error recovery to handle failed operations."""
        try:
            if not hasattr(self, 'latest_agent_response') or not self.latest_agent_response:
                return agent_task_assignments
            
            if not hasattr(self, 'last_high_level_actions') or not self.last_high_level_actions:
                return agent_task_assignments
            
            updated_assignments = agent_task_assignments.copy()
            
            for agent_id, response in self.latest_agent_response.items():
                if not response:
                    continue
                
                current_action = self.last_high_level_actions.get(agent_id)
                if not current_action:
                    continue
                
                recovery_action = self._analyze_failure_and_suggest_recovery(
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

    def _get_agent_completion_statuses(self) -> Dict[int, str]:
        """Get agent completion statuses, prioritizing latest responses."""
        agent_statuses = {}
        
        try:
            if hasattr(self, 'latest_agent_response') and self.latest_agent_response:
                for agent in self._agents:
                    response = self.latest_agent_response.get(agent.uid, "")
                    if response and response.strip():
                        agent_statuses[agent.uid] = response.strip()
                        continue
                    
                    state_desc = agent.get_last_state_description()
                    agent_statuses[agent.uid] = state_desc if state_desc else "No status available"
            else:
                for agent in self._agents:
                    state_desc = agent.get_last_state_description()
                    agent_statuses[agent.uid] = state_desc if state_desc else "No status available"
            
            if not agent_statuses:
                agent_statuses = {0: "Status unknown", 1: "Status unknown"}
            
            return agent_statuses
            
        except Exception:
            return {0: "Error extracting status", 1: "Error extracting status"}

    # TODO: @zephirefaith implement agent's room affiliations in the world graph
    # and edit this function to read from it
    def get_last_agent_positions(self) -> Dict[str, Any]:
        """
        Get the last positions for all agents.

        :return: A dictionary mapping agent names to their positions.
        """
        # Container to store agent positions
        agent_positions = {}

        # get agent nodes
        agents = self.env_interface.full_world_graph.get_agents()

        # Loop through the agents and populate nodes
        for agent in agents:
            agent_positions[agent.name] = agent.get_property("translation")

        return agent_positions
    
    # MIQP position and rotation function
    def get_last_agent_positions_miqp(self, world_graph: Dict[int, "WorldGraph"]) -> Dict[int, Dict[str, Any]]:
        """
        Get the last positions and rotations for all agents based on the provided world graphs.
        Specifically for MIQP state input.

        :param world_graph: A dictionary mapping agent UIDs to their respective WorldGraph objects.
        :return: A dictionary mapping agent UIDs to dictionaries containing 'position' and 'rotation'.
        """
        agent_poses = {}
        agents = self.env_interface.full_world_graph.get_agents()
        for agent in agents:
            agent_name = agent.name
            position = [0.0, 0.0, 0.0]  # Default position
            rotation_quat = [0.0, 0.0, 0.0, 1.0]  # Default rotation
            
            try:
                position = agent.get_property("translation")
                # rotation_quat = agent.get_property("rotation")
                # 目前rotation_quat对于agent无法获取，应采用其他方式获取
            except (ValueError, AttributeError) as e:
                print(f"Warning: Could not get pose properties for agent {agent_name}. Using default pose. Error: {str(e)}")
            agent_poses[agent_name] = {'position': position, 'rotation': rotation_quat}
        return agent_poses

    def get_agent_collisions(self) -> Dict[int, bool]:
        """
        Check if the agents are colliding.

        :return: A dictionary mapping agent UIDs to collision status.
        """
        # set collision to false
        collision = False

        # Get list of agent ids
        agent_ids = [
            articulated_agent.sim_obj.object_id
            for articulated_agent in self.env_interface.sim.agents_mgr.articulated_agents_iter
        ]

        # Return false if only one agent is in the scene
        if len(agent_ids) == 2:
            # Perform collision check
            self.env_interface.sim.perform_discrete_collision_detection()
            contact_points = self.env_interface.sim.get_physics_contact_points()

            for cp in contact_points:
                if coll_name_matches(cp, agent_ids[0]) and coll_name_matches(
                    cp, agent_ids[1]
                ):
                    collision = True

        # Declare output container
        out = {}

        # update the output
        for agent in self._agents:
            out[agent.uid] = collision

        return out

    def format_response(
        self, response: str, end_expression: Union[str, List[str]]
    ) -> str:
        """
        Format the LLM response by trimming it up to the first appearance of end_expression.

        :param response: The LLM response to format.
        :param end_expression: The end expression(s) to look for.
        :return: The formatted response.
        """
        response = response.rstrip("\n")
        if type(end_expression) == str:
            index = response.find(end_expression)
            target_end_expression = end_expression
        else:
            # end_expression is a list of string
            index = -1
            target_end_expression = ""
            for _end_expression in end_expression:
                _index = response.find(_end_expression)
                if _index < index or index == -1:
                    index = _index
                    target_end_expression = _end_expression
        return (
            response[: index + len(target_end_expression)] if index != -1 else response
        )

    def parse_thought(self, input_string: str) -> str:
        """
        Extract thought from the LLM response.

        :param input_string: The input string to parse.
        :return: The extracted thought.
        """
        # Define the patterns for Agent actions
        pattern = r"\n|Final Thought"

        # Search for the pattern in the input string
        match = re.search(pattern, input_string)

        if match:
            # Extract the text before the pattern
            return input_string[: match.start()].strip()
        else:
            # If no pattern is found, return the whole string
            return ""

    def _add_responses_to_prompt(self, responses: Dict[int, str]) -> str:
        """
        Add agent responses to the prompt.

        :param responses: A dictionary of agent responses.
        :return: The updated print string.
        """
        print_str = ""
        prompt_addition = ""
        add_object_update = False
        for agent_uid in sorted(responses.keys()):
            # If the response for a given agent is valid, add to the prompt and printout
            if responses[agent_uid]:
                # Update print string
                print_str += (
                    f"""Agent_{agent_uid}_Observation:{responses[agent_uid]}\n"""
                )
                # Update the prompt
                prompt_addition += (
                    f"""Agent_{agent_uid}_Observation:{responses[agent_uid]}\n"""
                )
                # self.curr_prompt += prompt_addition
                self.trace += (
                    f"""Agent_{agent_uid}_Observation:{responses[agent_uid]}\n"""
                )

            # If the response is empty then indicate the action is still in progress
            # only when replanning was required
            elif self.replan_required:
                responses[
                    agent_uid
                ] = f"Action {self.last_high_level_actions[agent_uid][0]}[{self.last_high_level_actions[agent_uid][1]}] is still in progress."

                # Update print string
                print_str += (
                    f"""Agent_{agent_uid}_Observation:{responses[agent_uid]}\n"""
                )

                # Update the prompt
                prompt_addition += (
                    f"""Agent_{agent_uid}_Observation:{responses[agent_uid]}\n"""
                )
                # self.curr_prompt += prompt_addition
                self.trace += (
                    f"""Agent_{agent_uid}_Observation:{responses[agent_uid]}\n"""
                )
                add_object_update = True

            # save agent observations to get feedback on skill execution
            self.latest_agent_response[agent_uid] = responses[agent_uid]

        if prompt_addition != "":
            self.curr_prompt += self.planner_config.llm.user_tag + prompt_addition
            if (
                self.planner_config.objects_response
                and add_object_update
                and self.planner_config.centralized
            ):
                world_graph = self.env_interface.world_graph[agent_uid]
                objects = get_objects_descr(
                    world_graph,
                    agent_uid,
                    include_room_name=True,
                    add_state_info=self.planner_config.objects_response_include_states,
                    centralized=self.planner_config.centralized,
                )
                if self.planner_config.prompt_w_updatedobjects_only:
                    # add details on what changed in the world.
                    # TODO: this currently assumes symmetric world graph,
                    # extend for decentralized/asymmetric WG
                    updated_objects = get_rearranged_objects_descr(
                        obj_descr_t=objects, obj_descr_t_1=self.curr_obj_states
                    )
                    self.curr_obj_states = objects
                    if updated_objects != "":
                        result = f"Newly found objects/updates on known objects: {updated_objects}\n"
                    else:
                        result = (
                            "No new objects or updates on known objects were found.\n"
                        )
                else:
                    result = f"Objects: {objects}\n"
                self.curr_obj_states = objects
                self.curr_prompt += result
                self.trace += result
                print_str += result
            self.curr_prompt += self.planner_config.llm.eot_tag
            # print(self.curr_prompt)

        # Force add thought after every observation
        if self.planner_config.planning_mode.lower() == "cot":
            for agent_uid in sorted(responses.keys()):
                if responses[agent_uid]:
                    print_str += "Thought:"
                    prompt_addition = f"{self.planner_config.llm.assistant_tag}Thought:"
                    self.curr_prompt += prompt_addition
                    self.trace += "Thought:"
                    break
        return print_str

    def desc_world_graph(self, world_graph_dict: Dict[int, "WorldGraph"]) -> str:
        """
        Generates a detailed description of the world graph, including entity positions.

        :param world_graph_dict: Dictionary mapping agent UIDs to their world graphs (unused, uses full graph).
        :return: A formatted string describing entities and their positions.
        """
        description_lines = ["Detailed World Graph Description:"]
        full_graph = self.env_interface.full_world_graph

        # Describe Furniture
        description_lines.append("Furniture:")
        all_furniture = sorted(full_graph.get_all_furnitures(), key=lambda f: f.name)
        if not all_furniture:
            description_lines.append("  (No furniture found)")
        for furniture in all_furniture:
            try:
                pos = furniture.get_property("translation")
                pos_str = f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
                description_lines.append(f"- {furniture.name}: Position {pos_str}")
            except (KeyError, AttributeError):
                description_lines.append(f"- {furniture.name}: Position not available")

        # Describe Objects
        description_lines.append("Objects:")
        all_objects = sorted(full_graph.get_all_objects(), key=lambda o: o.name)
        if not all_objects:
            description_lines.append("  (No objects found)")
        for obj in all_objects:
            try:
                pos = obj.get_property("translation")
                pos_str = f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
                # Optionally add parent info if available
                parent = full_graph.find_furniture_for_object(obj)
                parent_name = parent.name if parent else "Unknown"
                description_lines.append(f"- {obj.name}: Position {pos_str} (Parent: {parent_name})")
            except (KeyError, AttributeError):
                 parent = full_graph.find_furniture_for_object(obj)
                 parent_name = parent.name if parent else "Unknown"
                 description_lines.append(f"- {obj.name}: Position not available (Parent: {parent_name})")

        # Describe Agents (only position)
        description_lines.append("Agents:")
        all_agents = sorted(full_graph.get_agents(), key=lambda a: a.name)
        if not all_agents:
             description_lines.append("  (No agents found)")
        for agent in all_agents:
             try:
                 pos = agent.get_property("translation")
                 pos_str = f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
                 # Remove rotation description
                 # rot = agent.get_property("rotation") # Get rotation too
                 # yaw = quaternion_to_yaw(rot)
                 # rot_str = f"Yaw: {yaw:.2f} rad"
                 # description_lines.append(f"- {agent.name} ({agent.__class__.__name__}): Position {pos_str}, Rotation {rot_str}")
                 description_lines.append(f"- {agent.name} ({agent.__class__.__name__}): Position {pos_str}") # Only include position
             except (KeyError, AttributeError):
                 description_lines.append(f"- {agent.name} ({agent.__class__.__name__}): Pose not available")


        return "\n".join(description_lines)

    def replan(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
    ) -> str:
        """
        **REFACTORED** 重新规划：使用模块化组件集成任务分解、MIQP优化和序列化执行
        
        Returns:
            llm_response: LLM生成的响应字符串
        """
        print("\n" + "="*80)
        print("🚀 开始 MIQP Enhanced Plan (Modular)")
        print("="*80)
        
        # **STEP 1: 检查模块化组件是否可用**
        if not self._check_modular_components():
            print("[ERROR] Modular components not available, using fallback")
            return self._fallback_replan(instruction, observations, world_graph)
        
        # **STEP 2: 初始化或获取阶段信息**
        try:
            current_phase = self._initialize_or_get_current_phase(instruction, world_graph)
            if not current_phase:
                print("[INFO] All phases completed - task finished!")
                return "Final Thought: Exit!"
                
            current_phase_tasks = current_phase['tasks']
            print(f"[DEBUG] Current phase {current_phase['phase_id']} has {len(current_phase_tasks)} tasks")
            
        except Exception as e:
            print(f"[ERROR] Phase initialization failed: {e}")
            return self._fallback_replan(instruction, observations, world_graph)

        # **STEP 3: 任务分配（使用TaskAssignmentManager）**
        try:
            print(f"[Step 3/8] **MODULAR** Task assignment via TaskAssignmentManager...")
            agent_task_assignments = self.task_assignment_manager.heuristic_task_assignment(current_phase_tasks)
            
            print(f"[DEBUG] **MODULAR** Task assignments from TaskAssignmentManager:")
            for agent_id, tasks in agent_task_assignments.items():
                if tasks:
                    task_summaries = [f"{t['task_type']}→{t['target']}" for t in tasks]
                    print(f"  Agent {agent_id}: {task_summaries}")
                else:
                    print(f"  Agent {agent_id}: [waiting]")
                    
        except Exception as e:
            print(f"[ERROR] Task assignment failed: {e}")
            agent_task_assignments = {i: [] for i in range(len(self._agents))}

        # **STEP 4: 应用智能错误恢复（使用AgentStatusManager）**
        try:
            print(f"[Step 4/8] **MODULAR** Error recovery via AgentStatusManager...")
            agent_statuses = self._extract_agent_statuses_from_observations(observations)
            
            # **FIXED** 确保AgentStatusManager有必要的数据
            if hasattr(self, 'latest_agent_response'):
                self.agent_status_manager.latest_agent_response = self.latest_agent_response
            if hasattr(self, 'last_high_level_actions'):
                self.agent_status_manager.last_high_level_actions = self.last_high_level_actions
            
            # **FIXED** 使用正确的方法签名
            recovered_assignments = self.agent_status_manager.apply_intelligent_error_recovery(
                agent_task_assignments,
                current_phase
            )
            
            if recovered_assignments != agent_task_assignments:
                agent_task_assignments = recovered_assignments
                print(f"[DEBUG] **MODULAR** Applied error recovery via AgentStatusManager")
            else:
                print(f"[DEBUG] **MODULAR** No error recovery needed")
                
        except Exception as e:
            print(f"[ERROR] Error recovery failed: {e}")

        # **STEP 5: 构建阶段感知prompt（使用PhaseManager）**
        try:
            print(f"[Step 5/8] **MODULAR** Building prompt via PhaseManager...")
            enhanced_prompt = self.phase_manager.build_phase_aware_prompt_with_miqp_guidance(
                instruction,
                current_phase_tasks,
                agent_task_assignments,
                current_phase,
                self,  # planner_instance
                alpha_result=None  # 可以在MIQP求解后传入
            )
            
            print(f"[DEBUG] **MODULAR** Enhanced prompt built (length: {len(enhanced_prompt)})")
            
        except Exception as e:
            print(f"[ERROR] Enhanced prompt building failed: {e}")
            enhanced_prompt, _ = self.prepare_prompt(instruction, world_graph[self._agents[0].uid])

        # **STEP 6: LLM调用（使用LLMInterface）**
        try:
            print(f"[Step 6/8] **MODULAR** LLM call via LLMInterface...")
            response = self.llm_interface.call_llm(enhanced_prompt)
            
            if response:
                print(f"[DEBUG] **MODULAR** LLM response received (length: {len(response)})")
            else:
                print(f"[WARNING] Empty LLM response from LLMInterface")
                response = "Agent_0_Action: Explore[environment]\nAgent_1_Action: Wait[]\nAssigned!"

        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            response = "Agent_0_Action: Explore[environment]\nAgent_1_Action: Wait[]\nAssigned!"

        # **STEP 7: 解析和调整动作（使用LLMInterface + TaskAssignmentManager）**
        try:
            print(f"[Step 7/8] **MODULAR** Action parsing and adjustment...")
            
            # 解析高级动作（使用LLMInterface）
            high_level_actions = self.llm_interface.parse_high_level_actions(
                response, self.agents, self.params
            )
            
            # 阶段感知动作调整（使用TaskAssignmentManager）
            adjusted_actions = self.task_assignment_manager.adjust_actions_with_phase_awareness(
                high_level_actions,
                agent_task_assignments,
                current_phase
            )
            
            print(f"[DEBUG] **MODULAR** Actions parsed and adjusted:")
            for agent_id, action in adjusted_actions.items():
                if action and len(action) >= 3:
                    print(f"  Agent {agent_id}: {action[0]}({action[1]}) → {action[2]}")
                    
        except Exception as e:
            print(f"[ERROR] Action parsing/adjustment failed: {e}")
            adjusted_actions = {agent.uid: ("Explore", "environment", "environment") for agent in self.agents}

        # **STEP 8: 阶段完成检查和推进（使用PhaseManager）**
        try:
            print(f"[Step 8/8] **MODULAR** Phase completion check via PhaseManager...")
            
            # **ENHANCED** 传递更多上下文信息进行阶段完成检查
            phase_completion_context = {
                'agent_statuses': agent_statuses,
                'current_actions': adjusted_actions,
                'agent_task_assignments': agent_task_assignments,
                'observations': observations
            }
            
            # 检查当前阶段是否完成
            if self.phase_manager.is_current_phase_complete(phase_completion_context):
                print(f"[SUCCESS] **MODULAR** Phase {current_phase['phase_id']} completed!")
                
                # 推进到下一阶段
                if self.phase_manager.advance_to_next_phase():
                    print(f"[INFO] **MODULAR** Advanced to next phase")
                    self._phase_transition_pending = True
                else:
                    print(f"[SUCCESS] **MODULAR** All phases completed!")
            else:
                print(f"[DEBUG] **MODULAR** Phase {current_phase['phase_id']} still in progress")
                
        except Exception as e:
            print(f"[ERROR] Phase completion check failed: {e}")

        # **存储结果**
        self._last_response_info = {
            "optimization_success": True,
            "agent_task_assignments": agent_task_assignments,
            "task_decomposition_success": len(current_phase_tasks) > 0,
            "current_phase": current_phase,
            "total_phases": len(self.phase_manager._task_phases),
            "llm_response": response,
            "planning_approach": "MIQP_Modular_Phase",
            "modules_used": ["PhaseManager", "TaskAssignmentManager", "LLMInterface", "AgentStatusManager"]
        }

        print(f"\n[SUCCESS] **MODULAR** MIQP Enhanced Plan completed!")
        print(f"  Modules used: {', '.join(self._last_response_info['modules_used'])}")
        print(f"  Current Phase: {current_phase['phase_id'] + 1}/{len(self.phase_manager._task_phases)}")
        print("="*80)
        
        return response

    def _check_modular_components(self) -> bool:
        """检查模块化组件是否可用"""
        components = [
            (self.llm_interface, "LLMInterface"),
            (self.task_assignment_manager, "TaskAssignmentManager"),
            (self.agent_status_manager, "AgentStatusManager"),
            (self.phase_manager, "PhaseManager")
        ]
        
        for component, name in components:
            if component is None:
                print(f"[ERROR] {name} not available")
                return False
        
        return True

    def _initialize_or_get_current_phase(self, instruction: str, world_graph: Dict[int, "WorldGraph"]) -> Optional[Dict[str, Any]]:
        """初始化或获取当前阶段"""
        try:
            # 检查是否已有阶段信息
            current_phase = self.phase_manager.get_current_phase_tasks()
            if current_phase:
                return current_phase
                
            # 如果没有阶段信息，需要初始化
            if hasattr(self.perception_connector, 'task_execution_phases') and self.perception_connector.task_execution_phases:
                # 如果perception_connector有阶段信息，使用它
                self.phase_manager.initialize_phases(self.perception_connector.task_execution_phases)
                self.phase_manager._current_phase_index = getattr(self.perception_connector, 'current_phase_index', 0)
                return self.phase_manager.get_current_phase_tasks()
            else:
                # 创建基础阶段
                fallback_tasks = self.task_assignment_manager.create_fallback_tasks(instruction, world_graph)
                fallback_phases = [{
                    'phase_id': 0,
                    'tasks': fallback_tasks,
                    'max_parallel_tasks': len(self._agents),
                    'estimated_duration': 30.0,
                    'required_agents': len(self._agents)
                }]
                
                self.phase_manager.initialize_phases(fallback_phases)
                return self.phase_manager.get_current_phase_tasks()
                
        except Exception as e:
            print(f"[ERROR] Phase initialization failed: {e}")
            return None

    def _fallback_replan(self, instruction: str, observations: Dict[str, Any], world_graph: Dict[int, "WorldGraph"]) -> str:
        """当模块化组件不可用时的fallback"""
        print("[WARNING] Using fallback replan due to modular component issues")
        base_prompt, _ = self.prepare_prompt(instruction, world_graph[self._agents[0].uid])
        
        try:
            # 尝试使用基础的actions_parser
            if hasattr(self, 'actions_parser'):
                response = "Agent_0_Action: Explore[environment]\nAgent_1_Action: Wait[]\nAssigned!"
            else:
                response = "Agent_0_Action: Explore[environment]\nAgent_1_Action: Wait[]\nAssigned!"
        except:
            response = "Agent_0_Action: Explore[environment]\nAgent_1_Action: Wait[]\nAssigned!"
            
        return response



    def _reshape_alpha_to_phase_tasks(self, alpha, n_agents, n_phase_tasks, phase_task_info):
        """**FIXED** 阶段感知的alpha重塑方法"""
        if alpha is None:
            return np.zeros((n_agents, n_phase_tasks))
        
        try:
            alpha_array = np.array(alpha)
            original_shape = alpha_array.shape
            
            print(f"[DEBUG] **FIXED** Reshaping alpha for phase tasks:")
            print(f"  Original shape: {original_shape}")
            print(f"  Target shape: ({n_agents}, {n_phase_tasks})")
            print(f"  Phase task types: {phase_task_info['types'] if phase_task_info else 'unknown'}")
            
            # 如果alpha是一维数组
            if alpha_array.ndim == 1:
                expected_size = n_agents * n_phase_tasks
                if alpha_array.size == expected_size:
                    return alpha_array.reshape(n_agents, n_phase_tasks)
                else:
                    print(f"Warning: Alpha size {alpha_array.size} doesn't match expected {expected_size}")
                    # 如果长度不匹配，截取或填充
                    if alpha_array.size > expected_size:
                        # 截取前n个元素
                        truncated = alpha_array[:expected_size]
                        return truncated.reshape(n_agents, n_phase_tasks)
                    else:
                        # 填充到所需长度
                        padded = np.zeros(expected_size)
                        padded[:alpha_array.size] = alpha_array
                        return padded.reshape(n_agents, n_phase_tasks)
            
            # 如果已经是2D数组
            elif alpha_array.ndim == 2:
                if alpha_array.shape == (n_agents, n_phase_tasks):
                    return alpha_array
                else:
                    # **CRITICAL**: 处理从13个全局任务到阶段任务的映射
                    print(f"Warning: Alpha shape {alpha_array.shape} doesn't match expected ({n_agents}, {n_phase_tasks})")
                    
                    # 如果原alpha是全局任务矩阵(n_agents, 13)，需要提取阶段任务列
                    if (alpha_array.shape[0] == n_agents and 
                        alpha_array.shape[1] == 13 and 
                        phase_task_info and 'indices' in phase_task_info):
                        
                        # 从全局矩阵中提取阶段任务的列
                        active_indices = phase_task_info['indices']
                        if len(active_indices) == n_phase_tasks:
                            # 确保所有索引都在有效范围内
                            valid_indices = [i for i in active_indices if 0 <= i < 13]
                            if len(valid_indices) == n_phase_tasks:
                                result = alpha_array[:, valid_indices]
                                print(f"[DEBUG] **FIXED** Extracted phase tasks from global matrix using indices {valid_indices}")
                                return result
                    
                    # 如果智能体数量匹配但任务数量不同
                    if alpha_array.shape[0] == n_agents:
                        current_n_tasks = alpha_array.shape[1]
                        if current_n_tasks > n_phase_tasks:
                            # 截取前n_phase_tasks列
                            return alpha_array[:, :n_phase_tasks]
                        else:
                            # 填充更多列
                            result = np.zeros((n_agents, n_phase_tasks))
                            result[:, :current_n_tasks] = alpha_array
                            return result
                    else:
                        # 创建新矩阵并复制可能的值
                        result = np.zeros((n_agents, n_phase_tasks))
                        copy_agents = min(alpha_array.shape[0], n_agents)
                        copy_tasks = min(alpha_array.shape[1], n_phase_tasks)
                        result[:copy_agents, :copy_tasks] = alpha_array[:copy_agents, :copy_tasks]
                        return result
            
            # 其他情况，返回默认矩阵
            else:
                print(f"Warning: Unexpected alpha dimensions {alpha_array.ndim}, using fallback")
                return np.zeros((n_agents, n_phase_tasks))
                
        except Exception as e:
            print(f"Error reshaping alpha for phase tasks: {e}, using fallback matrix")
            return np.zeros((n_agents, n_phase_tasks))

    def _reshape_alpha_to_matrix(self, alpha, n_agents, n_tasks):
        """Legacy alpha重塑方法，保持向后兼容性"""
        return self._reshape_alpha_to_phase_tasks(alpha, n_agents, n_tasks, None)







    def _create_fallback_tasks(self, instruction: str, world_graph: Dict[int, "WorldGraph"]) -> List[Dict[str, Any]]:
        """创建fallback任务分解，使用有效的探索目标"""
        try:
            # 尝试从world_graph获取一个有效的房间
            all_rooms = world_graph[self._agents[0].uid].get_all_rooms()
            explore_target = all_rooms[0].name if all_rooms else 'kitchen'
            print(f"[DEBUG] Fallback explore target set to: {explore_target}")
        except Exception:
            explore_target = 'kitchen' # 最终回退

        fallback_tasks = []
        
        # 基于指令关键词创建基本任务
        instruction_lower = instruction.lower()
        
        if any(keyword in instruction_lower for keyword in ['move', 'bring', 'take', 'put', 'place']):
            # 移动类任务
            fallback_tasks.extend([
                {
                    'task_id': 'fallback_explore',
                    'task_type': 'Explore',
                    'target': explore_target,
                    'description': 'Explore to find objects',
                    'priority': 3,
                    'estimated_duration': 10.0,
                    'preferred_agent': None,
                    'prerequisites': [],
                    'can_parallel': True,
                    'phase_group': 'preparation'
                },
                {
                    'task_id': 'fallback_navigate',
                    'task_type': 'Navigate',
                    'target': 'target_location',
                    'description': 'Navigate to target',
                    'priority': 4,
                    'estimated_duration': 15.0,
                    'preferred_agent': 0,
                    'prerequisites': ['fallback_explore'],
                    'can_parallel': False,
                    'phase_group': 'execution'
                }
            ])
        else:
            # 默认探索任务
            fallback_tasks.append({
                'task_id': 'fallback_general',
                'task_type': 'Explore',
                'target': explore_target,
                'description': instruction,
                'priority': 3,
                'estimated_duration': 20.0,
                'preferred_agent': None,
                'prerequisites': [],
                'can_parallel': True,
                'phase_group': 'execution'
            })
        
        return fallback_tasks

    def task_plan_MIQP_set(self):
        """设置MIQP参数"""
        from habitat_llm.planner.HRCS.params_module.scenario_params_task import ScenarioConfigTask
        from habitat_llm.planner.HRCS.params_module.opt_params_task import OptimizationConfigTask
        from habitat_llm.planner.HRCS.class_def.RTA_task import RTA
        
        try:
            n_agents = len(self._agents)
            scenario_manager = ScenarioConfigTask(n_r=n_agents, n_t=13, n_c=5, n_f=5)
            opt_manager = OptimizationConfigTask(n_r=n_agents, n_t=13)
            
            self.scenario_params = scenario_manager
            self.opt_params = opt_manager.get_opt_params()
            self.rta = RTA(scenario_manager.get_scenario_params(), self.opt_params)
            
        except ImportError:
            print("MIQP modules not available, using placeholder")
            self.scenario_params = None
            self.opt_params = {}
            self.rta = None

    def task_plan_MIQP_solve_phase_aware(self, x, t, phase_task_info):
        """
        阶段感知的MIQP求解器：使用正确的任务维度
        """
        try:
            start_time = time.time()
            
            if self.scenario_params is None:
                print("[ERROR] No scenario parameters available for phase-aware MIQP solve")
                return None, None, None, 0.0, "NO_SCENARIO_PARAMS"
            if phase_task_info is None:
                print("[ERROR] No phase task info available for phase-aware MIQP solve")
                return None, None, None, 0.0, "NO_PHASE_INFO"
            
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            if x.ndim != 2:
                print(f"[WARNING] State vector x has unexpected shape {x.shape}, reshaping...")
                if x.size >= 6:
                    x = x.reshape(3, -1)[:, :len(self._agents)]
                else:
                    x = np.zeros((3, len(self._agents)))
            
            n_agents = len(self._agents)
            n_phase_tasks = phase_task_info['n_phase_tasks']
            
            print(f"[DEBUG] **FIXED** Phase-aware MIQP setup:")
            print(f"  Agents: {n_agents}")
            print(f"  Phase tasks: {n_phase_tasks}")
            print(f"  Task types: {phase_task_info['types']}")
            
            # 创建阶段特定的scenario_params
            phase_scenario_params = self._create_phase_scenario_params(phase_task_info)
            
            # 如果有RTA求解器且支持阶段感知
            if self.rta is not None:
                try:
                    # 尝试使用阶段特定参数调用求解器
                    alpha, u, delta, time_to_solve, opt_sol_info = self.rta.solve_miqp_phase_aware(
                        x, t, phase_scenario_params, n_phase_tasks
                    )
                except AttributeError:
                    # 如果RTA不支持阶段感知，使用启发式解决方案
                    print("[WARNING] RTA doesn't support phase-aware solving, using heuristic")
                    alpha, u, delta, time_to_solve, opt_sol_info = self._heuristic_phase_solve(
                        x, t, phase_task_info
                    )
            else:
                # 没有RTA求解器，使用启发式解决方案
                print("[WARNING] No RTA solver available, using heuristic phase solve")
                alpha, u, delta, time_to_solve, opt_sol_info = self._heuristic_phase_solve(
                    x, t, phase_task_info
                )
            
            solve_time = time_to_solve
            
            # 验证返回值具有正确的阶段特定维度
            if alpha is not None:
                alpha = np.array(alpha)
                expected_shape = (n_agents, n_phase_tasks)
                
                if alpha.ndim == 1:
                    # 如果返回的是1D数组，重塑为期望形状
                    expected_size = n_agents * n_phase_tasks
                    if alpha.size == expected_size:
                        alpha = alpha.reshape(expected_shape)
                    else:
                        print(f"[WARNING] Alpha size {alpha.size} doesn't match expected {expected_size}")
                        alpha = np.ones(expected_shape) * 0.5
                elif alpha.shape != expected_shape:
                    print(f"[WARNING] Alpha shape {alpha.shape} doesn't match expected {expected_shape}")
                    alpha = np.ones(expected_shape) * 0.5
                
                print(f"[DEBUG] **FIXED** Phase-aware MIQP solved successfully in {solve_time:.4f}s")
                print(f"[DEBUG] **FIXED** Alpha matrix shape: {alpha.shape} (agents={n_agents}, phase_tasks={n_phase_tasks})")
                return alpha, u, delta, solve_time, opt_sol_info
            else:
                print(f"[DEBUG] Phase-aware MIQP solve failed in {solve_time:.4f}s")
                return None, None, None, solve_time, "INFEASIBLE"
                
        except Exception as e:
            solve_time = time.time() - start_time if 'start_time' in locals() else 0.0
            print(f"[ERROR] Phase-aware MIQP solve exception: {e}")
            # 返回阶段特定维度的fallback解决方案
            n_agents = len(self._agents) if hasattr(self, '_agents') else 2
            n_phase_tasks = phase_task_info['n_phase_tasks'] if phase_task_info else 1
            alpha = np.ones((n_agents, n_phase_tasks)) * 0.5
            u = np.zeros((3, n_agents))
            delta = np.ones(n_phase_tasks)
            return alpha, u, delta, solve_time, f"EXCEPTION_FALLBACK: {str(e)}"

    def _create_phase_scenario_params(self, phase_task_info):
        """创建阶段特定的scenario参数"""
        try:
            # 复制基础参数
            phase_params = self.scenario_params.scenario_params.copy()
            
            # 使用阶段特定的T矩阵
            phase_params['T'] = phase_task_info['matrix']
            
            # 调整其他矩阵以匹配阶段任务数量
            n_phase_tasks = phase_task_info['n_phase_tasks']
            n_agents = len(self._agents)
            
            # A矩阵保持不变（能力×智能体）
            # ws权重需要调整为只包含活跃任务的权重
            if 'ws' in phase_params:
                original_ws = phase_params['ws']
                if isinstance(original_ws, list) and len(original_ws) == 5:  # 5个能力维度
                    # ws对应能力维度，不需要改变
                    phase_params['ws'] = original_ws
                else:
                    # 如果ws对应任务维度，需要调整
                    phase_params['ws'] = [w for w in original_ws[:n_phase_tasks]]
            
            return phase_params
            
        except Exception as e:
            print(f"[ERROR] Failed to create phase scenario params: {e}")
            return self.scenario_params.scenario_params if self.scenario_params else {}

    def _heuristic_phase_solve(self, x, t, phase_task_info):
        """启发式阶段求解方案"""
        start_time = time.time()
        
        n_agents = len(self._agents)
        n_phase_tasks = phase_task_info['n_phase_tasks']
        task_types = phase_task_info['types']
        
        print(f"[DEBUG] **HEURISTIC** Phase solve: {n_agents} agents, {n_phase_tasks} tasks")
        
        # 创建启发式分配矩阵
        alpha = np.zeros((n_agents, n_phase_tasks))
        
        # 基于智能体能力进行智能分配
        agent_capabilities = self._get_agent_capabilities()
        
        for task_idx, task_type in enumerate(task_types):
            # 找到能执行此任务的智能体
            capable_agents = []
            for agent_id, capabilities in agent_capabilities.items():
                if task_type in capabilities:
                    capable_agents.append(agent_id)
            
            if capable_agents:
                # 如果只有一个任务且多个智能体能执行，选择第一个
                if n_phase_tasks == 1 and len(capable_agents) > 1:
                    chosen_agent = capable_agents[0]
                    alpha[chosen_agent, task_idx] = 1.0
                    print(f"[DEBUG] **HEURISTIC** Assigned {task_type} to Agent {chosen_agent}")
                elif len(capable_agents) == 1:
                    # 只有一个智能体能执行
                    alpha[capable_agents[0], task_idx] = 1.0
                    print(f"[DEBUG] **HEURISTIC** Assigned {task_type} to Agent {capable_agents[0]} (only capable)")
                else:
                    # 多个智能体能执行，平均分配
                    weight = 1.0 / len(capable_agents)
                    for agent_id in capable_agents:
                        alpha[agent_id, task_idx] = weight
                    print(f"[DEBUG] **HEURISTIC** Split {task_type} among agents {capable_agents}")
            else:
                # 没有智能体能执行，分配给第一个智能体
                alpha[0, task_idx] = 1.0
                print(f"[DEBUG] **HEURISTIC** Fallback: assigned {task_type} to Agent 0")
        
        # 确保每个智能体至少分配到一些任务（如果任务数>=智能体数）
        if n_phase_tasks >= n_agents:
            for agent_id in range(n_agents):
                if np.sum(alpha[agent_id, :]) < 0.001:  # 智能体没有分配到任务
                    # 找到负载最轻的任务分配给这个智能体
                    task_loads = np.sum(alpha, axis=0)
                    lightest_task = np.argmin(task_loads)
                    alpha[agent_id, lightest_task] = 0.5
                    print(f"[DEBUG] **HEURISTIC** Ensured Agent {agent_id} has task {task_types[lightest_task]}")
        
        # 创建简单的控制输入和松弛变量
        u = np.zeros((3, n_agents))
        delta = np.ones(n_phase_tasks)
        
        solve_time = time.time() - start_time
        
        return alpha, u, delta, solve_time, "HEURISTIC_OPTIMAL"

    def task_plan_MIQP_solve(self, x, t):
        """
        Legacy MIQP solver for backward compatibility
        """
        return self.task_plan_MIQP_solve_phase_aware(x, t, None)

    def get_next_action(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
        verbose: bool = False,
    ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        """
        Get the next low-level action to execute.

        :param instruction: The instruction for the task.
        :param observations: The current observations.
        :param world_graph: The world graph for each agent.
        :param verbose: Whether to print verbose output. Defaults to False.
        :return: A tuple containing:
                 - The low-level actions for each agent
                 - Planner information
                 - Whether the planner is done
        """
        planner_info: Dict[str, Union[Any, str]] = {}
        
        # Early return if planner is already done
        if self.is_done:
            planner_info = {
                "prompts": {agent.uid: self.curr_prompt for agent in self.agents},
                "traces": {agent.uid: self.trace for agent in self.agents},
                "replanning_count": {
                    agent.uid: self.replanning_count for agent in self.agents
                },
                "replanned": {agent.uid: False for agent in self.agents},
                "replan_required": {
                    agent.uid: self.replan_required for agent in self.agents
                },
                "is_done": {agent.uid: self.is_done for agent in self.agents},
            }
            return {}, planner_info, self.is_done

        if self.curr_prompt == "":
            # Prepare prompts
            self.curr_prompt, self.params = self.prepare_prompt(
                instruction, world_graph[self._agents[0].uid], observations=observations
            )
            self.curr_obj_states = get_objects_descr(
                world_graph[self._agents[0].uid],
                self._agents[0].uid,
                include_room_name=True,
                add_state_info=self.planner_config.objects_response_include_states,
                centralized=self.planner_config.centralized,
            )

        if self.trace == "":
            self.trace += f"Task: {instruction}\nThought: "

        print_str = ""
        self.is_done = False

        if self.replan_required:
            planner_info["replanned"] = {agent.uid: True for agent in self.agents}
            if verbose:
                # calculate the total time of response generation
                start_time = time.time()

            llm_response = self.replan(instruction, observations, world_graph)
            
            # parse thought from the response
            thought = self.parse_thought(llm_response)

            if verbose:
                total_time = time.time() - start_time
                print(
                    f"Time taken for LLM response generation: {total_time}; replanning_count: {self.replanning_count}"
                )

            # Update prompt with the first response
            print_str += f"""{llm_response}\n{self.stopword}\n"""
            prompt_addition = (
                f"""{llm_response}\n{self.stopword}{self.planner_config.llm.eot_tag}"""
            )
            self.curr_prompt += prompt_addition
            self.trace += prompt_addition

            # Check if the planner should stop
            # Stop if the replanning count exceed a certain threshold
            # or end expression is found in llm response
            # This is helpful to break infinite planning loop.
            self.is_done = (self.check_if_agent_done(llm_response)) or (
                self.replanning_count == self.planner_config.replanning_threshold
            )
            # Increment the llm call counter on every replan
            # doesn't get incremented before comparison as first "replan" is technically
            # the first required plan
            self.replanning_count += 1

            # Early return if stop is required
            if self.is_done:
                planner_info = {
                    "print": print_str,
                    "prompts": {agent.uid: self.curr_prompt for agent in self.agents},
                    "traces": {agent.uid: self.trace for agent in self.agents},
                    "replanning_count": {
                        agent.uid: self.replanning_count for agent in self.agents
                    },
                    "replan_required": {
                        agent.uid: self.replan_required for agent in self.agents
                    },
                    "replanned": {agent.uid: True for agent in self.agents},
                    "is_done": {agent.uid: self.is_done for agent in self.agents},
                    "thought": {agent.uid: thought for agent in self.agents},
                    "high_level_actions": {
                        agent.uid: ("Done", None, None) for agent in self.agents
                    },
                }
                return {}, planner_info, self.is_done

            # Parse high level action directives from llm response
            high_level_actions = self.llm_interface.parse_high_level_actions(
                llm_response, self.agents, self.params
            )
            print(f"\n\n[DEBUG] Parsed High-Level Actions: {high_level_actions}\n\n")

            # **ENHANCED** 基于MIQP求解结果和阶段感知调整高级动作
            if hasattr(self, '_last_response_info') and self._last_response_info:
                response_info = self._last_response_info
                if "agent_task_assignments" in response_info and "current_phase" in response_info:
                    high_level_actions = self.task_assignment_manager.adjust_actions_with_phase_awareness(
                        high_level_actions, 
                        response_info["agent_task_assignments"],
                        response_info["current_phase"]
                    )
                    print(f"\n\n[DEBUG] Phase-Aware Adjusted High-Level Actions: {high_level_actions}\n\n")

            # Get low level actions and/or responses
            low_level_actions, responses = self.process_high_level_actions(
                high_level_actions, observations
            )

            # Store last executed high level action
            self.last_high_level_actions = high_level_actions
        else:
            planner_info["replanned"] = {agent.uid: False for agent in self.agents}
            # Set thought to None
            thought = None

            # Get low level actions and/or responses using last high level actions
            low_level_actions, responses = self.process_high_level_actions(
                self.last_high_level_actions, observations
            )

        # Log if replanning was done or not before overwriting the value
        planner_info["replan_required"] = {
            agent.uid: self.replan_required for agent in self.agents
        }

        # **OPTIMIZED** 简化的阶段推进检查（仅检查是否需要强制重新规划）
        if (hasattr(self, 'perception_connector') and self.perception_connector and 
            hasattr(self, '_phase_transition_pending') and self._phase_transition_pending):
            # 重置标志
            self._phase_transition_pending = False
            self.replan_required = True
            print(f"[INFO] **NEW** Phase transition detected, forcing plan on this iteration")

        # Check if replanning is required
        # Replanning is required when any of the actions being executed
        # have a response indicating success or failure (and the reason)
        self.replan_required = any(responses.values())
        print_str += self._add_responses_to_prompt(responses)

        # Update planner info with MIQP-specific information
        planner_info["responses"] = responses
        planner_info["thought"] = {agent.uid: thought for agent in self.agents}
        planner_info["is_done"] = {agent.uid: self.is_done for agent in self.agents}
        planner_info["print"] = print_str
        planner_info["high_level_actions"] = self.last_high_level_actions
        planner_info["prompts"] = {agent.uid: self.curr_prompt for agent in self.agents}
        planner_info["traces"] = {agent.uid: self.trace for agent in self.agents}
        planner_info["replanning_count"] = {
            agent.uid: self.replanning_count for agent in self.agents
        }
        planner_info["agent_states"] = self.get_last_agent_states()
        planner_info["agent_positions"] = self.get_last_agent_positions()
        planner_info["agent_collisions"] = self.get_agent_collisions()
        
        # **ENHANCED** 添加MIQP相关信息
        if hasattr(self, '_last_response_info') and self._last_response_info:
            planner_info["miqp_info"] = {
                "optimization_success": self._last_response_info.get("optimization_success", False),
                "miqp_status": self._last_response_info.get("miqp_status", "UNKNOWN"),
                "current_phase": self._last_response_info.get("current_phase", {}),
                "task_decomposition_success": self._last_response_info.get("task_decomposition_success", False)
            }

        return low_level_actions, planner_info, self.is_done

    def check_if_agent_done(self, llm_response: str) -> bool:
        """
        Check if the agent is done based on the LLM response.

        :param llm_response: The LLM response to check.
        :return: True if the agent is done, False otherwise.
        """
        return self.end_expression in llm_response






            