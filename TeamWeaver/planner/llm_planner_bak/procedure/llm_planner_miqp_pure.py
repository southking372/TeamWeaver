#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
MIQP Enhanced LLM Planner with Feedback Optimization

==============================================

Problem: Previous enhanced_prompt approach lost conversation history and execution feedback,
leading to poor feedback processing and inconsistent decision making.

Solution: Refactored Steps to follow old_llm_planner.py's proven approach:

1. **Base on self.curr_prompt**: Uses accumulated conversation history including:
   - Initial system prompt and task instruction
   - All previous LLM responses
   - All agent observations and execution feedback
   - Updated world state information

2. **Add MIQP guidance incrementally**: Via _build_miqp_guidance_addition()
   - Preserves conversation continuity
   - Adds phase-aware task assignments
   - Includes optimization status and guidelines

3. **Complete feedback integration**:
   - Agent failure reports → navigation suggestions
   - Task completion status → phase advancement
   - Error recovery → adjusted task assignments
   - World state changes → updated object positions

Benefits:
- LLM sees complete execution history for better decisions
- Maintains conversation continuity and context
- Enables effective learning from failures and feedback
- Supports iterative task refinement and error correction

Usage: The enhanced Step 12 now provides robust feedback processing while
maintaining all MIQP optimization benefits.
"""

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

        if self.perception_connector:
            self.perception_connector.reset()

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
        重新规划：集成任务分解、MIQP优化和序列化执行、支持阶段性任务执行和动态T矩阵生成
        
        Returns:
            llm_response: LLM生成的响应字符串
        """
        print("\n" + "="*80)
        print("🚀 开始 MIQP Enhanced Plan")
        print("="*80)
        
        t = 0.0
        
        # === Step 1: Extract Current Agent States ===
        print(f"[Step 1/15] Extracting current agent states...")
        try:
            agent_poses = self.get_last_agent_positions_miqp(world_graph)
            n_agents = len(self._agents)
            n_states = 3  # [x, y, theta]
            x = np.zeros((n_states, n_agents))
            for i, agent in enumerate(self._agents):
                agent_id = agent.uid
                if agent_id in agent_poses:
                    x[0, i] = agent_poses[agent_id]['position'][0]  # x
                    x[1, i] = agent_poses[agent_id]['position'][2]  # y (using z as y)
                    x[2, i] = agent_poses[agent_id].get('yaw', 0.0)  # theta
            # print(f"[DEBUG] Extracted states for {n_agents} agents: x.shape = {x.shape}")
        except Exception as e:
            print(f"[ERROR] Agent state extraction failed: {e}")
            x = np.zeros((3, len(self._agents)))
            
        # === Step 2: Initialize/Update Scenario Parameters ===
        print(f"[Step 2/15] Setting up MIQP scenario parameters...")
        if not hasattr(self, 'scenario_params') or self.scenario_params is None:
            self.task_plan_MIQP_set()
            # print(f"[DEBUG] MIQP scenario parameters initialized")
        
        # === Step 3: Extract World State ===
        print(f"[Step 3/15] Extracting world state...")
        try:
            if not hasattr(self, 'perception_connector') or self.perception_connector is None:
                self.perception_connector = PerceptionConnector(api_key_filename="api_key")
            
            world_state = self.perception_connector.extract_world_state(self.env_interface)
        except Exception as e:
            print(f"[ERROR] World state extraction failed: {e}")
            world_state = {'agent_poses': {}, 'object_positions': {}, 'furniture_positions': {}}

        # === Step 4: Pre-update Scenario Params ===
        print(f"[Step 4/15] Pre-updating scenario parameters...")
        try:
            if self.scenario_params is not None:
                self.perception_connector.pre_update_scenario_params(
                    self.scenario_params,
                    world_state
                )
                print(f"[DEBUG] Basic parameters updated based on world state")
            else:
                print(f"[WARNING] scenario_params is None, skipping basic updates")
        except Exception as e:
            print(f"[ERROR] Basic parameter update failed: {e}")

        # === Step 5: Sequenced Task Decomposition ===
        print(f"[Step 5/15] Decomposing task with sequencing...")
        try:
            llm_decompose_config = {
                "gpt_version": "moonshot-v1-32k",
                "max_tokens": 1200,
            }
            
            # 检查是否是首次分解，还是只需要获取当前阶段
            if not self.perception_connector.phase_manager.task_execution_phases:
                try:
                    structured_subtasks, execution_phases = self.perception_connector.structured_decompose_task_with_sequencing(
                        instruction,
                        self.env_interface,
                        llm_decompose_config,
                        max_agents=len(self._agents)
                    )
                    print(f"[DEBUG] Task decomposed into {len(structured_subtasks)} subtasks across {len(execution_phases)} phases")
                    
                    self._last_structured_subtasks = structured_subtasks
                    self._last_execution_phases = execution_phases
                except Exception as decompose_error:
                    print(f"[WARNING] Structured decomposition failed: {decompose_error}, using fallback")
                    raise decompose_error
            else:
                # 继续执行：检查当前阶段状态
                current_phase = self.perception_connector.get_current_phase_tasks()
                if current_phase:
                    print(f"[DEBUG] Continuing execution: Phase {self.perception_connector.phase_manager.current_phase_index + 1}/{len(self.perception_connector.phase_manager.task_execution_phases)}")
                    print(f"  Current phase tasks: {[t['task_type'] + '→' + t['target'] for t in current_phase['tasks']]}")
                else:
                    print(f"[DEBUG] All phases completed!")
                    return "Final Thought: Exit!"

        except Exception as e:
            print(f"[ERROR] Sequenced task decomposition failed: {e}, using fallback")
            try:
                execution_phases = self._create_fallback_tasks(instruction)
                self.perception_connector.phase_manager.set_execution_phases(execution_phases)
                print(f"[DEBUG] Fallback decomposition created {len(execution_phases)} phases")
            except Exception as fallback_error:
                print(f"[ERROR] Even fallback decomposition failed: {fallback_error}")
                return "Agent_0_Action: Explore[environment]\nAgent_1_Action: Wait[]\nAssigned!"

        # === Step 6: Get Current Phase Tasks ===
        print(f"[Step 6/15] Getting current phase tasks...")
        current_phase = self.perception_connector.get_current_phase_tasks()
        if not current_phase:
            print(f"[INFO] No more phases to execute - task completed!")
            return "Both agents have completed their tasks and are waiting. The task is now complete."
        
        current_phase_tasks = current_phase['tasks']
        print(f"[DEBUG] Current phase {current_phase['phase_id']} has {len(current_phase_tasks)} tasks:")
        for task in current_phase_tasks:
            print(f"  - {task['task_type']} → {task['target']} (Agent: {task.get('preferred_agent', 'Any')})")

        # === Step 7: Build Phase-Specific T Matrix ===
        print(f"[Step 7/15] Building phase-specific T matrix...")
        try:
            if self.scenario_params is not None:
                # 构建当前阶段的动态维度T矩阵
                phase_t_matrix, active_task_indices, active_task_types = self.perception_connector.build_phase_specific_t_matrix(
                    current_phase,
                    self.perception_connector.BASE_TASK_CAPABILITY_REQUIREMENTS
                )
                
                phase_task_info = {
                    'matrix': phase_t_matrix,
                    'indices': active_task_indices,
                    'types': active_task_types,
                    'n_phase_tasks': len(active_task_indices),
                    'n_total_tasks': 13
                }
                print(f"  Phase T matrix content:")
                for i, task_idx in enumerate(active_task_indices):
                    task_type = active_task_types[i]
                    non_zero_caps = np.where(phase_t_matrix[task_idx, :] > 0.001)[0]
                    cap_names = ['Movement', 'Object_Manip', 'Basic_Control', 'Liquid_Handle', 'Power_Control']
                    required_caps = [cap_names[j] for j in non_zero_caps]
                    print(f"    {task_type} (row {task_idx}): requires {required_caps}")
                    
            else:
                print(f"[WARNING] Cannot build phase-specific T matrix - missing scenario_params")
        except Exception as e:
            print(f"[ERROR] Phase-specific T matrix building failed: {e}")
            fallback_matrix = np.zeros_like(self.perception_connector.BASE_TASK_CAPABILITY_REQUIREMENTS)
            fallback_matrix[12, :] = self.perception_connector.BASE_TASK_CAPABILITY_REQUIREMENTS[12, :]  # 只激活Wait
            phase_task_info = {
                'matrix': fallback_matrix,
                'indices': [12],  # Wait任务索引
                'types': ['Wait'],
                'n_phase_tasks': 1,
                'n_total_tasks': 13
            }

        # === Step 8: Update Other MIQP Matrices ===
        print(f"[Step 8/15] Updating other MIQP matrices...")
        try:
            if self.scenario_params is not None and len(current_phase_tasks) > 0:
                self.perception_connector.update_miqp_matrices(
                    self.scenario_params,
                    current_phase_tasks,
                    world_state
                )
            else:
                print(f"[WARNING] Cannot update MIQP matrices - missing params or tasks")
        except Exception as e:
            print(f"[ERROR] MIQP matrices update failed: {e}")

        # === Step 9: MIQP Optimization ===
        print(f"[Step 9/15] Running MIQP optimization for current phase...")
        alpha, u, delta, time_to_solve, opt_sol_info = self.task_plan_MIQP_solve_phase_aware(
            x, t, phase_task_info
        )

        # === Step 10: Phase-Specific Task Assignment ===
        print(f"[Step 10/15] Mapping current phase tasks to agents...")
        agent_capabilities = self._get_agent_capabilities()
        
        for i, task in enumerate(current_phase_tasks):
            print(f"  Task {i}: {task.get('task_type', 'Unknown')} → {task.get('target', 'Unknown')}")
        
        if alpha is None or not self._validate_miqp_solution(alpha, agent_capabilities):
            print(f"[WARNING] MIQP solution validation failed, using heuristic assignment")
            agent_task_assignments = self._heuristic_task_assignment(current_phase_tasks)
        else:
            try:
                n_current_tasks = len(current_phase_tasks)
                n_total_tasks = phase_task_info.get('n_total_tasks', 13)
                active_indices = phase_task_info.get('indices', [])
                # print(f"[DEBUG] Phase task assignment:")
                # print(f"  Alpha shape: {alpha.shape if hasattr(alpha, 'shape') else 'unknown'}")
                # print(f"  Current phase tasks: {n_current_tasks}")
                # print(f"  Total matrix tasks: {n_total_tasks}")
                # print(f"  Active task indices: {active_indices}")
                
                if hasattr(alpha, 'shape') and alpha.shape == (len(self._agents), n_total_tasks):
                    if len(active_indices) == n_current_tasks:
                        phase_alpha = alpha[:, active_indices]
                    else:
                        print(f"[WARNING] Active indices mismatch: {len(active_indices)} vs {n_current_tasks}")
                        agent_task_assignments = self._heuristic_task_assignment(current_phase_tasks)
                        phase_alpha = None
                else:
                    print(f"[WARNING] Alpha dimensions unexpected: {getattr(alpha, 'shape', 'unknown')}")
                    phase_alpha = self._extract_phase_alpha_from_full_matrix(
                        alpha, len(self._agents), active_indices, n_current_tasks
                    )
                
                if phase_alpha is not None:
                    agent_task_assignments = self.perception_connector.map_subtasks_to_agents(
                        current_phase_tasks,
                        phase_alpha,
                        agent_capabilities
                    )
                    print(f"[DEBUG] Successfully assigned tasks using extracted phase alpha")
                else:
                    print(f"[WARNING] Could not extract phase alpha, using heuristic assignment")
                    agent_task_assignments = self._heuristic_task_assignment(current_phase_tasks)
                    
            except Exception as e:
                print(f"[ERROR] Phase task assignment failed: {e}, using heuristic")
                agent_task_assignments = self._heuristic_task_assignment(current_phase_tasks)

        print(f"[DEBUG] Phase {current_phase['phase_id']} task assignments:")
        for agent_id, tasks in agent_task_assignments.items():
            if tasks:
                task_summaries = [f"{t['task_type']}→{t['target']}" for t in tasks]
                print(f"  Agent {agent_id}: {task_summaries}")
            else:
                print(f"  Agent {agent_id}: []")

        # === Step 10.5: Apply Intelligent Error Recovery ===
        print(f"[Step 10.5/15] Applying intelligent error recovery...")
        try:
            # **CRITICAL FIX**: 智能错误恢复，处理Pick失败等问题
            recovered_assignments = self._apply_intelligent_error_recovery(
                agent_task_assignments, 
                current_phase
            )
            
            # 如果有恢复任务被添加，更新分配
            if recovered_assignments != agent_task_assignments:
                agent_task_assignments = recovered_assignments
                print(f"[DEBUG] **RECOVERY** Updated task assignments after error recovery:")
                for agent_id, tasks in agent_task_assignments.items():
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
                
        except Exception as e:
            print(f"[ERROR] Error recovery failed: {e}")

        # === Step 11: Build Phase-Aware Prompt ===
        print(f"[Step 11/15] Building phase-aware enhanced prompt...")
        try:
            miqp_guidance = self._build_miqp_guidance_addition(
                current_phase_tasks,
                agent_task_assignments,
                current_phase,
                alpha_result=alpha,
                world_state=world_state
            )
        except Exception as e:
            print(f"[ERROR] MIQP guidance building failed: {e}")
            miqp_guidance = ""

        # === Step 12: LLM Action Generation (Enhanced with Feedback) ===
        print(f"[Step 12/15] Generating actions via LLM with complete feedback history...")
        try:
            # 1. Start with the complete conversational history.
            prompt_for_llm = self.curr_prompt

            # 2. Append MIQP guidance as a new user instruction to maintain conversational flow.
            if miqp_guidance:
                prompt_for_llm += (
                    f"{self.planner_config.llm.user_tag}{miqp_guidance}"
                    f"{self.planner_config.llm.eot_tag}"
                )

            # 3. Add the assistant tag to prompt the LLM to start its response.
            # We add "Thought:" to encourage chain-of-thought reasoning.
            # prompt_for_llm += f"{self.planner_config.llm.assistant_tag}Thought:"

            print(f"[DEBUG] **FEEDBACK_ENHANCED** Using curr_prompt ({len(self.curr_prompt)} chars) + MIQP guidance ({len(miqp_guidance)} chars)")
            # print(f"[DEBUG-LYP] full_prompt_with_guidance (last 400 chars): \n{prompt_for_llm[-400:]}")
            # print(f"#######################[DEBUG-LYP] self.curr_prompt: \n{self.curr_prompt}")
            # print(f"#######################[DEBUG-LYP] miqp_guidance: \n{miqp_guidance}")
            
            # 4. Generate the response from the LLM.
            if self.planner_config.get("constrained_generation", False):
                print("[DEBUG-LYP] Now use constrained generation")
                raw_response = self.llm.generate(
                    # prompt_for_llm,
                    self.curr_prompt,
                    self.stopword,
                    generation_args={
                        "grammar_definition": self.build_response_grammar(
                            world_graph[self._agents[0].uid]
                        )
                    },
                )
                # compare_response = self.llm.generate(
                #     self.curr_prompt,
                #     self.stopword,
                #     generation_args={
                #         "grammar_definition": self.build_response_grammar(
                #             world_graph[self._agents[0].uid]
                #         )
                #     },
                # )
                # miqp_response = self.llm.generate(
                #     miqp_guidance,
                #     self.stopword,
                #     generation_args={
                #         "grammar_definition": self.build_response_grammar(
                #             world_graph[self._agents[0].uid]
                #         )
                #     },
                # )
            else:
                raw_response = self.llm.generate(self.curr_prompt, self.stopword)

            print(f"###################[DEBUG-LYP] raw_response: \n{raw_response}")
            # print(f"###################[DEBUG-LYP] compare_response: \n{compare_response}")
            # raw_response = compare_response
            
            # 5. Prepend the "Thought:" prompt to the raw response to form the complete turn,
            # then format it to remove any extraneous text after the end expression.
            llm_response = self.format_response(raw_response, self.end_expression)
            
            print(f"[DEBUG-LYP] Formatted LLM response: \n{llm_response}")

            if not llm_response or llm_response == "Thought:":
                print(f"[WARNING] Empty LLM response received after formatting.")
                llm_response = "Agent_0_Action: Wait[]\nAgent_1_Action: Wait[]\nAssigned!"

        except Exception as e:
            print(f"[ERROR] LLM action generation failed: {e}")
            llm_response = "Agent_0_Action: Wait[]\nAgent_1_Action: Wait[]\nAssigned!"

        # 添加调试输出，显示使用的prompt长度信息
        print(f"[DEBUG] **FEEDBACK_ENHANCED** Prompt components:")
        print(f"  - curr_prompt length: {len(self.curr_prompt)} chars")
        if miqp_guidance:
            print(f"  - MIQP guidance length: {len(miqp_guidance)} chars")
        print(f"  - Total prompt length: {len(prompt_for_llm) if 'prompt_for_llm' in locals() else 'unknown'} chars")
        print(f"  - Response length: {len(llm_response)} chars")

        # === Step 13: Parse High-Level Actions ===
        print(f"[Step 13/15] Parsing high-level actions...")
        try:
            high_level_actions = self._parse_high_level_actions(llm_response)
            
            # 根据当前阶段调整动作
            adjusted_actions = self._adjust_actions_with_phase_awareness(
                high_level_actions, 
                agent_task_assignments,
                current_phase
            )
            
            print(f"[DEBUG] Phase-adjusted actions:")
            for agent_id, action in adjusted_actions.items():
                if action and len(action) >= 3:
                    print(f"  Agent {agent_id}: {action[0]}({action[1]}) → {action[2]}")
        except Exception as e:
            print(f"[ERROR] High-level action parsing failed: {e}")
            adjusted_actions = {}

        # === Step 14: Update Scenario Parameters ===
        print(f"[Step 14/15] Updating scenario parameters for execution...")
        try:
            if self.scenario_params is not None and adjusted_actions:
                self.perception_connector.update_scenario_from_actions(
                    self.scenario_params,
                    world_state,
                    adjusted_actions
                )
            else:
                print(f"[WARNING] Cannot update scenario params - missing params or actions")
        except Exception as e:
            print(f"[ERROR] Scenario parameter update failed: {e}")

        # === Step 15: Phase Completion Check and Advancement ===
        # print(f"[Step 15/15] **NEW** Checking phase completion...")
        # try:
        #     # 智能获取agent状态 - 优先使用最新的responses
        #     agent_statuses = self._get_agent_completion_statuses()
            
        #     print(f"[DEBUG] Phase completion check for Phase {current_phase['phase_id']}:")
        #     print(f"  Agent completion statuses: {agent_statuses}")
            
        #     # 检查当前阶段是否完成
        #     if self.perception_connector.is_current_phase_complete(agent_statuses):
        #         print(f"[SUCCESS] **NEW** Phase {current_phase['phase_id']} completed!")
                
        #         # 尝试推进到下一阶段
        #         if self.perception_connector.advance_to_next_phase():
        #             next_phase = self.perception_connector.get_current_phase_tasks()
        #             print(f"[INFO] **NEW** Advanced to phase {next_phase['phase_id']}:")
        #             for task in next_phase['tasks']:
        #                 print(f"    Next: {task['task_type']} → {task['target']}")
                    
        #             # 设置阶段转换标志
        #             self._phase_transition_pending = True
        #             print(f"[DEBUG] Phase transition pending, will force plan on next iteration")
        #         else:
        #             print(f"[SUCCESS] All phases completed - Task sequence finished!")
        #             # return self.end_expression
        # except Exception as e:
        #     print(f"[ERROR] Phase completion check failed: {e}")
        #     import traceback
        #     traceback.print_exc()

        # === Store Results ===
        self._last_response_info = {
            "miqp_alpha": alpha,
            "miqp_u": u,
            "miqp_delta": delta,
            "miqp_time": time_to_solve,
            "miqp_status": opt_sol_info,
            "optimization_success": alpha is not None,
            "structured_subtasks": current_phase_tasks,
            "agent_task_assignments": agent_task_assignments,
            "task_decomposition_success": len(current_phase_tasks) > 0,
            "matrices_updated": True,
            "planning_approach": "MIQP_Sequential_Phase",
            "current_phase": current_phase,
            "total_phases": len(self.perception_connector.phase_manager.task_execution_phases),
            "llm_response": llm_response
        }

        print(f"\n[SUCCESS] **NEW** MIQP Sequential Phase Plan completed!")
        print(f"  Current Phase: {current_phase['phase_id'] + 1}/{len(self.perception_connector.phase_manager.task_execution_phases)}")
        print(f"  Phase Tasks: {len(current_phase_tasks)}")
        print(f"  Optimization: {opt_sol_info}")
        print(f"  Error Recovery: Applied")
        print("="*80)
        
        # # 验证反馈增强效果
        # print(f"[FEEDBACK_VALIDATION] Step 12 Enhancement Summary:")
        # print(f"  ✓ Used curr_prompt with complete conversation history ({len(self.curr_prompt)} chars)")
        # print(f"  ✓ Added MIQP guidance as enhancement ({len(miqp_guidance) if 'miqp_guidance' in locals() else 0} chars)")
        # print(f"  ✓ Preserved agent observations and execution feedback")
        # print(f"  ✓ Applied format_response for clean output")
        # print(f"  → Enhanced feedback processing compared to isolated enhanced_prompt approach")
        
        # 返回LLM响应
        return llm_response

    def _build_miqp_guidance_addition(
        self,
        current_phase_tasks: List[Dict[str, Any]],
        agent_task_assignments: Dict[int, List[Dict[str, Any]]],
        current_phase: Dict[str, Any],
        alpha_result: Optional[Any] = None,
        world_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        **NEW** 构建MIQP指导信息的增量部分，用于添加到现有的curr_prompt中。
        这确保保留了完整的对话历史和反馈，同时添加MIQP优化指导。
        格式化为更适合LLM理解的简洁指导。
        """
        # **CRITICAL FIX**: 使用更简洁、更符合对话格式的指导
        guidance_info = "Based on MIQP optimization analysis:\n\n"
        
        # 添加阶段信息（简化）
        try:
            phase_id = current_phase['phase_id']
            total_phases = len(self.perception_connector.phase_manager.task_execution_phases)
            guidance_info += f"Current Phase: {phase_id + 1}/{total_phases} - Focus on the following assignments:\n"
        except (KeyError, AttributeError):
            guidance_info += f"Current Phase - Focus on the following assignments:\n"
        
        # 添加任务分配信息（核心内容）
        if agent_task_assignments:
            for agent_id, tasks in agent_task_assignments.items():
                if tasks:
                    try:
                        task_summaries = []
                        for t in tasks:
                            task_type = t.get('task_type', 'Unknown')
                            target = t.get('target', 'Unknown')
                            task_summaries.append(f"{task_type}[{target}]")
                        task_list = ", ".join(task_summaries)
                        guidance_info += f"• Agent {agent_id}: {task_list}\n"
                    except Exception:
                        guidance_info += f"• Agent {agent_id}: Task assignment error\n"
                else:
                    guidance_info += f"• Agent {agent_id}: Wait[]\n"
        
        if world_state:
            guidance_info += "\n**Current Known World State:**\n"
            object_positions = world_state.get('object_positions')
            if object_positions:
                objects_info = []
                for name, info in object_positions.items():
                    if info and 'parent' in info:
                        objects_info.append(f"- {name} (on/in {info['parent']})")
                    else:
                        objects_info.append(f"- {name} (position unknown)")
                guidance_info += "Objects: " + ", ".join(objects_info) + "\n"
            else:
                guidance_info += "Objects: None found yet.\n"

        # 添加简化的执行提示
        guidance_info += f"\nRemember to explore if objects are not found yet, and coordinate to avoid conflicts."
        
        return guidance_info

    def _build_phase_aware_prompt_with_miqp_guidance(
        self,
        instruction: str,
        current_phase_tasks: List[Dict[str, Any]],
        agent_task_assignments: Dict[int, List[Dict[str, Any]]],
        current_phase: Dict[str, Any],
        alpha_result: Optional[Any] = None,
        llm_response: Optional[str] = None,
        world_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        **DEPRECATED** 构建包含阶段感知和MIQP优化指导的增强型prompt。
        现在使用_build_miqp_guidance_addition + curr_prompt的方式。
        """
        # 获取基础prompt
        base_prompt, _ = self.prepare_prompt(
            instruction, 
            self.env_interface.world_graph[self._agents[0].uid]
        )
        
        # 获取MIQP指导信息
        guidance_info = self._build_miqp_guidance_addition(
            current_phase_tasks, agent_task_assignments, current_phase, alpha_result, world_state
        )
        
        # 组合完整prompt
        enhanced_prompt = base_prompt + guidance_info
        return enhanced_prompt

    def _adjust_actions_with_phase_awareness(
        self,
        high_level_actions: Dict[int, Tuple[str, str, Optional[str]]],
        agent_task_assignments: Dict[int, List[Dict[str, Any]]],
        current_phase: Dict[str, Any]
    ) -> Dict[int, Tuple[str, str, Optional[str]]]:
        """
        根据当前阶段和任务分配调整动作，确保符合阶段约束。
        """
        adjusted_actions = {}
        # current_phase_task_types = {task['task_type'] for task in current_phase['tasks']}
        
        print(f"[DEBUG] **NEW** Adjusting actions for phase {current_phase['phase_id']}:")
        
        all_assigned_tasks = []
        for agent_id in agent_task_assignments:
            all_assigned_tasks.extend(agent_task_assignments[agent_id])

        for agent_id, action_tuple in high_level_actions.items():
            assigned_tasks = agent_task_assignments.get(agent_id, [])

            if not assigned_tasks:
                adjusted_actions[agent_id] = ("Wait", "", "")
                print(f"    Agent {agent_id}: No task assigned, will Wait.")
                continue

            # Agent has tasks, generate action from the first one
            assigned_task = assigned_tasks[0]
            replacement_action = self._generate_action_from_subtask(assigned_task)

            if replacement_action:
                adjusted_actions[agent_id] = replacement_action
                print(f"    Agent {agent_id}: Assigned task is {assigned_task['task_type']}, executing {replacement_action[0]}.")
            else:
                adjusted_actions[agent_id] = ("Wait", "", "")
                print(f"    Agent {agent_id}: Could not generate action from task, will Wait.")

        # Ensure all agents have an action
        for agent_id in range(len(self._agents)):
            if agent_id not in adjusted_actions:
                adjusted_actions[agent_id] = ("Wait", "", "")
                print(f"    Agent {agent_id}: No action from LLM, will Wait.")

        return adjusted_actions

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

    def _get_agent_capabilities(self) -> Dict[int, List[str]]:
        return {
            0: ['Navigate', 'Explore', 'Pick', 'Place', 'Open', 'Close', 'Rearrange', 'Wait'],
            1: ['Navigate', 'Explore', 'Pick', 'Place', 'Open', 'Close', 'Clean', 'Fill', 'Pour', 'PowerOn', 'PowerOff', 'Rearrange', 'Wait']
        }

    def _validate_miqp_solution(self, alpha, agent_capabilities: Dict[int, List[str]]) -> bool:
        """验证MIQP解决方案"""
        if alpha is None:
            return False
        
        if isinstance(alpha, np.ndarray):
            return alpha.any()
        elif hasattr(alpha, '__iter__'):
            return not all(a == 0 for a in alpha)
        else:
            return alpha != 0

    def _heuristic_task_assignment(self, tasks: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """**FIXED** 阶段感知的启发式任务分配，特别处理单任务阶段"""
        assignments = {i: [] for i in range(len(self._agents))}
        
        if not tasks:
            print("[DEBUG] No tasks to assign")
            return assignments
        
        print(f"[DEBUG] Assigning {len(tasks)} phase tasks to {len(self._agents)} agents")
        
        # 基于智能体能力进行智能分配
        agent_capabilities = self._get_agent_capabilities()
        
        # 特殊处理单任务阶段
        if len(tasks) == 1:
            task = tasks[0]
            task_type = task.get('task_type', 'Wait')
            target = task.get('target', '')
            
            capable_agents = []
            for agent_id, capabilities in agent_capabilities.items():
                if task_type in capabilities:
                    capable_agents.append(agent_id)
            
            # 选择能力值最高的智能体（能力值为capabilities列表长度），如有多个则取第一个，否则为0
            if capable_agents:
                chosen_agent = max(capable_agents, key=lambda aid: len(agent_capabilities[aid]))
            else:
                chosen_agent = 0
            
            # 添加分配信息到任务中
            assigned_task = task.copy()
            assigned_task.update({
                'assigned_agent': chosen_agent,
                'assignment_method': 'heuristic_single_task',
                'assignment_confidence': 'High' if capable_agents else 'Low'
            })
            
            assignments[chosen_agent].append(assigned_task)
            print(f"[DEBUG] **HEURISTIC** Single task assignment: {task_type}({target}) → Agent {chosen_agent}")
            print(f"[DEBUG] **HEURISTIC** Agent {chosen_agent} capabilities: {agent_capabilities.get(chosen_agent, [])}")
            
            # 其他智能体设置为空（会在action生成时处理Wait）
            for agent_id in range(len(self._agents)):
                if agent_id != chosen_agent:
                    print(f"[DEBUG] **HEURISTIC** Agent {agent_id} will wait (no task assigned)")
        
        else:
            # 多任务阶段，使用原来的分配逻辑
            for task in tasks:
                task_type = task.get('task_type', 'Wait')
                target = task.get('target', '')
                
                # 找到能执行此任务的智能体
                capable_agents = []
                for agent_id, capabilities in agent_capabilities.items():
                    if task_type in capabilities:
                        capable_agents.append(agent_id)
                
                if capable_agents:
                    # 选择负载最轻的智能体
                    chosen_agent = min(capable_agents, key=lambda aid: len(assignments[aid]))
                    
                    # 添加分配信息到任务中
                    assigned_task = task.copy()
                    assigned_task.update({
                        'assigned_agent': chosen_agent,
                        'assignment_method': 'heuristic_multi_task',
                        'assignment_confidence': 'Medium'
                    })
                    
                    assignments[chosen_agent].append(assigned_task)
                    print(f"[DEBUG] **HEURISTIC** Multi-task assignment: {task_type}({target}) → Agent {chosen_agent}")
                else:
                    # 如果没有智能体能执行，分配给第一个智能体
                    assigned_task = task.copy()
                    assigned_task.update({
                        'assigned_agent': 0,
                        'assignment_method': 'heuristic_fallback',
                        'assignment_confidence': 'Low'
                    })
                    
                    assignments[0].append(assigned_task)
                    print(f"[DEBUG] **HEURISTIC** Fallback assignment: {task_type}({target}) → Agent 0 (no capable agent found)")
        
        # 显示最终分配结果
        for agent_id, agent_tasks in assignments.items():
            if agent_tasks:
                task_summaries = [f"{t['task_type']}→{t.get('target', '')}" for t in agent_tasks]
                print(f"[DEBUG] **HEURISTIC** Agent {agent_id}: {task_summaries}")
            else:
                print(f"[DEBUG] **HEURISTIC** Agent {agent_id}: [idle/waiting]")
        
        return assignments

    def _generate_action_from_subtask(self, subtask: Dict[str, Any]) -> Optional[Tuple[str, str, Optional[str]]]:
        """从子任务生成动作"""
        task_type = subtask.get('task_type', '')
        target = subtask.get('target', '')
        
        if not task_type or not target:
            return None
            
        return (task_type, target, None)

    def _action_matches_task_type(self, action_name: str, task_type: str) -> bool:
        return action_name == task_type or (
            action_name in ['Pick', 'Place'] and task_type == 'Rearrange'
        )

    def _create_fallback_tasks(self, instruction: str) -> List[Dict[str, Any]]:
        """创建更稳健的、分阶段的后備任務計劃。"""
        
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
            'max_parallel_tasks': len(self._agents),
            'estimated_duration': 20.0,
            'required_agents': len(self._agents)
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
            'max_parallel_tasks': len(self._agents),
            'estimated_duration': 5.0,
            'required_agents': len(self._agents)
        }

        print(f"[DEBUG] Created a robust 2-phase fallback plan: Explore -> Wait")
        return [explore_phase, wait_phase]

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
        阶段感知的MIQP求解器：使用固定13×5的T矩阵维度
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
            n_total_tasks = phase_task_info.get('n_total_tasks', 13) 
            n_phase_tasks = phase_task_info['n_phase_tasks']
            
            phase_scenario_params = self._create_phase_scenario_params(phase_task_info)
            
            if self.rta is not None:
                try:
                    alpha, u, delta, time_to_solve, opt_sol_info = self.rta.solve_miqp_phase_aware(
                        x, t, phase_scenario_params, n_total_tasks 
                    )
                except AttributeError:
                    print("[WARNING] RTA doesn't support phase-aware solving, using heuristic")
                    alpha, u, delta, time_to_solve, opt_sol_info = self._heuristic_phase_solve(
                        x, t, phase_task_info
                    )
            else:
                print("[WARNING] No RTA solver available, using heuristic phase solve")
                alpha, u, delta, time_to_solve, opt_sol_info = self._heuristic_phase_solve(
                    x, t, phase_task_info
                )
            
            solve_time = time_to_solve
            
            # 验证返回值具有正确的固定维度
            if alpha is not None:
                alpha = np.array(alpha)
                expected_shape = (n_agents, n_total_tasks)  # 固定13任务维度
                
                if alpha.ndim == 1:
                    # 如果返回的是1D数组，重塑为期望形状
                    expected_size = n_agents * n_total_tasks
                    if alpha.size == expected_size:
                        alpha = alpha.reshape(expected_shape)
                    else:
                        print(f"[WARNING] Alpha size {alpha.size} doesn't match expected {expected_size}")
                        alpha = np.ones(expected_shape) * 0.5
                elif alpha.shape != expected_shape:
                    print(f"[WARNING] Alpha shape {alpha.shape} doesn't match expected {expected_shape}")
                    # 使用启发式方法重建alpha
                    alpha = self._rebuild_alpha_for_fixed_dimensions(alpha, n_agents, n_total_tasks, phase_task_info)
                
                print(f"[DEBUG] **FIXED** Phase-aware MIQP solved successfully in {solve_time:.4f}s")
                print(f"[DEBUG] **FIXED** Alpha matrix shape: {alpha.shape} (agents={n_agents}, total_tasks={n_total_tasks})")
                print(f"[DEBUG] **FIXED** Active tasks: {n_phase_tasks}/{n_total_tasks}")
                return alpha, u, delta, solve_time, opt_sol_info
            else:
                print(f"[DEBUG] Phase-aware MIQP solve failed in {solve_time:.4f}s")
                return None, None, None, solve_time, "INFEASIBLE"
                
        except Exception as e:
            solve_time = time.time() - start_time if 'start_time' in locals() else 0.0
            print(f"[ERROR] Phase-aware MIQP solve exception: {e}")
            # 返回固定维度的fallback解决方案
            n_agents = len(self._agents) if hasattr(self, '_agents') else 2
            n_total_tasks = phase_task_info.get('n_total_tasks', 13) if phase_task_info else 13
            alpha = np.ones((n_agents, n_total_tasks)) * 0.5
            u = np.zeros((3, n_agents))
            delta = np.ones(n_total_tasks)
            return alpha, u, delta, solve_time, f"EXCEPTION_FALLBACK: {str(e)}"

    def _rebuild_alpha_for_fixed_dimensions(self, original_alpha, n_agents, n_total_tasks, phase_task_info):
        """重建alpha矩阵以适应固定的13×5维度"""
        try:
            fixed_alpha = np.zeros((n_agents, n_total_tasks))
            active_indices = phase_task_info.get('indices', [])
            
            if hasattr(original_alpha, 'shape') and len(active_indices) > 0:
                if original_alpha.shape == (n_agents, len(active_indices)):
                    for i, task_idx in enumerate(active_indices):
                        if task_idx < n_total_tasks:
                            fixed_alpha[:, task_idx] = original_alpha[:, i]
                elif original_alpha.size == n_agents * len(active_indices):
                    reshaped = original_alpha.reshape(n_agents, len(active_indices))
                    for i, task_idx in enumerate(active_indices):
                        if task_idx < n_total_tasks:
                            fixed_alpha[:, task_idx] = reshaped[:, i]
                else:
                    for task_idx in active_indices:
                        if task_idx < n_total_tasks:
                            fixed_alpha[:, task_idx] = 0.5
            else:
                for task_idx in active_indices:
                    if task_idx < n_total_tasks:
                        fixed_alpha[:, task_idx] = 0.5
            
            print(f"[DEBUG] Rebuilt alpha from shape {getattr(original_alpha, 'shape', 'unknown')} to {fixed_alpha.shape}")
            return fixed_alpha
            
        except Exception as e:
            print(f"[ERROR] Failed to rebuild alpha: {e}")
            # 最终fallback
            fallback_alpha = np.zeros((n_agents, n_total_tasks))
            active_indices = phase_task_info.get('indices', [12]) 
            for task_idx in active_indices:
                if task_idx < n_total_tasks:
                    fallback_alpha[:, task_idx] = 0.5
            return fallback_alpha

    def _create_phase_scenario_params(self, phase_task_info):
        """创建阶段特定的scenario参数"""
        try:
            phase_params = self.scenario_params.scenario_params.copy()
            phase_params['T'] = phase_task_info['matrix']
            n_phase_tasks = phase_task_info['n_phase_tasks']
            if 'ws' in phase_params:
                original_ws = phase_params['ws']
                if isinstance(original_ws, list) and len(original_ws) == 5:
                    phase_params['ws'] = original_ws
                else:
                    phase_params['ws'] = [w for w in original_ws[:n_phase_tasks]]
            return phase_params
            
        except Exception as e:
            print(f"[ERROR] Failed to create phase scenario params: {e}")
            return self.scenario_params.scenario_params if self.scenario_params else {}

    def _heuristic_phase_solve(self, x, t, phase_task_info):
        """启发式阶段求解方案，使用固定13×5维度"""
        start_time = time.time()
        
        n_agents = len(self._agents)
        n_total_tasks = phase_task_info.get('n_total_tasks', 13)  # 固定维度
        n_phase_tasks = phase_task_info['n_phase_tasks']
        task_types = phase_task_info['types']
        active_indices = phase_task_info.get('indices', [])
        
        print(f"[DEBUG] **HEURISTIC** Phase solve: {n_agents} agents, {n_phase_tasks} active tasks out of {n_total_tasks} total")
        
        alpha = np.zeros((n_agents, n_total_tasks))
        agent_capabilities = self._get_agent_capabilities()
        
        for i, task_idx in enumerate(active_indices):
            if i < len(task_types):
                task_type = task_types[i]
                
                capable_agents = []
                for agent_id, capabilities in agent_capabilities.items():
                    if task_type in capabilities:
                        capable_agents.append(agent_id)
                
                if capable_agents:
                    if n_phase_tasks == 1 and len(capable_agents) > 1:
                        chosen_agent = capable_agents[0]
                        alpha[chosen_agent, task_idx] = 1.0
                        print(f"[DEBUG] **HEURISTIC** Assigned {task_type} (task {task_idx}) to Agent {chosen_agent}")
                    elif len(capable_agents) == 1:
                        alpha[capable_agents[0], task_idx] = 1.0
                        print(f"[DEBUG] **HEURISTIC** Assigned {task_type} (task {task_idx}) to Agent {capable_agents[0]} (only capable)")
                    else:
                        weight = 1.0 / len(capable_agents)
                        for agent_id in capable_agents:
                            alpha[agent_id, task_idx] = weight
                        print(f"[DEBUG] **HEURISTIC** Split {task_type} (task {task_idx}) among agents {capable_agents}")
                else:
                    alpha[0, task_idx] = 1.0
                    print(f"[DEBUG] **HEURISTIC** Fallback: assigned {task_type} (task {task_idx}) to Agent 0")
        
        if n_phase_tasks >= n_agents:
            for agent_id in range(n_agents):
                agent_total = np.sum(alpha[agent_id, :])
                if agent_total < 0.001:
                    if active_indices:
                        task_loads = np.array([np.sum(alpha[:, task_idx]) for task_idx in active_indices])
                        lightest_task_idx = np.argmin(task_loads)
                        actual_task_idx = active_indices[lightest_task_idx]
                        alpha[agent_id, actual_task_idx] = 0.5
                        task_type = task_types[lightest_task_idx] if lightest_task_idx < len(task_types) else "Unknown"
                        # print(f"[DEBUG] **HEURISTIC** Ensured Agent {agent_id} has task {task_type} (task {actual_task_idx})")
        
        u = np.zeros((3, n_agents))
        delta = np.ones(n_total_tasks)
        
        solve_time = time.time() - start_time
        
        print(f"[DEBUG] **HEURISTIC** Final alpha matrix shape: {alpha.shape}")
        print(f"[DEBUG] **HEURISTIC** Non-zero assignments: {np.sum(alpha > 0.001)} out of {n_agents * n_total_tasks}")
        
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
                start_time = time.time()

            llm_response = self.replan(instruction, observations, world_graph)
            thought = self.parse_thought(llm_response)

            if verbose:
                total_time = time.time() - start_time
                print(
                    f"Time taken for LLM response generation: {total_time}; replanning_count: {self.replanning_count}"
                )
            
            # high_level_actions = self._parse_high_level_actions(llm_response)
            print_str += f"""{llm_response}\n{self.stopword}\n"""
            prompt_addition = (
                f"""{llm_response}\n{self.stopword}{self.planner_config.llm.eot_tag}"""
            )
            self.curr_prompt += prompt_addition
            self.trace += prompt_addition
            self.is_done = (self.check_if_agent_done(llm_response)) or (
                self.replanning_count == self.planner_config.replanning_threshold
            )
            self.replanning_count += 1
            
            if self.is_done:
                planner_info = {
                    "print": print_str,
                    # "print_no_tags": print_str_no_tags,
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
                    # "high_level_actions": {
                    #     agent.uid: high_level_actions.get(agent.uid, ("Done", None, None)) for agent in self.agents
                    # },
                    "high_level_actions": {
                        agent.uid: ("Done", None, None) for agent in self.agents
                    },
                }
                return {}, planner_info, self.is_done
            
            high_level_actions = self.actions_parser(
                self.agents, llm_response, self.params # or current_params
            )
            print(f"\n\n[DEBUG-LYP-v3] Parsed High-Level Actions before done check: {high_level_actions}\n\n")
            # Get low level actions and/or responses
            low_level_actions, responses = self.process_high_level_actions(
                high_level_actions, observations
            )
            # Store last executed high level action
            self.last_high_level_actions = high_level_actions

        else:
            planner_info["replanned"] = {agent.uid: False for agent in self.agents}
            thought = None

            # Get low level actions and/or responses using last high level actions
            low_level_actions, responses = self.process_high_level_actions(
                self.last_high_level_actions, observations
            )

        # Log if replanning was done or not before overwriting the value
        planner_info["replan_required"] = {
            agent.uid: self.replan_required for agent in self.agents
        }

        # 简化的阶段推进检查（仅检查是否需要强制重新规划）
        # if (hasattr(self, 'perception_connector') and self.perception_connector and 
        #     hasattr(self, '_phase_transition_pending') and self._phase_transition_pending):
        #     # 重置标志
        #     self._phase_transition_pending = False
        #     self.replan_required = True
        #     print(f"[INFO] **NEW** Phase transition detected, forcing plan on this iteration")

        # Check if replanning is required
        # Replanning is required when any of the actions being executed
        # have a response indicating success or failure (and the reason)
        self.replan_required = any(responses.values())
        print_str += self._add_responses_to_prompt(responses)

        # Update planner info
        planner_info.update({
            "responses": responses,
            "thought": {agent.uid: thought for agent in self.agents},
            "is_done": {agent.uid: self.is_done for agent in self.agents},
            "print": print_str,
            "high_level_actions": self.last_high_level_actions,
            "prompts": {agent.uid: self.curr_prompt for agent in self.agents},
            "traces": {agent.uid: self.trace for agent in self.agents},
            "replanning_count": {
                agent.uid: self.replanning_count for agent in self.agents
            },
            "agent_states": self.get_last_agent_states(),
            "agent_positions": self.get_last_agent_positions(),
            "agent_collisions": self.get_agent_collisions(),
        })
        
        # if hasattr(self, '_last_response_info') and self._last_response_info:
        #     planner_info["miqp_info"] = {
        #         "optimization_success": self._last_response_info.get("optimization_success", False),
        #         "miqp_status": self._last_response_info.get("miqp_status", "UNKNOWN"),
        #         "current_phase": self._last_response_info.get("current_phase", {}),
        #         "task_decomposition_success": self._last_response_info.get("task_decomposition_success", False)
        #     }

        return low_level_actions, planner_info, self.is_done

    def check_if_agent_done(self, llm_response: str) -> bool:
        """
        Check if the agent is done based on the LLM response.

        :param llm_response: The LLM response to check.
        :return: True if the agent is done, False otherwise.
        """
        # 1. Check for the hardcoded end expression (most reliable)
        if self.end_expression in llm_response:
            return True

        # 2. Check for semantic completion phrases
        response_lower = llm_response.lower()
        positive_phrases = [
            "task is complete",
            "task is now complete",
            "task is finished",
            "task is now finished",
            "task has been completed",
            "all tasks are complete",
            "the goal has been achieved",
            "i am done",
            "mission accomplished",
            "actions are finished",
        ]
        
        negative_phrases = [
            "not complete",
            "not finished",
            "unable to complete",
            "in progress",
        ]

        # Avoid false positives if negative phrases are present
        if any(phrase in response_lower for phrase in negative_phrases):
            return False
            
        # Check if any of the positive completion phrases are present
        if any(phrase in response_lower for phrase in positive_phrases):
            return True

        return False

    def _parse_high_level_actions(self, llm_response: str) -> Dict[int, Tuple[str, str, Optional[str]]]:
        """
        解析LLM响应中的高级动作

        :param llm_response: LLM生成的响应
        :return: 解析出的高级动作字典
        """
        try:
            if not hasattr(self, 'params') or self.params is None:
                self.params = {}
            
            return self.actions_parser(self.agents, llm_response, self.params)
        except Exception as e:
            print(f"[ERROR] 高级动作解析失败: {e}")
            return {
                agent.uid: ("Explore", "environment", None) 
                for agent in self.agents
            }
            
    def _extract_phase_alpha_from_full_matrix(self, full_alpha, n_agents, active_indices, n_current_tasks):
        """从固定13×5的alpha矩阵中提取当前阶段的任务分配"""
        try:
            if not hasattr(full_alpha, 'shape'):
                full_alpha = np.array(full_alpha)
            
            if full_alpha.ndim == 1:
                if full_alpha.size == n_agents * 13:
                    full_alpha = full_alpha.reshape(n_agents, 13)
                else:
                    print(f"[WARNING] Cannot reshape 1D alpha of size {full_alpha.size}")
                    return None
            
            if full_alpha.shape[0] != n_agents:
                print(f"[WARNING] Alpha agent dimension mismatch: {full_alpha.shape[0]} vs {n_agents}")
                return None
            
            # 提取活跃任务列
            if len(active_indices) == n_current_tasks and all(0 <= idx < full_alpha.shape[1] for idx in active_indices):
                phase_alpha = full_alpha[:, active_indices]
                print(f"[DEBUG] Extracted phase alpha {phase_alpha.shape} from full matrix {full_alpha.shape}")
                print(f"[DEBUG] Active indices used: {active_indices}")
                return phase_alpha
            else:
                print(f"[WARNING] Invalid active indices: {active_indices} for matrix shape {full_alpha.shape}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Failed to extract phase alpha: {e}")
            return None
            