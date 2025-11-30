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

from habitat_llm.planner.HRCS.plan_module.action_manager import ActionManager
from habitat_llm.planner.HRCS.plan_module.error_handler import ErrorHandler
from habitat_llm.planner.HRCS.plan_module.miqp_solver_wrapper import MIQPSolverWrapper
from habitat_llm.planner.HRCS.plan_module.prompt_builder import PromptBuilder
from habitat_llm.planner.HRCS.plan_module.task_helper import TaskHelper
from habitat_llm.planner.HRCS.plan_module.feedback_manager import FeedbackManager
from habitat_llm.planner.HRCS.plan_module.execution_manager import ExecutionManager


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
        
        self.prompt_builder = PromptBuilder(plan_config, env_interface)

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

        self.perception_connector.reset()
        self.miqp_solver_wrapper.reset()
        self.task_helper.reset()
        self.error_handler.reset()
        self.action_manager.reset()
        self.feedback_manager.reset()
        self.execution_manager.reset()
        self.prompt_builder.reset()

        # Reset agents
        for agent in self._agents:
            agent.reset()

        print("[DEBUG] LLMPlanner reset completed - all components reset")

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
        
        # MIQP components
        self.miqp_solver_wrapper = MIQPSolverWrapper()
        self.perception_connector = PerceptionConnector()
        self.task_helper = TaskHelper(self._agents)
        self.error_handler = ErrorHandler()
        self.action_manager = ActionManager(self.actions_parser)
        self.feedback_manager = FeedbackManager()
        self.execution_manager = ExecutionManager()

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
            return self.feedback_manager.get_agent_completion_statuses(self._agents, self.latest_agent_response)

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

    def get_last_agent_positions_miqp(self, world_graph: Dict[int, "WorldGraph"]) -> Dict[int, Dict[str, Any]]:
        """
        Get the last positions and rotations for all agents from the full world graph.
        This version is specifically for MIQP state input.

        :return: A dictionary mapping agent UIDs to dictionaries containing 'position' and 'rotation'.
        """
        agent_poses = {}
        agents_from_graph = self.env_interface.full_world_graph.get_agents()
        # agents_from_graph = world_graph.get_agents()
        agent_map = {f"agent_{agent.uid}": agent for agent in self._agents}

        for agent_entity in agents_from_graph:
            agent_name = agent_entity.name
            position = [0.0, 0.0, 0.0]
            try:
                position = agent_entity.get_property("translation")
            except (ValueError, AttributeError) as e:
                print(f"Warning: Could not get pose properties for agent '{agent_name}'. Using default pose. Error: {str(e)}")

            if agent_name in agent_map:
                agent_uid = agent_map[agent_name].uid
                agent_poses[agent_uid] = {'position': position}
            else:
                try:
                    agent_uid = int(agent_name.split('_')[-1])
                    agent_poses[agent_uid] = {'position': position}
                    print(f"Warning: Agent '{agent_name}' not found in agent map, but successfully parsed UID {agent_uid}.")
                except (ValueError, IndexError):
                    print(f"Warning: Could not determine UID for agent '{agent_name}'. Skipping this agent.")
        return agent_poses

    def _setup_replan_state(self, world_graph: Dict[int, "WorldGraph"]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Initializes the state for replanning by extracting world and agent states,
        and preparing scenario parameters. This combines the initial setup steps
        of the replan method.
        """
        # Step 1: Extract World State (including agent poses)
        # print(f"[Setup] Extracting world state...")
        try:
            if not hasattr(self, 'perception_connector') or self.perception_connector is None:
                self.perception_connector = PerceptionConnector(api_key_filename="api_key")
            world_state = self.perception_connector.extract_world_state(self.env_interface)
        except Exception as e:
            print(f"[ERROR] World state extraction failed during setup: {e}")
            world_state = {'agent_poses': {}, 'object_positions': {}, 'furniture_positions': {}}

        # Step 2: Extract Agent States from world_state and format for MIQP
        # print(f"[Setup] Extracting agent states for MIQP...")
        n_agents = len(self._agents)
        n_states = 3  # [x, y, z]
        x = np.zeros((n_states, n_agents))
        agent_poses = self.get_last_agent_positions_miqp(world_graph)
            
        for i, agent in enumerate(self._agents):
            agent_id = agent.uid
            if agent_id in agent_poses:
                pose_info = agent_poses[agent_id]
                position = np.array(pose_info['position'])
                x[0, i] = position[0]  # x (right & left)
                x[1, i] = position[2]  # z (front & back)
                x[2, i] = position[1]  # y (up & down)
            else:
                print(f"[WARNING] Pose info for agent {agent_id} not found. Using default zero state.")
                x[:, i] = 0.0
        # Step 3: Pre-update Scenario Params
        # print(f"[Setup] Pre-updating scenario parameters...")
        try:
            if self.miqp_solver_wrapper.scenario_params is not None:
                self.miqp_solver_wrapper.scenario_params.update_scenario_from_world_state(world_state)
                self.perception_connector.pre_update_scenario_params(
                    self.miqp_solver_wrapper.scenario_params,
                    world_state
                )
            else:
                print(f"[WARNING] scenario_params is None, skipping basic updates")
        except Exception as e:
            print(f"[ERROR] Basic parameter update failed during setup: {e}")
        return x, world_state

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

        # === Step 1: Initialize/Update Scenario Parameters ===
        print(f"[Step 1/12] Setting up MIQP scenario parameters...")
        # if not self.miqp_solver_wrapper.scenario_params:
        if self.miqp_solver_wrapper.scenario_params == None:
            self.miqp_solver_wrapper.task_plan_MIQP_set(self._agents)  
        # === Step 2: SetUp Stage ===
        print(f"[Step 2/12] Initializing replan state...")
        x, world_state = self._setup_replan_state(world_graph)
        # world_description = self.desc_world_graph(world_graph)
        # === Step 2.5: Update Agent Resumes with Context ===
        print(f"[Step 2.5/12] Updating Agent Resumes with latest context...")
        try:
            # Use the latest agent responses/feedback for updates
            self.task_helper.update_resumes_from_context(
                world_state, self.latest_agent_response, self.last_high_level_actions
            )
            # print("[DEBUG] Agent resumes updated successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to update agent resumes: {e}")

        # === Step 3: Sequenced Task Decomposition ===
        print(f"[Step 3/12] Decomposing task with sequencing...")
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
                execution_phases = self.task_helper.create_fallback_tasks(instruction, self._agents)
                self.perception_connector.phase_manager.set_execution_phases(execution_phases)
                print(f"[DEBUG] Fallback decomposition created {len(execution_phases)} phases")
            except Exception as fallback_error:
                print(f"[ERROR] Even fallback decomposition failed: {fallback_error}")
                return "Agent_0_Action: Explore[bedroom_1]\nAgent_1_Action: Wait[]\nAssigned!"

        # === Step 4: Get Current Phase Tasks ===
        print(f"[Step 4/12] Getting and enriching current phase tasks...")
        current_phase = self.perception_connector.get_enriched_current_phase(world_state)
        if not current_phase:
            print(f"[INFO] No more phases to execute - task completed!")
            return "Final Thought: Exit!"

        current_phase_tasks = current_phase.get('tasks', [])
        print(f"[DEBUG] Current phase {current_phase['phase_id']} has {len(current_phase_tasks)} tasks:")
        for task in current_phase_tasks:
            task_info = f"  - {task['task_type']} → {task['target']}"
            if 'target_pos' in task:
                pos = task['target_pos']
                task_info += f" (Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}])"
            print(task_info)

        # === Step 5: Build Phase-Specific T Matrix ===
        print(f"[Step 5/12] Building phase-specific T matrix...")
        try:
            if self.miqp_solver_wrapper.scenario_params is not None:
                phase_t_matrix, active_task_indices, active_task_types = self.perception_connector.build_phase_specific_t_matrix(
                    current_phase,
                    self.perception_connector.BASE_TASK_CAPABILITY_REQUIREMENTS
                )
                n_instances = phase_t_matrix.shape[0]
                aptitude_matrix = self.task_helper.get_aptitude_matrix(current_phase_tasks, self._agents)
                phase_task_info = {
                    'matrix': phase_t_matrix,
                    'indices': active_task_indices,
                    'types': active_task_types,
                    'tasks': current_phase_tasks,
                    'n_phase_tasks': n_instances,
                    'n_total_tasks': n_instances,
                    'aptitude_matrix': aptitude_matrix  # Add to phase info
                }
            else:
                print(f"[WARNING] Cannot build phase-specific T matrix - missing scenario_params")
        except Exception as e:
            print(f"[ERROR] Phase-specific T matrix building failed: {e}")
            wait_t_matrix, wait_indices, wait_types = self.perception_connector.build_phase_specific_t_matrix({'tasks': []})
            phase_task_info = {
                'matrix': wait_t_matrix,
                'indices': wait_indices,
                'types': wait_types,
                'n_phase_tasks': 1,
                'n_total_tasks': 1,
                'aptitude_matrix': np.ones((len(self._agents), 1))
            }

        # === Step 6: Update Other MIQP Matrices ===
        print(f"[Step 6/12] Updating other MIQP matrices...")
        try:
            if self.miqp_solver_wrapper.scenario_params is not None and len(current_phase_tasks) > 0:
                if self.miqp_solver_wrapper.opt_params:
                    instance_specific_bounds = self.miqp_solver_wrapper.opt_params.get_instance_robot_bounds(current_phase_tasks)
                    phase_task_info['n_r_bounds'] = instance_specific_bounds
                
                # Refer to current phase task instances
                self.perception_connector.update_scenario_for_phase_execution(
                    self.miqp_solver_wrapper.scenario_params,
                    current_phase_tasks,
                    world_state
                )
                
                self.perception_connector.update_miqp_matrices(
                    self.miqp_solver_wrapper.scenario_params,
                    phase_t_matrix,
                    current_phase_tasks,
                    world_state
                )
            else:
                print(f"[WARNING] Cannot update MIQP matrices - missing params or tasks")
        except Exception as e:
            print(f"[ERROR] MIQP matrices update failed: {e}")

        # === Step 7: MIQP Optimization ===
        print(f"[Step 7/12] Running MIQP optimization for current phase...")
        # print("====="*10)
        # print(f"[DEBUG] x: {x}")
        # print(f"[DEBUG] phase_task_info: {phase_task_info}")
        # print(f"[DEBUG] self._agents: {self._agents}")
        # print("====="*10)
        alpha, u, delta, time_to_solve, opt_sol_info = self.miqp_solver_wrapper.task_plan_MIQP_solve_phase_aware(
            x, t, phase_task_info, self._agents
        )
        print("[DEBUG-LYP-v3] MIQP optimization results:")
        print(f"  - Alpha: {alpha}")
        
        # === Step 8: Phase-Specific Task Assignment ===
        print(f"[Step 8/12] Mapping current phase tasks to agents...")
        agent_task_assignments = self.task_helper.assign_tasks_for_phase(
            current_phase_tasks,
            alpha,
            phase_task_info,
            self._agents,
            self.perception_connector,
            self.miqp_solver_wrapper
        )

        # === Step 9: Apply Intelligent Error Recovery ===
        print(f"[Step 9/12] Applying intelligent error recovery...")
        agent_task_assignments = self.error_handler.recover_and_log_assignments(
            agent_task_assignments,
            self.last_high_level_actions,
            self.latest_agent_response
        )

        # === Step 10: Build Phase-Aware Prompt ===
        print(f"[Step 10/12] Building phase-aware enhanced prompt...")
        try:
            miqp_guidance = self.prompt_builder.build_miqp_guidance_addition(
                current_phase_tasks,
                agent_task_assignments,
                current_phase,
                self.perception_connector,
                alpha_result=alpha,
                world_state=world_state
            )
        except Exception as e:
            print(f"[ERROR] MIQP guidance building failed: {e}")
            miqp_guidance = ""

        # === Step 11: LLM Action Generation (Enhanced with Feedback) ===
        print(f"[Step 11/12] Generating actions via LLM with complete feedback history...")
        try:
            # 1. Prepare prompt
            prompt_for_llm = self.prompt_builder.prepare_llm_prompt(self.curr_prompt, miqp_guidance)
            
            # Debug info
            if miqp_guidance:
                print(f"[DEBUG] MIQP guidance injected ({len(miqp_guidance)} chars):")
                print(f"  {miqp_guidance}")
            else:
                print(f"[DEBUG] No MIQP guidance to inject")
            
            # 2. Generate Response
            if self.planner_config.get("constrained_generation", False):
                print("[DEBUG-LYP] Now use constrained generation")
                raw_response = self.llm.generate(
                    self.curr_prompt,
                    self.stopword,
                    generation_args={
                        "grammar_definition": self.build_response_grammar(
                            world_graph[self._agents[0].uid]
                        )
                    },
                )
                compare_response = self.llm.generate(
                    prompt_for_llm,
                    self.stopword,
                    generation_args={
                        "grammar_definition": self.build_response_grammar(
                            world_graph[self._agents[0].uid]
                        )
                    },
                )
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
            print(f"###################[DEBUG-LYP] compare_response: \n{compare_response}")
            # raw_response = compare_response
            
            # 3. Form expression
            llm_response = self.format_response(raw_response, self.end_expression)
            # print(f"[DEBUG-LYP] Formatted LLM response: \n{llm_response}")
            if not llm_response or llm_response == "Thought:":
                print(f"[WARNING] Empty LLM response received after formatting.")
                llm_response = "Agent_0_Action: Wait[]\nAgent_1_Action: Wait[]\nAssigned!"
        except Exception as e:
            print(f"[ERROR] LLM action generation failed: {e}")
            llm_response = "Agent_0_Action: Wait[]\nAgent_1_Action: Wait[]\nAssigned!"

        # === Step 12: Parse and Validate High-Level Actions with Closed-Loop Feedback ===
        print(f"[Step 12/12] Parsing and validating actions with closed-loop feedback...")
        try:
            current_agent_positions = self.get_last_agent_positions_miqp(world_graph)
            validated_actions = self.action_manager.parse_and_validate_actions(
                llm_response, 
                self._agents, 
                world_graph, 
                self.execution_manager
            )
            adjusted_actions = self.action_manager.adjust_actions_with_phase_awareness(
                validated_actions, 
                agent_task_assignments,
                current_phase,
                self._agents
            )
            self.execution_manager.record_execution_context(
                adjusted_actions,
                observations,
                current_agent_positions
            )
            validation_summary = self.action_manager.get_action_validation_summary()
            # print(f"[DEBUG] CLOSED-LOOP FEEDBACK Summary:")
            # print(f"  Total actions processed: {validation_summary['total_actions']}")
            # print(f"  Corrections made: {validation_summary['corrections_made']}")
            # print(f"  Correction rate: {validation_summary['correction_rate']:.2%}")
            print(f"[DEBUG] FINAL Validated actions:")
            for agent_id, action in adjusted_actions.items():
                if action and len(action) >= 3:
                    print(f"  Agent {agent_id}: {action[0]}({action[1]}) → {action[2]}")
        except Exception as e:
            print(f"[ERROR] Closed-loop action validation failed: {e}")
            try:
                adjusted_actions = self.action_manager.parse_high_level_actions(llm_response, self._agents)
            except:
                adjusted_actions = {}

        # === Step 13: Update Scenario Parameters ===
        print(f"[Step Feedback] Updating scenario parameters for execution...")
        try:
            if self.miqp_solver_wrapper.scenario_params is not None and adjusted_actions:
                self.execution_manager.update_scenario_for_execution(
                    self.miqp_solver_wrapper.scenario_params,
                    world_state,
                    adjusted_actions
                )
            else:
                print(f"[WARNING] Cannot update scenario params - missing params or actions")
        except Exception as e:
            print(f"[ERROR] Scenario parameter update failed: {e}")

        # === Step 14: Phase Completion Check and Advancement ===
        self._phase_transition_pending = self.feedback_manager.check_and_advance_phase(
            self.perception_connector,
            self._agents,
            self.latest_agent_response,
            current_phase
        )
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
            "llm_response": llm_response,
            "adjusted_actions": adjusted_actions
        }

        print(f"\n[SUCCESS] MIQP Sequential Phase Plan completed!")
        print(f"  Current Phase: {current_phase['phase_id'] + 1}/{len(self.perception_connector.phase_manager.task_execution_phases)}")
        print(f"  Phase Tasks: {len(current_phase_tasks)}")
        print(f"  Optimization: {opt_sol_info}")
        print("="*80)
        
        # 返回LLM响应
        return llm_response

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

        # 简化的阶段推进检查
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
        if self.end_expression in llm_response:
            return True
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
        if any(phrase in response_lower for phrase in negative_phrases):
            return False
        if any(phrase in response_lower for phrase in positive_phrases):
            return True
        return False
            