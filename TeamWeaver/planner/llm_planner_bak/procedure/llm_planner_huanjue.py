#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import re
import time
import os
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

# MIQP Planner 相关 imports (修改为引用 HRCS 子目录)
from habitat_llm.planner.HRCS.params_module.scenario_params_task import ScenarioConfigTask
from habitat_llm.planner.HRCS.params_module.opt_params_task import OptimizationConfigTask
from habitat_llm.planner.HRCS.class_def.GlobalVarsManager_task import GlobalVarsManager_task
from habitat_llm.planner.HRCS.class_def.RTA_task import RTA
from habitat_llm.planner.perception_connector import PerceptionConnector
import numpy as np
import logging # 添加日志模块

# 导入幻觉评估模块
from habitat_llm.evaluation.coherence_evaluator import CoherenceEvaluator
from habitat_llm.evaluation.truthfulness_evaluator import TruthfulnessEvaluator

# 将四元数转换为弧度值
def quaternion_to_yaw(quaternion):
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

        # MIQP Planner 相关初始化 (移到 task_plan_MIQP_set 中动态创建)
        self.scenario_params = None
        self.opt_params = None
        self.miqp_globalvars = None
        self.rta = None

        # 初始化 PerceptionConnector
        self.perception_connector = PerceptionConnector()

        # 初始化幻觉评估器 - 始终启用
        self.enable_hallucination_eval = True # 强制始终启用
        self.coherence_evaluator = CoherenceEvaluator()
        self.truthfulness_evaluator = TruthfulnessEvaluator()
        
        # 存储评估历史
        self.planning_trace_history = []
        self.world_state_history = []
        self.ground_truth_history = []
        self.miqp_assignment_history = []
        self.hallucination_metrics_history = []

        # 初始化幻觉评估日志记录器
        self.hallucination_logger = logging.getLogger("HallucinationEvalLogger")
        # 如果logger没有handler, 才添加，避免重复添加
        if not self.hallucination_logger.handlers:
            # 使用与 planner_demo.py 一致的日志文件
            # 注意：确保 planner_demo.py 中的日志配置允许此 logger 写入
            # 或者为此 logger 配置单独的文件处理器
            # 如果 plan_config 中没有提供 hallucination_log_path，则默认在当前目录下生成日志文件
            log_file_path = plan_config.get("hallucination_log_path", "hallucination_eval.log")
            # 确保日志文件直接在当前工作目录下创建
            if "/" not in log_file_path and "\\" not in log_file_path: # 简单判断是否为相对路径且不包含子目录
                log_file_path = os.path.join(os.getcwd(), log_file_path)

            file_handler = logging.FileHandler(log_file_path)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.hallucination_logger.addHandler(file_handler)
            self.hallucination_logger.setLevel(logging.INFO)
            self.hallucination_logger.propagate = False # 防止日志向上传播到root logger，避免重复打印到控制台

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

        # save agent observations to get feedback on skill execution
        self.latest_agent_response: Dict[int, str] = {}
        self.curr_prompt: str = ""
        self.trace: str = ""
        self.curr_obj_states: str = ""
        self.params: Dict[str, Any] = {}

        # 重置幻觉评估历史
        if self.enable_hallucination_eval:
            self.planning_trace_history = []
            self.world_state_history = []
            self.ground_truth_history = []
            self.miqp_assignment_history = []
            self.hallucination_metrics_history = []

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
    ):
        """
        Replan a high level action using the LLM/VLM and MIQP solver.
        """
        # 1. 设置/更新 MIQP 相关参数和 RTA 实例
        self.task_plan_MIQP_set()

        # === Call the new description method ===
        # detailed_world_desc = self.desc_world_graph(world_graph)
        # print(f"\n{detailed_world_desc}\n") # Print for debugging
        # =======================================

        # 2. 提取当前机器人状态 x = [x, y, theta] for MIQP
        alpha, u, delta, time_to_solve, opt_sol_info = None, None, None, 0, "Not Solved"
        # if self.rta is not None:
        try:
            n_r = len(self._agents)
            n_x = 3
            x = np.zeros((n_x, n_r))
            agent_poses = self.get_last_agent_positions_miqp(world_graph)

            for i, agent in enumerate(self._agents):
                agent_uid = agent.uid
                pose_info = agent_poses.get(agent_uid)

                if pose_info:
                    position = pose_info['position']
                    rotation_quat = pose_info['rotation']
                else:
                    # print(f"Warning: Pose info not found for agent {agent_uid} in get_last_agent_positions result. Using default pose.\n")
                    position = [0.0, 0.0, 0.0]
                    rotation_quat = [0.0, 0.0, 0.0, 1.0]

                theta = quaternion_to_yaw(rotation_quat)

                # Habitat 的坐标系: X 正方向为右，Y 正方向为上，Z 正方向为后 (屏幕向内)
                # position[0] 是 X, position[2] 是 Z (映射到 MIQP 的 Y)
                x[0, i] = position[0]
                x[1, i] = position[2] # 使用 Z 坐标作为 MIQP 的 Y 坐标
                x[2, i] = theta        # yaw angle (theta)

            # 时间变量 (保持现有逻辑，或根据需要调整为仿真时间)
            t = (time.time() % 300) + 2

            # 3. 使用 PerceptionConnector 提取世界状态并更新任务参数
            try:
                # 提取世界状态
                world_state = self.perception_connector.extract_world_state(self.env_interface)
                # print(f"\n#######################[DEBUG-LYP] Extracted World State: {world_state}\n")

                # 调用 pre_update (仅基于世界状态)
                if self.scenario_params is not None:
                    self.perception_connector.pre_update_scenario_params(
                        self.scenario_params,
                        world_state
                    )
                    # print(f"\n#######################[DEBUG-LYP] Executed Pre-Update Scenario Parameters\n")

            except Exception as e:
                print(f"\n\n#######################[DEBUG-LYP] Error during world state extraction or parameter update: {str(e)}\n\n")

            # 4. 调用 MIQP 求解器
            # print(f"\n#######################[DEBUG-LYP] Calling task_plan_MIQP_solve with x: {x}, t: {t}\n")
            alpha, u, delta, time_to_solve, opt_sol_info = self.task_plan_MIQP_solve(x, t)

            # print(f"#######################[DEBUG-LYP] MIQP Result - Alpha (Task Assign): {alpha}, U (Control): {u}, Delta: {delta}, Time: {time_to_solve}, Info: {opt_sol_info}")

        except Exception as e:
            print(f"\n\n#######################[DEBUG-LYP] Error during MIQP state preparation or solving: {str(e)}\n\n")
            # 保留 alpha 等为 None 或默认值

        # 5. 调用 LLM 生成响应 (保持原有逻辑)
        # (MIQP 的结果 alpha 目前没有直接用于影响 LLM 的 prompt 或生成，但可以根据需要加入)
        if self.planner_config.get("constrained_generation", False):
             # 确保 world_graph[self._agents[0].uid] 存在
            current_world_graph = world_graph.get(self._agents[0].uid)
            if current_world_graph is None:
                 print(f"Error: World graph for agent {self._agents[0].uid} not found.")
                 # 可以选择使用默认语法或抛出错误
                 grammar = "" # 或者一个基础语法
            else:
                grammar = self.build_response_grammar(current_world_graph)

            llm_response = self.llm.generate(
                self.curr_prompt,
                self.stopword,
                generation_args={
                    "grammar_definition": grammar
                },
            )
        else:
            llm_response = self.llm.generate(self.curr_prompt, self.stopword)
        print(f"\n#######################[DEBUG-LYP] LLM response: {llm_response}")

        # 6. 格式化 LLM 响应 (保持原有逻辑)
        llm_response = self.format_response(llm_response, self.end_expression)
        print(f"\n#######################[DEBUG-LYP] LLM response after formatting: {llm_response}")
        
        # 7. 调用 ft_update (基于世界状态和高级动作)
        high_level_actions = self.actions_parser(
            self.agents, llm_response, self.params # 或 current_params
        )
        if self.scenario_params is not None:
            self.perception_connector.ft_update_scenario_params(
                self.scenario_params,
                world_state,
                high_level_actions
            )
            # print(f"\n#######################[DEBUG-LYP] Executed FT-Update Scenario Parameters\n")
        else:
            print(f"\n#######################[DEBUG-LYP] Warning: scenario_params is None, skipping FT-parameter update\n")

        # 8. 幻觉评估 (新增)
        self._evaluate_hallucination(
            llm_response, world_graph, alpha, observations
        )

        # 7. 返回信息 (保持原有结构, 增加 detailed_world_desc)
        info = {"llm_response": llm_response,
                "miqp_alpha": alpha,
                "miqp_u": u,
                "miqp_delta": delta,
                "miqp_time": time_to_solve,
                "miqp_status": opt_sol_info}
        # print(f"\n###################[DEBUG-LYP] agent_states: {self.get_last_agent_states()}")
        # print(f"\n###################[DEBUG-LYP] agent_positions: {self.get_last_agent_positions()}") # 这个是原始的 getter
        # print(f"\n###################[DEBUG-LYP] agent_collisions: {self.get_agent_collisions()}")

        return info

    def _evaluate_hallucination(self,
                               llm_response: str,
                               world_graph: Dict[int, "WorldGraph"],
                               miqp_assignment: Optional[np.ndarray],
                               observations: Dict[str, Any]):
        """
        评估当前规划步骤的幻觉程度
        """
            
        try:
            # 1. 记录当前规划步骤
            current_step = f"Thought: {self.parse_thought(llm_response)}\n{llm_response}"
            self.planning_trace_history.append(current_step)
            
            # 2. 提取世界状态
            world_state = self._extract_world_state_for_evaluation(world_graph)
            self.world_state_history.append(world_state)
            
            # 3. 提取真实状态 (从仿真环境)
            ground_truth_state = self._extract_ground_truth_state(observations)
            self.ground_truth_history.append(ground_truth_state)
            
            # 4. 记录MIQP分配
            if miqp_assignment is not None:
                self.miqp_assignment_history.append(miqp_assignment)
            else:
                self.miqp_assignment_history.append(None)
            
            # 5. 执行幻觉评估 (每隔几步或在任务结束时)
            if len(self.planning_trace_history) >= 3:  # 至少有3步才能评估
                self._perform_hallucination_evaluation()
                
        except Exception as e:
            print(f"幻觉评估过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _extract_world_state_for_evaluation(self, world_graph: Dict[int, "WorldGraph"]) -> Dict[str, Any]:
        """从世界图中提取用于评估的状态信息"""
        world_state = {
            'objects': {},
            'furniture': {},
            'spatial_relations': [],
            'agents': {}
        }
        
        # 使用第一个agent的世界图作为参考
        if world_graph:
            first_agent_graph = list(world_graph.values())[0]
            
            # 提取对象信息
            for obj in first_agent_graph.get_all_objects():
                world_state['objects'][obj.name] = {
                    'name': obj.name,
                    'position': obj.properties.get('translation', [0, 0, 0]),
                    'states': obj.properties.get('states', {}),
                    'parent': getattr(obj, 'parent', None)
                }
            
            # 提取家具信息
            for furniture in first_agent_graph.get_all_furnitures():
                world_state['furniture'][furniture.name] = {
                    'name': furniture.name,
                    'position': furniture.properties.get('translation', [0, 0, 0]),
                    'type': furniture.properties.get('type', 'unknown')
                }
            
            # 提取agent信息
            for agent in first_agent_graph.get_agents():
                world_state['agents'][agent.name] = {
                    'name': agent.name,
                    'position': agent.properties.get('translation', [0, 0, 0])
                }
        
        return world_state
    
    def _extract_ground_truth_state(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """从仿真环境观察中提取真实状态"""
        # 这里需要根据具体的观察格式来实现
        # 简化实现，实际应该从仿真器中获取真实的对象位置和状态
        ground_truth = {
            'objects': {},
            'furniture': {},
            'spatial_relations': [],
            'agents': {}
        }
        
        # 从环境接口获取真实状态
        if hasattr(self.env_interface, 'full_world_graph'):
            full_graph = self.env_interface.full_world_graph
            
            # 提取真实对象信息
            for obj in full_graph.get_all_objects():
                ground_truth['objects'][obj.name] = {
                    'name': obj.name,
                    'position': obj.properties.get('translation', [0, 0, 0]),
                    'states': obj.properties.get('states', {})
                }
        
        return ground_truth
    
    def _perform_hallucination_evaluation(self):
        """执行幻觉评估"""
        try:
            # 1. 上下文一致性评估
            coherence_metrics = self.coherence_evaluator.evaluate_coherence(
                self.planning_trace_history,
                self.world_state_history,
                self.miqp_assignment_history
            )
            
            # 2. 事实一致性评估
            truthfulness_metrics = self.truthfulness_evaluator.evaluate_truthfulness(
                self.planning_trace_history,
                self.world_state_history,
                self.ground_truth_history
            )
            
            # 3. 记录评估结果
            evaluation_result = {
                'step': len(self.planning_trace_history),
                'coherence_metrics': coherence_metrics,
                'truthfulness_metrics': truthfulness_metrics,
                'timestamp': time.time()
            }
            
            self.hallucination_metrics_history.append(evaluation_result)
            
            # 4. 输出评估结果
            print(f"\n=== 幻觉评估结果 (步骤 {evaluation_result['step']}) ===")
            print(f"语义连贯性: {coherence_metrics.semantic_coherence:.3f}")
            print(f"时序连贯性: {coherence_metrics.temporal_coherence:.3f}")
            print(f"动作连贯性: {coherence_metrics.action_coherence:.3f}")
            print(f"总体连贯性: {coherence_metrics.overall_coherence:.3f}")
            print(f"事实准确性: {truthfulness_metrics.factual_accuracy:.3f}")
            print(f"世界一致性: {truthfulness_metrics.world_consistency:.3f}")
            print(f"对象存在性: {truthfulness_metrics.object_existence:.3f}")
            print(f"空间准确性: {truthfulness_metrics.spatial_accuracy:.3f}")
            print(f"总体真实性: {truthfulness_metrics.overall_truthfulness:.3f}")
            print("=" * 50)

            # 将评估结果记录到日志文件
            self.hallucination_logger.info(f"幻觉评估 (步骤 {evaluation_result['step']}):")
            self.hallucination_logger.info(f"  Coherence Metrics: {coherence_metrics}")
            self.hallucination_logger.info(f"  Truthfulness Metrics: {truthfulness_metrics}")
            self.hallucination_logger.info(f"  Planning Trace History (last 3): {self.planning_trace_history[-3:]}") # 只记录最近几步的trace，避免日志过大
            self.hallucination_logger.info("-" * 30)
            
            # 保存详细的评估数据到单独的JSON文件
            self._save_hallucination_metrics_to_json(evaluation_result)
                
        except Exception as e:
            print(f"执行幻觉评估时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _save_hallucination_metrics_to_json(self, evaluation_result):
        """将幻觉评估指标保存到JSON文件"""
        try:
            import json
            import os
            
            # 创建metrics目录（如果不存在）
            metrics_dir = os.path.join(os.getcwd(), "hallucination_metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            
            # 准备要保存的数据
            # 将数据类转换为字典
            serializable_result = {
                'step': evaluation_result['step'],
                'timestamp': evaluation_result['timestamp'],
                'coherence_metrics': {
                    'semantic_coherence': float(evaluation_result['coherence_metrics'].semantic_coherence),
                    'temporal_coherence': float(evaluation_result['coherence_metrics'].temporal_coherence),
                    'action_coherence': float(evaluation_result['coherence_metrics'].action_coherence),
                    'overall_coherence': float(evaluation_result['coherence_metrics'].overall_coherence)
                },
                'truthfulness_metrics': {
                    'factual_accuracy': float(evaluation_result['truthfulness_metrics'].factual_accuracy),
                    'world_consistency': float(evaluation_result['truthfulness_metrics'].world_consistency),
                    'object_existence': float(evaluation_result['truthfulness_metrics'].object_existence),
                    'spatial_accuracy': float(evaluation_result['truthfulness_metrics'].spatial_accuracy),
                    'overall_truthfulness': float(evaluation_result['truthfulness_metrics'].overall_truthfulness)
                }
            }
            
            # 生成文件名
            filename = f"hallucination_metrics_step_{evaluation_result['step']}.json"
            filepath = os.path.join(metrics_dir, filename)
            
            # 保存到JSON文件
            with open(filepath, 'w') as f:
                json.dump(serializable_result, f, indent=2)
            
            print(f"幻觉评估指标已保存到: {filepath}")
            
            # 同时更新汇总文件
            summary_path = os.path.join(metrics_dir, "hallucination_metrics_summary.json")
            
            # 读取现有汇总数据（如果存在）
            all_metrics = []
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, 'r') as f:
                        all_metrics = json.load(f)
                except:
                    all_metrics = [] # 如果读取失败，则从空列表开始
            
            # 添加新数据
            all_metrics.append(serializable_result)
            
            # 保存更新后的汇总数据
            with open(summary_path, 'w') as f:
                json.dump(all_metrics, f, indent=2)
                
        except Exception as e:
            print(f"保存幻觉评估指标到JSON时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_hallucination_metrics(self) -> List[Dict[str, Any]]:
        """获取幻觉评估历史"""
        return self.hallucination_metrics_history

    def task_plan_MIQP_set(self):
        """
        用户设置相关的任务信息，包括全局变量管理器，RTA，scenario_params等更新参数；
        同时应包括任务的目标，任务的约束条件等。（这里需要加入条件吗）
        使用 task_utils 中的模块设置 MIQP 相关的参数和 RTA 实例。
        """
        try:
            # 1. 获取实际机器人数量
            actual_n_r = len(self._agents)
            if actual_n_r == 0:
                 print("Warning: No agents found, skipping MIQP setup.")
                 self.rta = None
                 return

            # 2. 初始化配置管理器
            # 注意: n_r 参数传递给配置类以确保内部矩阵维度正确
            scenario_manager = ScenarioConfigTask(n_r=actual_n_r)
            opt_manager = OptimizationConfigTask(n_r=actual_n_r)
            self.miqp_globalvars = GlobalVarsManager_task()

            # 3. 获取参数
            self.scenario_params = scenario_manager.get_scenario_params()
            self.opt_params = opt_manager.get_opt_params()

            # 4. 初始化并注册全局变量 (使用 ScenarioConfigTask 中的默认值)
            global_task_vars = scenario_manager.get_global_task_vars()
            self.miqp_globalvars.register_vars_from_dict(global_task_vars)

            # 可以在这里根据需要覆盖或添加特定的全局变量
            # 例如: self.miqp_globalvars.set_var('p_goal', np.array([new_x, new_y]))
            # 检查场景参数中的任务数量是否与优化参数中的边界数量匹配
            n_t_scenario = len(self.scenario_params.get('tasks', []))
            n_t_opt = self.opt_params.get('n_r_bounds', np.array([])).shape[0]

            if n_t_scenario != n_t_opt:
                 print(f"Warning: Mismatch in task count between Scenario ({n_t_scenario}) and Opt ({n_t_opt}). Adjusting Opt bounds.")
                 # 调整 n_r_bounds 以匹配 scenario 中的任务数量
                 default_bounds = [0, actual_n_r] # 默认允许 0 到所有机器人执行
                 new_bounds = np.array([self.opt_params['n_r_bounds'][i] if i < n_t_opt else default_bounds
                                       for i in range(n_t_scenario)])
                 self.opt_params['n_r_bounds'] = new_bounds
                 print(f"Adjusted n_r_bounds shape: {self.opt_params['n_r_bounds'].shape}")


            # 5. 初始化 RTA_task
            # 确保将 global_vars_manager 传递给 RTA
            self.rta = RTA(self.scenario_params, self.opt_params) # 使用 RTA
            self.rta.global_vars_manager_ = self.miqp_globalvars # 手动设置 manager

            # 6. 评估映射和特化 (RTA_task 初始化时会自动调用)
            # self.rta.evaluate_mappings_and_specializations() # 不再需要显式调用

            print(f"\n#######################[DEBUG-LYP] MIQP RTA setup complete for {actual_n_r} agents.\n")

        except ImportError as e:
            print(f"\n\n#######################[DEBUG-LYP] MIQP 模块导入失败: {str(e)}. MIQP planning will be disabled.\n\n")
            self.scenario_params = {}
            self.opt_params = {}
            self.miqp_globalvars = None
            self.rta = None
        except Exception as e:
            print(f"\n\n#######################[DEBUG-LYP] MIQP 参数设置失败: {str(e)}\n\n")
            # 打印更详细的错误追踪信息
            import traceback
            traceback.print_exc()
            self.scenario_params = {}
            self.opt_params = {}
            self.miqp_globalvars = None
            self.rta = None

    def task_plan_MIQP_solve(self, x, t):
        """
        调用MIQP的solve函数，进行任务规划，返回相应的alpha分配结果；
        过程中给出过程中判断，更新等过程，根据机器人的状态进行实时的调整。
        调用 MIQP 的 solve 函数，进行任务规划。
        """
        # 如果 RTA 初始化失败或未设置，直接返回
        if self.rta is None:
            print(f"\n\n#######################[DEBUG-LYP] RTA 未初始化，无法执行 MIQP 求解\n\n")
            return None, None, None, 0, "RTA not initialized"
            
        try: 
            # print(f"\n#######################[DEBUG-LYP] 执行 MIQP 求解，输入状态 x (shape: {x.shape}):\n{x}\nt: {t}\n")
            
            # 在solve_miqp之前重新构建约束，确保使用最新的状态 (RTA_task 的 solve_miqp 内部会做)
            # self.rta.build_constraints(x, t)
            # self.rta.build_projector() # 这个通常在 evaluate_mappings_and_specializations 或 set_specializations 时更新
            
            # RTA_task 的 solve_miqp 需要 x 和 t
            alpha, u, delta, time_to_solve, opt_sol_info = self.rta.solve_miqp(x, t)
            
            # 打印结果用于调试
            print(f"\n#######################[DEBUG-LYP] MIQP Solution:")
            print(f"  Task Assignment Matrix (alpha - flattened):\n{alpha}")
            print(f"  Control Inputs (u - flattened):\n{u}")
            print(f"  Slack Variables (delta - flattened):\n{delta}")
            print(f"  Solution Time: {time_to_solve:.4f}s")
            print(f"  Optimization Status: {opt_sol_info}\n")
            
            return alpha, u, delta, time_to_solve, opt_sol_info
        
        except Exception as e:
            print(f"\n\n#######################[DEBUG-LYP] MIQP 求解失败: {str(e)}\n\n")
            # 可以在 traceback 中打印更详细的错误信息
            import traceback
            traceback.print_exc()
            return None, None, None, 0, f"Error: {str(e)}"

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

            response_info = self.replan(instruction, observations, world_graph)
            llm_response = response_info["llm_response"]
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

            # Parse high level action directives from llm response *before* checking if done
            # Note: self.params might need miqp_alpha if parser needs it
            # current_params = {**self.params, "miqp_assignment": response_info.get("miqp_alpha")} # Example if needed
            high_level_actions = self.actions_parser(
                self.agents, llm_response, self.params # or current_params
            )
            print(f"\n\n[DEBUG] Parsed High-Level Actions before done check: {high_level_actions}\n\n")

            self.is_done = (self.check_if_agent_done(llm_response, high_level_actions)) or (
                self.replanning_count == self.planner_config.replanning_threshold
            )

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
                    "high_level_actions": {
                        # If done by keywords+wait, actions might be Wait, otherwise Done
                        agent.uid: high_level_actions.get(agent.uid, ("Done", None, None)) for agent in self.agents
                    },
                }
                return {}, planner_info, self.is_done

            """Insert MIQP Optimizer
            (给出LLM一个参考Example，根据World Graph) / (距离/电量 来做一个能力值的评估标准，启发式)
            需要设计评分标准，及其对应的Prompt
            Features and capabilities -> A
            Task need abilities -> T
            
            给出相应的这个Post到Low_level_actions. (分配安排 + 原内容)
            """  

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

        # Check if replanning is required for the *next* step
        # Original: self.replan_required = any(responses.values())
        # New logic: Only require replan if the task is NOT done AND any agent response indicates it's needed.
        if self.is_done:
            self.replan_required = False
            # 如果任务完成且启用了幻觉评估，执行最后一次评估并记录历史
            if self.planning_trace_history:
                # 确保所有步骤都被评估
                if len(self.planning_trace_history) > len(self.hallucination_metrics_history) * 3 : 
                     self._perform_hallucination_evaluation() # 执行最后一次评估
                
                self.hallucination_logger.info("任务完成. 最终幻觉评估历史:")
                for i, metrics_data in enumerate(self.hallucination_metrics_history):
                    self.hallucination_logger.info(f"  评估批次 {i+1} (步骤 {metrics_data['step']}):")
                    self.hallucination_logger.info(f"    Coherence: {metrics_data['coherence_metrics']}")
                    self.hallucination_logger.info(f"    Truthfulness: {metrics_data['truthfulness_metrics']}")
                self.hallucination_logger.info("=" * 50 + " 任务结束 " + "=" * 50)
                
                # 保存最终的幻觉评估汇总结果
                self._save_final_hallucination_summary()
            print("[DEBUG] Task marked as done, setting replan_required to False.")
        else:
            self.replan_required = any(responses.values())
            if self.replan_required:
                 print("[DEBUG] Task not done and agent responses indicate need for replan.")
            # else:
            #      print("[DEBUG] Task not done, but agent responses do not require replan.")

        print_str += self._add_responses_to_prompt(responses)

        # Update planner info (合并公共部分)
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
        return low_level_actions, planner_info, self.is_done

    def check_if_agent_done(self, llm_response: str, high_level_actions: Dict[int, Tuple[str, str, Optional[str]]]) -> bool:
        """
        Check if the agent is done based on the LLM response and actions.

        :param llm_response: The LLM response to check.
        :param high_level_actions: Parsed high-level actions from the LLM response.
        :return: True if the agent is done, False otherwise.
        """
        # Original check for the explicit end expression
        if self.end_expression in llm_response:
            print(f"[DEBUG] Detected task completion via end expression: '{self.end_expression}'")
            # 如果任务完成且启用了幻觉评估，执行最后一次评估并记录历史
            if self.planning_trace_history: # Assuming planning_trace_history exists if eval is on
                # 确保所有步骤都被评估
                if len(self.planning_trace_history) > len(self.hallucination_metrics_history) * 3 : # 简化的判断，实际应更精确
                    self._perform_hallucination_evaluation() # 执行最后一次评估
                
                self.hallucination_logger.info("任务完成. 最终幻觉评估历史:")
                for i, metrics_data in enumerate(self.hallucination_metrics_history):
                    self.hallucination_logger.info(f"  评估批次 {i+1} (步骤 {metrics_data['step']}):")
                    self.hallucination_logger.info(f"    Coherence: {metrics_data['coherence_metrics']}")
                    self.hallucination_logger.info(f"    Truthfulness: {metrics_data['truthfulness_metrics']}")
                self.hallucination_logger.info("=" * 50 + " 任务结束 " + "=" * 50)
                
                # 保存最终的幻觉评估汇总结果
                self._save_final_hallucination_summary()
            return True

        # Enhanced check: Keywords + All agents waiting
        # Define keywords indicating completion in the LLM's natural language response
        completion_keywords = ["task is now complete", "successfully placed", "task is done"]
        # Check if any completion keyword is present in the LLM response (case-insensitive)
        contains_completion_keyword = any(keyword in llm_response.lower() for keyword in completion_keywords)

        # Check if all parsed high-level actions are 'Wait'
        all_agents_waiting = False
        if high_level_actions and len(high_level_actions) == len(self.agents): # Ensure actions for all agents
            all_agents_waiting = all(action_tuple[0] == 'Wait' for action_tuple in high_level_actions.values())

        # If completion keywords are found AND all agents are assigned 'Wait', consider it done
        if contains_completion_keyword and all_agents_waiting:
            print("[DEBUG] Detected task completion via keywords and all agents assigned 'Wait'.")
            # 如果任务完成且启用了幻觉评估，执行最后一次评估并记录历史
            if self.planning_trace_history: # Assuming planning_trace_history exists if eval is on
                # 确保所有步骤都被评估
                if len(self.planning_trace_history) > len(self.hallucination_metrics_history) * 3 : # 简化的判断，实际应更精确
                    self._perform_hallucination_evaluation() # 执行最后一次评估
                
                self.hallucination_logger.info("任务完成. 最终幻觉评估历史:")
                for i, metrics_data in enumerate(self.hallucination_metrics_history):
                    self.hallucination_logger.info(f"  评估批次 {i+1} (步骤 {metrics_data['step']}):")
                    self.hallucination_logger.info(f"    Coherence: {metrics_data['coherence_metrics']}")
                    self.hallucination_logger.info(f"    Truthfulness: {metrics_data['truthfulness_metrics']}")
                self.hallucination_logger.info("=" * 50 + " 任务结束 " + "=" * 50)
                
                # 保存最终的幻觉评估汇总结果
                self._save_final_hallucination_summary()
            return True

        # If neither condition is met, the task is not considered done by this check
        return False

    def _save_final_hallucination_summary(self):
        """将幻觉评估汇总结果保存到文件"""
        try:
            import json
            import os
            import numpy as np
            
            # 检查必要的属性是否存在
            if not self.hallucination_metrics_history: # Direct check as it's always initialized
                print("没有幻觉评估历史数据可供保存")
                return
            
            # 创建metrics目录（如果不存在）
            metrics_dir = os.path.join(os.getcwd(), "hallucination_metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            
            # 从历史记录中计算平均指标
            semantic_coherence_values = []
            temporal_coherence_values = []
            action_coherence_values = []
            overall_coherence_values = []
            
            factual_accuracy_values = []
            world_consistency_values = []
            object_existence_values = []
            spatial_accuracy_values = []
            overall_truthfulness_values = []
            
            # 收集所有评估指标
            for metrics_data in self.hallucination_metrics_history:
                try:
                    coherence_metrics = metrics_data['coherence_metrics'] # Direct access
                    truthfulness_metrics = metrics_data['truthfulness_metrics'] # Direct access
                    
                    semantic_coherence_values.append(float(coherence_metrics.semantic_coherence)) # Direct access
                    temporal_coherence_values.append(float(coherence_metrics.temporal_coherence)) # Direct access
                    action_coherence_values.append(float(coherence_metrics.action_coherence)) # Direct access
                    overall_coherence_values.append(float(coherence_metrics.overall_coherence)) # Direct access
                    
                    factual_accuracy_values.append(float(truthfulness_metrics.factual_accuracy)) # Direct access
                    world_consistency_values.append(float(truthfulness_metrics.world_consistency)) # Direct access
                    object_existence_values.append(float(truthfulness_metrics.object_existence)) # Direct access
                    spatial_accuracy_values.append(float(truthfulness_metrics.spatial_accuracy)) # Direct access
                    overall_truthfulness_values.append(float(truthfulness_metrics.overall_truthfulness)) # Direct access
                except Exception as e:
                    print(f"处理评估指标时出错: {str(e)}")
                    continue # Skip this entry and continue with others
            
            # 确保有数据可供计算
            if not semantic_coherence_values: # Check if any data was actually collected
                print("没有有效的评估指标数据可供计算平均值")
                return
            
            # 计算平均值
            summary_data = {
                'coherence_metrics': {
                    'semantic_coherence': float(np.mean(semantic_coherence_values)) if semantic_coherence_values else 0,
                    'temporal_coherence': float(np.mean(temporal_coherence_values)) if temporal_coherence_values else 0,
                    'action_coherence': float(np.mean(action_coherence_values)) if action_coherence_values else 0,
                    'overall_coherence': float(np.mean(overall_coherence_values)) if overall_coherence_values else 0
                },
                'truthfulness_metrics': {
                    'factual_accuracy': float(np.mean(factual_accuracy_values)) if factual_accuracy_values else 0,
                    'world_consistency': float(np.mean(world_consistency_values)) if world_consistency_values else 0,
                    'object_existence': float(np.mean(object_existence_values)) if object_existence_values else 0,
                    'spatial_accuracy': float(np.mean(spatial_accuracy_values)) if spatial_accuracy_values else 0,
                    'overall_truthfulness': float(np.mean(overall_truthfulness_values)) if overall_truthfulness_values else 0
                },
                'steps_evaluated': len(self.hallucination_metrics_history),
                'timestamp': time.time()
            }
            
            # 生成文件名
            filename = "hallucination_final_summary.json"
            filepath = os.path.join(metrics_dir, filename)
            
            # 保存到JSON文件
            with open(filepath, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            print(f"最终幻觉评估汇总结果已保存到: {filepath}")
            
            # 打印最终评估结果
            print("\n" + "=" * 20 + " 最终幻觉评估汇总 " + "=" * 20)
            print(f"评估步骤数: {summary_data['steps_evaluated']}")
            print(f"平均语义连贯性: {summary_data['coherence_metrics']['semantic_coherence']:.3f}")
            print(f"平均时序连贯性: {summary_data['coherence_metrics']['temporal_coherence']:.3f}")
            print(f"平均动作连贯性: {summary_data['coherence_metrics']['action_coherence']:.3f}")
            print(f"平均总体连贯性: {summary_data['coherence_metrics']['overall_coherence']:.3f}")
            print(f"平均事实准确性: {summary_data['truthfulness_metrics']['factual_accuracy']:.3f}")
            print(f"平均世界一致性: {summary_data['truthfulness_metrics']['world_consistency']:.3f}")
            print(f"平均对象存在性: {summary_data['truthfulness_metrics']['object_existence']:.3f}")
            print(f"平均空间准确性: {summary_data['truthfulness_metrics']['spatial_accuracy']:.3f}")
            print(f"平均总体真实性: {summary_data['truthfulness_metrics']['overall_truthfulness']:.3f}")
            print("=" * 60)
                
        except Exception as e:
            print(f"保存幻觉评估汇总结果到文件时出错: {str(e)}")
            import traceback
            traceback.print_exc()
