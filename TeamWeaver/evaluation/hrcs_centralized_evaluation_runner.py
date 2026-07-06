# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

"""
Update centralized evaluation runner to use our (Perception -> Planner -> Low Skiller) algorithm.
Otherwise, it also support Human and RL agent modeling interaction.

So here we have a Perception Layer, and a Perception2Planner Connection, so we can realize HRCS system.
This class is used to run the evaluation of the agent in a centralized manner.
"""
from typing import TYPE_CHECKING, Any, Dict, Tuple, List, Optional, Union
import numpy as np
import copy
from hydra.utils import instantiate

from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.evaluation import CentralizedEvaluationRunner
from habitat_llm.planner.planner import Planner
from habitat_llm.tools.motor_skills.motor_skill_tool import MotorSkillTool

# from optimizer.adaptive_optimizer_v2 import AdaptiveTaskOptimizer
from habitat_llm.planner.human_modeling.LLMProfessionTagger import LLMProfessionTagger
from habitat_llm.planner.human_modeling.HumanModelingSystem import HumanModelingSystem

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from habitat_llm.world_model import WorldGraph

class HRCSEvaluationRunner(CentralizedEvaluationRunner):
    def __init__(
        self,
        evaluation_runner_config_arg: "DictConfig",  # Hydra config type
        env_arg: EnvironmentInterface,
    ) -> None:
        """
        Initialize the CentralizedEvaluationRunner.

        :param evaluation_runner_config_arg: Configuration object containing evaluation parameters
                                           including agent and planner configurations
        :param env_arg: Environment interface instance that provides access to the simulation
        """
        # optimizer_config = evaluation_runner_config_arg.optimizer
        super().__init__(evaluation_runner_config_arg, env_arg)
        
        #Initialize human modeling system related properties
        self.human_system = None
        self.human_scenario_matrices = None
        self.task_optimizer = None
        self.human_agent_uid = None  #humanmanipulationmember'sUID

    def get_low_level_actions(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
    ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        """
        Given a set of observations, gets a vector of low level actions for all agents.
        """
        #according toplannertypes and human modeling system states to determine how to obtain low-level actions
        if self.human_system is not None and hasattr(self.planner, "get_next_action_with_human"):
            #ifplannerSupport humanitymanipulationstaff, using specialized methods
            low_level_actions, planner_info, should_end = self.planner.get_next_action_with_human(
                instruction, 
                observations, 
                world_graph,
                self.human_system,
                self.human_agent_uid
            )
        else:
            #Use standardget_next_actionmethod
            low_level_actions, planner_info, should_end = self.planner.get_next_action(
                instruction, observations, world_graph
            )
            
            if self.human_system is not None and self.human_agent_uid in low_level_actions:
                low_level_actions, planner_info = self.apply_human_capability_feedback(
                    low_level_actions, planner_info
                )
        
        return low_level_actions, planner_info, should_end
    
    def reset_planners(self) -> None:
        """Reset the centralized planner to prepare for a new episode."""
        assert isinstance(self.planner, Planner)
        self.planner.reset()
        
        #Reset human modeling system
        if self.human_system is not None:
            #Logic to reset the human modeling system can be added here
            pass

    def _initialize_planners(self) -> None:
        """
        Initialize the centralized planner based on the evaluation runner configuration.
        Sets up the planner with appropriate agents and configures special planning modes
        if specified.
        """
        planner_conf = self.evaluation_runner_config.planner
        planner = instantiate(planner_conf)
        self.planner: Planner = planner(env_interface=self.env_interface)

        # Set both agents to the planner
        self.planner.agents = [
            self.agents[agent_id] for agent_id in sorted(self.agents.keys())
        ]
        if (
            "plan_config" in planner_conf
            and "planning_mode" in planner_conf.plan_config
            and planner_conf.plan_config.planning_mode == "st"
        ):
            for agent in self.planner.agents:
                for tool in agent.tools.values():
                    if isinstance(tool, MotorSkillTool):
                        tool.error_mode = "st"
        
        if hasattr(self.evaluation_runner_config, "human_planner_model"):
            self._initialize_human_modeling(self.evaluation_runner_config.human_planner_model)
            if self.human_agent_uid in self.agents:
                print(f"humanmanipulationmember(UID: {self.human_agent_uid}) ) added to planner")
                print(f"humanmanipulationemployee occupation type: {self.human_system.human_models[0].profession_type}")
                print(f"humanmanipulationStaff ability value: {self.human_system.human_models[0].capabilities}")
            else:
                print(f"warn:humanmanipulationmember(UID: {self.human_agent_uid}) ) not found in agent list")
    
    def apply_human_capability_feedback(
        self, 
        low_level_actions: Dict[int, Any], 
        planner_info: Dict[str, Any]
    ) -> Tuple[Dict[int, Any], Dict[str, Any]]:
        if self.human_system is None or self.human_agent_uid not in low_level_actions:
            return low_level_actions, planner_info
        human_capabilities = self.human_system.human_models[0].capabilities
        precision_factor = human_capabilities.get('precision', 0.5)
        human_action = low_level_actions[self.human_agent_uid]
        if isinstance(human_action, dict) and "action" in human_action:
            adjusted_action = copy.deepcopy(human_action)
            if "action" in adjusted_action and isinstance(adjusted_action["action"], (list, np.ndarray)):
                adjusted_action["action"] = np.array(adjusted_action["action"]) * (0.5 + 0.5 * precision_factor)
            low_level_actions[self.human_agent_uid] = adjusted_action
            
            #record humansmanipulationemployee’s task effectiveness
            if "task_assignment" in planner_info and self.human_agent_uid in planner_info["task_assignment"]:
                task_idx = planner_info["task_assignment"][self.human_agent_uid]
                if task_idx > 0:
                    task_type = "transport" if task_idx == 1 else "coverage"
                    print(f"humanmanipulationstaff are performing tasks: {task_type}")
                    # if self.task_optimizer is not None:
                    #     #Here you can add the update logic of the task optimizer
                    #     pass
        return low_level_actions, planner_info

    def _initialize_human_modeling(self, human_planner_model):
        api_key = human_planner_model.get("api_key")
        tagger = LLMProfessionTagger(api_key=api_key)
        self.human_system = HumanModelingSystem(tagger, num_humans=1)
        
        human_descriptions = human_planner_model.get(
            "human_descriptions", 
            ["Experienced operator skilled in precision control and fast response, with many years of operation experience"] # Here can be changed to different descriptions
        )
        self.human_system.initialize_humans(human_descriptions)
        self.human_scenario_matrices = self.human_system.generate_scenario_matrices()
        self.human_agent_uid = human_planner_model.get("human_agent_uid", 0)
        # if "scenario_params" in human_planner_model and "global_vars" in human_planner_model:
        #     self.task_optimizer = AdaptiveTaskOptimizer(
        #         human_planner_model["scenario_params"], 
        #         human_planner_model["global_vars"]
        #     )
    
    def add_human_planner_model(self, human_planner_model):
        self._initialize_human_modeling(human_planner_model)
        
        if hasattr(self, "planner") and isinstance(self.planner, Planner):
            #ensure humanitymanipulationmember is in the agent list
            if self.human_agent_uid in self.agents:
                print(f"humanmanipulationmember(UID: {self.human_agent_uid}) ) added to planner")
                print(f"humanmanipulationemployee occupation type: {self.human_system.human_models[0].profession_type}")
                print(f"humanmanipulationStaff ability value: {self.human_system.human_models[0].capabilities}")
            else:
                print(f"warn:humanmanipulationmember(UID: {self.human_agent_uid}) ) not found in agent list")
