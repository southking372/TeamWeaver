from typing import TYPE_CHECKING, Dict, Any, Tuple, Optional, Union, List
from habitat_llm.planner.HRCS.connector.action_updater import ActionUpdater

if TYPE_CHECKING:
    from habitat_llm.planner.HRCS.params_module.scenario_params_task import ScenarioConfigTask

class ExecutionManager:
    """
    Manages the final steps of the planning cycle before execution,
    specifically updating scenario parameters based on the chosen high-level actions.
    Enhanced with historical context tracking and feedback recording.
    """
    def __init__(self):
        self.action_updater = ActionUpdater()
        self.execution_state = {}
        self.active_updates = []
        
        self.execution_history = {
            'actions': [],  # List[Dict[str, Any]] - Record actions for each round
            'observations': [],  # List[Dict[str, Any]] - Record observations for each round
            'agent_positions': [],  # List[Dict[int, Dict]] - Record the agent position in each round
            'timestamps': []  # List[float] - Record timestamp
        }
        self.max_history_length = 10

    def reset(self):
        """
        Reset the ExecutionManager state.
        Clear execution tracking, active updates, and historical context.
        """
        self.execution_state = {}
        self.active_updates = []
        
        # Reset history
        self.execution_history = {
            'actions': [],
            'observations': [],
            'agent_positions': [],
            'timestamps': []
        }
        print("[DEBUG] ExecutionManager reset completed - history cleared")

    def record_execution_context(
        self,
        high_level_actions: Dict[int, Tuple[str, str, Optional[str]]],
        observations: Dict[str, Any],
        agent_positions: Dict[int, Dict[str, Any]],
        timestamp: float = None
    ) -> None:
        """
        Record execution context history information
        """
        import time
        if timestamp is None:
            timestamp = time.time()
        
        # Record current round information
        self.execution_history['actions'].append(high_level_actions.copy())
        self.execution_history['observations'].append(observations.copy())
        self.execution_history['agent_positions'].append(agent_positions.copy())
        self.execution_history['timestamps'].append(timestamp)
        
        # Keep history within limits
        iflen(self.execution_history['actions']) > self.max_history_length:
            for key in self.execution_history:
                self.execution_history[key] = self.execution_history[key][-self.max_history_length:]
        
        print(f"[DEBUG] Recorded execution context - history length: {len(self.execution_history['actions'])}")

    def get_recent_actions(self, agent_id: int, lookback_steps: int = 3) -> List[Tuple[str, str, Optional[str]]]:
        """
        Get the actions of the specified agent in recent steps
        
        Args:
            agent_id: agent ID
            lookback_steps: Look back at steps
            
        Returns:
            List of recent actions
        """
        recent_actions = []
        
        # Start with the most recent history and look forward
        for i in range(min(lookback_steps, len(self.execution_history['actions']))):
            index = -(i + 1)  # Start from the last one
            action_dict = self.execution_history['actions'][index]
            
            if agent_id in action_dict:
                recent_actions.append(action_dict[agent_id])
        
        return recent_actions

    def get_latest_observation(self, agent_id: int) -> Optional[str]:
        """
        Get the latest observation results of the specified agent
        Returns:
            latest observation string
        """
        if not self.execution_history['observations']:
            return None
        
        latest_obs = self.execution_history['observations'][-1]
        return latest_obs.get(f'Agent_{agent_id}_Observation', None)

    def get_agent_position_history(self, agent_id: int, lookback_steps: int = 3) -> List[Dict[str, Any]]:
        """
        Get the historical location information of the specified agent
        Returns:
            List of historical location information
        """
        position_history = []
        
        for i in range(min(lookback_steps, len(self.execution_history['agent_positions']))):
            index = -(i + 1)
            position_dict = self.execution_history['agent_positions'][index]
            
            if agent_id in position_dict:
                position_history.append(position_dict[agent_id])
        
        return position_history

    def update_scenario_for_execution(
        self,
        scenario_config: Union[Dict[str, Any], "ScenarioConfigTask"],
        world_state: Dict[str, Any],
        high_level_actions: Dict[int, Tuple[str, str, Optional[str]]]
    ) -> None:
        """
        Updates the scenario configuration based on the planned high-level actions.

        This method uses an ActionUpdater to process the actions and translate them
        into specific parameter updates for the simulation environment.

        Args:
            scenario_config: The scenario configuration object or dictionary.
            world_state: The current state of the world.
            high_level_actions: The dictionary of high-level actions to be executed.
        """
        if not high_level_actions:
            print("[ExecutionManager] No actions provided, skipping scenario update.")
            return

        # print(f"[ExecutionManager] Updating scenario with {len(high_level_actions)} actions.")
        all_updates = self.action_updater.process_and_get_updates(high_level_actions, world_state)
        from habitat_llm.planner.HRCS.connector.planner_utils import update_param_value
        for param_name, value in all_updates.items():
            update_param_value(scenario_config, param_name, value)
        print(f"[ExecutionManager] Applied {len(all_updates)} parameter updates for execution.") 