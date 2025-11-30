import numpy as np
import random
from collections import defaultdict

class RLTaskAdapter:
    """
    A reinforcement learning agent that adapts task functions in multi-robot systems.
    Uses Q-learning to learn optimal task function modifications based on system performance.
    """
    
    def __init__(self, scenario_config, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        self.scenario_config = scenario_config
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Initialize Q-table as a nested defaultdict to handle new state-action pairs
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Define action space for task modifications
        # Each action is a tuple (task_index, modification_type, value)
        self.actions = self._init_action_space()
        
    def _init_action_space(self):
        """Initialize the action space for task modifications."""
        actions = []
        
        # For transport task (index 0)
        actions.append((0, "speed", 0.8))   # Reduce speed parameter
        actions.append((0, "speed", 1.0))   # Default speed parameter
        actions.append((0, "speed", 1.2))   # Increase speed parameter
        
        # For coverage control task (index 1)
        actions.append((1, "strategy", "conservative"))  # More cautious coverage
        actions.append((1, "strategy", "balanced"))      # Balanced approach
        actions.append((1, "strategy", "aggressive"))    # More aggressive coverage
        
        return actions
    
    def get_state(self):
        """
        Extract the current state of the multi-robot system.
        Discretizes continuous values into a manageable state representation.
        """
        # Get robot features from scenario_config
        A = self.scenario_config.scenario_params['A']
        
        # Count robots with wheels and propellers (for locomotion capability)
        wheels_count = np.sum(A[0])
        propellers_count = np.sum(A[1])
        
        # Count robots with cameras (for monitoring capability)
        cameras_count = np.sum(A[2])
        
        # Get task performance metrics (would come from external monitoring)
        # In a real implementation, these would be actual metrics from task execution
        transport_progress = self._get_transport_progress()  # Low, Medium, High
        coverage_quality = self._get_coverage_quality()      # Low, Medium, High
        
        # Combine discretized features into a state tuple
        state = (
            min(wheels_count, 3),      # Discretize: 0, 1, 2, 3+
            min(propellers_count, 2),  # Discretize: 0, 1, 2+
            min(cameras_count, 3),     # Discretize: 0, 1, 2, 3+
            transport_progress,        # Already discrete: 0, 1, 2
            coverage_quality           # Already discrete: 0, 1, 2
        )
        
        return state
    
    def _get_transport_progress(self):
        """
        Get current progress of transport task.
        This would normally come from task monitoring.
        For demonstration, returns a simple discretized progress value.
        """
        # In a real implementation, this would measure actual task progress
        # For simulation, we'll use a placeholder random value (0=Low, 1=Medium, 2=High)
        # This should be replaced with actual performance metrics
        return random.randint(0, 2)
    
    def _get_coverage_quality(self):
        """
        Get current quality of coverage control task.
        This would normally come from task monitoring.
        For demonstration, returns a simple discretized quality value.
        """
        # In a real implementation, this would measure actual coverage performance
        # For simulation, we'll use a placeholder random value (0=Low, 1=Medium, 2=High)
        # This should be replaced with actual performance metrics
        return random.randint(0, 2)
    
    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        Either explore randomly or exploit the best known action.
        """
        if random.random() < self.exploration_rate:
            # Exploration: choose a random action
            return random.choice(self.actions)
        else:
            # Exploitation: choose the best action based on Q-values
            if state in self.q_table and self.q_table[state]:
                return max(self.q_table[state], key=self.q_table[state].get)
            else:
                # If state is new or has no Q-values, choose random action
                return random.choice(self.actions)
    
    def apply_action(self, action):
        """
        Apply the selected action to modify the task functions.
        """
        task_idx, mod_type, value = action
        
        # Update the corresponding task function or parameter
        if task_idx == 0:  # Transport task
            if mod_type == "speed":
                # Modify transport task speed parameter
                self._modify_transport_speed(value)
        
        elif task_idx == 1:  # Coverage control task
            if mod_type == "strategy":
                # Select different coverage strategy
                self._select_coverage_strategy(value)
    
    def _modify_transport_speed(self, speed_value):
        """
        Modify the speed parameter in the transport task function.
        In a real implementation, this would update a parameter in TransportTask.
        """
        # Placeholder for modifying transport task parameters
        # In a real implementation, this would update the actual function
        print(f"Modifying transport speed to {speed_value}")
        
        # Example of how this might be implemented:
        # self.scenario_config.scenario_params['tasks'][0]['speed_param'] = speed_value
    
    def _select_coverage_strategy(self, strategy):
        """
        Select a different implementation of the coverage control strategy.
        In a real implementation, this would switch between different function implementations.
        """
        # Placeholder for selecting coverage strategy
        # In a real implementation, this would switch the function reference
        print(f"Selecting coverage strategy: {strategy}")
        
        # Example of how this might be implemented:
        # if strategy == "conservative":
        #     self.scenario_config.scenario_params['tasks'][1]['function'] = CoverageControl.conservative_coverage
        # elif strategy == "aggressive":
        #     self.scenario_config.scenario_params['tasks'][1]['function'] = CoverageControl.aggressive_coverage
    
    def calculate_reward(self):
        """
        Calculate the reward based on task performance after applying modifications.
        This would normally measure the actual performance improvements.
        """
        # In a real implementation, this would calculate actual performance metrics
        # For demonstration, we'll use a placeholder reward calculation
        
        # Get task performance metrics (simulated here)
        transport_speed = self._evaluate_transport_speed()
        transport_safety = self._evaluate_transport_safety()
        coverage_area = self._evaluate_coverage_area()
        coverage_efficiency = self._evaluate_coverage_efficiency()
        
        # Calculate combined reward
        transport_reward = 0.7 * transport_speed + 0.3 * transport_safety
        coverage_reward = 0.6 * coverage_area + 0.4 * coverage_efficiency
        
        # Combined reward with task weights
        total_reward = 0.5 * transport_reward + 0.5 * coverage_reward
        
        return total_reward
    
    def _evaluate_transport_speed(self):
        """Evaluate transport task speed. Higher is better."""
        # Placeholder for actual performance evaluation
        return random.uniform(0, 1)
    
    def _evaluate_transport_safety(self):
        """Evaluate transport task safety. Higher is better."""
        # Placeholder for actual performance evaluation
        return random.uniform(0, 1)
    
    def _evaluate_coverage_area(self):
        """Evaluate coverage area. Higher is better."""
        # Placeholder for actual performance evaluation
        return random.uniform(0, 1)
    
    def _evaluate_coverage_efficiency(self):
        """Evaluate coverage efficiency. Higher is better."""
        # Placeholder for actual performance evaluation
        return random.uniform(0, 1)
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Update Q-table using the Q-learning algorithm.
        """
        # Get the best action for the next state
        best_next_action = None
        best_next_q_value = 0
        
        if next_state in self.q_table and self.q_table[next_state]:
            best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
            best_next_q_value = self.q_table[next_state][best_next_action]
        
        # Q-learning update rule
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * best_next_q_value - current_q
        )
        
        # Update Q-value
        self.q_table[state][action] = new_q
    
    def train(self, episodes=100):
        """
        Train the RL agent over a number of episodes.
        """
        for episode in range(episodes):
            # Get current state
            state = self.get_state()
            
            # Select and apply action
            action = self.select_action(state)
            self.apply_action(action)
            
            # Execute tasks and get performance (simulated)
            # In a real implementation, this would actually run the tasks
            
            # Calculate reward based on performance
            reward = self.calculate_reward()
            
            # Get new state after action
            next_state = self.get_state()
            
            # Update Q-table
            self.update_q_table(state, action, reward, next_state)
            
            # Print episode information (optional)
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {reward:.4f}")
    
    def get_best_action(self, state):
        """
        Get the best action for a given state based on learned Q-values.
        """
        if state in self.q_table and self.q_table[state]:
            return max(self.q_table[state], key=self.q_table[state].get)
        else:
            return random.choice(self.actions)
    
    def suggest_modifications(self):
        """
        Suggest task function modifications based on current state.
        This is the main interface for getting adaptation suggestions.
        """
        # Get current state
        state = self.get_state()
        
        # Get best action for current state
        best_action = self.get_best_action(state)
        
        # Return the suggested modification
        return best_action