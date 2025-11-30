from habitat_llm.planner.miqp_planner.task_utils.scenario_params import ScenarioConfig
from habitat_llm.planner.miqp_planner.optimizer.rl_adapter.adaptive_scenario_config import AdaptiveScenarioConfig

def main():
    # Initialize the original ScenarioConfig
    original_config = ScenarioConfig()
    
    # Create the adaptive wrapper with RL capabilities
    adaptive_config = AdaptiveScenarioConfig(
        original_config,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.2
    )
    
    # Train the RL agent
    print("Training the RL agent for task adaptation...")
    adaptive_config.train_adaptation(episodes=100)
    
    # Use the trained agent for real-time adaptation
    print("\nDemonstrating real-time task adaptation:")
    
    # Scenario 1: Standard operation
    state_description = {
        'wheels_count': 4,
        'propellers_count': 1,
        'cameras_count': 5,
        'transport_progress': 1,  # Medium progress
        'coverage_quality': 1     # Medium quality
    }
    
    recommendation = adaptive_config.get_best_adaptation_for_state(state_description)
    print("\nScenario 1: Standard operation")
    print(f"Recommended adaptation: {recommendation['task']} - {recommendation['modification']}")
    
    # Scenario 2: Transport task falling behind
    state_description = {
        'wheels_count': 4,
        'propellers_count': 1,
        'cameras_count': 5,
        'transport_progress': 0,  # Low progress
        'coverage_quality': 2     # High quality
    }
    
    recommendation = adaptive_config.get_best_adaptation_for_state(state_description)
    print("\nScenario 2: Transport task falling behind")
    print(f"Recommended adaptation: {recommendation['task']} - {recommendation['modification']}")
    
    # Scenario 3: Coverage control issues
    state_description = {
        'wheels_count': 4,
        'propellers_count': 1,
        'cameras_count': 5,
        'transport_progress': 2,  # High progress
        'coverage_quality': 0     # Low quality
    }
    
    recommendation = adaptive_config.get_best_adaptation_for_state(state_description)
    print("\nScenario 3: Coverage control issues")
    print(f"Recommended adaptation: {recommendation['task']} - {recommendation['modification']}")
    
    # Real-time adaptation during operation
    print("\nSimulating real-time adaptation during operation:")
    for step in range(1, 6):
        print(f"\nStep {step}:")
        task_idx, mod_type, value = adaptive_config.adapt_task_functions()
        
        # Print the adaptation applied
        if task_idx == 0:
            print(f"Adapted Transport Task: {mod_type} = {value}")
        else:
            print(f"Adapted Coverage Control Task: {mod_type} = {value}")
        
        # Simulate task execution with adapted functions
        performance = adaptive_config.evaluate_performance()
        print(f"Performance after adaptation: {performance:.4f}")

if __name__ == "__main__":
    main()