import time

class HumanCareerModel:
    def __init__(self, profession_type, initial_capabilities):
        self.profession_type = profession_type  # profession type
        self.capabilities = initial_capabilities  # initial capability values
        self.adaptation_history = []  # adaptation history
        
    def get_capability(self, task_type):
        """Get capability value for a specific task type."""
        return self.capabilities.get(task_type, 0.5)  # default medium capability
        
    def adapt_capability(self, task_type, performance_data):
        """Adjust capability value based on performance data."""
        # small incremental adjustment
        current_value = self.capabilities.get(task_type, 0.5)
        adaptation_factor = self._calculate_adaptation_factor(performance_data)
        new_value = current_value + 0.1 * adaptation_factor  # max adjustment magnitude 0.1
        
        # keep capability in [0, 1]
        new_value = max(0, min(1, new_value))
        self.capabilities[task_type] = new_value
        
        # record adaptation history
        self.adaptation_history.append({
            'task_type': task_type,
            'old_value': current_value,
            'new_value': new_value,
            'performance_data': performance_data,
            'timestamp': time.time()
        })
        
    def _calculate_adaptation_factor(self, performance_data):
        """Compute adaptation factor from performance data."""
        # based on performance data; may be linear or a more complex function
        return performance_data - 0.5  # assume performance data is in [0, 1]
