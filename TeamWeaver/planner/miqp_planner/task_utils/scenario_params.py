import numpy as np
from habitat_llm.planner.miqp_planner.task_utils.transport_module import TransportTask
from habitat_llm.planner.miqp_planner.task_utils.coverage_module import CoverageControl

class ScenarioConfig:
    """
    Configuration class for robot scenario parameters.
    Manages robot features, tasks, capabilities, and system dynamics.
    """
    
    def __init__(self):
        # Initialize basic configuration
        self.n_r = 5  # Number of robots
        self.n_t = 2  # Number of tasks (1: transport, 2: perimeter defense)
        self.n_c = 2  # Number of capabilities (1: locomotion, 2: monitoring)
        self.n_f = 3  # Number of features (1: wheels, 2: propellers, 3: camera)
        self.n_x = 3  # State dimension
        self.n_u = 3  # Control input dimension
        
        # Initialize scenario parameters
        self.scenario_params = self._initialize_scenario_params()
    
    def _initialize_scenario_params(self):
        """Initialize all scenario parameters including robot features and capabilities"""
        # Initialize robots' features and capabilities
        A = np.zeros((self.n_f, self.n_r))
        # Robot features configuration (rows: features, columns: robots)
        # Feature 1: wheels, Feature 2: propellers, Feature 3: camera
        A[0, 0] = 1; A[1, 0] = 0; A[2, 0] = 1  # Robot 1's features
        A[0, 1] = 1; A[1, 1] = 0; A[2, 1] = 1  # Robot 2's features
        A[0, 2] = 1; A[1, 2] = 0; A[2, 2] = 1  # Robot 3's features
        A[0, 3] = 1; A[1, 3] = 0; A[2, 3] = 1  # Robot 4's features
        A[0, 4] = 0; A[1, 4] = 1; A[2, 4] = 1  # Robot 5's features
        
        # Task capabilities requirement matrix
        T = np.zeros((self.n_t, self.n_c))
        T[0, 0] = 1; T[0, 1] = 0  # Task 1 (transport)'s capabilities
        T[1, 0] = 3; T[1, 1] = 3  # Task 2 (perimeter defense)'s capabilities
        
        # Define capabilities in terms of features
        Hs = [None] * self.n_c  
        Hs[0] = np.zeros((2, self.n_f))
        Hs[0][0, 0] = 1  # Wheels contribute to capability 1 (locomotion)
        Hs[0][1, 1] = 1  # Propellers contribute to capability 1 (locomotion)
        Hs[1] = np.zeros((1, self.n_f))
        Hs[1][0, 2] = 1  # Camera contributes to capability 2 (monitoring)
        
        # Weights for capabilities
        ws = [None] * self.n_c
        ws[0] = np.eye(2)  # Weights for Hs[0] capabilities
        ws[1] = np.eye(1)  # Weights for Hs[1] capabilities
        
        # Create scenario parameters dictionary
        scenario_params = {
            'A': A,
            'T': T,
            'Hs': Hs,
            'ws': ws,
            'robot_dyn': self._initialize_robot_dynamics(),
            'tasks': self._initialize_tasks()
        }
        
        return scenario_params
    
    def _initialize_robot_dynamics(self):
        """Initialize robot dynamics functions"""
        def f(x):
            return np.zeros_like(x)
        
        def g(x):
            return np.eye(self.n_x)
        
        def sys_dyn(x, u):    
            return f(x) + g(x) @ u
        
        return {
            'f': f,
            'g': g,
            'n_x': self.n_x,
            'n_u': self.n_u,
            'sys_dyn': sys_dyn
        }
    
    def _initialize_tasks(self):
        """Initialize task functions"""
        tasks = [None] * self.n_t
        
        # Task 1: Transport
        tasks[0] = {
            'function': TransportTask.transport_function,
            'gradient': TransportTask.transport_gradient,
            'time_derivative': TransportTask.transport_time_derivative
        }
        
        # Task 2: Coverage Control
        tasks[1] = {
            'function': CoverageControl.coverage_control_task_function,
            'gradient': CoverageControl.coverage_control_task_gradient,
            'time_derivative': CoverageControl.coverage_control_task_time_derivative
        }
        
        return tasks
    
    def get_scenario_params(self):
        """Get the complete scenario parameters dictionary"""
        return self.scenario_params
    
    def update_robot_features(self, robot_idx, features):
        """
        Update the features of a specific robot
        
        Parameters:
            robot_idx: Index of the robot to update
            features: List of feature values [wheels, propellers, camera]
        """
        for i, feature_val in enumerate(features):
            if i < self.n_f:
                self.scenario_params['A'][i, robot_idx] = feature_val
    
    def get_updated_scenario_params(self):
        """Get a copy of the scenario parameters with only the A matrix"""
        return {'A': self.scenario_params['A'].copy()}