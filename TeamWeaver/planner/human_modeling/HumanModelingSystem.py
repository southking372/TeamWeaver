import numpy as np
from habitat_llm.planner.human_modeling.LLMProfessionTagger import LLMProfessionTagger
from habitat_llm.planner.human_modeling.HumanCareerModel import HumanCareerModel
from habitat_llm.planner.human_modeling.PerformanceEvaluator import PerformanceEvaluator

class HumanModelingSystem:
    """
    Human Modeling System
    Integrates LLM profession tagging and heuristic capability adaptation
    to produce matrices and parameters for human-robot swarm collaboration.
    """
    
    def __init__(self, llm_tagger, num_humans=3):
        """
        Initialize human modeling system
        
        Parameters:
            llm_tagger: LLMProfessionTagger instance, for profession tagging
            num_humans: Number of humans, default 3
        """
        self.llm_tagger = llm_tagger
        self.num_humans = num_humans
        self.human_models = []
        self.performance_evaluator = None
        
        # task type mapping
        self.task_type_mapping = {
            "transport": 0,  # transport task
            "coverage": 1,   # coverage control task
            "precision": 2   # precision control task
        }
        
        # capability type mapping
        self.capability_mapping = {
            "locomotion": 0,  # locomotion capability
            "monitoring": 1,  # monitoring capability
            "coordination": 2 # coordination capability
        }
        
    def initialize_humans(self, human_descriptions):
        """
        Initialize human models based on human descriptions
        
        Parameters:
            human_descriptions: list of human descriptions
        """
        num_humans = min(self.num_humans, len(human_descriptions))
        
        for i in range(num_humans):
            profession = self.llm_tagger.tag_human_profession(human_descriptions[i])
            # get initial capabilities for the profession
            initial_capabilities = self.llm_tagger.get_initial_capabilities(profession)
            
            # create HumanCareerModel instance
            human_model = HumanCareerModel(
                profession_type=profession,
                initial_capabilities=initial_capabilities
            )
            
            # add to human model list
            self.human_models.append(human_model)
        
        # initialize performance evaluator
        self.performance_evaluator = PerformanceEvaluator(self.human_models)
        
    def update_performance(self, human_idx, task_type, execution_data):
        """
        Update human performance
        
        Parameters:
            human_idx: human model index
            task_type: task type
            execution_data: execution data
        """
        # validate index
        if human_idx < 0 or human_idx >= len(self.human_models):
            raise ValueError(f"Invalid human model index: {human_idx}")
            
        # validate task type
        if task_type not in self.task_type_mapping:
            raise ValueError(f"Invalid task type: {task_type}")
            
        # evaluate performance
        performance_score = self.performance_evaluator.evaluate_performance(
            human_idx, task_type, execution_data
        )
        
        return performance_score
        
    def generate_scenario_matrices(self):
        """
        Generate matrices and parameters compatible with scenario_params.py
        
        Returns:
            dict containing A, T, Hs, ws matrices and nr, nt, nc, nf, nx, nu parameters
        """
        # compute parameters
        n_r = self.num_humans  # number of humans
        n_t = len(self.task_type_mapping)  # number of task types
        n_c = len(self.capability_mapping)  # number of capability types
        n_f = 3  # number of features (aligned with robot features)
        n_x = 3  # state dimension
        n_u = 3  # control input dimension
        
        # initialize feature matrix A (n_f x n_r)
        A = np.zeros((n_f, n_r))
        
        # initialize task capability requirement matrix T (n_t x n_c)
        T = np.zeros((n_t, n_c))
        
        # initialize capability feature matrices Hs (n_c matrices)
        Hs = [None] * n_c
        
        # initialize capability weight matrices ws (n_c matrices)
        ws = [None] * n_c
        
        # fill matrices
        for i, human_model in enumerate(self.human_models):
            # fill feature matrix A
            # feature 1: locomotion capability
            A[0, i] = human_model.get_capability("transport")
            # feature 2: monitoring capability
            A[1, i] = human_model.get_capability("coverage")
            # feature 3: precision control capability
            A[2, i] = human_model.get_capability("precision")
        
        # fill task capability requirement matrix T
        # transport task requires locomotion and coordination
        T[0, 0] = 1  # locomotion
        T[0, 2] = 0.5  # coordination
        
        # coverage control task requires monitoring and locomotion
        T[1, 1] = 1  # monitoring
        T[1, 0] = 0.7  # locomotion
        
        # precision control task requires precision control and coordination
        T[2, 2] = 1  # coordination
        T[2, 0] = 0.3  # locomotion
        
        # fill capability feature matrices Hs
        # locomotion contributed by feature 1
        Hs[0] = np.zeros((1, n_f))
        Hs[0][0, 0] = 1
        
        # monitoring contributed by feature 2
        Hs[1] = np.zeros((1, n_f))
        Hs[1][0, 1] = 1
        
        # coordination contributed by feature 3
        Hs[2] = np.zeros((1, n_f))
        Hs[2][0, 2] = 1
        
        # fill capability weight matrices ws
        for i in range(n_c):
            ws[i] = np.eye(1)  # identity matrix
        
        # return result
        return {
            'A': A,
            'T': T,
            'Hs': Hs,
            'ws': ws,
            'n_r': n_r,
            'n_t': n_t,
            'n_c': n_c,
            'n_f': n_f,
            'n_x': n_x,
            'n_u': n_u
        }
