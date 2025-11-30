import numpy as np
from habitat_llm.planner.human_modeling.LLMProfessionTagger import LLMProfessionTagger
from habitat_llm.planner.human_modeling.HumanCareerModel import HumanCareerModel
from habitat_llm.planner.human_modeling.PerformanceEvaluator import PerformanceEvaluator

class HumanModelingSystem:
    """
    Human Modeling System
    整合LLM职业打标和启发式能力适应，生成与机器人集群协作的矩阵和参数
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
        
        # 任务类型映射
        self.task_type_mapping = {
            "transport": 0,  # 运输任务
            "coverage": 1,   # 覆盖控制任务
            "precision": 2   # 精确控制任务
        }
        
        # 能力类型映射
        self.capability_mapping = {
            "locomotion": 0,  # 移动能力
            "monitoring": 1,  # 监控能力
            "coordination": 2 # 协调能力
        }
        
    def initialize_humans(self, human_descriptions):
        """
        Initialize human models based on human descriptions
        
        参数:
            human_descriptions: 人类描述列表
        """
        num_humans = min(self.num_humans, len(human_descriptions))
        
        for i in range(num_humans):
            profession = self.llm_tagger.tag_human_profession(human_descriptions[i])
            # 获取职业的初始能力值
            initial_capabilities = self.llm_tagger.get_initial_capabilities(profession)
            
            # 创建HumanCareerModel实例
            human_model = HumanCareerModel(
                profession_type=profession,
                initial_capabilities=initial_capabilities
            )
            
            # 添加到人类模型列表
            self.human_models.append(human_model)
        
        # 初始化性能评估器
        self.performance_evaluator = PerformanceEvaluator(self.human_models)
        
    def update_performance(self, human_idx, task_type, execution_data):
        """
        更新人类性能
        
        参数:
            human_idx: 人类模型索引
            task_type: 任务类型
            execution_data: 执行数据
        """
        # 确保索引有效
        if human_idx < 0 or human_idx >= len(self.human_models):
            raise ValueError(f"无效的人类模型索引: {human_idx}")
            
        # 确保任务类型有效
        if task_type not in self.task_type_mapping:
            raise ValueError(f"无效的任务类型: {task_type}")
            
        # 评估性能
        performance_score = self.performance_evaluator.evaluate_performance(
            human_idx, task_type, execution_data
        )
        
        return performance_score
        
    def generate_scenario_matrices(self):
        """
        生成与scenario_params.py兼容的矩阵和参数
        
        返回:
            包含A, T, Hs, ws矩阵以及nr, nt, nc, nf, nx, nu参数的字典
        """
        # 计算参数
        n_r = self.num_humans  # 人类数量
        n_t = len(self.task_type_mapping)  # 任务类型数量
        n_c = len(self.capability_mapping)  # 能力类型数量
        n_f = 3  # 特征数量 (与机器人特征对应)
        n_x = 3  # 状态维度
        n_u = 3  # 控制输入维度
        
        # 初始化特征矩阵 A (n_f x n_r)
        A = np.zeros((n_f, n_r))
        
        # 初始化任务能力需求矩阵 T (n_t x n_c)
        T = np.zeros((n_t, n_c))
        
        # 初始化能力特征矩阵 Hs (n_c个矩阵)
        Hs = [None] * n_c
        
        # 初始化能力权重矩阵 ws (n_c个矩阵)
        ws = [None] * n_c
        
        # 填充矩阵
        for i, human_model in enumerate(self.human_models):
            # 填充特征矩阵 A
            # 特征1: 移动能力
            A[0, i] = human_model.get_capability("transport")
            # 特征2: 监控能力
            A[1, i] = human_model.get_capability("coverage")
            # 特征3: 精确控制能力
            A[2, i] = human_model.get_capability("precision")
        
        # 填充任务能力需求矩阵 T
        # 运输任务需要移动能力和协调能力
        T[0, 0] = 1  # 移动能力
        T[0, 2] = 0.5  # 协调能力
        
        # 覆盖控制任务需要监控能力和移动能力
        T[1, 1] = 1  # 监控能力
        T[1, 0] = 0.7  # 移动能力
        
        # 精确控制任务需要精确控制能力和协调能力
        T[2, 2] = 1  # 协调能力
        T[2, 0] = 0.3  # 移动能力
        
        # 填充能力特征矩阵 Hs
        # 移动能力由特征1贡献
        Hs[0] = np.zeros((1, n_f))
        Hs[0][0, 0] = 1
        
        # 监控能力由特征2贡献
        Hs[1] = np.zeros((1, n_f))
        Hs[1][0, 1] = 1
        
        # 协调能力由特征3贡献
        Hs[2] = np.zeros((1, n_f))
        Hs[2][0, 2] = 1
        
        # 填充能力权重矩阵 ws
        for i in range(n_c):
            ws[i] = np.eye(1)  # 单位矩阵
        
        # 返回结果
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