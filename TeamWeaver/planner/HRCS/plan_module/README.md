# HRCS Plan Module架构说明

## 📋 概述

`plan_module/`文件夹包含了MIQP-LLM混合规划系统的核心执行模块，负责将MIQP优化结果与LLM推理能力相结合，实现智能、稳定的任务规划和执行管理。这些模块采用了**目标导向设计**，专注于实现高质量的规划决策和执行控制。

## 🏗️ 架构设计理念

### 核心设计原则
- **目标导向**：提供目标指导而非直接动作命令，保持LLM的决策自主性
- **智能增强**：MIQP优化为LLM提供全局最优的宏观指导
- **反馈驱动**：基于执行反馈进行学习和调整，提升系统稳定性
- **模块解耦**：每个模块职责明确，便于独立测试和扩展

### 执行流程
```
MIQP优化 → TaskHelper分配 → ActionManager验证 → 
PromptBuilder增强 → ErrorHandler恢复 → ExecutionManager更新 → 
FeedbackManager评估
```

## 📦 模块详解

### 1. `action_manager.py` - 智能动作管理器
**核心职责**：解析LLM响应并验证动作的目标一致性

#### 主要功能：
- **动作解析**：将LLM响应转换为结构化的高级动作
- **目标导向验证**：验证LLM动作是否有助于实现阶段目标
- **智能调整**：在保持LLM决策自主性的前提下提供目标导向建议

#### 关键方法：
```python
# 解析LLM响应中的高级动作
def parse_high_level_actions(llm_response: str, agents) -> Dict[int, Tuple[str, str, Optional[str]]]

# 目标导向的动作验证和调整
def adjust_actions_with_phase_awareness(
    high_level_actions, agent_task_assignments, current_phase, agents
) -> Dict[int, Tuple[str, str, Optional[str]]]

# 验证动作是否推进目标实现
def _action_advances_objective(action_name, action_target, objective) -> bool

# 为目标建议合适的动作
def _suggest_action_for_objective(objective, original_action) -> Tuple[str, str, Optional[str]]
```

#### 设计特点：
```python
# 目标导向验证逻辑示例
if self._action_advances_objective(llm_action, llm_target, primary_objective):
    # LLM的动作有助于目标实现，保持不变
    adjusted_actions[agent_id] = action_tuple
    print(f"LLM action '{llm_action}[{llm_target}]' advances objective ✓")
else:
    # 提供目标导向的建议但保持一定灵活性
    if self._is_exploration_reasonable(llm_action, primary_objective):
        adjusted_actions[agent_id] = action_tuple
```

### 2. `task_helper.py` - 基于"简历"的智能任务分配系统
**核心职责**：基于Agent能力特征进行智能任务分配

#### 主要功能：
- **Agent简历管理**：维护每个Agent的能力、性能特质和动态状态
- **智能评分分配**：基于硬约束+软约束的双层匹配机制
- **动态学习**：根据执行反馈更新Agent性能评估
- **负载均衡**：避免单个Agent过载

#### Agent简历结构：
```python
@dataclass
class AgentResume:
    agent_id: int
    name: str
    
    # 硬约束：Agent能执行的任务类型
    capabilities: List[str] = field(default_factory=list)
    
    # 软约束：性能特质评分 (0-1)
    performance_traits: Dict[str, float] = field(default_factory=dict)
    
    # 动态状态：实时决策信息
    current_task_load: int = 0
    position: Optional[np.ndarray] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
```

#### 关键方法：
```python
# 核心评分函数：计算Agent对任务的适应性分数
def _calculate_assignment_score(task: Dict[str, Any], resume: AgentResume) -> float

# 基于评分的智能任务分配
def _distribute_tasks_with_scoring(tasks, agent_ids) -> Dict[int, List[Dict[str, Any]]]

# 根据反馈更新性能特质
def _update_traits_from_feedback(resume: AgentResume, feedback: str)

# 更新Agent简历上下文
def update_resumes_from_context(world_state, agent_feedback)
```

#### 智能分配示例：
```python
# Agent 0: "Workhorse" - 基础能力强，速度快
Agent_0_Resume = {
    'capabilities': ['Navigate', 'Explore', 'Pick', 'Place', 'Rearrange'],
    'performance_traits': {
        'speed': 0.9,           # 高速度
        'precision': 0.6,       # 中等精度
        'exploration_bias': 0.5 # 中等探索倾向
    }
}

# Agent 1: "Specialist" - 精密操作，特殊技能
Agent_1_Resume = {
    'capabilities': ['Navigate', 'Pick', 'Place', 'Clean', 'Fill', 'PowerOn'],
    'performance_traits': {
        'speed': 0.6,                    # 中等速度
        'precision': 0.9,                # 高精度
        'liquid_handling_skill': 0.95,   # 液体处理专长
        'power_control_skill': 0.9       # 电源控制专长
    }
}
```

### 3. `miqp_solver_wrapper.py` - MIQP求解器包装器
**核心职责**：管理MIQP优化参数和求解过程

#### 主要功能：
- **参数管理**：初始化和维护ScenarioConfigTask和OptimizationConfigTask
- **阶段感知求解**：支持针对当前阶段的动态任务实例求解
- **RTA集成**：封装RTA求解器的调用接口

#### 关键方法：
```python
# 设置MIQP参数
def task_plan_MIQP_set(agents: List["Agent"])

# 阶段感知的MIQP求解
def task_plan_MIQP_solve_phase_aware(x, t, phase_task_info, agents)

# 重置求解器状态
def reset()
```

#### 求解流程：
```python
# 典型求解调用
alpha, u, delta, time_to_solve, opt_sol_info = self.miqp_solver_wrapper.task_plan_MIQP_solve_phase_aware(
    x,                  # 当前状态
    t,                  # 时间
    phase_task_info,    # 阶段任务信息
    self._agents        # Agent列表
)
```

### 4. `prompt_builder.py` - 智能提示构建器
**核心职责**：构建MIQP指导的目标导向提示

#### 主要功能：
- **MIQP指导构建**：将优化结果转换为LLM可理解的目标描述
- **目标导向格式化**：描述要实现的目标而非具体执行步骤
- **上下文增强**：整合世界状态和阶段信息

#### 关键方法：
```python
# 构建MIQP指导信息
def build_miqp_guidance_addition(
    current_phase_tasks, agent_task_assignments, current_phase, 
    perception_connector, alpha_result=None, world_state=None
) -> str

# 构建阶段感知的增强提示（已废弃，使用增量方式）
def build_phase_aware_prompt_with_miqp_guidance(...)
```

#### 目标导向设计示例：
```python
# 传统动作指令（避免）
"Execute Pick[apple]"

# 目标导向指导（推荐）
"achieve pickup of apple (requires navigation first if not nearby)"
```

### 5. `error_handler.py` - 智能错误处理器
**核心职责**：检测执行失败并提供智能恢复策略

#### 主要功能：
- **失败分析**：分析Agent状态反馈，识别失败模式
- **恢复策略**：基于失败类型生成针对性的恢复动作
- **历史跟踪**：维护错误历史和恢复尝试记录

#### 关键方法：
```python
# 应用智能错误恢复
def recover_and_log_assignments(
    agent_task_assignments, last_high_level_actions, latest_agent_response
) -> Dict[int, List[Dict[str, Any]]]

# 分析失败并建议恢复动作
def analyze_failure_and_suggest_recovery(
    agent_id, status, current_action
) -> Optional[Tuple[str, str, Optional[str]]]
```

#### 智能恢复逻辑：
```python
# 恢复策略示例
if action_name == "Pick" and "not close enough" in status_lower:
    return ("Navigate", target, target)  # 导航到目标位置
elif action_name == "Pick" and "object not found" in status_lower:
    return ("Explore", "environment", "environment")  # 探索寻找对象
elif "collision" in status_lower:
    return ("Wait", "", "")  # 等待避免冲突
```

### 6. `execution_manager.py` - 执行管理器
**核心职责**：管理规划周期的最终执行步骤

#### 主要功能：
- **场景参数更新**：基于选定的高级动作更新场景配置
- **ActionUpdater集成**：调用ActionUpdater处理动作转换
- **执行状态跟踪**：维护执行相关的状态信息

#### 关键方法：
```python
# 为执行更新场景配置
def update_scenario_for_execution(
    scenario_config, world_state, high_level_actions
) -> None
```

#### 调用示例：
```python
# 在llm_planner_miqp.py中的使用
self.execution_manager.update_scenario_for_execution(
    self.miqp_solver_wrapper.scenario_params,
    world_state,
    adjusted_actions
)
```

### 7. `feedback_manager.py` - 反馈管理器
**核心职责**：管理执行反馈和阶段推进

#### 主要功能：
- **状态提取**：从多种来源获取Agent完成状态
- **阶段评估**：检查当前阶段是否完成
- **阶段推进**：管理阶段转换逻辑
- **性能跟踪**：维护Agent性能指标

#### 关键方法：
```python
# 获取Agent完成状态
def get_agent_completion_statuses(agents, latest_agent_response) -> Dict[int, str]

# 检查并推进阶段
def check_and_advance_phase(
    perception_connector, agents, latest_agent_response, current_phase
) -> bool
```

#### 智能状态获取：
```python
# 优先级策略：最新响应 > Agent状态 > 默认值
def get_agent_completion_statuses(agents, latest_agent_response):
    if latest_agent_response:
        # 优先使用最新的响应
        return latest_agent_response
    else:
        # 备用：获取Agent状态描述
        return {agent.uid: agent.get_last_state_description() for agent in agents}
```

## 🔄 模块间协作流程

### 在replan方法中的调用序列

```python
# Step 8: 智能任务分配
agent_task_assignments = self.task_helper.assign_tasks_for_phase(
    current_phase_tasks, alpha, phase_task_info, self._agents, ...
)

# Step 9: 智能错误恢复  
agent_task_assignments = self.error_handler.recover_and_log_assignments(
    agent_task_assignments, self.last_high_level_actions, self.latest_agent_response
)

# Step 10: 提示构建
miqp_guidance = self.prompt_builder.build_miqp_guidance_addition(
    current_phase_tasks, agent_task_assignments, current_phase, ...
)

# Step 11: LLM调用（反馈增强）
llm_response = self.llm.generate(self.curr_prompt + miqp_guidance, self.stopword)

# Step 12: 动作解析和验证
high_level_actions = self.action_manager.parse_high_level_actions(llm_response, self._agents)
adjusted_actions = self.action_manager.adjust_actions_with_phase_awareness(
    high_level_actions, agent_task_assignments, current_phase, self._agents
)

# Step 13: 执行参数更新
self.execution_manager.update_scenario_for_execution(
    self.miqp_solver_wrapper.scenario_params, world_state, adjusted_actions
)

# Step 15: 反馈和阶段管理
self._phase_transition_pending = self.feedback_manager.check_and_advance_phase(
    self.perception_connector, self._agents, self.latest_agent_response, current_phase
)
```

### 模块依赖关系图
```
MIQPSolverWrapper ─────┐
                       ├──→ TaskHelper ──→ ActionManager ──→ PromptBuilder
ErrorHandler ──────────┘                           │
                                                   ▼
ExecutionManager ←─────────────────────────── LLM Response
    │                                              │
    ▼                                              ▼
FeedbackManager ←────────────────────── Agent Execution Results
```

## 📚 使用指南

### 初始化
```python
from habitat_llm.planner.HRCS.plan_module.action_manager import ActionManager
from habitat_llm.planner.HRCS.plan_module.task_helper import TaskHelper
from habitat_llm.planner.HRCS.plan_module.miqp_solver_wrapper import MIQPSolverWrapper
# ... 其他模块

# 在LLMPlanner.__init__中初始化
self.miqp_solver_wrapper = MIQPSolverWrapper()
self.task_helper = TaskHelper(self._agents)
self.action_manager = ActionManager(self.actions_parser)
self.error_handler = ErrorHandler()
self.prompt_builder = PromptBuilder(plan_config, env_interface)
self.execution_manager = ExecutionManager()
self.feedback_manager = FeedbackManager()
```

### 典型使用流程
```python
# 1. 设置MIQP参数
self.miqp_solver_wrapper.task_plan_MIQP_set(self._agents)

# 2. 智能任务分配
assignments = self.task_helper.assign_tasks_for_phase(...)

# 3. 错误恢复
recovered_assignments = self.error_handler.recover_and_log_assignments(...)

# 4. 构建增强提示
guidance = self.prompt_builder.build_miqp_guidance_addition(...)

# 5. 解析和验证动作
actions = self.action_manager.parse_high_level_actions(...)
adjusted_actions = self.action_manager.adjust_actions_with_phase_awareness(...)

# 6. 更新执行参数
self.execution_manager.update_scenario_for_execution(...)

# 7. 管理反馈和阶段
phase_transition = self.feedback_manager.check_and_advance_phase(...)
```

## 🎯 设计特色

### 1. 目标导向而非动作导向
- **传统方式**：直接指定"Execute Pick[object]"
- **本系统**：描述目标"achieve pickup of object (requires navigation first)"

### 2. 智能Agent简历系统
- **能力边界**：硬约束确保任务可执行性
- **性能优化**：软约束优化分配效率
- **动态学习**：根据反馈调整Agent评估

### 3. 反馈增强的规划循环
- **历史保持**：维护完整的对话历史和执行反馈
- **MIQP增强**：优化指导作为增量信息而非替换
- **阶段感知**：基于执行阶段的动态规划调整

## 🧪 测试建议

### 单元测试重点
- **ActionManager**：测试目标验证逻辑和动作调整策略
- **TaskHelper**：验证评分算法和分配公平性
- **ErrorHandler**：测试失败识别和恢复策略的准确性
- **PromptBuilder**：验证提示格式和目标描述的清晰性

### 集成测试重点
- **MIQP-LLM一致性**：验证优化建议与LLM决策的协调性
- **反馈闭环**：测试执行反馈对后续规划的影响
- **阶段推进**：验证阶段转换的正确性和及时性

## 🔧 扩展指南

### 添加新的Agent类型
1. 在`TaskHelper`中扩展`_initialize_agent_resumes()`
2. 添加新的能力特质和评分逻辑
3. 更新`_calculate_assignment_score()`方法

### 增加新的错误类型
1. 在`ErrorHandler`中扩展`analyze_failure_and_suggest_recovery()`
2. 添加新的失败模式识别逻辑
3. 设计相应的恢复策略

### 扩展MIQP求解器
1. 在`MIQPSolverWrapper`中添加新的求解模式
2. 扩展phase_task_info结构
3. 更新求解器调用接口
