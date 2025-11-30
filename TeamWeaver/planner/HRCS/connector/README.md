# HRCS Connector模块架构说明

## 📋 概述

`connector/`文件夹包含了连接感知系统与MIQP优化系统的核心模块，负责将高层指令转换为可执行的任务计划，并管理整个规划生命周期。这些模块采用了**模块化解耦设计**，每个模块专注于特定的功能领域。

## 🏗️ 架构设计理念

### 核心设计原则
- **单一职责**：每个模块专注于特定的功能领域
- **高内聚低耦合**：模块内部功能紧密相关，模块间接口清晰
- **可扩展性**：新功能可以通过添加新模块或扩展现有模块实现
- **可测试性**：每个模块都可以独立测试和验证

### 数据流向
```
Input Instruction → PhaseManager → TaskDependencyEnhancer → 
MatrixUpdater → ActionUpdater → planner_utils
```

## 📦 模块详解

### 1. `phase_manager.py` - 阶段管理器
**核心职责**：管理任务执行的生命周期和阶段推进

#### 主要功能：
- **LLM任务分解**：使用专门的序列化分解Prompt将复杂指令分解为结构化子任务
- **阶段状态管理**：维护当前执行阶段和完成状态跟踪
- **进度监控**：跟踪任务完成进度和阶段转换条件

#### 关键方法：
```python
# 使用LLM进行任务分解并初始化阶段
def decompose_and_initialize_phases(instruction, world_description, agent_info_string, llm_config)

# 获取当前阶段的任务
def get_current_phase_tasks() -> Optional[Dict[str, Any]]

# 推进到下一个执行阶段
def advance_to_next_phase() -> bool

# 检查当前阶段是否完成
def is_current_phase_complete(agent_statuses) -> bool
```

#### 调用示例：
```python
# 在PerceptionConnector中的使用
self.phase_manager = PhaseManager(self.llm_client)
structured_subtasks = self.phase_manager.decompose_and_initialize_phases(
    instruction, world_desc_string, agent_info_string, llm_config
)
```

### 2. `dependency_enhancer.py` - 依赖关系增强器
**核心职责**：基于任务语义构建智能依赖关系和执行阶段

#### 主要功能：
- **语义依赖分析**：识别Pick-Navigate-Place等复合动作序列的隐式依赖
- **依赖图构建**：自动添加必要的前置条件（如Pick前必须Navigate）
- **阶段组织**：将任务组织为可并行执行的阶段，优化整体执行效率

#### 关键方法：
```python
# 主入口：结构化和分阶段处理
def structure_and_phase(structured_subtasks, max_agents) -> Tuple[List, List, Dict]

# 语义依赖增强
def _enhance_semantic_dependencies(structured_subtasks) -> List[Dict[str, Any]]

# 构建任务依赖图
def _build_dependency_graph(enhanced_tasks) -> Dict[str, List[str]]

# 组织任务为执行阶段
def _organize_tasks_into_phases(tasks, max_agents) -> List[Dict[str, Any]]
```

#### 调用示例：
```python
# 在PerceptionConnector中的使用
enhancer = TaskDependencyEnhancer()
enhanced_subtasks, execution_phases, dependency_graph = enhancer.structure_and_phase(
    structured_subtasks, max_agents
)
```

### 3. `matrix_updater.py` - 矩阵更新器
**核心职责**：维护MIQP优化所需的参数矩阵

#### 主要功能：
- **T矩阵管理**：任务-能力需求矩阵（13×5维度）
- **A矩阵管理**：智能体能力矩阵（5×2维度）  
- **权重向量管理**：任务优先级权重向量(ws)
- **动态更新**：基于任务特性和世界状态动态调整矩阵参数

#### 矩阵定义：
```python
# 基础任务-能力需求矩阵 [13任务 × 5能力]
BASE_TASK_CAPABILITY_REQUIREMENTS = np.array([
    # [移动, 操作, 控制, 液体, 电源]
    [1, 0, 0, 0, 0],  # Navigate
    [1, 0, 0, 0, 0],  # Explore
    [0, 1, 0, 0, 0],  # Pick
    [0, 1, 0, 0, 0],  # Place
    # ... 其他任务
])

# 基础智能体能力矩阵 [5能力 × 2智能体]
BASE_ROBOT_CAPABILITIES = np.array([
    [2.0, 1.8],  # 移动能力
    [2.0, 1.8],  # 操作能力
    [2.0, 1.8],  # 控制能力
    [0.0, 1.4],  # 液体处理（仅Agent 1）
    [0.0, 1.3]   # 电源控制（仅Agent 1）
])
```

#### 关键方法：
```python
# 更新所有MIQP矩阵
def update_matrices(structured_subtasks, world_state) -> Dict[str, Union[np.ndarray, List]]

# 基于任务优先级更新任务-能力矩阵
def _update_task_capability_matrix_enhanced(subtasks, llm_analysis)

# 基于智能体状态更新能力矩阵
def _update_robot_capability_matrix_enhanced(subtasks, llm_analysis)
```

### 4. `action_updater.py` - 动作更新器
**核心职责**：将高级动作转换为场景参数更新

#### 主要功能：
- **运动技能处理**：处理Navigate, Pick, Place, Rearrange, Explore, Wait等动作
- **状态操控处理**：处理Clean, Fill, Pour, PowerOn, PowerOff等特殊动作
- **参数生成**：将动作转换为MIQP求解器可用的参数更新

#### 关键方法：
```python
# 主处理方法
def process_and_get_updates(high_level_actions, world_state) -> Dict[str, Any]

# 处理运动技能相关动作
def _process_motor_skill_actions(high_level_actions, world_state) -> Dict[str, Any]

# 处理状态操控相关动作
def _process_state_manipulation_actions(high_level_actions, world_state) -> Dict[str, Any]

# 更新导航参数
def _update_navigation_params(target_name, world_state) -> Dict[str, Any]
```

#### 调用示例：
```python
# 在ExecutionManager中的使用
action_updater = ActionUpdater()
all_updates = action_updater.process_and_get_updates(high_level_actions, world_state)
```

### 5. `world_describer.py` - 世界描述器
**核心职责**：生成用于LLM提示的详细世界描述

#### 主要功能：
- **环境布局描述**：生成房间、家具的详细布局信息
- **智能体状态描述**：提供智能体位置和持有物信息
- **观察模式支持**：支持全观察和部分观察两种模式

#### 关键方法：
```python
# 生成完整世界描述
def get_world_description(env_interface, world_state, is_partial_obs=False) -> str

# 生成环境布局描述
def _get_layout_description(env_interface, is_partial_obs=False) -> str

# 生成智能体状态描述
def _get_agent_status_description(world_state) -> str
```

### 6. `planner_utils.py` - 工具函数集
**核心职责**：提供通用的辅助函数

#### 主要功能：
- **JSON解析**：从LLM响应中提取结构化数据
- **参数更新**：支持字典和ScenarioConfigTask对象的统一更新接口
- **位置查找**：在世界状态中查找目标位置信息
- **配置管理**：提供LLM配置的统一接口

#### 关键方法：
```python
# 从文本中提取JSON
def extract_json_from_text(text: str, target_type: type = dict) -> Optional[str]

# 统一参数更新接口
def update_param_value(scenario_config, key: str, value: Any) -> None

# 查找目标位置
def find_target_position(target_name: str, world_state: Dict[str, Any]) -> Optional[List[float]]

# 获取LLM配置
def get_llm_config() -> Dict[str, Any]
```

## 🔄 模块间协作流程

### 典型调用序列

1. **任务分解阶段**：
   ```python
   PhaseManager.decompose_and_initialize_phases()  # LLM分解
   ↓
   TaskDependencyEnhancer.structure_and_phase()   # 依赖增强
   ```

2. **矩阵更新阶段**：
   ```python
   MatrixUpdater.update_matrices()                # 参数矩阵更新
   ↓
   planner_utils.update_param_value()            # 参数应用
   ```

3. **动作处理阶段**：
   ```python
   ActionUpdater.process_and_get_updates()       # 动作转换
   ↓
   planner_utils.update_param_value()            # 参数更新
   ```

### 依赖关系图
```
PhaseManager ──────┐
                   ├──→ PerceptionConnector ──→ LLMPlanner
TaskDependencyEnhancer ┘
                   
MatrixUpdater ─────┐
                   ├──→ MIQP Solver
ActionUpdater ─────┘

planner_utils ────→ (被所有模块使用)
```

## 📚 使用指南

### 初始化
```python
from habitat_llm.planner.HRCS.connector.phase_manager import PhaseManager
from habitat_llm.planner.HRCS.connector.dependency_enhancer import TaskDependencyEnhancer
from habitat_llm.planner.HRCS.connector.matrix_updater import MatrixUpdater
from habitat_llm.planner.HRCS.connector.action_updater import ActionUpdater

# 初始化组件
phase_manager = PhaseManager(llm_client)
dependency_enhancer = TaskDependencyEnhancer()
matrix_updater = MatrixUpdater(llm_client)
action_updater = ActionUpdater()
```

### 典型使用流程
```python
# 1. 任务分解
structured_tasks = phase_manager.decompose_and_initialize_phases(
    instruction, world_description, agent_info, llm_config
)

# 2. 依赖增强和阶段组织
enhanced_tasks, phases, deps = dependency_enhancer.structure_and_phase(
    structured_tasks, max_agents
)

# 3. 矩阵更新
matrices = matrix_updater.update_matrices(enhanced_tasks, world_state)

# 4. 动作处理
updates = action_updater.process_and_get_updates(actions, world_state)
```

## 🧪 测试建议

### 单元测试重点
- **PhaseManager**：测试LLM分解的稳定性和阶段管理逻辑
- **TaskDependencyEnhancer**：测试依赖关系识别的准确性
- **MatrixUpdater**：验证矩阵维度和数值的正确性
- **ActionUpdater**：测试动作到参数的转换逻辑

### 集成测试重点
- 模块间数据传递的完整性
- 端到端规划流程的正确性
- 异常情况下的鲁棒性

## 🔧 扩展指南

### 添加新任务类型
1. 在`MatrixUpdater`中更新`BASE_TASK_CAPABILITY_REQUIREMENTS`
2. 在`ActionUpdater`中添加相应的处理逻辑
3. 在`TaskDependencyEnhancer`中添加语义依赖规则

### 添加新智能体能力
1. 扩展`BASE_ROBOT_CAPABILITIES`矩阵
2. 更新`BASE_CAPABILITY_WEIGHTS`权重
3. 修改相关的处理逻辑
