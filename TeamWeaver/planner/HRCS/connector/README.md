# HRCS Connector Module Architecture

## 📋 Overview

The `connector/` folder contains the core modules that connect the perception system with the MIQP optimization system. They convert high-level instructions into executable task plans and manage the entire planning lifecycle. These modules follow a **modular decoupling design**, with each module focused on a specific functional domain.

## 🏗️ Architecture Design Philosophy

### Core Design Principles
- **Single responsibility**: Each module focuses on a specific functional domain
- **High cohesion, low coupling**: Tightly related functionality within modules, clear interfaces between modules
- **Extensibility**: New features can be added via new modules or extensions to existing ones
- **Testability**: Each module can be tested and validated independently

### Data Flow
```
Input Instruction → PhaseManager → TaskDependencyEnhancer → 
MatrixUpdater → ActionUpdater → planner_utils
```

## 📦 Module Details

### 1. `phase_manager.py` - Phase Manager
**Core responsibility**: Manage task execution lifecycle and phase advancement

#### Main features:
- **LLM task decomposition**: Use specialized serialization decomposition prompts to break complex instructions into structured subtasks
- **Phase state management**: Maintain current execution phase and completion status tracking
- **Progress monitoring**: Track task completion progress and phase transition conditions

#### Key methods:
```python
# Decompose tasks with LLM and initialize phases
def decompose_and_initialize_phases(instruction, world_description, agent_info_string, llm_config)

# Get tasks for the current phase
def get_current_phase_tasks() -> Optional[Dict[str, Any]]

# Advance to the next execution phase
def advance_to_next_phase() -> bool

# Check whether the current phase is complete
def is_current_phase_complete(agent_statuses) -> bool
```

#### Usage example:
```python
# Usage in PerceptionConnector
self.phase_manager = PhaseManager(self.llm_client)
structured_subtasks = self.phase_manager.decompose_and_initialize_phases(
    instruction, world_desc_string, agent_info_string, llm_config
)
```

### 2. `dependency_enhancer.py` - Dependency Enhancer
**Core responsibility**: Build intelligent dependencies and execution phases based on task semantics

#### Main features:
- **Semantic dependency analysis**: Identify implicit dependencies in composite action sequences such as Pick-Navigate-Place
- **Dependency graph construction**: Automatically add necessary preconditions (e.g., Navigate before Pick)
- **Phase organization**: Organize tasks into parallelizable phases to optimize overall execution efficiency

#### Key methods:
```python
# Main entry: structure and phase processing
def structure_and_phase(structured_subtasks, max_agents) -> Tuple[List, List, Dict]

# Semantic dependency enhancement
def _enhance_semantic_dependencies(structured_subtasks) -> List[Dict[str, Any]]

# Build task dependency graph
def _build_dependency_graph(enhanced_tasks) -> Dict[str, List[str]]

# Organize tasks into execution phases
def _organize_tasks_into_phases(tasks, max_agents) -> List[Dict[str, Any]]
```

#### Usage example:
```python
# Usage in PerceptionConnector
enhancer = TaskDependencyEnhancer()
enhanced_subtasks, execution_phases, dependency_graph = enhancer.structure_and_phase(
    structured_subtasks, max_agents
)
```

### 3. `matrix_updater.py` - Matrix Updater
**Core responsibility**: Maintain parameter matrices required for MIQP optimization

#### Main features:
- **T matrix management**: Task-capability requirement matrix (13×5 dimensions)
- **A matrix management**: Agent capability matrix (5×2 dimensions)  
- **Weight vector management**: Task priority weight vector (ws)
- **Dynamic updates**: Dynamically adjust matrix parameters based on task characteristics and world state

#### Matrix definitions:
```python
# Base task-capability requirement matrix [13 tasks × 5 capabilities]
BASE_TASK_CAPABILITY_REQUIREMENTS = np.array([
    # [Mobility, Manipulation, Control, Liquid, Power]
    [1, 0, 0, 0, 0],  # Navigate
    [1, 0, 0, 0, 0],  # Explore
    [0, 1, 0, 0, 0],  # Pick
    [0, 1, 0, 0, 0],  # Place
    # ... other tasks
])

# Base agent capability matrix [5 capabilities × 2 agents]
BASE_ROBOT_CAPABILITIES = np.array([
    [2.0, 1.8],  # Mobility capability
    [2.0, 1.8],  # Manipulation capability
    [2.0, 1.8],  # Control capability
    [0.0, 1.4],  # Liquid handling (Agent 1 only)
    [0.0, 1.3]   # Power control (Agent 1 only)
])
```

#### Key methods:
```python
# Update all MIQP matrices
def update_matrices(structured_subtasks, world_state) -> Dict[str, Union[np.ndarray, List]]

# Update task-capability matrix based on task priority
def _update_task_capability_matrix_enhanced(subtasks, llm_analysis)

# Update capability matrix based on agent state
def _update_robot_capability_matrix_enhanced(subtasks, llm_analysis)
```

### 4. `action_updater.py` - Action Updater
**Core responsibility**: Convert high-level actions into scenario parameter updates

#### Main features:
- **Motor skill handling**: Handle Navigate, Pick, Place, Rearrange, Explore, Wait, and similar actions
- **State manipulation handling**: Handle Clean, Fill, Pour, PowerOn, PowerOff, and other special actions
- **Parameter generation**: Convert actions into parameter updates usable by the MIQP solver

#### Key methods:
```python
# Main processing method
def process_and_get_updates(high_level_actions, world_state) -> Dict[str, Any]

# Process motor skill related actions
def _process_motor_skill_actions(high_level_actions, world_state) -> Dict[str, Any]

# Process state manipulation related actions
def _process_state_manipulation_actions(high_level_actions, world_state) -> Dict[str, Any]

# Update navigation parameters
def _update_navigation_params(target_name, world_state) -> Dict[str, Any]
```

#### Usage example:
```python
# Usage in ExecutionManager
action_updater = ActionUpdater()
all_updates = action_updater.process_and_get_updates(high_level_actions, world_state)
```

### 5. `world_describer.py` - World Describer
**Core responsibility**: Generate detailed world descriptions for LLM prompts

#### Main features:
- **Environment layout description**: Generate detailed layout information for rooms and furniture
- **Agent state description**: Provide agent position and held-object information
- **Observation mode support**: Support full observation and partial observation modes

#### Key methods:
```python
# Generate complete world description
def get_world_description(env_interface, world_state, is_partial_obs=False) -> str

# Generate environment layout description
def _get_layout_description(env_interface, is_partial_obs=False) -> str

# Generate agent state description
def _get_agent_status_description(world_state) -> str
```

### 6. `planner_utils.py` - Utility Functions
**Core responsibility**: Provide common helper functions

#### Main features:
- **JSON parsing**: Extract structured data from LLM responses
- **Parameter updates**: Unified update interface for dictionaries and ScenarioConfigTask objects
- **Position lookup**: Find target position information in world state
- **Configuration management**: Provide a unified LLM configuration interface

#### Key methods:
```python
# Extract JSON from text
def extract_json_from_text(text: str, target_type: type = dict) -> Optional[str]

# Unified parameter update interface
def update_param_value(scenario_config, key: str, value: Any) -> None

# Find target position
def find_target_position(target_name: str, world_state: Dict[str, Any]) -> Optional[List[float]]

# Get LLM configuration
def get_llm_config() -> Dict[str, Any]
```

## 🔄 Inter-Module Collaboration Flow

### Typical call sequence

1. **Task decomposition phase**:
   ```python
   PhaseManager.decompose_and_initialize_phases()  # LLM decomposition
   ↓
   TaskDependencyEnhancer.structure_and_phase()   # Dependency enhancement
   ```

2. **Matrix update phase**:
   ```python
   MatrixUpdater.update_matrices()                # Parameter matrix update
   ↓
   planner_utils.update_param_value()            # Parameter application
   ```

3. **Action processing phase**:
   ```python
   ActionUpdater.process_and_get_updates()       # Action conversion
   ↓
   planner_utils.update_param_value()            # Parameter update
   ```

### Dependency diagram
```
PhaseManager ──────┐
                   ├──→ PerceptionConnector ──→ LLMPlanner
TaskDependencyEnhancer ┘
                   
MatrixUpdater ─────┐
                   ├──→ MIQP Solver
ActionUpdater ─────┘

planner_utils ────→ (used by all modules)
```

## 📚 Usage Guide

### Initialization
```python
from habitat_llm.planner.HRCS.connector.phase_manager import PhaseManager
from habitat_llm.planner.HRCS.connector.dependency_enhancer import TaskDependencyEnhancer
from habitat_llm.planner.HRCS.connector.matrix_updater import MatrixUpdater
from habitat_llm.planner.HRCS.connector.action_updater import ActionUpdater

# Initialize components
phase_manager = PhaseManager(llm_client)
dependency_enhancer = TaskDependencyEnhancer()
matrix_updater = MatrixUpdater(llm_client)
action_updater = ActionUpdater()
```

### Typical usage flow
```python
# 1. Task decomposition
structured_tasks = phase_manager.decompose_and_initialize_phases(
    instruction, world_description, agent_info, llm_config
)

# 2. Dependency enhancement and phase organization
enhanced_tasks, phases, deps = dependency_enhancer.structure_and_phase(
    structured_tasks, max_agents
)

# 3. Matrix update
matrices = matrix_updater.update_matrices(enhanced_tasks, world_state)

# 4. Action processing
updates = action_updater.process_and_get_updates(actions, world_state)
```

## 🧪 Testing Recommendations

### Unit test focus
- **PhaseManager**: Test LLM decomposition stability and phase management logic
- **TaskDependencyEnhancer**: Test accuracy of dependency identification
- **MatrixUpdater**: Verify matrix dimensions and numerical correctness
- **ActionUpdater**: Test action-to-parameter conversion logic

### Integration test focus
- Completeness of inter-module data transfer
- Correctness of end-to-end planning flow
- Robustness under exceptional conditions

## 🔧 Extension Guide

### Adding new task types
1. Update `BASE_TASK_CAPABILITY_REQUIREMENTS` in `MatrixUpdater`
2. Add corresponding handling logic in `ActionUpdater`
3. Add semantic dependency rules in `TaskDependencyEnhancer`

### Adding new agent capabilities
1. Extend the `BASE_ROBOT_CAPABILITIES` matrix
2. Update `BASE_CAPABILITY_WEIGHTS` weights
3. Modify related processing logic
