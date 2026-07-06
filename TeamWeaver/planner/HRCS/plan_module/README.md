# HRCS Plan Module Architecture

## 📋 Overview

The `plan_module/` folder contains the core execution modules of the MIQP-LLM hybrid planning system. It combines MIQP optimization results with LLM reasoning to deliver intelligent, stable task planning and execution management. These modules follow a **goal-oriented design**, focused on high-quality planning decisions and execution control.

## 🏗️ Architecture Design Philosophy

### Core Design Principles
- **Goal-oriented**: Provide goal guidance rather than direct action commands, preserving LLM decision autonomy
- **Intelligent enhancement**: MIQP optimization supplies globally optimal macro-level guidance to the LLM
- **Feedback-driven**: Learn and adapt from execution feedback to improve system stability
- **Module decoupling**: Each module has a clear responsibility for independent testing and extension

### Execution Flow
```
MIQP Optimization → TaskHelper Assignment → ActionManager Validation → 
PromptBuilder Enhancement → ErrorHandler Recovery → ExecutionManager Update → 
FeedbackManager Evaluation
```

## 📦 Module Details

### 1. `action_manager.py` - Intelligent Action Manager
**Core responsibility**: Parse LLM responses and validate action goal consistency

#### Main features:
- **Action parsing**: Convert LLM responses into structured high-level actions
- **Goal-oriented validation**: Verify whether LLM actions help achieve phase objectives
- **Intelligent adjustment**: Provide goal-oriented suggestions while preserving LLM decision autonomy

#### Key methods:
```python
# Parse high-level actions from LLM response
def parse_high_level_actions(llm_response: str, agents) -> Dict[int, Tuple[str, str, Optional[str]]]

# Goal-oriented action validation and adjustment
def adjust_actions_with_phase_awareness(
    high_level_actions, agent_task_assignments, current_phase, agents
) -> Dict[int, Tuple[str, str, Optional[str]]]

# Verify whether an action advances objective achievement
def _action_advances_objective(action_name, action_target, objective) -> bool

# Suggest appropriate actions for an objective
def _suggest_action_for_objective(objective, original_action) -> Tuple[str, str, Optional[str]]
```

#### Design characteristics:
```python
# Goal-oriented validation logic example
if self._action_advances_objective(llm_action, llm_target, primary_objective):
    # LLM action advances the objective, keep unchanged
    adjusted_actions[agent_id] = action_tuple
    print(f"LLM action '{llm_action}[{llm_target}]' advances objective ✓")
else:
    # Provide goal-oriented suggestions while maintaining flexibility
    if self._is_exploration_reasonable(llm_action, primary_objective):
        adjusted_actions[agent_id] = action_tuple
```

### 2. `task_helper.py` - Resume-Based Intelligent Task Assignment System
**Core responsibility**: Perform intelligent task assignment based on Agent capability profiles

#### Main features:
- **Agent resume management**: Maintain each Agent's capabilities, performance traits, and dynamic state
- **Intelligent scoring assignment**: Dual-layer matching with hard constraints + soft constraints
- **Dynamic learning**: Update Agent performance estimates from execution feedback
- **Load balancing**: Avoid overloading individual Agents

#### Agent resume structure:
```python
@dataclass
class AgentResume:
    agent_id: int
    name: str
    
    # Hard constraints: task types the Agent can execute
    capabilities: List[str] = field(default_factory=list)
    
    # Soft constraints: performance trait scores (0-1)
    performance_traits: Dict[str, float] = field(default_factory=dict)
    
    # Dynamic state: real-time decision information
    current_task_load: int = 0
    position: Optional[np.ndarray] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
```

#### Key methods:
```python
# Core scoring function: compute Agent suitability score for a task
def _calculate_assignment_score(task: Dict[str, Any], resume: AgentResume) -> float

# Intelligent task assignment based on scores
def _distribute_tasks_with_scoring(tasks, agent_ids) -> Dict[int, List[Dict[str, Any]]]

# Update performance traits from feedback
def _update_traits_from_feedback(resume: AgentResume, feedback: str)

# Update Agent resume context
def update_resumes_from_context(world_state, agent_feedback)
```

#### Intelligent assignment example:
```python
# Agent 0: "Workhorse" - strong baseline capabilities, high speed
Agent_0_Resume = {
    'capabilities': ['Navigate', 'Explore', 'Pick', 'Place', 'Rearrange'],
    'performance_traits': {
        'speed': 0.9,           # High speed
        'precision': 0.6,       # Medium precision
        'exploration_bias': 0.5 # Medium exploration tendency
    }
}

# Agent 1: "Specialist" - precision operations, specialized skills
Agent_1_Resume = {
    'capabilities': ['Navigate', 'Pick', 'Place', 'Clean', 'Fill', 'PowerOn'],
    'performance_traits': {
        'speed': 0.6,                    # Medium speed
        'precision': 0.9,                # High precision
        'liquid_handling_skill': 0.95,   # Liquid handling expertise
        'power_control_skill': 0.9       # Power control expertise
    }
}
```

### 3. `miqp_solver_wrapper.py` - MIQP Solver Wrapper
**Core responsibility**: Manage MIQP optimization parameters and the solving process

#### Main features:
- **Parameter management**: Initialize and maintain ScenarioConfigTask and OptimizationConfigTask
- **Phase-aware solving**: Support dynamic task instance solving for the current phase
- **RTA integration**: Encapsulate the RTA solver call interface

#### Key methods:
```python
# Set MIQP parameters
def task_plan_MIQP_set(agents: List["Agent"])

# Phase-aware MIQP solving
def task_plan_MIQP_solve_phase_aware(x, t, phase_task_info, agents)

# Reset solver state
def reset()
```

#### Solving flow:
```python
# Typical solve call
alpha, u, delta, time_to_solve, opt_sol_info = self.miqp_solver_wrapper.task_plan_MIQP_solve_phase_aware(
    x,                  # Current state
    t,                  # Time
    phase_task_info,    # Phase task information
    self._agents        # Agent list
)
```

### 4. `prompt_builder.py` - Intelligent Prompt Builder
**Core responsibility**: Build goal-oriented prompts guided by MIQP

#### Main features:
- **MIQP guidance construction**: Convert optimization results into goal descriptions the LLM can understand
- **Goal-oriented formatting**: Describe objectives to achieve rather than specific execution steps
- **Context enhancement**: Integrate world state and phase information

#### Key methods:
```python
# Build MIQP guidance information
def build_miqp_guidance_addition(
    current_phase_tasks, agent_task_assignments, current_phase, 
    perception_connector, alpha_result=None, world_state=None
) -> str

# Build phase-aware enhanced prompt (deprecated; use incremental approach)
def build_phase_aware_prompt_with_miqp_guidance(...)
```

#### Goal-oriented design example:
```python
# Traditional action instruction (avoid)
"Execute Pick[apple]"

# Goal-oriented guidance (recommended)
"achieve pickup of apple (requires navigation first if not nearby)"
```

### 5. `error_handler.py` - Intelligent Error Handler
**Core responsibility**: Detect execution failures and provide intelligent recovery strategies

#### Main features:
- **Failure analysis**: Analyze Agent status feedback and identify failure patterns
- **Recovery strategies**: Generate targeted recovery actions based on failure type
- **History tracking**: Maintain error history and recovery attempt records

#### Key methods:
```python
# Apply intelligent error recovery
def recover_and_log_assignments(
    agent_task_assignments, last_high_level_actions, latest_agent_response
) -> Dict[int, List[Dict[str, Any]]]

# Analyze failure and suggest recovery actions
def analyze_failure_and_suggest_recovery(
    agent_id, status, current_action
) -> Optional[Tuple[str, str, Optional[str]]]
```

#### Intelligent recovery logic:
```python
# Recovery strategy example
if action_name == "Pick" and "not close enough" in status_lower:
    return ("Navigate", target, target)  # Navigate to target location
elif action_name == "Pick" and "object not found" in status_lower:
    return ("Explore", "environment", "environment")  # Explore to find object
elif "collision" in status_lower:
    return ("Wait", "", "")  # Wait to avoid conflict
```

### 6. `execution_manager.py` - Execution Manager
**Core responsibility**: Manage the final execution steps of the planning cycle

#### Main features:
- **Scenario parameter updates**: Update scenario configuration based on selected high-level actions
- **ActionUpdater integration**: Call ActionUpdater to handle action conversion
- **Execution state tracking**: Maintain execution-related state information

#### Key methods:
```python
# Update scenario configuration for execution
def update_scenario_for_execution(
    scenario_config, world_state, high_level_actions
) -> None
```

#### Usage example:
```python
# Usage in llm_planner_miqp.py
self.execution_manager.update_scenario_for_execution(
    self.miqp_solver_wrapper.scenario_params,
    world_state,
    adjusted_actions
)
```

### 7. `feedback_manager.py` - Feedback Manager
**Core responsibility**: Manage execution feedback and phase advancement

#### Main features:
- **Status extraction**: Obtain Agent completion status from multiple sources
- **Phase evaluation**: Check whether the current phase is complete
- **Phase advancement**: Manage phase transition logic
- **Performance tracking**: Maintain Agent performance metrics

#### Key methods:
```python
# Get Agent completion statuses
def get_agent_completion_statuses(agents, latest_agent_response) -> Dict[int, str]

# Check and advance phase
def check_and_advance_phase(
    perception_connector, agents, latest_agent_response, current_phase
) -> bool
```

#### Intelligent status retrieval:
```python
# Priority strategy: latest response > Agent status > default value
def get_agent_completion_statuses(agents, latest_agent_response):
    if latest_agent_response:
        # Prefer the latest response
        return latest_agent_response
    else:
        # Fallback: get Agent status descriptions
        return {agent.uid: agent.get_last_state_description() for agent in agents}
```

## 🔄 Inter-Module Collaboration Flow

### Call sequence in the replan method

```python
# Step 8: Intelligent task assignment
agent_task_assignments = self.task_helper.assign_tasks_for_phase(
    current_phase_tasks, alpha, phase_task_info, self._agents, ...
)

# Step 9: Intelligent error recovery  
agent_task_assignments = self.error_handler.recover_and_log_assignments(
    agent_task_assignments, self.last_high_level_actions, self.latest_agent_response
)

# Step 10: Prompt construction
miqp_guidance = self.prompt_builder.build_miqp_guidance_addition(
    current_phase_tasks, agent_task_assignments, current_phase, ...
)

# Step 11: LLM call (feedback-enhanced)
llm_response = self.llm.generate(self.curr_prompt + miqp_guidance, self.stopword)

# Step 12: Action parsing and validation
high_level_actions = self.action_manager.parse_high_level_actions(llm_response, self._agents)
adjusted_actions = self.action_manager.adjust_actions_with_phase_awareness(
    high_level_actions, agent_task_assignments, current_phase, self._agents
)

# Step 13: Execution parameter update
self.execution_manager.update_scenario_for_execution(
    self.miqp_solver_wrapper.scenario_params, world_state, adjusted_actions
)

# Step 15: Feedback and phase management
self._phase_transition_pending = self.feedback_manager.check_and_advance_phase(
    self.perception_connector, self._agents, self.latest_agent_response, current_phase
)
```

### Module dependency diagram
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

## 📚 Usage Guide

### Initialization
```python
from habitat_llm.planner.HRCS.plan_module.action_manager import ActionManager
from habitat_llm.planner.HRCS.plan_module.task_helper import TaskHelper
from habitat_llm.planner.HRCS.plan_module.miqp_solver_wrapper import MIQPSolverWrapper
# ... other modules

# Initialize in LLMPlanner.__init__
self.miqp_solver_wrapper = MIQPSolverWrapper()
self.task_helper = TaskHelper(self._agents)
self.action_manager = ActionManager(self.actions_parser)
self.error_handler = ErrorHandler()
self.prompt_builder = PromptBuilder(plan_config, env_interface)
self.execution_manager = ExecutionManager()
self.feedback_manager = FeedbackManager()
```

### Typical usage flow
```python
# 1. Set MIQP parameters
self.miqp_solver_wrapper.task_plan_MIQP_set(self._agents)

# 2. Intelligent task assignment
assignments = self.task_helper.assign_tasks_for_phase(...)

# 3. Error recovery
recovered_assignments = self.error_handler.recover_and_log_assignments(...)

# 4. Build enhanced prompt
guidance = self.prompt_builder.build_miqp_guidance_addition(...)

# 5. Parse and validate actions
actions = self.action_manager.parse_high_level_actions(...)
adjusted_actions = self.action_manager.adjust_actions_with_phase_awareness(...)

# 6. Update execution parameters
self.execution_manager.update_scenario_for_execution(...)

# 7. Manage feedback and phases
phase_transition = self.feedback_manager.check_and_advance_phase(...)
```

## 🎯 Design Highlights

### 1. Goal-oriented rather than action-oriented
- **Traditional approach**: Directly specify "Execute Pick[object]"
- **This system**: Describe the goal "achieve pickup of object (requires navigation first)"

### 2. Intelligent Agent resume system
- **Capability boundaries**: Hard constraints ensure task executability
- **Performance optimization**: Soft constraints optimize assignment efficiency
- **Dynamic learning**: Adjust Agent evaluation based on feedback

### 3. Feedback-enhanced planning loop
- **History preservation**: Maintain complete dialogue history and execution feedback
- **MIQP enhancement**: Optimization guidance as incremental information rather than replacement
- **Phase awareness**: Dynamic planning adjustments based on execution phase

## 🧪 Testing Recommendations

### Unit test focus
- **ActionManager**: Test goal validation logic and action adjustment strategies
- **TaskHelper**: Verify scoring algorithm and assignment fairness
- **ErrorHandler**: Test failure identification and recovery strategy accuracy
- **PromptBuilder**: Verify prompt format and clarity of goal descriptions

### Integration test focus
- **MIQP-LLM consistency**: Verify coordination between optimization suggestions and LLM decisions
- **Feedback loop**: Test the impact of execution feedback on subsequent planning
- **Phase advancement**: Verify correctness and timeliness of phase transitions

## 🔧 Extension Guide

### Adding new Agent types
1. Extend `_initialize_agent_resumes()` in `TaskHelper`
2. Add new capability traits and scoring logic
3. Update the `_calculate_assignment_score()` method

### Adding new error types
1. Extend `analyze_failure_and_suggest_recovery()` in `ErrorHandler`
2. Add new failure pattern recognition logic
3. Design corresponding recovery strategies

### Extending the MIQP solver
1. Add new solving modes in `MIQPSolverWrapper`
2. Extend the phase_task_info structure
3. Update the solver call interface
