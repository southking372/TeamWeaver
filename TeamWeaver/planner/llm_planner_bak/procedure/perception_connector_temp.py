"""
PerceptionConnector模块 - 连接感知系统与MIQP优化系统
提供世界状态提取、任务分解、矩阵更新等功能
"""

from typing import Dict, List, Any, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import json
import abc
import requests
import os
import sys
from pathlib import Path
from habitat_llm.planner.miqp_prompts import get_miqp_prompt, MIQPAnalysisPrompt, TaskDecompositionPrompt
from habitat_llm.planner.HRCS.connector.dependency_enhancer import TaskDependencyEnhancer
from habitat_llm.planner.HRCS.connector.planner_utils import (
    extract_json_from_text,
    get_llm_config,
    update_param_value,
)
from habitat_llm.planner.HRCS.connector.action_updater import ActionUpdater
from habitat_llm.planner.HRCS.connector.matrix_updater import MatrixUpdater
import openai

if TYPE_CHECKING:
    from habitat_llm.planner.HRCS.params_module.scenario_params_task import ScenarioConfigTask
    from habitat_llm.agent.env import EnvironmentInterface


def quaternion_to_yaw(quaternion: List[float]) -> float:
    """将四元数转换为 Z 轴旋转（偏航角 Yaw）。"""
    quat = np.array(quaternion)
    x, y, z, w = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)

class PerceptionConnector:
    """
    连接感知信息（来自 WorldGraph/环境）和任务规划参数（ScenarioConfigTask）。
    
    主要功能:
    1. 任务分解: 使用LLM将指令分解为结构化子任务
    2. 世界状态提取: 从环境获取智能体、物体、家具位置信息
    3. MIQP矩阵更新: 基于Agent配置的能力维度更新优化参数
    4. 场景参数更新: 根据高级动作更新任务目标和约束
    5. 任务分配: 基于MIQP优化结果将子任务分配给智能体
    6.  任务序列化: 将子任务组织为有依赖关系的执行阶段
    
    能力维度组织:
    - Motor Skills: nav, pick, place, open, close, rearrange, explore, wait
    - Object States: power_on, power_off, clean, fill, pour  
    - Perception: find_receptacle, find_object, find_agent_action, find_room
    """
    
    # --- MIQP矩阵初始化常量 ---
    BASE_TASK_CAPABILITY_REQUIREMENTS = np.array([
        # 任务: [Navigate, Explore, Pick, Place, Open, Close, Clean, Fill, Pour, PowerOn, PowerOff, Rearrange, Wait]
        # 能力: [移动, 操作, 控制, 液体, 电源]
        [1, 0, 0, 0, 0],  # Navigate
        [1, 0, 0, 0, 0],  # Explore
        [0, 1, 0, 0, 0],  # Pick
        [0, 1, 0, 0, 0],  # Place
        [0, 0, 1, 0, 0],  # Open
        [0, 0, 1, 0, 0],  # Close
        [0, 0, 0, 1, 0],  # Clean
        [0, 0, 0, 1, 0],  # Fill
        [0, 0, 0, 1, 0],  # Pour
        [0, 0, 0, 0, 1],  # PowerOn
        [0, 0, 0, 0, 1],  # PowerOff
        [0, 1, 0, 0, 0],  # Rearrange
        [0, 0, 0, 0, 0],  # Wait
    ], dtype=float)

    BASE_ROBOT_CAPABILITIES = np.array([
        [2.0, 1.8],  # 移动
        [2.0, 1.8],  # 操作
        [2.0, 1.8],  # 控制
        [0.0, 1.3],  # 液体 (仅Agent 1)
        [0.0, 1.3]   # 电源 (仅Agent 1)
    ], dtype=float)
    
    BASE_CAPABILITY_WEIGHTS = [
        2.0 * np.eye(1),  # 移动
        2.5 * np.eye(1),  # 操作
        2.0 * np.eye(1),  # 控制
        1.8 * np.eye(1),  # 液体
        1.5 * np.eye(1)   # 电源
    ]
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        api_key_filename: Optional[str] = None,
        llm_base_url: Optional[str] = "https://api.moonshot.cn/v1"
    ):
        self.last_world_state: Dict[str, Any] = {}
        self.llm_client = llm_client
        self.matrix_updater = MatrixUpdater(llm_client)
        api_key_filename = "api_key"

        #  添加任务序列管理
        self.task_execution_phases: List[Dict[str, Any]] = []
        self.current_phase_index: int = 0
        self.task_dependency_graph: Dict[str, List[str]] = {}
        self.completed_tasks: List[str] = []
        self.active_tasks: List[str] = []
        
        #  阶段性T矩阵缓存
        self.phase_t_matrices: Dict[int, np.ndarray] = {}
        
        if self.llm_client is None and api_key_filename:
            try:
                api_key_path = Path(api_key_filename + '.txt')
                if not api_key_path.exists():
                    api_key_path = Path(api_key_filename)

                api_key = api_key_path.read_text().strip()
                self.llm_client = openai.OpenAI(
                    api_key=api_key,
                    base_url=llm_base_url,
                )
                print(f"PerceptionConnector: LLM client initialized using API key from {api_key_path} and base URL {llm_base_url}")
            except Exception as e:
                print(f"Error initializing LLM client in PerceptionConnector: {e}")
        else:
            print("PerceptionConnector: LLM client not provided and api_key_filename not specified. Task decomposition will not be available.")

    # --- 主要公共接口 ---

    def extract_world_state(self, env_interface: "EnvironmentInterface") -> Dict[str, Any]:
        """
        从环境接口提取当前世界状态。
        """
        world_state: Dict[str, Any] = {
            'agent_poses': {},
            'object_positions': {},
            'furniture_positions': {},
        }
        full_graph = env_interface.full_world_graph

        # 1. 提取 Agent 位姿
        agents = full_graph.get_agents()
        for agent_node in agents:
            agent_id = agent_node.name
            try:
                pos = agent_node.get_property("translation")
                try:
                    rot_quat = agent_node.get_property("rotation")
                    yaw = quaternion_to_yaw(rot_quat)
                except (KeyError, AttributeError, ValueError):
                    rot_quat = [0.0, 0.0, 0.0, 1.0]
                    yaw = 0.0
                world_state['agent_poses'][agent_id] = {'position': pos, 'rotation': rot_quat, 'yaw': yaw}
            except (KeyError, AttributeError, ValueError) as e:
                print(f"Error: 无法获取 Agent '{agent_id}' 的位姿: {e}")
                world_state['agent_poses'][agent_id] = {'position': [0.0, 0.0, 0.0], 'rotation': [0.0, 0.0, 0.0, 1.0], 'yaw': 0.0}

        # 2. 提取 Object 位置和父物体
        all_objects = full_graph.get_all_objects()
        for obj_node in all_objects:
            obj_name = obj_node.name
            try:
                pos = obj_node.get_property("translation")
                parent_node = full_graph.find_furniture_for_object(obj_node)
                parent_name = parent_node.name if parent_node else None
                world_state['object_positions'][obj_name] = {'position': pos, 'parent': parent_name}
            except (KeyError, AttributeError) as e:
                print(f"Error: 无法获取 Object '{obj_name}' 的位置: {e}")
                world_state['object_positions'][obj_name] = None

        # 3. 提取 Furniture 位置
        all_furniture = full_graph.get_all_furnitures()
        for furn_node in all_furniture:
            furn_name = furn_node.name
            try:
                pos = furn_node.get_property("translation")
                world_state['furniture_positions'][furn_name] = {'position': pos}
            except (KeyError, AttributeError) as e:
                print(f"Error: 无法获取 Furniture '{furn_name}' 的位置: {e}")
                world_state['furniture_positions'][furn_name] = None # 或者一个默认值

        self.last_world_state = world_state
        return world_state

    def structured_decompose_task_with_sequencing(
        self,
        instruction: str,
        env_interface: "EnvironmentInterface",
        llm_config: Dict[str, Any],
        max_agents: int = 2
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        使用LLM将指令分解为带序列依赖的结构化子任务，并组织为执行阶段。
            
        Returns:
            (structured_subtasks, execution_phases)
            structured_subtasks: 完整的子任务列表
            execution_phases: 按阶段组织的任务执行计划
        """
        if not self.llm_client:
            raise ValueError("LLM client not initialized. Cannot decompose task.")

        # 1. 使用增强的序列化分解prompt
        structured_subtasks = self._decompose_with_sequencing_prompt(
            instruction, env_interface, llm_config
        )
        
        print(f"DEBUG:  Initial LLM decomposition: {len(structured_subtasks)} tasks")
        
        # 2. 语义增强依赖关系构建
        enhancer = TaskDependencyEnhancer()
        enhanced_subtasks, execution_phases = enhancer.structure_and_phase(
            structured_subtasks, max_agents
        )
        
        # 3. 分析任务依赖关系
        self.task_dependency_graph = self._build_dependency_graph(enhanced_subtasks)
        
        # 4. 将任务组织为执行阶段
        execution_phases = self._organize_tasks_into_phases(
            enhanced_subtasks, max_agents
        )
        
        # 4. 缓存阶段信息
        self.task_execution_phases = execution_phases
        self.current_phase_index = 0
        
        print(f"DEBUG: Task decomposed into {len(enhanced_subtasks)} subtasks across {len(execution_phases)} phases")
        for i, phase in enumerate(execution_phases):
            task_summaries = [f"{t['task_type']}→{t['target']}" for t in phase['tasks']]
            carrying_tasks = [t for t in phase['tasks'] if t.get('is_carrying_navigation', False)]
            carrying_note = f" [CARRYING: {len(carrying_tasks)}]" if carrying_tasks else ""
            print(f"  Phase {i+1}: {task_summaries} (max_parallel: {phase['max_parallel_tasks']}){carrying_note}")
        
        return enhanced_subtasks, execution_phases

    def _decompose_with_sequencing_prompt(
        self,
        instruction: str,
        env_interface: "EnvironmentInterface",
        llm_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """使用专门的序列化分解prompt"""
        objects_prompt_string = self._get_objects_string_for_prompt(env_interface)
        current_world_state = self.last_world_state or self.extract_world_state(env_interface)
        agent_info_string = self._get_agent_status_for_prompt(current_world_state)
        prompt_template = get_miqp_prompt("sequencing_decomposition", get_llm_config())
        sequencing_prompt = prompt_template(
            instruction, objects_prompt_string, agent_info_string
        )

        try:
            messages = [
                {"role": "system", "content": "You are a precise task sequencing assistant. Return only valid JSON."},
                {"role": "user", "content": sequencing_prompt}
            ]
            api_params = {
                "model": llm_config.get("gpt_version", "moonshot-v1-32k"),
                "messages": messages,
                "max_tokens": llm_config.get("max_tokens", 1500),
                "temperature": 0.1
            }

            response = self.llm_client.chat.completions.create(**api_params)
            decomposed_text = response.choices[0].message.content.strip()
            
            structured_tasks = self._parse_sequenced_decomposition_response(decomposed_text)
            
            if structured_tasks:
                print(f"DEBUG: Successfully parsed {len(structured_tasks)} sequenced subtasks")
                return structured_tasks
            else:
                print("Warning: Failed to parse sequenced subtasks, using simple decomposition fallback.")
                return self.structured_decompose_task(instruction, env_interface, llm_config)
                
        except Exception as e:
            print(f"Error calling LLM for sequenced task decomposition: {e}")
            raise

    def _parse_sequenced_decomposition_response(self, response_text: str) -> List[Dict[str, Any]]:
        """解析带序列信息的任务分解响应"""
        try:
            json_text = extract_json_from_text(response_text, list)
            if not json_text:
                return []

            structured_tasks = json.loads(json_text)
            validated_tasks = []
            required_fields = ['task_type', 'target', 'description']
            
            for i, task in enumerate(structured_tasks):
                if not isinstance(task, dict) or not all(field in task for field in required_fields):
                    continue
                
                task_id = f"task_{i}"
                prerequisites = task.get('prerequisites', [])
                
                cleaned_prerequisites = []
                for prereq in prerequisites:
                    if prereq == task_id:
                        continue # Remove self-reference
                    if isinstance(prereq, str) and prereq.strip():
                        cleaned_prerequisites.append(prereq.strip())
                
                validated_task = {
                    'task_id': task_id,
                    'task_type': task['task_type'],
                    'target': task['target'],
                    'description': task['description'],
                    'priority': max(1, min(5, task.get('priority', 3))),  # 限制优先级范围
                    'estimated_duration': max(1.0, min(60.0, task.get('estimated_duration', 5.0))),  # 限制持续时间
                    'preferred_agent': task.get('preferred_agent'),
                    'prerequisites': cleaned_prerequisites,
                    'can_parallel': bool(task.get('can_parallel', False)),
                    'phase_group': task.get('phase_group', 'execution')
                }
                validated_tasks.append(validated_task)
            
            # 最终验证：确保所有前置条件都指向有效的任务
            task_ids = {task['task_id'] for task in validated_tasks}
            for task in validated_tasks:
                valid_prereqs = [prereq for prereq in task['prerequisites'] if prereq in task_ids]
                if len(valid_prereqs) != len(task['prerequisites']):
                    invalid_prereqs = [prereq for prereq in task['prerequisites'] if prereq not in task_ids]
                    print(f"WARNING: Removed invalid prerequisites {invalid_prereqs} from task {task['task_id']}")
                    task['prerequisites'] = valid_prereqs
            
            print(f"DEBUG: Validated {len(validated_tasks)} tasks from LLM response")
            return validated_tasks
            
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: JSON parsing for sequenced decomposition failed: {e}")
            return []

    def _build_dependency_graph(self, structured_subtasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """ 构建任务依赖关系图并检测循环依赖"""
        dependency_graph = {}
        task_ids = {task['task_id'] for task in structured_subtasks}
        
        for task in structured_subtasks:
            task_id = task['task_id']
            prerequisites = task.get('prerequisites', [])
            
            # 清理前置条件：移除自循环和无效引用
            cleaned_prerequisites = []
            for prereq in prerequisites:
                if prereq == task_id:
                    print(f"WARNING: Detected self-dependency in {task_id}, removing")
                    continue
                if prereq not in task_ids:
                    print(f"WARNING: Invalid prerequisite '{prereq}' for task {task_id}, removing")
                    continue
                cleaned_prerequisites.append(prereq)
            
            dependency_graph[task_id] = cleaned_prerequisites
        
        # 检测更复杂的循环依赖
        cycles = self._detect_dependency_cycles(dependency_graph)
        if cycles:
            print(f"WARNING: Detected {len(cycles)} dependency cycles:")
            for i, cycle in enumerate(cycles):
                print(f"  Cycle {i+1}: {' -> '.join(cycle + [cycle[0]])}")
            
            # 尝试修复循环依赖
            dependency_graph = self._break_dependency_cycles(dependency_graph, cycles)
            
        print(f"DEBUG: Built dependency graph with {len(dependency_graph)} tasks")
        for task_id, deps in dependency_graph.items():
            if deps:
                print(f"  {task_id} depends on: {deps}")
        
        return dependency_graph

    def _detect_dependency_cycles(self, dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
        """检测依赖图中的循环"""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                cycle_start = path.index(node)
                cycle = path[cycle_start:]
                cycles.append(cycle)
                return True
            if node in visited:
                return False
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependency_graph.get(node, []):
                if dfs(neighbor, path + [neighbor]):
                    pass
            
            rec_stack.remove(node)
            return False
        
        for task_id in dependency_graph:
            if task_id not in visited:
                dfs(task_id, [task_id])
        
        return cycles

    def _break_dependency_cycles(self, dependency_graph: Dict[str, List[str]], cycles: List[List[str]]) -> Dict[str, List[str]]:
        """打破依赖循环"""
        cleaned_graph = dependency_graph.copy()
        
        for cycle in cycles:
            # 简单策略：移除循环中优先级最低的任务的一个依赖
            if len(cycle) >= 2:
                # 选择要打破的边：通常是最后一条边
                source_task = cycle[-1]
                target_task = cycle[0]
                
                if target_task in cleaned_graph.get(source_task, []):
                    cleaned_graph[source_task].remove(target_task)
                    print(f"INFO: Broke cycle by removing dependency: {source_task} -> {target_task}")
        
        return cleaned_graph

    def _organize_tasks_into_phases(
        self, 
        structured_subtasks: List[Dict[str, Any]], 
        max_agents: int
    ) -> List[Dict[str, Any]]:
        """将任务组织为可并行执行的阶段"""
        
        # 1. 按依赖关系进行拓扑排序
        sorted_tasks = self._topological_sort_tasks(structured_subtasks)
        
        # 2. 分组为执行阶段
        phases = []
        remaining_tasks = sorted_tasks.copy()
        safety_counter = 0  # 防止无限循环
        max_iterations = len(sorted_tasks) * 2
        
        while remaining_tasks and safety_counter < max_iterations:
            safety_counter += 1
            current_phase_tasks = []
            tasks_to_remove = []
            
            for task in remaining_tasks:
                # 检查此任务的前置条件是否都已完成
                prerequisites = task.get('prerequisites', [])
                all_prerequisites_done = True
                
                if prerequisites:
                    all_prerequisites_done = all(
                        any(completed_task['task_id'] == prereq 
                            for phase in phases 
                            for completed_task in phase['tasks'])
                        for prereq in prerequisites
                    )
                
                # 检查是否可以并行执行
                can_add_to_phase = (
                    all_prerequisites_done and 
                    len(current_phase_tasks) < max_agents and
                    (task.get('can_parallel', False) or len(current_phase_tasks) == 0)
                )
                
                if can_add_to_phase:
                    current_phase_tasks.append(task)
                    tasks_to_remove.append(task)
                    
                    # 如果任务不能并行，这个阶段就只有这一个任务
                    if not task.get('can_parallel', False):
                        break
            
            # 如果没有任务可以添加到当前阶段，强制添加一个任务以避免死锁
            if not current_phase_tasks and remaining_tasks:
                print(f"WARNING: Force adding task to break potential deadlock")
                forced_task = remaining_tasks[0]
                current_phase_tasks.append(forced_task)
                tasks_to_remove.append(forced_task)
            
            # 移除已添加到当前阶段的任务
            for task in tasks_to_remove:
                remaining_tasks.remove(task)
            
            # 创建阶段信息
            if current_phase_tasks:
                phase_info = {
                    'phase_id': len(phases),
                    'tasks': current_phase_tasks,
                    'max_parallel_tasks': len(current_phase_tasks),
                    'estimated_duration': max(task.get('estimated_duration', 5.0) 
                                             for task in current_phase_tasks),
                    'required_agents': min(len(current_phase_tasks), max_agents)
                }
                phases.append(phase_info)
        
        # 检查是否还有剩余任务
        if remaining_tasks:
            print(f"WARNING: {len(remaining_tasks)} tasks could not be properly organized into phases")
            # 将剩余任务强制添加到最后一个阶段
            if phases:
                phases[-1]['tasks'].extend(remaining_tasks)
                phases[-1]['max_parallel_tasks'] = len(phases[-1]['tasks'])
                phases[-1]['required_agents'] = min(len(phases[-1]['tasks']), max_agents)
            else:
                # 创建一个新阶段包含所有剩余任务
                phases.append({
                    'phase_id': 0,
                    'tasks': remaining_tasks,
                    'max_parallel_tasks': len(remaining_tasks),
                    'estimated_duration': max(task.get('estimated_duration', 5.0) 
                                             for task in remaining_tasks),
                    'required_agents': min(len(remaining_tasks), max_agents)
                })
        
        print(f"DEBUG: Organized {len(sorted_tasks)} tasks into {len(phases)} phases")
        return phases

    def _topological_sort_tasks(self, structured_subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ 对任务进行拓扑排序，自动处理循环依赖"""
        from collections import deque, defaultdict
        
        # 首先清理依赖关系，移除自循环和无效依赖
        cleaned_tasks = self._clean_task_dependencies(structured_subtasks)
        
        # 构建入度表
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        task_dict = {task['task_id']: task for task in cleaned_tasks}
        
        # 初始化入度
        for task in cleaned_tasks:
            in_degree[task['task_id']] = 0
        
        # 构建图和入度
        for task in cleaned_tasks:
            task_id = task['task_id']
            for prereq in task.get('prerequisites', []):
                if prereq in task_dict and prereq != task_id:  # 避免自循环
                    graph[prereq].append(task_id)
                    in_degree[task_id] += 1
        
        # 拓扑排序
        queue = deque([task_id for task_id in in_degree if in_degree[task_id] == 0])
        sorted_task_ids = []
        
        while queue:
            current = queue.popleft()
            sorted_task_ids.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 检查是否还有环（剩余的任务）
        remaining_tasks = [task_id for task_id in in_degree if task_id not in sorted_task_ids]
        if remaining_tasks:
            print(f"WARNING: {len(remaining_tasks)} tasks still have circular dependencies after cleaning:")
            for task_id in remaining_tasks:
                print(f"  - {task_id}: {task_dict[task_id].get('prerequisites', [])}")
            
            # 将剩余任务按优先级排序后添加到结果中
            remaining_task_objs = [task_dict[task_id] for task_id in remaining_tasks]
            remaining_task_objs.sort(key=lambda x: x.get('priority', 3), reverse=True)
            sorted_task_ids.extend([task['task_id'] for task in remaining_task_objs])
            
            print(f"INFO: Added {len(remaining_tasks)} tasks with circular dependencies in priority order")
        
        return [task_dict[task_id] for task_id in sorted_task_ids]

    def _clean_task_dependencies(self, structured_subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清理任务依赖关系，移除自循环和无效引用"""
        cleaned_tasks = []
        task_ids = {task['task_id'] for task in structured_subtasks}
        
        for task in structured_subtasks:
            cleaned_task = task.copy()
            original_prereqs = task.get('prerequisites', [])
            
            # 清理前置条件
            cleaned_prereqs = []
            for prereq in original_prereqs:
                # 移除自循环
                if prereq == task['task_id']:
                    print(f"WARNING: Removed self-dependency for task {task['task_id']}")
                    continue
                
                # 移除无效的任务引用
                if prereq not in task_ids:
                    print(f"WARNING: Removed invalid prerequisite '{prereq}' for task {task['task_id']}")
                    continue
                
                cleaned_prereqs.append(prereq)
            
            cleaned_task['prerequisites'] = cleaned_prereqs
            cleaned_tasks.append(cleaned_task)
            
            if len(cleaned_prereqs) != len(original_prereqs):
                print(f"INFO: Cleaned prerequisites for {task['task_id']}: {original_prereqs} -> {cleaned_prereqs}")
        
        return cleaned_tasks

    def get_current_phase_tasks(self) -> Optional[Dict[str, Any]]:
        """ 获取当前阶段的任务"""
        if (self.current_phase_index < len(self.task_execution_phases)):
            return self.task_execution_phases[self.current_phase_index]
        return None

    def advance_to_next_phase(self) -> bool:
        """ 推进到下一个执行阶段"""
        if self.current_phase_index < len(self.task_execution_phases) - 1:
            # 标记当前阶段的任务为完成
            current_phase = self.task_execution_phases[self.current_phase_index]
            for task in current_phase['tasks']:
                self.completed_tasks.append(task['task_id'])
            
            self.current_phase_index += 1
            print(f"DEBUG: Advanced to phase {self.current_phase_index + 1}/{len(self.task_execution_phases)}")
            return True
        return False

    def is_current_phase_complete(self, agent_statuses: Dict[int, str]) -> bool:
        """ 检查当前阶段是否完成"""
        current_phase = self.get_current_phase_tasks()
        if not current_phase:
            return True
        
        # 增强的完成检查逻辑
        try:
            if not agent_statuses:
                print(f"[DEBUG] Phase completion check: No agent statuses available")
                return False
            
            print(f"[DEBUG] Phase completion check for phase {current_phase['phase_id']}:")
            for agent_id, status in agent_statuses.items():
                print(f"  Agent {agent_id}: '{status}'")
            
            #  更全面的完成关键词检测
            completion_keywords = [
                'completed', 'done', 'finished', 'success', 'successful', 'achievement',
                'reached', 'arrived', 'found', 'picked', 'placed', 'opened', 'closed',
                'execution!', 'goal', 'target', 'complete', 'accomplish'
            ]
            
            failure_keywords = [
                'failed', 'error', 'stuck', 'impossible', 'cannot', 'unable', 'fail',
                'blocked', 'obstacle', 'collision', 'timeout', 'unreachable'
            ]
            
            completed_agents = 0
            failed_agents = 0
            total_relevant_agents = 0
            
            # 统计各种状态的智能体
            for agent_id, status in agent_statuses.items():
                if status and isinstance(status, str) and status.strip():
                    total_relevant_agents += 1
                    status_lower = status.lower()
                    
                    # 检查完成状态
                    has_completion = any(keyword in status_lower for keyword in completion_keywords)
                    has_failure = any(keyword in status_lower for keyword in failure_keywords)
                    
                    if has_completion:
                        completed_agents += 1
                        print(f"    Agent {agent_id}: COMPLETED (detected completion keywords)")
                    elif has_failure:
                        failed_agents += 1
                        print(f"    Agent {agent_id}: FAILED (detected failure keywords)")
                    else:
                        print(f"    Agent {agent_id}: IN_PROGRESS (no clear completion/failure signal)")
            
            #  阶段完成条件
            current_phase_tasks = current_phase.get('tasks', [])
            required_agents = current_phase.get('required_agents', 1)
            
            print(f"[DEBUG] Phase analysis:")
            print(f"  Required agents: {required_agents}")
            print(f"  Completed agents: {completed_agents}")
            print(f"  Failed agents: {failed_agents}")
            print(f"  Total relevant agents: {total_relevant_agents}")
            print(f"  Phase tasks: {len(current_phase_tasks)}")
            
            # 阶段完成条件（更宽松但安全）：
            # 1. 至少有一个智能体完成任务
            # 2. 失败的智能体数量不超过总数的一半
            # 3. 对于单任务阶段，一个智能体完成即可
            is_single_task_phase = len(current_phase_tasks) == 1
            
            if is_single_task_phase:
                # 单任务阶段：一个智能体完成即可
                phase_complete = completed_agents >= 1 and failed_agents == 0
                completion_reason = "single task phase with at least one completion"
            else:
                # 多任务阶段：需要足够的智能体完成
                min_required_completions = min(required_agents, len(current_phase_tasks))
                phase_complete = (completed_agents >= min_required_completions and 
                                failed_agents < total_relevant_agents // 2)
                completion_reason = f"multi-task phase with {completed_agents}/{min_required_completions} required completions"
            
            if phase_complete:
                print(f"[SUCCESS] Phase {current_phase['phase_id']} marked as COMPLETE!")
                print(f"  Reason: {completion_reason}")
                print(f"  Statistics: {completed_agents} completed, {failed_agents} failed")
            else:
                print(f"[DEBUG] Phase {current_phase['phase_id']} still IN_PROGRESS")
                print(f"  Reason: {completion_reason}")
            
            return phase_complete
            
        except Exception as e:
            print(f"[ERROR] Phase completion check failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def build_phase_specific_t_matrix(
        self, 
        phase_info: Dict[str, Any],
        base_t_matrix: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[int], List[str]]:
        """
        为特定阶段构建T矩阵，保持13×5维度，非活跃任务行设为全零
        
        Returns:
            Tuple[phase_t_matrix, active_task_indices, active_task_types]
            - phase_t_matrix: 阶段特定的T矩阵 [13 × 5] (非活跃任务行为全零)
            - active_task_indices: 活跃任务在全局任务列表中的索引
            - active_task_types: 活跃任务类型列表
        """
        
        try:
            if base_t_matrix is None:
                base_t_matrix = self.BASE_TASK_CAPABILITY_REQUIREMENTS.copy()
            
            # 确保base_t_matrix是正确的numpy数组
            if not isinstance(base_t_matrix, np.ndarray):
                base_t_matrix = np.array(base_t_matrix)
            
            # 获取当前阶段涉及的任务类型
            phase_task_types = []
            if 'tasks' in phase_info:
                phase_task_types = [task.get('task_type', 'Wait') for task in phase_info['tasks']]
            
            # 去重但保持顺序
            unique_phase_task_types = []
            seen = set()
            for task_type in phase_task_types:
                if task_type not in seen:
                    unique_phase_task_types.append(task_type)
                    seen.add(task_type)
            
            # 任务类型到矩阵行的映射
            task_type_to_index = {
                'Navigate': 0, 'Explore': 1, 'Pick': 2, 'Place': 3,
                'Open': 4, 'Close': 5, 'Clean': 6, 'Fill': 7,
                'Pour': 8, 'PowerOn': 9, 'PowerOff': 10, 'Rearrange': 11, 'Wait': 12
            }
            
            # 保持13×5维度，非活跃任务行设为全零
            phase_t_matrix = np.zeros_like(base_t_matrix)  # 初始化为全零矩阵
            active_task_indices = []
            
            for task_type in unique_phase_task_types:
                if task_type in task_type_to_index:
                    task_idx = task_type_to_index[task_type]
                    if task_idx < base_t_matrix.shape[0]:
                        active_task_indices.append(task_idx)
                        
                        # 获取基础需求行
                        base_row = base_t_matrix[task_idx, :].copy()
                        
                        # 根据任务优先级调整需求
                        for task in phase_info.get('tasks', []):
                            if task.get('task_type') == task_type:
                                priority = task.get('priority', 3)
                                priority_multiplier = 1.0 + (priority - 3) * 0.1
                                priority_multiplier = np.clip(priority_multiplier, 0.5, 2.0)
                                base_row *= priority_multiplier
                                break  # 只处理第一个匹配的任务
                        
                        # 将调整后的行赋值到对应位置
                        phase_t_matrix[task_idx, :] = base_row
            
            # 如果没有有效的活跃任务，默认激活Wait任务
            if not active_task_indices:
                wait_idx = task_type_to_index['Wait']
                phase_t_matrix[wait_idx, :] = base_t_matrix[wait_idx, :]
                active_task_indices = [wait_idx]
                unique_phase_task_types = ['Wait']
            
            print(f"DEBUG: Built phase-specific T matrix for phase {phase_info.get('phase_id', 'unknown')}")
            print(f"  Matrix shape: {phase_t_matrix.shape} (maintained 13×5 dimensions)")
            print(f"  Active task count: {len(active_task_indices)} out of 13 total tasks")
            print(f"  Active task types: {unique_phase_task_types}")
            print(f"  Active task indices: {active_task_indices}")
            print(f"  Non-zero rows: {np.sum(np.any(phase_t_matrix != 0, axis=1))} (rest are zero-disabled)")
            
            return phase_t_matrix, active_task_indices, unique_phase_task_types
            
        except Exception as e:
            print(f"[ERROR] Failed to build phase-specific T matrix: {e}")
            # 返回基础矩阵作为fallback，但只激活Wait任务
            fallback_matrix = np.zeros_like(self.BASE_TASK_CAPABILITY_REQUIREMENTS)
            fallback_matrix[12, :] = self.BASE_TASK_CAPABILITY_REQUIREMENTS[12, :]  # 只激活Wait
            return fallback_matrix, [12], ['Wait']

    def structured_decompose_task(
        self,
        instruction: str,
        env_interface: "EnvironmentInterface",
        llm_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        使用 LLM 将给定的指令分解为结构化的子任务列表。
        """
        if not self.llm_client:
            raise ValueError("LLM client not initialized. Cannot decompose task.")

        # 1. 准备对象和智能体状态信息
        objects_prompt_string = self._get_objects_string_for_prompt(env_interface)
        current_world_state = self.last_world_state or self.extract_world_state(env_interface)
        agent_info_string = self._get_agent_status_for_prompt(current_world_state)

        # 2. 构建结构化prompt
        world_state_desc = f"Environment State:\n{objects_prompt_string}\n\nAgent Status:\n{agent_info_string}"
        prompt_template = TaskDecompositionPrompt("task_decomposition", get_llm_config())
        structured_prompt = prompt_template(instruction, world_state_desc)

        # 3. 调用 LLM
        messages = [
            {"role": "system", "content": "You are a precise task decomposition assistant. Return only valid JSON."},
            {"role": "user", "content": structured_prompt}
        ]
        api_params = {
            "model": llm_config.get("gpt_version", "moonshot-v1-32k"),
            "messages": messages,
            "max_tokens": llm_config.get("max_tokens", 1500),
            "temperature": 0.1
        }

        try:
            response = self.llm_client.chat.completions.create(**api_params)
            decomposed_text = response.choices[0].message.content.strip()
            
            structured_tasks = self._parse_decomposition_response(decomposed_text)
            
            if structured_tasks:
                print(f"DEBUG: Successfully parsed {len(structured_tasks)} structured subtasks")
                return structured_tasks
            else:
                print("Warning: Failed to parse structured subtasks, using simple decomposition fallback.")
                return self._simple_task_decomposition(instruction)
                
        except Exception as e:
            print(f"Error calling LLM for structured task decomposition: {e}")
            raise

    def update_miqp_matrices(
        self,
        scenario_config: Union[Dict[str, Any], "ScenarioConfigTask"],
        structured_subtasks: List[Dict[str, Any]],
        world_state: Dict[str, Any]
    ) -> None:
        """
        根据任务分析更新MIQP的参数矩阵(A, T, ws)。
        此方法现在调用MatrixUpdater来处理所有矩阵更新逻辑。
        """
        updated_matrices = self.matrix_updater.update_matrices(
            structured_subtasks,
            world_state
        )
        
        if updated_matrices:
            update_param_value(scenario_config, 'T', updated_matrices['T'])
            update_param_value(scenario_config, 'A', updated_matrices['A'])
            update_param_value(scenario_config, 'ws', updated_matrices['ws'])
            print("DEBUG: MIQP matrices updated successfully via MatrixUpdater.")

    def pre_update_scenario_params(
        self,
        scenario_config: Union[Dict[str, Any], "ScenarioConfigTask"],
        world_state: Dict[str, Any]
    ) -> None:
        """
        根据世界状态更新场景参数（例如，是否有智能体持有物体）。
        此操作在任务分解和规划之前进行。

        Args:
            scenario_config: ScenarioConfigTask 实例或参数字典。
            world_state: 从 extract_world_state 获取的世界状态字典。
        """
        agent_names = list(world_state.get('agent_poses', {}).keys())
        
        for obj_name, obj_info in world_state.get('object_positions', {}).items():
            if obj_info and obj_info.get('parent') in agent_names:
                holding_robot_id = obj_info['parent']
                update_param_value(scenario_config, 'is_holding', True)
                update_param_value(scenario_config, 'holding_robot_id', holding_robot_id)
                print(f"DEBUG: Pre-update: Agent '{holding_robot_id}' is holding '{obj_name}'.")
                return

            update_param_value(scenario_config, 'is_holding', False)
            update_param_value(scenario_config, 'holding_robot_id', None)

    def update_scenario_from_actions(
        self,
        scenario_config: Union[Dict[str, Any], "ScenarioConfigTask"],
        world_state: Dict[str, Any],
        high_level_actions: Dict[int, Tuple[str, str, Optional[str]]]
    ) -> None:
        """
        基于规划后生成的高级动作，更新场景参数以供执行。

        Args:
            scenario_config: ScenarioConfigTask实例或参数字典
            world_state: 世界状态字典
            high_level_actions: LLM解析的高级动作字典 {agent_id: (tool_name, args_str, target_name)}
        """
        print(f"DEBUG: Updating scenario params from {len(high_level_actions)} high-level actions.")
            
        action_updater = ActionUpdater()
        all_updates = action_updater.process_and_get_updates(high_level_actions, world_state)
            
        for param_name, value in all_updates.items():
            update_param_value(scenario_config, param_name, value)
                
        print(f"DEBUG: Applied {len(all_updates)} parameter updates for execution.")

     # --- LLM 任务分解与分析 ---

    def _get_objects_string_for_prompt(self, env_interface: "EnvironmentInterface") -> str:
        """从WorldGraph格式化对象列表以用于LLM prompt。"""
        full_graph = env_interface.full_world_graph
        objects_for_prompt = []

        for obj_node in full_graph.get_all_objects():
            try:
                category = obj_node.category
            except AttributeError:
                category = "Unknown"
            objects_for_prompt.append({'name': obj_node.name, 'category': category or "Unknown"})
        
        for furn_node in full_graph.get_all_furnitures():
            try:
                category = furn_node.category
            except AttributeError:
                category = "Furniture"
            objects_for_prompt.append({'name': furn_node.name, 'category': category or "Furniture"})
        
        return f"objects = {json.dumps(objects_for_prompt)}"

    def _get_agent_status_for_prompt(self, world_state: Dict[str, Any]) -> str:
        """从世界状态格式化智能体状态以用于LLM prompt。"""
        if not world_state:
            return "No agent status available"
            
        agent_status_lines = []
        agent_poses = world_state.get('agent_poses', {})
        held_objects_by_agent = {agent: [] for agent in agent_poses}
        
        for obj_name, obj_info in world_state.get('object_positions', {}).items():
            if obj_info and obj_info.get('parent') in agent_poses:
                held_objects_by_agent[obj_info['parent']].append(obj_name)
        
        for agent_name, pose_info in agent_poses.items():
            pos_str = "Position unknown"
            if pose_info and 'position' in pose_info:
                pos = pose_info['position']
                pos_str = f"Position [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
            
            held_str = ", holding: " + ", ".join(held_objects_by_agent[agent_name]) if held_objects_by_agent.get(agent_name) else ", hands free"
            agent_status_lines.append(f"- {agent_name}: {pos_str}{held_str}")
        
        return "\n".join(agent_status_lines) if agent_status_lines else "No agents found."

    def _parse_decomposition_response(self, response_text: str) -> List[Dict[str, Any]]:
        """健壮地解析LLM返回的结构化任务分解JSON。"""
        try:
            # 尝试从文本中提取并解析JSON数组
            json_text = extract_json_from_text(response_text, list)
            if not json_text:
                print(f"Warning: No valid JSON array found in LLM response.")
                return []

            structured_tasks = json.loads(json_text)
            validated_tasks = []
            required_fields = ['task_type', 'target', 'description']
            
            for i, task in enumerate(structured_tasks):
                if not isinstance(task, dict) or not all(field in task for field in required_fields):
                    print(f"Warning: Task {i} is malformed: {task}")
                    continue
                
                # 填充默认值
                validated_task = {
                    'task_type': task['task_type'],
                    'target': task['target'],
                    'description': task['description'],
                    'priority': task.get('priority', 3),
                    'estimated_duration': task.get('estimated_duration', 5.0),
                    'preferred_agent': task.get('preferred_agent'),
                    'prerequisites': task.get('prerequisites', [])
                }
                validated_tasks.append(validated_task)
            
            return validated_tasks
            
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: JSON parsing for decomposition failed: {e}")
            return []

    def _simple_task_decomposition(self, instruction: str) -> List[Dict[str, Any]]:
        """当LLM分解失败时使用的基于规则的简单任务分解。"""
        # 基于关键词的简单分解策略
        tasks = []
        if "pick" in instruction.lower() and "place" in instruction.lower():
            tasks.extend([
                {'task_type': 'Explore', 'target': 'environment', 'description': 'Explore to find targets', 'priority': 2},
                {'task_type': 'Navigate', 'target': 'object_location', 'description': 'Navigate to object', 'priority': 3},
                {'task_type': 'Pick', 'target': 'target_object', 'description': 'Pick up object', 'priority': 4},
                {'task_type': 'Navigate', 'target': 'receptacle_location', 'description': 'Navigate to receptacle', 'priority': 3},
                {'task_type': 'Place', 'target': 'target_receptacle', 'description': 'Place object', 'priority': 4},
            ])
        else:
            tasks.append({'task_type': 'Explore', 'target': 'environment', 'description': instruction, 'priority': 3})

        # 为所有简单任务添加默认值
        for task in tasks:
            task.setdefault('estimated_duration', 10.0)
            task.setdefault('preferred_agent', None)
            task.setdefault('prerequisites', [])
        return tasks
    
    def _initialize_base_T_matrix(self) -> np.ndarray:
        """返回基础的任务-能力需求矩阵T"""
        return self.BASE_TASK_CAPABILITY_REQUIREMENTS.copy()
