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
from habitat_llm.planner.miqp_prompts import MIQPAnalysisPrompt, TaskDecompositionPrompt, get_miqp_prompt
import openai

if TYPE_CHECKING:
    from habitat_llm.planner.centralized_llm_planner import ScenarioConfigTask
    from habitat_llm.environment_interface import EnvironmentInterface


def quaternion_to_yaw(quaternion: List[float]) -> float:
    """将四元数转换为 Z 轴旋转（偏航角 Yaw）。"""
    quat = np.array(quaternion)
    x, y, z, w = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)

def safe_extract_position(position_data, dimensions=3):
    if position_data is None:
        return [0.0] * dimensions
        
    if hasattr(position_data, '__getitem__') and hasattr(position_data, '__len__'):
        try:
            coords = []
            for i in range(min(dimensions, len(position_data))):
                coords.append(float(position_data[i]))
            while len(coords) < dimensions:
                coords.append(0.0)
            return coords
        except (IndexError, TypeError, ValueError):
            return [0.0] * dimensions
    else:
        return [0.0] * dimensions

class PerceptionConnector:
    """
    连接感知信息（来自 WorldGraph/环境）和任务规划参数（ScenarioConfigTask）。
    
    主要功能:
    1. 任务分解: 使用LLM将指令分解为结构化子任务
    2. 世界状态提取: 从环境获取智能体、物体、家具位置信息
    3. MIQP矩阵更新: 基于Agent配置的能力维度更新优化参数
    4. 场景参数更新: 根据高级动作更新任务目标和约束
    5. 任务分配: 基于MIQP优化结果将子任务分配给智能体
    6. **NEW** 任务序列化: 将子任务组织为有依赖关系的执行阶段
    
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
        [1, 0, 0, 0, 0],  # Wait
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
        api_key_filename = "api_key"

        # **NEW** 添加任务序列管理
        self.task_execution_phases: List[Dict[str, Any]] = []
        self.current_phase_index: int = 0
        self.task_dependency_graph: Dict[str, List[str]] = {}
        self.completed_tasks: List[str] = []
        self.active_tasks: List[str] = []
        
        # **NEW** 阶段性T矩阵缓存
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
        **NEW** 使用LLM将指令分解为带序列依赖的结构化子任务，并组织为执行阶段。
        
        Args:
            instruction: 原始指令
            env_interface: 环境接口
            llm_config: LLM配置
            max_agents: 最大智能体数量
            
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
        
        print(f"DEBUG: **ENHANCED** Initial LLM decomposition: {len(structured_subtasks)} tasks")
        
        # 2. **NEW** 语义增强依赖关系构建
        enhanced_subtasks = self._enhance_dependency_relationships(structured_subtasks)
        
        print(f"DEBUG: **SEMANTIC** Enhanced dependency relationships:")
        for task in enhanced_subtasks:
            task_id = task['task_id']
            task_type = task['task_type']
            prerequisites = task.get('prerequisites', [])
            is_carrying = task.get('is_carrying_navigation', False)
            carrying_marker = " [CARRYING]" if is_carrying else ""
            print(f"  {task_id} ({task_type}){carrying_marker}: depends on {prerequisites}")
        
        # 3. 分析任务依赖关系
        self.task_dependency_graph = self._build_dependency_graph(enhanced_subtasks)
        
        # 4. 将任务组织为执行阶段
        execution_phases = self._organize_tasks_into_phases(
            enhanced_subtasks, max_agents
        )
        
        # 4. 缓存阶段信息
        self.task_execution_phases = execution_phases
        self.current_phase_index = 0
        
        print(f"DEBUG: **ENHANCED** Task decomposed into {len(enhanced_subtasks)} subtasks across {len(execution_phases)} phases")
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

        # **ENHANCED** 使用增强的语义分解prompt
        sequencing_prompt = self._enhance_llm_decomposition_prompt(
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
                "max_tokens": llm_config.get("max_tokens", 2000),
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
            json_text = self._extract_json_from_text(response_text, list)
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
                
                # 验证并清理前置条件
                cleaned_prerequisites = []
                for prereq in prerequisites:
                    # 防止自引用
                    if prereq == task_id:
                        print(f"WARNING: Removed self-reference in task {task_id}")
                        continue
                    # 确保前置条件格式正确
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
        """**NEW** 构建任务依赖关系图并检测循环依赖"""
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
                # 找到循环
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
                    pass  # 继续查找其他循环
            
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
        """**NEW** 将任务组织为可并行执行的阶段"""
        
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
        """**NEW** 对任务进行拓扑排序，自动处理循环依赖"""
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
        """**NEW** 获取当前阶段的任务"""
        if (self.current_phase_index < len(self.task_execution_phases)):
            return self.task_execution_phases[self.current_phase_index]
        return None

    def advance_to_next_phase(self) -> bool:
        """**NEW** 推进到下一个执行阶段"""
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
        """**ENHANCED** 检查当前阶段是否完成"""
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
            
            # **ENHANCED** 更全面的完成关键词检测
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
            
            # **ENHANCED** 阶段完成条件
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
            
            # **NEW APPROACH**: 保持13×5维度，非活跃任务行设为全零
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
        prompt_template = TaskDecompositionPrompt("task_decomposition", self._get_llm_config())
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
        此方法优先尝试通过LLM直接生成矩阵。如果失败，则回退到基于规则的增强方法。
        """
        try:
            print("DEBUG: Attempting to generate MIQP matrices directly from LLM response.")
            T, A, ws = self._generate_matrices_from_llm(structured_subtasks)
            A = self.BASE_ROBOT_CAPABILITIES

            if T is not None and A is not None and ws is not None:
                self._update_param_value(scenario_config, 'T', T)
                self._update_param_value(scenario_config, 'A', A)
                self._update_param_value(scenario_config, 'ws', ws)
            #     print(f"DEBUG: Updated T matrix:\n{T}")
            #     print(f"DEBUG: Updated A matrix:\n{A}")
            #     print(f"DEBUG: Updated ws matrix:\n{ws}")
                print("DEBUG: Successfully generated and updated MIQP matrices from LLM.")
            #     return

            print("Warning: Failed to generate matrices directly from LLM, falling back to rule-based analysis.")
            llm_analysis = self._llm_analyze_task_constraints(structured_subtasks, world_state)
            
            updated_T = self._update_task_capability_matrix_enhanced(structured_subtasks, llm_analysis)
            self._update_param_value(scenario_config, 'T', updated_T)
            
            updated_A = self._update_robot_capability_matrix_enhanced(structured_subtasks, llm_analysis)
            self._update_param_value(scenario_config, 'A', updated_A)
            
            updated_ws = self._update_capability_weights_enhanced(structured_subtasks, llm_analysis)
            self._update_param_value(scenario_config, 'ws', updated_ws)
            
            # print(f"DEBUG: Updated T matrix:\n{updated_T}")
            # print(f"DEBUG: Updated A matrix:\n{updated_A}")
            # print(f"DEBUG: Updated ws matrix:\n{updated_ws}")
            # print("DEBUG: MIQP matrices updated successfully with LLM-enhanced analysis.")
            
        except Exception as e:
            print(f"ERROR: Matrix generation failed entirely: {e}. Using simple fallback.")
            self._update_matrices_fallback(scenario_config, structured_subtasks, world_state)

    def map_subtasks_to_agents(
        self,
        subtasks: List[Dict[str, Any]],
        alpha_matrix: np.ndarray,
        agent_capabilities: Dict[int, List[str]] = None
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        基于MIQP的alpha分配矩阵将子任务分配给智能体。
        
        Args:
            subtasks: 结构化的子任务列表
            alpha_matrix: MIQP求解得到的任务分配矩阵 [n_agents × n_tasks]
            agent_capabilities: 每个智能体的能力限制（可选）
            
        Returns:
            每个智能体分配到的子任务字典: {agent_id: [subtask1, subtask2, ...]}
        """
        try:
            # 输入验证
            if alpha_matrix is None:
                print("[WARNING] Alpha matrix is None, using simple assignment")
                return self._simple_task_assignment(subtasks)
            
            if not isinstance(alpha_matrix, np.ndarray):
                print("[WARNING] Alpha matrix is not numpy array, converting")
                alpha_matrix = np.array(alpha_matrix)
            
            if alpha_matrix.ndim != 2:
                print(f"[WARNING] Alpha matrix has unexpected dimensions {alpha_matrix.ndim}, using simple assignment")
                return self._simple_task_assignment(subtasks)
            
            n_agents, n_tasks_matrix = alpha_matrix.shape
            
            if not subtasks:
                return {i: [] for i in range(n_agents)}

            # 确保子任务列表与矩阵维度对齐
            aligned_subtasks = self._align_subtasks_with_matrix(subtasks, n_tasks_matrix)
            
            agent_assignments = {i: [] for i in range(n_agents)}
            
            for task_idx, subtask in enumerate(aligned_subtasks):
                if task_idx >= n_tasks_matrix:
                    print(f"[WARNING] Task index {task_idx} exceeds matrix size {n_tasks_matrix}")
                    break
                    
                # 找到最适合执行此任务的智能体
                try:
                    task_weights = alpha_matrix[:, task_idx]
                    best_agent_idx = np.argmax(task_weights)
                    assignment_weight = task_weights[best_agent_idx]
                    
                    # 应用分配阈值和能力检查
                    if self._should_assign_task(subtask, best_agent_idx, assignment_weight, agent_capabilities):
                        assigned_subtask = self._create_assigned_subtask(subtask, best_agent_idx, assignment_weight)
                        agent_assignments[best_agent_idx].append(assigned_subtask)
                        print(f"DEBUG: Assigned '{subtask['task_type']}' to Agent {best_agent_idx} (weight: {assignment_weight:.3f})")
                except Exception as e:
                    print(f"[ERROR] Task assignment failed for task {task_idx}: {e}")
                    continue
            
            return agent_assignments
            
        except Exception as e:
            print(f"[ERROR] Map subtasks to agents failed: {e}, using simple assignment")
            return self._simple_task_assignment(subtasks)
    
    def _simple_task_assignment(self, subtasks: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """简单的任务分配fallback方法"""
        assignments = {0: [], 1: []}
        for i, task in enumerate(subtasks):
            agent_id = i % 2  # 简单轮询分配
            assignments[agent_id].append(task)
        return assignments

    def _align_subtasks_with_matrix(self, subtasks: List[Dict[str, Any]], n_tasks_matrix: int) -> List[Dict[str, Any]]:
        """确保子任务列表与MIQP矩阵维度对齐"""
        if len(subtasks) <= n_tasks_matrix:
            # 如果子任务数量不足，用等待任务填充
            aligned_subtasks = subtasks.copy()
            while len(aligned_subtasks) < n_tasks_matrix:
                aligned_subtasks.append({
                    'task_type': 'Wait',
                    'target': '',
                    'description': 'Wait for other agents',
                    'priority': 1,
                    'estimated_duration': 1.0
                })
            return aligned_subtasks
        else:
            # 如果子任务过多，截断到矩阵大小
            return subtasks[:n_tasks_matrix]

    def _should_assign_task(
        self, 
        subtask: Dict[str, Any], 
        agent_idx: int, 
        assignment_weight: float,
        agent_capabilities: Dict[int, List[str]] = None
    ) -> bool:
        """判断是否应该将任务分配给指定智能体"""
        # 检查分配权重阈值
        if assignment_weight < 0.1:  # 权重太低
            return False
        
        # 检查智能体能力
        if agent_capabilities:
            agent_caps = agent_capabilities.get(agent_idx, [])
            task_type = subtask.get('task_type', '')
            if task_type and task_type not in agent_caps:
                return False
        
        return True

    def _create_assigned_subtask(
        self, 
        subtask: Dict[str, Any], 
        agent_idx: int, 
        assignment_weight: float
    ) -> Dict[str, Any]:
        """创建分配给智能体的子任务"""
        assigned_subtask = subtask.copy()
        assigned_subtask.update({
            'assigned_agent': agent_idx,
            'assignment_weight': assignment_weight,
            'assignment_confidence': 'High' if assignment_weight > 0.7 else 'Medium' if assignment_weight > 0.3 else 'Low',
            'assignment_method': 'MIQP'
        })
        return assigned_subtask

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
                self._update_param_value(scenario_config, 'is_holding', True)
                self._update_param_value(scenario_config, 'holding_robot_id', holding_robot_id)
                print(f"DEBUG: Pre-update: Agent '{holding_robot_id}' is holding '{obj_name}'.")
                return

            self._update_param_value(scenario_config, 'is_holding', False)
            self._update_param_value(scenario_config, 'holding_robot_id', None)

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
            
        all_updates = {}
        motor_skill_updates = self._process_motor_skill_actions(high_level_actions, world_state)
        state_manipulation_updates = self._process_state_manipulation_actions(high_level_actions, world_state)
        all_updates.update(motor_skill_updates)
        all_updates.update(state_manipulation_updates)
            
        for param_name, value in all_updates.items():
            self._update_param_value(scenario_config, param_name, value)
                
        print(f"DEBUG: Applied {len(all_updates)} parameter updates for execution.")

    def ft_update_scenario_params(
        self,
        scenario_config: Union[Dict[str, Any], "ScenarioConfigTask"],
        world_state: Dict[str, Any],
        high_level_actions: Dict[int, Tuple[str, str, Optional[str]]]
    ) -> None:
        """
        保持向后兼容性，调用update_scenario_from_actions。
        """
        self.update_scenario_from_actions(scenario_config, world_state, high_level_actions)

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
            json_text = self._extract_json_from_text(response_text, list)
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
    
    # --- MIQP矩阵更新 ---
    def _update_matrices_fallback(
        self, 
        scenario_config: Union[Dict[str, Any], "ScenarioConfigTask"],
        structured_subtasks: List[Dict[str, Any]],
        world_state: Dict[str, Any]
    ) -> None:
        """简化的fallback矩阵更新方法"""
        print(f"DEBUG: Using fallback method to update MIQP matrices.")
        updated_T = self._update_task_capability_matrix(world_state, structured_subtasks, scenario_config)
        self._update_param_value(scenario_config, 'T', updated_T)
        print(f"DEBUG: Updated T matrix:\n{updated_T}")
        
        updated_A = self._update_robot_capability_matrix(world_state, structured_subtasks, scenario_config)
        self._update_param_value(scenario_config, 'A', updated_A)
        print(f"DEBUG: Updated A matrix:\n{updated_A}")
        
        updated_ws = self._update_capability_weights(world_state, structured_subtasks, scenario_config)
        self._update_param_value(scenario_config, 'ws', updated_ws)
        print(f"DEBUG: Updated weights matrix:\n{updated_ws}")
        
        print(f"DEBUG: Fallback matrix update completed.")

    def _generate_matrices_from_llm(
        self, 
        structured_subtasks: List[Dict[str, Any]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[np.ndarray]]]:
        """
        通过LLM直接生成MIQP矩阵。
        
        Args:
            structured_subtasks: 结构化的子任务列表。
            
        Returns:
            一个包含T, A, ws矩阵的元组，如果失败则返回(None, None, None)。
        """
        if not self.llm_client:
            return None, None, None

        capability_names = ["Movement", "Object Manipulation", "Basic Control", "Liquid Handling", "Power Control"]
        task_type_names = [
            'Navigate', 'Explore', 'Pick', 'Place', 'Open', 'Close', 'Clean', 'Fill', 
            'Pour', 'PowerOn', 'PowerOff', 'Rearrange', 'Wait'
        ]
        agent_descriptions = {
            "Agent 0 (Standard)": "Can perform basic movement, object manipulation (pick, place, rearrange), and basic control (open, close). Cannot handle liquids or power.",
            "Agent 1 (Advanced)": "Can perform all tasks Agent 0 can, and is also equipped to handle liquids (clean, fill, pour) and power (power on/off)."
        }

        prompt = f"""
You are an expert AI assistant for a robotic task planner. Your goal is to generate optimal parameters for a Mixed-Integer Quadratic Program (MIQP) solver that assigns tasks to two robots.

Based on the mission goal and decomposed subtasks, you must generate three matrices in a single JSON object:
1.  **Task-Capability Matrix (T)**: How much each task requires each capability.
2.  **Agent-Capability Matrix (A)**: How proficient each agent is at each capability.
3.  **Capability Weights (ws)**: How important each capability is for the overall mission.

**Mission Context:**
- **Subtasks**: {json.dumps(structured_subtasks, indent=2)}
- **Agent Descriptions**: {json.dumps(agent_descriptions, indent=2)}
- **Capabilities**: {json.dumps(capability_names, indent=2)}
- **Task Types**: {json.dumps(task_type_names, indent=2)}

**Instructions for Generating JSON:**

1.  **`task_capability_matrix` (T)**:
    - A list of lists with shape (13 tasks x 5 capabilities).
    - Each value should be between 0.0 (not required) and 1.0 (highly required).
    - Rows correspond to Task Types, columns correspond to Capabilities in the order provided above.
    - A task should only have non-zero values for capabilities it actually uses. For example, 'Navigate' should only require 'Movement'. 'Pick' should only require 'Object Manipulation'.

2.  **`agent_capability_matrix` (A)**:
    - A list of lists with shape (5 capabilities x 2 agents).
    - Each value represents proficiency, from 0.0 (incapable) to 1.0 (highly proficient).
    - Rows correspond to Capabilities, columns correspond to Agents (Agent 0, Agent 1).
    - Values should reflect the agent descriptions. If an agent cannot perform a task, its score for that capability must be 0.0. A score of 1.0 is considered baseline proficiency.

3.  **`capability_weights` (ws)**:
    - A list of 5 numbers, one for each capability.
    - Each value represents the importance of that capability for this specific mission, from 1.0 (less important) to 5.0 (critically important).
    - Consider which capabilities are central to completing the given subtasks.

4.  **`reasoning`**:
    - A brief text string explaining your choices for the weights and any non-obvious matrix values.

**Output Format (JSON only):**
```json
{{
  "reasoning": "Brief explanation here.",
  "task_capability_matrix": [],
  "agent_capability_matrix": [],
  "capability_weights": []
}}
```
"""
        try:
            api_params = {
                "model": "moonshot-v1-8k",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.1,
            }
            try:
                # 优先使用JSON模式
                api_params["response_format"] = {"type": "json_object"}
                response = self.llm_client.chat.completions.create(**api_params)
            except Exception:
                # 如果不支持，则回退到普通模式
                print("Warning: `response_format` not supported, retrying without it.")
                del api_params["response_format"]
                response = self.llm_client.chat.completions.create(**api_params)

            response_text = response.choices[0].message.content
            return self._parse_llm_matrix_response(response_text)
        except Exception as e:
            print(f"Error calling LLM for matrix generation: {e}")
            return None, None, None

    def _parse_llm_matrix_response(self, response_text: str):
        """解析并验证从LLM返回的包含MIQP矩阵的JSON。"""
        try:
            data = json.loads(response_text)

            T_matrix = data.get("task_capability_matrix")
            if not isinstance(T_matrix, list) or len(T_matrix) != 13 or not all(isinstance(r, list) and len(r) == 5 for r in T_matrix):
                print(f"Validation Error: T matrix is malformed. Shape: {len(T_matrix)}x{len(T_matrix[0]) if T_matrix and T_matrix[0] else 'N/A'}")
                return None, None, None
            T = np.array(T_matrix, dtype=float)

            A_matrix = data.get("agent_capability_matrix")
            if not isinstance(A_matrix, list) or len(A_matrix) != 5 or not all(isinstance(r, list) and len(r) == 2 for r in A_matrix):
                print("Validation Error: A matrix is malformed.")
                return None, None, None
            A = np.array(A_matrix, dtype=float)

            ws_weights = data.get("capability_weights")
            if not isinstance(ws_weights, list) or len(ws_weights) != 5:
                print("Validation Error: ws weights are malformed.")
                return None, None, None
            ws = [w * np.eye(1) for w in ws_weights]

            print(f"DEBUG: LLM Reasoning: {data.get('reasoning', 'N/A')}")
            return T, A, ws
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"Error parsing LLM matrix response: {e}\nResponse was:\n{response_text[:500]}...")
            return None, None, None

    def _llm_analyze_task_constraints(
        self, 
        structured_subtasks: List[Dict[str, Any]], 
        world_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用LLM分析任务约束，为矩阵生成提供指导。"""
        if not self.llm_client or not structured_subtasks:
            return self._get_default_analysis()
        
        try:
            prompt_template = MIQPAnalysisPrompt("miqp_analysis", self._get_llm_config())
            analysis_prompt = prompt_template(structured_subtasks, world_state)
            
            response = self.llm_client.chat.completions.create(
                model="moonshot-v1-8k", 
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=12000,
                temperature=0.1
            )
            analysis_text = response.choices[0].message.content.strip()
            
            llm_analysis = self._parse_miqp_analysis_response(analysis_text)
            if llm_analysis:
                print(f"DEBUG: LLM analysis completed successfully.")
            return llm_analysis
            
        except Exception as e:
            print(f"Warning: LLM analysis failed: {e}, using default analysis.")
            return self._get_default_analysis()

    def _parse_miqp_analysis_response(self, analysis_text: str) -> Dict[str, Any]:
        """解析MIQP分析的LLM响应"""
        try:
            # 尝试提取JSON
            json_text = self._extract_json_from_text(analysis_text, dict)
            if json_text:
                analysis = json.loads(json_text)
                
                # 验证必要的字段
                required_fields = ["task_complexity", "agent_suitability", "constraints"]
                for field in required_fields:
                    if field not in analysis:
                        analysis[field] = {}
                
                return analysis
            else:
                print("Warning: No valid JSON found in MIQP analysis response")
                return self._get_default_analysis()
                
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Failed to parse MIQP analysis response: {e}")
            return self._get_default_analysis()
            
    def _update_task_capability_matrix_enhanced(
        self, 
        structured_subtasks: List[Dict[str, Any]], 
        llm_analysis: Dict[str, Any]
    ) -> np.ndarray:
        """基于LLM分析更新任务-能力需求矩阵T（增强版）。"""
        base_T = self._initialize_base_T_matrix()
        
        complexity = llm_analysis.get("task_complexity", {})
        conservative_factor = llm_analysis.get("constraints", {}).get("conservative_factors", {}).get("safety", 0.9)
        
        # 定义任务类型到T矩阵行的映射
        task_map = {'Navigate': 0, 'Explore': 1, 'Pick': 2, 'Place': 3}

        # 根据复杂度适度增强任务需求
        for task_name, task_idx in task_map.items():
            if any(subtask['task_type'] == task_name for subtask in structured_subtasks):
                task_complexity = complexity.get(task_name.lower(), 0.5)
                # 增强因子，确保不会过大
                enhancement = min(task_complexity * conservative_factor * 0.2, 0.3)
                # T矩阵中只有一个非零元素，代表所需能力
                capability_idx = np.argmax(base_T[task_idx, :])
                base_T[task_idx, capability_idx] += enhancement
        
        # 确保值在合理范围内
        base_T = np.clip(base_T, 0.0, 1.5)
        return base_T

    def _update_robot_capability_matrix_enhanced(
        self, 
        structured_subtasks: List[Dict[str, Any]],
        llm_analysis: Dict[str, Any]
    ) -> np.ndarray:
        """基于LLM分析更新机器人-能力矩阵A（增强版）。"""
        base_A = self.BASE_ROBOT_CAPABILITIES.copy()
        
        agent_suitability = llm_analysis.get("agent_suitability", {})
        reliability_factor = llm_analysis.get("constraints", {}).get("conservative_factors", {}).get("reliability", 0.9)
        
        task_types = {sub['task_type'] for sub in structured_subtasks}
        
        # 定义能力需求 -> (能力索引, LLM适用性key)
        cap_map = {
            'Navigate': (0, 'movement'), 'Explore': (0, 'movement'), 'Wait': (0, 'movement'),
            'Pick': (1, 'manipulation'), 'Place': (1, 'manipulation'), 'Rearrange': (1, 'manipulation'),
            'Open': (2, 'manipulation'), 'Close': (2, 'manipulation'),
            'Clean': (3, 'liquid'), 'Fill': (3, 'liquid'), 'Pour': (3, 'liquid'),
            'PowerOn': (4, 'power'), 'PowerOff': (4, 'power'),
        }

        needed_caps = {cap_map[t] for t in task_types if t in cap_map}

        for agent_idx in range(base_A.shape[1]):
            agent_key = f"agent_{agent_idx}"
            suitability = agent_suitability.get(agent_key, {})
            
            for cap_idx, cap_key in needed_caps:
                if base_A[cap_idx, agent_idx] > 0: # 如果智能体具备该能力
                    suitability_factor = suitability.get(cap_key, 0.9) * reliability_factor
                    # 增强能力值，确保不低于1.0
                    enhancement = base_A[cap_idx, agent_idx] * suitability_factor * 1.1
                    base_A[cap_idx, agent_idx] = max(1.0, min(1.3, enhancement))

        return base_A

    def _update_capability_weights_enhanced(
        self, 
        structured_subtasks: List[Dict[str, Any]],
        llm_analysis: Dict[str, Any]
    ) -> List[np.ndarray]:
        """基于LLM分析更新能力权重ws（增强版）。"""
        base_weights = [w.copy() for w in self.BASE_CAPABILITY_WEIGHTS]
        task_type_counts = {}
        for subtask in structured_subtasks:
            task_type = subtask['task_type']
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
        
        # 能力索引 -> 相关任务类型
        cap_to_tasks = {
            0: ['Navigate', 'Explore'],
            1: ['Pick', 'Place', 'Rearrange'],
            2: ['Open', 'Close'],
            3: ['Clean', 'Fill', 'Pour'],
        }
        
        for cap_idx, related_tasks in cap_to_tasks.items():
            if any(task in task_type_counts for task in related_tasks):
                base_weights[cap_idx] *= 1.2
        
        enhanced_weights = [np.clip(w, 1.0, 4.0) for w in base_weights]
        return enhanced_weights

    def _update_task_capability_matrix(
        self, 
        world_state: Dict[str, Any], 
        structured_subtasks: List[Dict[str, Any]], 
        scenario_config: Union[Dict[str, Any], "ScenarioConfigTask"]
    ) -> np.ndarray:
        """
        基于世界状态更新任务-能力需求矩阵T。
        """
        try:
            base_T = self._initialize_base_T_matrix()
            if not structured_subtasks:
                return base_T
                
            # 根据世界状态更新任务需求
            task_importance = {}
            for subtask in structured_subtasks:
                task_type = subtask.get('task_type')
                if task_type:
                    task_importance[task_type] = task_importance.get(task_type, 0) + subtask.get('priority', 3)
            
            task_map = {
                'Navigate': 0, 'Explore': 1, 'Pick': 2, 'Place': 3, 
                'Open': 4, 'Close': 5, 'Clean': 6, 'Fill': 7, 
                'Pour': 8, 'PowerOn': 9, 'PowerOff': 10, 'Rearrange': 11, 'Wait': 12
            }
            
            for task_type, importance in task_importance.items():
                if task_type in task_map:
                    task_idx = task_map[task_type]
                    importance_factor = min(1.3, 1.0 + (importance - 3) * 0.1)
                    base_T[task_idx, :] *= importance_factor
            
            return base_T
            
        except Exception as e:
            print(f"[ERROR] 任务能力需求矩阵更新失败: {e}")
            # 返回基础矩阵，确保能力值至少为1.0
            return self.BASE_TASK_CAPABILITY_REQUIREMENTS.copy()

    def _update_robot_capability_matrix(
        self, 
        world_state: Dict[str, Any], 
        structured_subtasks: List[Dict[str, Any]],
        scenario_config: Union[Dict[str, Any], "ScenarioConfigTask"]
    ) -> np.ndarray:
        """
        基于世界状态更新机器人能力矩阵A（简化版）。
        
        Args:
            world_state: 当前世界状态
            structured_subtasks: 结构化子任务列表
            scenario_config: ScenarioConfigTask实例或参数字典
            
        Returns:
            更新后的A矩阵
        """
        try:
            # 获取基础PARTNR能力配置
            base_A = self.BASE_ROBOT_CAPABILITIES.copy()
            
            # 简单的能力调整 - 基于任务类型，确保满足需求
            updated_A = base_A.copy()
            
            # 根据世界状态更新能力需求
            task_types = {sub.get('task_type') for sub in structured_subtasks}
            
            if any(t in ['Pick', 'Place', 'Rearrange'] for t in task_types):
                updated_A[1, :] = np.clip(base_A[1, :] * 1.1, 1.0, 1.5) # 增强操作能力
            if any(t in ['Clean', 'Fill', 'Pour'] for t in task_types):
                updated_A[3, 1] = np.clip(base_A[3, 1] * 1.05, 1.0, 1.5) # 增强Agent 1的液体处理能力
                
            return updated_A
            
        except Exception as e:
            print(f"[ERROR] 机器人能力矩阵更新失败: {e}")
            # 返回基础配置，确保能力值至少为1.0
            return self.BASE_ROBOT_CAPABILITIES.copy()

    def _update_capability_weights(
        self, 
        world_state: Dict[str, Any], 
        structured_subtasks: List[Dict[str, Any]],
        scenario_config: Union[Dict[str, Any], "ScenarioConfigTask"]
    ) -> List[np.ndarray]:
        """
        基于世界状态更新能力权重ws（简化版）。
        
        Args:
            world_state: 当前世界状态
            structured_subtasks: 结构化子任务列表
            scenario_config: ScenarioConfigTask实例或参数字典
            
        Returns:
            更新后的ws权重矩阵列表
        """
        try:
            # 基础权重配置
            base_weights = [w.copy() for w in self.BASE_CAPABILITY_WEIGHTS]
            
            # 基于世界状态调整权重
            task_counts = {}
            for subtask in structured_subtasks:
                task_type = subtask.get('task_type', '')
                task_counts[task_type] = task_counts.get(task_type, 0) + 1
            
            if task_counts.get('Navigate', 0) > 2: base_weights[0] *= 1.2
            if sum(task_counts.get(t, 0) for t in ['Pick', 'Place']) > 2: base_weights[1] *= 1.3
            if sum(task_counts.get(t, 0) for t in ['Clean', 'Fill', 'Pour']) > 0: base_weights[3] *= 1.2
            
            return base_weights
            
        except Exception as e:
            print(f"[ERROR] 能力权重更新失败: {e}")
            # 返回基础权重
            return [w.copy() for w in self.BASE_CAPABILITY_WEIGHTS]

    

    def _get_default_analysis(self) -> Dict[str, Any]:
        """获取默认的任务分析，作为LLM分析失败时的fallback"""
        return {
            "task_complexity": {"navigate": 0.5, "pick": 0.7, "place": 0.7, "explore": 0.4},
            "agent_suitability": {
                "agent_0": {"movement": 0.9, "manipulation": 0.8},
                "agent_1": {"movement": 0.9, "manipulation": 0.9, "liquid": 0.8, "power": 0.7}
            },
            "constraints": {"conservative_factors": {"safety": 0.9, "reliability": 0.9}},
            "critical_tasks": ["Pick", "Place", "Navigate"]
        }

    def _update_param_value(self, scenario_config: Union[Dict[str, Any], "ScenarioConfigTask"], key: str, value: Any) -> None:
        """
        更新参数值，同时支持字典类型和ScenarioConfigTask类型。
        """
        if isinstance(scenario_config, dict):
            scenario_config[key] = value
        else:
            # 对于ScenarioConfigTask实例，需要更新scenario_params字典
            try:
                if hasattr(scenario_config, 'scenario_params') and key in ['A', 'T', 'ws', 'Hs']:
                    # 直接更新scenario_params中的矩阵
                    scenario_config.scenario_params[key] = value
                    print(f"DEBUG: Updated scenario_params['{key}'] in ScenarioConfigTask")
                else:
                    # 尝试使用全局任务变量更新方法
                    scenario_config.update_global_task_var(key, value)
            except AttributeError:
                print(f"Error: scenario_config 既不是字典也没有适当的更新方法")

    def _get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置，模拟llm_planner中的配置格式"""
        return {
            'system_tag': '[SYSTEM]',
            'user_tag': '[USER]', 
            'assistant_tag': '[ASSISTANT]',
            'eot_tag': '[EOT]'
        }

    def _extract_json_from_text(self, text: str, target_type: type = dict) -> Optional[str]:
        """从可能包含额外文本的LLM响应中提取JSON片段。"""
        if target_type == dict:
            start_char, end_char = '{', '}'
        elif target_type == list:
            start_char, end_char = '[', ']'
        else:
            return None

        try:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end > start:
                return text[start:end+1]
        except Exception:
            pass
        return None

    def _find_target_position(self, target_name: str, world_state: Dict[str, Any]) -> Optional[List[float]]:
        """查找目标在世界状态中的位置"""
        if not target_name:
            return None
            
        # 首先在家具中查找
        furniture_pos = world_state.get('furniture_positions', {}).get(target_name)
        if furniture_pos and 'position' in furniture_pos:
            return furniture_pos['position']
        
        # 然后在物体中查找
        object_pos = world_state.get('object_positions', {}).get(target_name)
        if object_pos and 'position' in object_pos:
            return object_pos['position']
        
        return None

    def _process_motor_skill_actions(
        self, 
        high_level_actions: Dict[int, Tuple[str, str, Optional[str]]], 
        world_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理Motor Skills相关的动作更新
        基于Agent工具配置: Navigate, Pick, Place, Rearrange, Explore, Wait, Open, Close
        """
        updates = {}
        
        for agent_id, action_tuple in high_level_actions.items():
            if not action_tuple or action_tuple[2] is None:  # 跳过无效动作
                continue
                
            tool_name, args_str, target_name = action_tuple
            
            # Navigation相关 (精确匹配 Navigate)
            if tool_name == 'Navigate':
                nav_updates = self._update_navigation_params(target_name, world_state)
                updates.update(nav_updates)
                print(f"DEBUG: Agent {agent_id} using Navigate to {target_name}")
            
            # Manipulation相关 (精确匹配 Pick, Place, Rearrange)
            elif tool_name in ['Pick', 'Place', 'Rearrange']:
                manip_updates = self._update_manipulation_params(tool_name, target_name, world_state)
                updates.update(manip_updates)
                print(f"DEBUG: Agent {agent_id} using {tool_name} on {target_name}")
            
            # Exploration相关 (精确匹配 Explore)
            elif tool_name == 'Explore':
                explore_updates = self._update_exploration_params(target_name, world_state)
                updates.update(explore_updates)
                print(f"DEBUG: Agent {agent_id} using Explore in {target_name}")
            
            # 铰接控制 (精确匹配 Open, Close)
            elif tool_name in ['Open', 'Close']:
                # Open/Close 主要影响环境状态，通常不需要更新MIQP参数
                print(f"DEBUG: Agent {agent_id} using {tool_name} on {target_name}")
                
            # Wait动作 (精确匹配 Wait)
            elif tool_name == 'Wait':
                print(f"DEBUG: Agent {agent_id} waiting - no parameter updates needed")
            
            # 未识别的动作
            else:
                print(f"WARNING: Unrecognized motor skill action: {tool_name} for Agent {agent_id}")
        
        return updates

    def _update_navigation_params(self, target_name: str, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """更新导航相关参数"""
        updates = {}
        
        # 查找目标位置
        target_pos = self._find_target_position(target_name, world_state)
        if target_pos:
            # MIQP使用XZ平面坐标
            nav_goal = np.array([target_pos[0], target_pos[2]])
            updates['p_goal'] = nav_goal
            updates['theta_goal'] = 0.0  # 默认朝向
            print(f"DEBUG: Updated navigation goal to {nav_goal} for target '{target_name}'")
        else:
            print(f"WARNING: Could not find position for navigation target '{target_name}'")
        
        return updates

    def _update_manipulation_params(
        self, 
        tool_name: str, 
        target_name: str, 
        world_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """更新操作相关参数，基于精确的工具名称匹配"""
        updates = {}
        
        # Pick动作 (精确匹配)
        if tool_name == 'Pick':
            target_pos = self._find_target_position(target_name, world_state)
            if target_pos:
                obj_pos = np.array([target_pos[0], target_pos[2]])
                updates['target_object_position'] = obj_pos
                print(f"DEBUG: Updated pick target to {obj_pos} for object '{target_name}'")
        
        # Place动作 (精确匹配)
        elif tool_name == 'Place':
            target_pos = self._find_target_position(target_name, world_state)
            if target_pos:
                place_pos = np.array([target_pos[0], target_pos[2]])
                updates['target_receptacle_position'] = place_pos
                print(f"DEBUG: Updated place target to {place_pos} for receptacle '{target_name}'")
        
        # Rearrange动作 (精确匹配)
        elif tool_name == 'Rearrange':
            # Rearrange[object, spatial_relation, furniture, spatial_constraint, reference_object]
            # 主要关注目标家具位置
            target_pos = self._find_target_position(target_name, world_state)
            if target_pos:
                rearrange_pos = np.array([target_pos[0], target_pos[2]])
                updates['target_receptacle_position'] = rearrange_pos
                print(f"DEBUG: Updated rearrange target to {rearrange_pos} for '{target_name}'")
        
        return updates

    def _update_exploration_params(self, target_name: str, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """更新探索相关参数"""
        updates = {}
        
        # 基于目标生成探索点
        exploration_targets = []
        
        if target_name and target_name != 'environment':
            # 查找与目标相关的家具/区域
            for furn_name, furn_info in world_state.get('furniture_positions', {}).items():
                if (target_name.lower() in furn_name.lower() and 
                    furn_info and 'position' in furn_info):
                    
                    exploration_targets.append({
                        'position': np.array([furn_info['position'][0], furn_info['position'][2]]),
                        'explored': False,
                        'id': hash(furn_name)
                    })
        
        if exploration_targets:
            updates['exploration_targets'] = exploration_targets
            print(f"DEBUG: Generated {len(exploration_targets)} exploration targets for '{target_name}'")
        else:
            print(f"DEBUG: Using default exploration targets for '{target_name}'")
        
        return updates

    def _process_state_manipulation_actions(
        self, 
        high_level_actions: Dict[int, Tuple[str, str, Optional[str]]], 
        world_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理状态操作工具相关的动作
        包括: Clean, Fill, Pour, PowerOn, PowerOff (Agent 1专有)
        """
        updates = {}
        
        for agent_id, action_tuple in high_level_actions.items():
            if not action_tuple or action_tuple[2] is None:
                continue
                
            tool_name, args_str, target_name = action_tuple
            
            # 状态操作工具 (Agent 1 专有能力)
            if tool_name in ['Clean', 'Fill', 'Pour', 'PowerOn', 'PowerOff']:
                state_updates = self._update_state_manipulation_params(tool_name, target_name, world_state, agent_id)
                updates.update(state_updates)
                print(f"DEBUG: Agent {agent_id} using state manipulation tool '{tool_name}' on '{target_name}'")
        
        return updates

    def _update_state_manipulation_params(
        self, 
        tool_name: str, 
        target_name: str, 
        world_state: Dict[str, Any],
        agent_id: int
    ) -> Dict[str, Any]:
        """更新状态操作相关参数"""
        updates = {}
        
        # 检查Agent是否有该工具的权限 
        if agent_id == 0 and tool_name in ['Clean', 'Fill', 'Pour', 'PowerOn', 'PowerOff']:
            print(f"WARNING: Agent {agent_id} attempting to use {tool_name} but lacks this capability")
            return updates
        
        # 获取目标对象位置
        target_pos = self._find_target_position(target_name, world_state)
        
        if tool_name == 'Clean':
            if target_pos:
                clean_pos = np.array([target_pos[0], target_pos[2]])
                updates['target_object_position'] = clean_pos
                updates['operation_type'] = 'clean'
                print(f"DEBUG: Updated clean target to {clean_pos} for object '{target_name}'")
                
        elif tool_name in ['Fill', 'Pour']:
            if target_pos:
                fluid_pos = np.array([target_pos[0], target_pos[2]])
                updates['target_receptacle_position'] = fluid_pos
                updates['operation_type'] = 'fluid_manipulation'
                print(f"DEBUG: Updated {tool_name.lower()} target to {fluid_pos} for '{target_name}'")
                
        elif tool_name in ['PowerOn', 'PowerOff']:
            if target_pos:
                power_pos = np.array([target_pos[0], target_pos[2]])
                updates['target_object_position'] = power_pos
                updates['operation_type'] = 'power_control'
                updates['power_state'] = tool_name == 'PowerOn'
                print(f"DEBUG: Updated power control target to {power_pos} for '{target_name}' (state: {tool_name})")
        
        return updates

    def _initialize_base_T_matrix(self) -> np.ndarray:
        """返回基础的任务-能力需求矩阵T"""
        return self.BASE_TASK_CAPABILITY_REQUIREMENTS.copy()

    def _enhance_dependency_relationships(self, structured_subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """**NEW** 基于任务语义增强依赖关系，特别处理复合动作"""
        enhanced_tasks = [task.copy() for task in structured_subtasks]
        
        print(f"[DEBUG] **SEMANTIC** Enhancing dependency relationships for {len(enhanced_tasks)} tasks")
        
        # 1. 识别任务类型和目标
        pick_tasks = []
        navigate_tasks = []
        place_tasks = []
        
        for i, task in enumerate(enhanced_tasks):
            task_type = task.get('task_type', '')
            target = task.get('target', '')
            
            if task_type == 'Pick':
                pick_tasks.append((i, task))
            elif task_type == 'Navigate':
                navigate_tasks.append((i, task))
            elif task_type == 'Place':
                place_tasks.append((i, task))
        
        # 2. 增强Navigate任务的依赖关系（携带物品导航逻辑）
        for nav_idx, nav_task in navigate_tasks:
            nav_target = nav_task.get('target', '').lower()
            nav_description = nav_task.get('description', '').lower()
            
            # 检查是否为"携带物品导航"
            is_carrying_navigation = self._is_carrying_navigation_task(nav_task, pick_tasks, place_tasks)
            
            if is_carrying_navigation:
                # 找到所有应该完成的Pick任务
                required_pick_tasks = self._find_required_pick_tasks(nav_task, pick_tasks, enhanced_tasks)
                
                current_prereqs = set(nav_task.get('prerequisites', []))
                
                for pick_idx, pick_task in required_pick_tasks:
                    pick_task_id = pick_task['task_id']
                    if pick_task_id not in current_prereqs:
                        current_prereqs.add(pick_task_id)
                        print(f"[SEMANTIC] Added pick dependency: {nav_task['task_id']} now depends on {pick_task_id}")
                
                enhanced_tasks[nav_idx]['prerequisites'] = list(current_prereqs)
                
                # 标记为携带导航任务
                enhanced_tasks[nav_idx]['is_carrying_navigation'] = True
                enhanced_tasks[nav_idx]['description'] += " (carrying items)"
        
        # 3. 增强Place任务的依赖关系
        for place_idx, place_task in place_tasks:
            place_target = place_task.get('target', '').lower()
            
            # Place任务应该依赖对应的Pick任务和携带导航任务
            required_picks = self._find_pick_tasks_for_place(place_task, pick_tasks)
            required_navs = self._find_carrying_navigation_for_place(place_task, navigate_tasks, enhanced_tasks)
            
            current_prereqs = set(place_task.get('prerequisites', []))
            
            for pick_idx, pick_task in required_picks:
                pick_task_id = pick_task['task_id']
                if pick_task_id not in current_prereqs:
                    current_prereqs.add(pick_task_id)
                    print(f"[SEMANTIC] Added pick dependency for place: {place_task['task_id']} now depends on {pick_task_id}")
            
            for nav_idx, nav_task in required_navs:
                nav_task_id = nav_task['task_id']
                if nav_task_id not in current_prereqs:
                    current_prereqs.add(nav_task_id)
                    print(f"[SEMANTIC] Added navigation dependency for place: {place_task['task_id']} now depends on {nav_task_id}")
            
            enhanced_tasks[place_idx]['prerequisites'] = list(current_prereqs)
        
        # 4. 验证和清理依赖关系
        self._validate_enhanced_dependencies(enhanced_tasks)
        
        return enhanced_tasks

    def _is_carrying_navigation_task(
        self, 
        nav_task: Dict[str, Any], 
        pick_tasks: List[Tuple[int, Dict[str, Any]]], 
        place_tasks: List[Tuple[int, Dict[str, Any]]]
    ) -> bool:
        """判断是否为携带物品的导航任务"""
        nav_target = nav_task.get('target', '').lower()
        nav_description = nav_task.get('description', '').lower()
        
        # 检查目标是否为放置位置
        place_targets = set()
        for _, place_task in place_tasks:
            place_target = place_task.get('target', '').lower()
            place_targets.add(place_target)
        
        # 如果导航目标是放置目标之一，很可能是携带导航
        if nav_target in place_targets:
            return True
        
        # 检查描述中是否包含携带、运送等关键词
        carrying_keywords = ['carry', 'carrying', 'bring', 'take', 'transport', 'move', 'deliver']
        if any(keyword in nav_description for keyword in carrying_keywords):
            return True
        
        # 检查是否在Pick任务之后且Place任务之前
        nav_task_id = nav_task.get('task_id', '')
        
        # 获取任务序号
        try:
            nav_order = int(nav_task_id.split('_')[1]) if '_' in nav_task_id else 0
        except:
            nav_order = 0
        
        # 检查是否有Pick任务在它之前
        has_pick_before = any(
            int(pick_task['task_id'].split('_')[1]) < nav_order 
            for _, pick_task in pick_tasks 
            if '_' in pick_task.get('task_id', '')
        )
        
        # 检查是否有Place任务在它之后
        has_place_after = any(
            int(place_task['task_id'].split('_')[1]) > nav_order 
            for _, place_task in place_tasks 
            if '_' in place_task.get('task_id', '')
        )
        
        return has_pick_before and has_place_after

    def _find_required_pick_tasks(
        self, 
        nav_task: Dict[str, Any], 
        pick_tasks: List[Tuple[int, Dict[str, Any]]],
        all_tasks: List[Dict[str, Any]]
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """找到导航任务所需的Pick任务"""
        nav_task_id = nav_task.get('task_id', '')
        nav_target = nav_task.get('target', '').lower()
        
        try:
            nav_order = int(nav_task_id.split('_')[1]) if '_' in nav_task_id else 0
        except:
            nav_order = 999
        
        required_picks = []
        
        # 查找在此导航任务之前的所有Pick任务
        for pick_idx, pick_task in pick_tasks:
            pick_task_id = pick_task.get('task_id', '')
            try:
                pick_order = int(pick_task_id.split('_')[1]) if '_' in pick_task_id else 0
            except:
                pick_order = 0
            
            if pick_order < nav_order:
                required_picks.append((pick_idx, pick_task))
        
        # 如果没有找到基于顺序的Pick任务，查找所有相关的Pick任务
        if not required_picks:
            # 查找目标相关的Pick任务
            for pick_idx, pick_task in pick_tasks:
                pick_target = pick_task.get('target', '').lower()
                # 如果Pick的目标对象和导航任务的目标位置相关
                if self._are_targets_related(pick_target, nav_target):
                    required_picks.append((pick_idx, pick_task))
        
        return required_picks

    def _find_pick_tasks_for_place(
        self, 
        place_task: Dict[str, Any], 
        pick_tasks: List[Tuple[int, Dict[str, Any]]]
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """找到Place任务对应的Pick任务"""
        place_description = place_task.get('description', '').lower()
        place_target = place_task.get('target', '').lower()
        
        related_picks = []
        
        # 从描述中提取物品名称
        for pick_idx, pick_task in pick_tasks:
            pick_target = pick_task.get('target', '').lower()
            pick_description = pick_task.get('description', '').lower()
            
            # 检查是否为相同物品
            if pick_target in place_description or any(
                word in place_description for word in pick_target.split('_')
            ):
                related_picks.append((pick_idx, pick_task))
        
        return related_picks

    def _find_carrying_navigation_for_place(
        self, 
        place_task: Dict[str, Any], 
        navigate_tasks: List[Tuple[int, Dict[str, Any]]],
        all_tasks: List[Dict[str, Any]]
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """找到Place任务对应的携带导航任务"""
        place_target = place_task.get('target', '').lower()
        place_task_id = place_task.get('task_id', '')
        
        try:
            place_order = int(place_task_id.split('_')[1]) if '_' in place_task_id else 0
        except:
            place_order = 999
        
        related_navs = []
        
        for nav_idx, nav_task in navigate_tasks:
            nav_target = nav_task.get('target', '').lower()
            nav_task_id = nav_task.get('task_id', '')
            
            try:
                nav_order = int(nav_task_id.split('_')[1]) if '_' in nav_task_id else 0
            except:
                nav_order = 0
            
            # **ENHANCED** 多重条件匹配Place任务所需的Navigate任务
            is_target_match = (nav_target == place_target or self._are_targets_related(nav_target, place_target))
            is_before_place = nav_order < place_order
            is_carrying = nav_task.get('is_carrying_navigation', False)
            
            # **NEW** 增强匹配逻辑：
            # 1. 直接目标匹配 + 时序正确 + 是携带导航
            # 2. 或者是在Place之前的最近携带导航任务（即使目标不完全匹配）
            # 3. 或者是描述中包含相同物品的导航任务
            
            condition1 = is_target_match and is_before_place and is_carrying
            condition2 = is_before_place and is_carrying and self._is_navigation_for_placing(nav_task, place_task)
            condition3 = is_before_place and self._has_shared_objects_in_description(nav_task, place_task)
            
            if condition1 or condition2 or condition3:
                related_navs.append((nav_idx, nav_task))
                print(f"[SEMANTIC] Found navigation dependency: Place[{place_task['task_id']}] needs Navigate[{nav_task['task_id']}] (condition: {'target_match' if condition1 else 'carrying_logic' if condition2 else 'shared_objects'})")
        
        # **FALLBACK** 如果没有找到精确匹配，查找最近的携带导航任务
        if not related_navs:
            carrying_navs = [(idx, task) for idx, task in navigate_tasks 
                           if task.get('is_carrying_navigation', False) and 
                           int(task.get('task_id', 'task_0').split('_')[1]) < place_order]
            
            if carrying_navs:
                # 选择最接近Place任务的携带导航
                closest_nav = max(carrying_navs, key=lambda x: int(x[1].get('task_id', 'task_0').split('_')[1]))
                related_navs.append(closest_nav)
                print(f"[SEMANTIC] Fallback navigation dependency: Place[{place_task['task_id']}] needs closest carrying Navigate[{closest_nav[1]['task_id']}]")
        
        return related_navs

    def _are_targets_related(self, target1: str, target2: str) -> bool:
        """判断两个目标是否相关"""
        if not target1 or not target2:
            return False
        
        # 直接匹配
        if target1 == target2:
            return True
        
        # 词汇匹配
        words1 = set(target1.replace('_', ' ').split())
        words2 = set(target2.replace('_', ' ').split())
        
        # 如果有共同词汇，认为相关
        common_words = words1.intersection(words2)
        return len(common_words) > 0

    def _is_navigation_for_placing(self, nav_task: Dict[str, Any], place_task: Dict[str, Any]) -> bool:
        """判断Navigate任务是否是为Place任务准备的"""
        nav_description = nav_task.get('description', '').lower()
        place_description = place_task.get('description', '').lower()
        place_target = place_task.get('target', '').lower()
        
        # 检查导航任务的描述是否暗示了放置意图
        placement_keywords = ['place', 'put', 'move', 'bring', 'carry', 'transport', 'deliver']
        has_placement_intent = any(keyword in nav_description for keyword in placement_keywords)
        
        # 检查目标位置相关性
        nav_target_words = set(nav_task.get('target', '').replace('_', ' ').lower().split())
        place_target_words = set(place_target.replace('_', ' ').split())
        has_target_overlap = bool(nav_target_words.intersection(place_target_words))
        
        return has_placement_intent and has_target_overlap

    def _has_shared_objects_in_description(self, nav_task: Dict[str, Any], place_task: Dict[str, Any]) -> bool:
        """判断Navigate和Place任务是否涉及相同的物品"""
        nav_description = nav_task.get('description', '').lower()
        place_description = place_task.get('description', '').lower()
        
        # 提取可能的物品名称（常见物品关键词）
        object_keywords = ['toy', 'food', 'truck', 'box', 'cup', 'plate', 'bottle', 'book', 'pen', 'key']
        
        nav_objects = set()
        place_objects = set()
        
        for keyword in object_keywords:
            if keyword in nav_description:
                nav_objects.add(keyword)
            if keyword in place_description:
                place_objects.add(keyword)
        
        # 也检查任务目标中的物品名称
        nav_target_words = set(nav_task.get('target', '').replace('_', ' ').lower().split())
        place_target_words = set(place_task.get('target', '').replace('_', ' ').lower().split())
        
        for keyword in object_keywords:
            if keyword in nav_target_words:
                nav_objects.add(keyword)
            if keyword in place_target_words:
                place_objects.add(keyword)
        
        # 如果有共同的物品关键词，认为相关
        shared_objects = nav_objects.intersection(place_objects)
        return len(shared_objects) > 0

    def _validate_enhanced_dependencies(self, enhanced_tasks: List[Dict[str, Any]]) -> None:
        """验证增强后的依赖关系，防止循环依赖"""
        task_ids = {task['task_id'] for task in enhanced_tasks}
        
        for task in enhanced_tasks:
            task_id = task['task_id']
            prerequisites = task.get('prerequisites', [])
            
            # 移除无效引用
            valid_prereqs = [prereq for prereq in prerequisites if prereq in task_ids and prereq != task_id]
            
            if len(valid_prereqs) != len(prerequisites):
                removed = len(prerequisites) - len(valid_prereqs)
                print(f"[SEMANTIC] Cleaned {removed} invalid dependencies from {task_id}")
                task['prerequisites'] = valid_prereqs

    def _enhance_llm_decomposition_prompt(self, instruction: str, objects_info: str, agent_info: str) -> str:
        
        enhanced_prompt = f"""Decompose the instruction into a JSON array of subtasks with dependencies, focusing on carrying-state logic.

INSTRUCTION: {instruction}

WORLD STATE:
{objects_info}

AGENT STATUS:
{agent_info}

AGENT CAPABILITIES:
- Agent 0: Navigate, Explore, Pick, Place, Open, Close, Rearrange, Wait
- Agent 1: All Agent 0 skills + Clean, Fill, Pour, PowerOn, PowerOff

DEPENDENCY RULES:
- A "Navigate" task for placing items is a "carrying navigation" and MUST depend on ALL relevant "Pick" tasks.
- Example: For "Put A and B on C", the "Navigate to C" task must have prerequisites pointing to "Pick A" and "Pick B".
- A "Place" task must depend on its corresponding "Pick" and "carrying navigation" tasks.

JSON OUTPUT FORMAT (JSON only, no comments):
[
  {{
    "task_type": "Navigate|Pick|Place|etc.",
    "target": "object_or_location",
    "description": "description of the subtask",
    "priority": 1-5,
    "prerequisites": ["task_id_1"],
    "can_parallel": boolean
  }}
]
"""
        return enhanced_prompt
