import numpy as np
import re
from typing import Dict, Any, List, Optional, Tuple

class TaskPriorityConfig:
    """任务优先级配置类 - 简化的PARTNR 13任务到5能力映射"""
    
    def __init__(self):
        # PARTNR 13任务类型映射 (与scenario_params_task.py一致)
        self.partnr_task_mapping = {
            'Navigate': 0,   'Explore': 1,    'Pick': 2,       'Place': 3,
            'Open': 4,       'Close': 5,      'Clean': 6,      'Fill': 7,
            'Pour': 8,       'PowerOn': 9,    'PowerOff': 10,  'Rearrange': 11,
            'Wait': 12
        }
        
        # 13任务到5能力分类的映射关系
        self.task_to_capability_mapping = {
            # 基础移动能力 (能力0)
            'Navigate': 0, 'Explore': 0, 'Wait': 0,
            # 物体操作能力 (能力1)
            'Pick': 1, 'Place': 1, 'Rearrange': 1,
            # 基本控制能力 (能力2)
            'Open': 2, 'Close': 2,
            # 液体处理能力 (能力3)
            'Clean': 3, 'Fill': 3, 'Pour': 3,
            # 电源控制能力 (能力4)
            'PowerOn': 4, 'PowerOff': 4
        }
        
        self.base_priority_weights = {
            0: 1.5,   # Navigate - 重要的基础移动
            1: 2.0,   # Explore - 早期探索重要
            2: 3.0,   # Pick - 核心操作任务
            3: 3.0,   # Place - 核心操作任务
            4: 2.5,   # Open - 铰接控制
            5: 2.5,   # Close - 铰接控制
            6: 2.0,   # Clean - 状态操作
            7: 2.0,   # Fill - 状态操作
            8: 2.0,   # Pour - 状态操作
            9: 1.5,   # PowerOn - 设备控制
            10: 1.5,  # PowerOff - 设备控制
            11: 3.5,  # Rearrange - 复合高优先级任务
            12: 0.1   # Wait - 最低优先级
        }
        self.early_time_threshold = 20.0
        self.explore_early_boost = 1.0
        self.wait_penalty = 0.05
        
        self.llm_response_boost = {
            'active_action': 4.0,      # 当前正在执行的动作
            'next_logical': 3.0,       # 逻辑上的下一步动作
            'prerequisite_completed': 2.5,  # 前置条件完成后的动作
            'agent_capability_match': 2.0,  # 与Agent能力匹配的动作
            'task_progression': 3.5,   # 任务进展相关的动作
        }
        
        # 序列执行任务进展权重
        self.task_sequence_weights = {
            'navigate_to_pick': 2.5,    # Navigate完成后提升Pick
            'pick_to_place': 3.0,       # Pick完成后提升Place
            'explore_to_pick': 2.0,     # Explore完成后提升Pick
            'place_to_navigate': 1.5,   # Place完成后提升Navigate
        }

class TaskUtilityNormalizer:
    """任务效用归一化器 - LLM响应驱动的优先级调整"""
    
    def __init__(self, dim: Dict[str, int], tasks: List[Dict[str, Any]], 
                 task_priority_config: Optional[TaskPriorityConfig] = None):
        """
        初始化任务效用归一化器
        """
        self.dim = dim
        self.tasks = tasks
        self.config = task_priority_config or TaskPriorityConfig()
        
        self.action_history: List[Dict[str, Any]] = []
        self.last_llm_response = ""
        self.last_parsed_actions: Dict[int, Tuple[str, str]] = {}
        self.debug_info = {
            'last_scaling_factors': {},
            'priority_adjustments': {},
            'llm_influence': {}
        }

    def update_config(self, **kwargs):
        """动态更新优先级配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def parse_llm_response(self, llm_response: str) -> Dict[int, Tuple[str, str]]:
        """解析LLM响应，提取Agent动作信息"""
        parsed_actions = {}
        
        # 【TODO】正则表达式匹配 Agent_X_Action: ActionName[target] 格式
        action_pattern = r'Agent_(\d+)_Action:\s*(\w+)\[([^\]]*)\]'
        matches = re.findall(action_pattern, llm_response)
        
        for agent_id_str, action_name, action_target in matches:
            try:
                agent_id = int(agent_id_str)
                parsed_actions[agent_id] = (action_name.strip(), action_target.strip())
                print(f"[DEBUG] Parsed action: Agent {agent_id} → {action_name}[{action_target}]")
            except ValueError:
                print(f"[WARNING] Invalid agent ID in LLM response: {agent_id_str}")
        
        return parsed_actions

    def analyze_task_progression(self, current_actions: Dict[int, Tuple[str, str]]) -> Dict[str, float]:
        """分析任务进展状态，动作序列逻辑递进"""
        progression_weights = {}
        if self.last_parsed_actions:
            for agent_id, (last_action, last_target) in self.last_parsed_actions.items():
                current_action_info = current_actions.get(agent_id)
                if current_action_info:
                    current_action, current_target = current_action_info
                    if last_action == 'Navigate' and current_action == 'Pick':
                        progression_weights['navigate_to_pick'] = self.config.task_sequence_weights['navigate_to_pick']
                        print(f"[DEBUG] Task progression: Navigate → Pick (Agent {agent_id})")
                    elif last_action == 'Pick' and current_action == 'Place':
                        progression_weights['pick_to_place'] = self.config.task_sequence_weights['pick_to_place']
                        print(f"[DEBUG] Task progression: Pick → Place (Agent {agent_id})")
                    elif last_action == 'Explore' and current_action == 'Pick':
                        progression_weights['explore_to_pick'] = self.config.task_sequence_weights['explore_to_pick']
                        print(f"[DEBUG] Task progression: Explore → Pick (Agent {agent_id})")
                    elif last_action == 'Place' and current_action == 'Navigate':
                        progression_weights['place_to_navigate'] = self.config.task_sequence_weights['place_to_navigate']
                        print(f"[DEBUG] Task progression: Place → Navigate (Agent {agent_id})")
        return progression_weights

    def calculate_llm_driven_priorities(self, llm_response: str, global_vars_dict: Dict[str, Any]) -> Dict[int, float]:
        """基于LLM响应计算任务优先级权重"""
        
        current_actions = self.parse_llm_response(llm_response)
        progression_weights = self.analyze_task_progression(current_actions)
        llm_priorities = {j: 1.0 for j in range(self.dim['n_t'])}
        
        # 基于当前动作调整优先级
        for agent_id, (action_name, action_target) in current_actions.items():
            task_idx = self.config.partnr_task_mapping.get(action_name)
            
            if task_idx is not None:
                llm_priorities[task_idx] *= self.config.llm_response_boost['active_action']
                # print(f"[DEBUG] LLM boost: Task {task_idx} ({action_name}) priority × {self.config.llm_response_boost['active_action']}")
                
                # 特殊处理Rearrange任务 (涉及Pick和Place)
                if action_name == 'Rearrange':
                    pick_idx = self.config.partnr_task_mapping.get('Pick', 2)
                    place_idx = self.config.partnr_task_mapping.get('Place', 3)
                    llm_priorities[pick_idx] *= self.config.llm_response_boost['next_logical']
                    llm_priorities[place_idx] *= self.config.llm_response_boost['next_logical']
                    # print(f"[DEBUG] Rearrange boost: Pick({pick_idx}) and Place({place_idx}) enhanced")
        
        for progression_type, weight in progression_weights.items():
            if progression_type == 'navigate_to_pick':
                pick_idx = self.config.partnr_task_mapping.get('Pick', 2)
                llm_priorities[pick_idx] *= weight
            elif progression_type == 'pick_to_place':
                place_idx = self.config.partnr_task_mapping.get('Place', 3)
                llm_priorities[place_idx] *= weight
            elif progression_type == 'explore_to_pick':
                pick_idx = self.config.partnr_task_mapping.get('Pick', 2)
                llm_priorities[pick_idx] *= weight
            elif progression_type == 'place_to_navigate':
                nav_idx = self.config.partnr_task_mapping.get('Navigate', 0)
                llm_priorities[nav_idx] *= weight
        
        self.last_llm_response = llm_response
        self.last_parsed_actions = current_actions
        self.action_history.append({
            'timestamp': global_vars_dict.get('current_time', 0),
            'actions': current_actions,
            'progression': progression_weights
        })
        
        self.debug_info['llm_influence'] = {
            'current_actions': current_actions,
            'progression_weights': progression_weights,
            'llm_priorities': llm_priorities
        }
        
        return llm_priorities

    def calculate_scaling_factors(self, x: np.ndarray, t: float, global_vars_dict: Dict[str, Any] = None,
                                 llm_response: str = "") -> Dict[int, float]:
        """
        计算任务效用缩放因子 - 整合LLM响应、时间和状态信息
        """
        n_r = self.dim['n_r']
        n_t = self.dim['n_t']
        
        if global_vars_dict is None:
            global_vars_dict = {}
        
        # === 1. 计算基础任务效用值 ===
        task_values = {j: [] for j in range(n_t)}
        for i in range(n_r):
            for j in range(n_t):
                if j < len(self.tasks):
                    task = self.tasks[j]
                    try:
                        task_func_value = task['function'](x[:, i], t, i, vars_dict=global_vars_dict)
                        task_values[j].append(abs(task_func_value))
                    except Exception as e:
                        print(f"[WARNING] Task {j} utility calculation failed: {e}")
                        task_values[j].append(1.0)
                else:
                    task_values[j].append(1.0)  # 任务未定义时的默认值
        
        # === 2. 计算基础统计信息 ===
        task_stats = {}
        for j in range(n_t):
            values = task_values[j]
            if values:
                task_stats[j] = {
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'std': np.std(values) if len(values) > 1 else 0.0
                }
            else:
                task_stats[j] = {'mean': 1.0, 'max': 1.0, 'min': 1.0, 'std': 0.0}
        
        # === 3. 基础优先级权重 ===
        base_weights = self.config.base_priority_weights.copy()
        
        # === 4. 时间和状态驱动的调整 ===
        time_weights = {j: 1.0 for j in range(n_t)}
        wait_idx = self.config.partnr_task_mapping.get('Wait', 12)
        time_weights[wait_idx] = self.config.wait_penalty
        
        # 早期探索提升
        explore_idx = self.config.partnr_task_mapping.get('Explore', 1)
        if t < self.config.early_time_threshold:
            time_weights[explore_idx] = self.config.explore_early_boost
        
        # 探索状态检查
        exploration_targets = global_vars_dict.get('exploration_targets', [])
        num_unexplored = sum(1 for target in exploration_targets if not target.get('explored', False))
        if num_unexplored > 0:
            time_weights[explore_idx] *= 1.5
        
        # === 5. LLM响应驱动的优先级调整 ===
        llm_weights = {j: 1.0 for j in range(n_t)}
        if llm_response.strip():
            llm_weights = self.calculate_llm_driven_priorities(llm_response, global_vars_dict)
        
        # === 6. 计算最终缩放因子 ===
        task_means = np.array([task_stats[j]['mean'] for j in range(n_t)])
        valid_means = task_means[task_means > 1e-6]
        max_mean = np.max(valid_means) if len(valid_means) > 0 else 1.0
        if max_mean < 1e-6:
            max_mean = 1.0
        
        scaling_factors = {}
        priority_adjustments = {}
        
        for j in range(n_t):
            if task_stats[j]['mean'] > 1e-6:
                norm_factor = max_mean / task_stats[j]['mean']
                norm_factor = np.clip(norm_factor, 0.01, 100.0)
            else:
                norm_factor = 1.0
            
            base_priority = base_weights.get(j, 1.0)
            time_priority = time_weights.get(j, 1.0)
            llm_priority = llm_weights.get(j, 1.0)
            
            combined_priority = base_priority * time_priority * llm_priority
            final_factor = combined_priority * norm_factor
            
            scaling_factors[j] = final_factor
            priority_adjustments[j] = {
                'base': base_priority,
                'time': time_priority,
                'llm': llm_priority,
                'combined': combined_priority,
                'norm_factor': norm_factor
            }
        
        # === 7. 保存调试信息 ===
        self.debug_info['last_scaling_factors'] = scaling_factors
        self.debug_info['priority_adjustments'] = priority_adjustments
        
        # 输出关键调整信息
        if llm_response.strip():
            print(f"[DEBUG] LLM-driven scaling factors updated:")
            for j, factor in scaling_factors.items():
                task_name = self._get_task_name_by_index(j)
                if factor > 2.0:  # 只显示显著提升的任务
                    print(f"  Task {j} ({task_name}): {factor:.2f}")
        
        return scaling_factors

    def _get_task_name_by_index(self, task_idx: int) -> str:
        index_to_name = {v: k for k, v in self.config.partnr_task_mapping.items()}
        return index_to_name.get(task_idx, f"Task_{task_idx}")
    def get_debug_info(self) -> Dict[str, Any]:
        return self.debug_info
    def get_action_history(self) -> List[Dict[str, Any]]:
        return self.action_history
    def reset_history(self):
        self.action_history.clear()
        self.last_llm_response = ""
        self.last_parsed_actions.clear()