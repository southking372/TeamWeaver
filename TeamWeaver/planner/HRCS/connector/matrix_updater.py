from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import json

from habitat_llm.planner.miqp_prompts import get_miqp_prompt, MIQPAnalysisPrompt
from habitat_llm.planner.HRCS.connector.planner_utils import extract_json_from_text, get_llm_config


class MatrixUpdater:
    """
    负责更新MIQP优化器所需的矩阵 (T, A, ws)。
    此类将所有矩阵更新逻辑（基于LLM或规则）与PerceptionConnector解耦。
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
        [0.0, 1.4],  # 液体 (仅Agent 1)
        [0.0, 1.3]   # 电源 (仅Agent 1)
    ], dtype=float)
    
    BASE_CAPABILITY_WEIGHTS = [
        2.0 * np.eye(1),  # 移动
        2.5 * np.eye(1),  # 操作
        2.0 * np.eye(1),  # 控制
        1.8 * np.eye(1),  # 液体
        1.5 * np.eye(1)   # 电源
    ]
    
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client

    def update_matrices(
        self,
        structured_subtasks: List[Dict[str, Any]],
        world_state: Dict[str, Any]
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        主入口方法，根据任务分析更新MIQP的参数矩阵。
        优先尝试LLM生成，失败则回退到基于规则的增强方法。
        """
        try:
            print("DEBUG: Attempting to generate MIQP matrices directly from LLM response.")
            T, A, ws = self._generate_matrices_from_llm(structured_subtasks)
            A = self.BASE_ROBOT_CAPABILITIES.copy()

            if T is not None and A is not None and ws is not None:
                print("DEBUG: Successfully generated and updated MIQP matrices from LLM.")
                return {'T': T, 'A': A, 'ws': ws}

            print("Warning: Failed to generate matrices directly from LLM, falling back to rule-based analysis.")
            llm_analysis = self._llm_analyze_task_constraints(structured_subtasks, world_state)
            
            updated_T = self._update_task_capability_matrix_enhanced(structured_subtasks, llm_analysis)
            updated_A = self._update_robot_capability_matrix_enhanced(structured_subtasks, llm_analysis)
            updated_ws = self._update_capability_weights_enhanced(structured_subtasks, llm_analysis)
            
            return {'T': updated_T, 'A': updated_A, 'ws': updated_ws}
            
        except Exception as e:
            print(f"ERROR: Matrix generation failed entirely: {e}. Using simple fallback.")
            return self._update_matrices_fallback(structured_subtasks, world_state)

    def _generate_matrices_from_llm(
        self, 
        structured_subtasks: List[Dict[str, Any]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[np.ndarray]]]:
        """通过LLM直接生成MIQP矩阵。"""
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

        prompt_template = get_miqp_prompt("matrix_generation", get_llm_config())
        prompt = prompt_template(
            structured_subtasks=structured_subtasks,
            agent_descriptions=agent_descriptions,
            capability_names=capability_names,
            task_type_names=task_type_names
        )

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
                print(f"Validation Error: T matrix is malformed.")
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
            prompt_template = MIQPAnalysisPrompt("miqp_analysis", get_llm_config())
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
            json_text = extract_json_from_text(analysis_text, dict)
            if json_text:
                analysis = json.loads(json_text)
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
        base_T = self.BASE_TASK_CAPABILITY_REQUIREMENTS.copy()
        
        complexity = llm_analysis.get("task_complexity", {})
        conservative_factor = llm_analysis.get("constraints", {}).get("conservative_factors", {}).get("safety", 0.9)
        
        task_map = {'Navigate': 0, 'Explore': 1, 'Pick': 2, 'Place': 3}

        for task_name, task_idx in task_map.items():
            if any(subtask['task_type'] == task_name for subtask in structured_subtasks):
                task_complexity = complexity.get(task_name.lower(), 0.5)
                enhancement = min(task_complexity * conservative_factor * 0.2, 0.3)
                capability_idx = np.argmax(base_T[task_idx, :])
                base_T[task_idx, capability_idx] += enhancement
        
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
                if base_A[cap_idx, agent_idx] > 0:
                    suitability_factor = suitability.get(cap_key, 0.9) * reliability_factor
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

    def _update_matrices_fallback(
        self, 
        structured_subtasks: List[Dict[str, Any]],
        world_state: Dict[str, Any]
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """简化的fallback矩阵更新方法"""
        print(f"DEBUG: Using fallback method to update MIQP matrices.")
        updated_T = self._update_task_capability_matrix(world_state, structured_subtasks)
        updated_A = self._update_robot_capability_matrix(world_state, structured_subtasks)
        updated_ws = self._update_capability_weights(world_state, structured_subtasks)
        
        return {'T': updated_T, 'A': updated_A, 'ws': updated_ws}

    def _update_task_capability_matrix(
        self, 
        world_state: Dict[str, Any], 
        structured_subtasks: List[Dict[str, Any]], 
    ) -> np.ndarray:
        """基于世界状态更新任务-能力需求矩阵T。"""
        try:
            base_T = self.BASE_TASK_CAPABILITY_REQUIREMENTS.copy()
            if not structured_subtasks:
                return base_T
                
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
            return self.BASE_TASK_CAPABILITY_REQUIREMENTS.copy()

    def _update_robot_capability_matrix(
        self, 
        world_state: Dict[str, Any], 
        structured_subtasks: List[Dict[str, Any]],
    ) -> np.ndarray:
        """基于世界状态更新机器人能力矩阵A（简化版）。"""
        try:
            base_A = self.BASE_ROBOT_CAPABILITIES.copy()
            updated_A = base_A.copy()
            task_types = {sub.get('task_type') for sub in structured_subtasks}
            
            if any(t in ['Pick', 'Place', 'Rearrange'] for t in task_types):
                updated_A[1, :] = np.clip(base_A[1, :] * 1.1, 1.0, 1.5)
            if any(t in ['Clean', 'Fill', 'Pour'] for t in task_types):
                updated_A[3, 1] = np.clip(base_A[3, 1] * 1.05, 1.0, 1.5)
                
            return updated_A
            
        except Exception as e:
            print(f"[ERROR] 机器人能力矩阵更新失败: {e}")
            return self.BASE_ROBOT_CAPABILITIES.copy()

    def _update_capability_weights(
        self, 
        world_state: Dict[str, Any], 
        structured_subtasks: List[Dict[str, Any]],
    ) -> List[np.ndarray]:
        """基于世界状态更新能力权重ws。"""
        try:
            base_weights = [w.copy() for w in self.BASE_CAPABILITY_WEIGHTS]
            
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
            return [w.copy() for w in self.BASE_CAPABILITY_WEIGHTS]

    def _get_default_analysis(self) -> Dict[str, Any]:
        return {
            "task_complexity": {"navigate": 0.5, "pick": 0.7, "place": 0.7, "explore": 0.4},
            "agent_suitability": {
                "agent_0": {"movement": 0.9, "manipulation": 0.8},
                "agent_1": {"movement": 0.9, "manipulation": 0.9, "liquid": 0.8, "power": 0.7}
            },
            "constraints": {"conservative_factors": {"safety": 0.9, "reliability": 0.9}},
            "critical_tasks": ["Pick", "Place", "Navigate"]
        } 