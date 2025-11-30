#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import re
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

@dataclass
class CoherenceMetrics:
    """上下文一致性评估指标"""
    semantic_coherence: float  # 语义连贯性 [0,1]
    temporal_coherence: float  # 时序连贯性 [0,1] 
    action_coherence: float    # 动作连贯性 [0,1]
    overall_coherence: float   # 总体连贯性 [0,1]
    
class CoherenceEvaluator:
    """
    上下文一致性评估器
    用于评估LLM生成的规划序列在语义、时序和动作层面的连贯性
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化评估器
        
        Args:
            model_name: 用于语义相似度计算的句子嵌入模型
        """
        self.sentence_model = SentenceTransformer(model_name)
        self.action_patterns = {
            'navigation': ['navigate', 'go', 'move', 'walk'],
            'manipulation': ['pick', 'place', 'grab', 'put', 'drop'],
            'observation': ['look', 'find', 'search', 'observe'],
            'communication': ['wait', 'done', 'inform']
        }
        
    def evaluate_coherence(self, 
                          planning_trace: List[str], 
                          world_state_sequence: List[Dict[str, Any]],
                          miqp_assignments: Optional[List[np.ndarray]] = None) -> CoherenceMetrics:
        """
        评估规划轨迹的上下文一致性
        
        Args:
            planning_trace: LLM生成的规划步骤序列
            world_state_sequence: 对应的世界状态序列
            miqp_assignments: MIQP优化器的任务分配结果序列
            
        Returns:
            CoherenceMetrics: 一致性评估结果
        """
        # 1. 语义连贯性评估
        semantic_score = self._evaluate_semantic_coherence(planning_trace)
        
        # 2. 时序连贯性评估
        temporal_score = self._evaluate_temporal_coherence(
            planning_trace, world_state_sequence
        )
        
        # 3. 动作连贯性评估
        action_score = self._evaluate_action_coherence(
            planning_trace, miqp_assignments
        )
        
        # 4. 计算总体连贯性（加权平均）
        overall_score = (
            0.4 * semantic_score + 
            0.3 * temporal_score + 
            0.3 * action_score
        )
        
        return CoherenceMetrics(
            semantic_coherence=semantic_score,
            temporal_coherence=temporal_score,
            action_coherence=action_score,
            overall_coherence=overall_score
        )
    
    def _evaluate_semantic_coherence(self, planning_trace: List[str]) -> float:
        """
        评估语义连贯性
        通过计算相邻规划步骤之间的语义相似度
        """
        if len(planning_trace) < 2:
            return 1.0
            
        # 提取思考过程和动作描述
        thoughts = []
        actions = []
        
        for step in planning_trace:
            thought = self._extract_thought(step)
            action = self._extract_action(step)
            if thought:
                thoughts.append(thought)
            if action:
                actions.append(action)
        
        # 计算思考过程的语义连贯性
        thought_coherence = self._calculate_sequence_coherence(thoughts)
        
        # 计算动作序列的语义连贯性
        action_coherence = self._calculate_sequence_coherence(actions)
        
        # 加权平均
        return 0.6 * thought_coherence + 0.4 * action_coherence
    
    def _evaluate_temporal_coherence(self, 
                                   planning_trace: List[str],
                                   world_state_sequence: List[Dict[str, Any]]) -> float:
        """
        评估时序连贯性
        检查规划步骤是否符合时间逻辑和因果关系
        """
        if len(planning_trace) < 2:
            return 1.0
            
        coherence_scores = []
        
        for i in range(1, len(planning_trace)):
            prev_step = planning_trace[i-1]
            curr_step = planning_trace[i]
            
            # 检查前置条件是否满足
            precondition_score = self._check_preconditions(
                prev_step, curr_step, 
                world_state_sequence[i-1] if i-1 < len(world_state_sequence) else {}
            )
            
            # 检查动作序列的逻辑性
            logical_score = self._check_action_logic(prev_step, curr_step)
            
            step_score = 0.5 * precondition_score + 0.5 * logical_score
            coherence_scores.append(step_score)
        
        return np.mean(coherence_scores) if coherence_scores else 1.0
    
    def _evaluate_action_coherence(self, 
                                 planning_trace: List[str],
                                 miqp_assignments: Optional[List[np.ndarray]]) -> float:
        """
        评估动作连贯性
        检查动作序列是否合理，以及与MIQP分配的一致性
        """
        if len(planning_trace) < 2:
            return 1.0
            
        # 提取动作序列
        actions = [self._extract_action(step) for step in planning_trace]
        actions = [a for a in actions if a]  # 过滤空值
        
        if len(actions) < 2:
            return 1.0
        
        # 1. 动作类型转换的合理性
        transition_score = self._evaluate_action_transitions(actions)
        
        # 2. 与MIQP分配的一致性（如果提供）
        miqp_consistency = 1.0
        if miqp_assignments:
            miqp_consistency = self._evaluate_miqp_consistency(
                actions, miqp_assignments
            )
        
        return 0.7 * transition_score + 0.3 * miqp_consistency
    
    def _calculate_sequence_coherence(self, text_sequence: List[str]) -> float:
        """计算文本序列的语义连贯性"""
        if len(text_sequence) < 2:
            return 1.0
            
        # 生成句子嵌入
        embeddings = self.sentence_model.encode(text_sequence)
        
        # 计算相邻句子的余弦相似度
        similarities = []
        for i in range(1, len(embeddings)):
            sim = cosine_similarity(
                embeddings[i-1].reshape(1, -1),
                embeddings[i].reshape(1, -1)
            )[0][0]
            similarities.append(sim)
        
        # 返回平均相似度
        return np.mean(similarities) if similarities else 1.0
    
    def _extract_thought(self, planning_step: str) -> str:
        """从规划步骤中提取思考过程"""
        # 匹配 "Thought:" 后的内容
        thought_pattern = r"Thought:\s*(.*?)(?=\n|Agent_|$)"
        match = re.search(thought_pattern, planning_step, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _extract_action(self, planning_step: str) -> str:
        """从规划步骤中提取动作描述"""
        # 匹配 "Agent_X_Action:" 后的内容
        action_pattern = r"Agent_\d+_Action:\s*(.*?)(?=\n|$)"
        match = re.search(action_pattern, planning_step)
        return match.group(1).strip() if match else ""
    
    def _check_preconditions(self, 
                           prev_step: str, 
                           curr_step: str,
                           world_state: Dict[str, Any]) -> float:
        """检查动作的前置条件是否满足"""
        curr_action = self._extract_action(curr_step)
        if not curr_action:
            return 1.0
            
        # 检查导航动作的前置条件
        if any(nav in curr_action.lower() for nav in self.action_patterns['navigation']):
            # 导航动作通常不需要特殊前置条件
            return 1.0
            
        # 检查操作动作的前置条件
        if any(manip in curr_action.lower() for manip in self.action_patterns['manipulation']):
            # 操作动作需要目标对象存在且可达
            return 0.8  # 简化评分
            
        return 1.0
    
    def _check_action_logic(self, prev_step: str, curr_step: str) -> float:
        """检查动作序列的逻辑性"""
        prev_action = self._extract_action(prev_step)
        curr_action = self._extract_action(curr_step)
        
        if not prev_action or not curr_action:
            return 1.0
            
        # 检查动作序列的合理性
        # 例如：pick之后应该是place或navigate，而不是再次pick同一物体
        
        prev_type = self._classify_action_type(prev_action)
        curr_type = self._classify_action_type(curr_action)
        
        # 定义合理的动作转换
        valid_transitions = {
            'navigation': ['manipulation', 'observation', 'navigation'],
            'manipulation': ['navigation', 'manipulation', 'communication'],
            'observation': ['navigation', 'manipulation', 'observation'],
            'communication': ['navigation', 'observation', 'communication']
        }
        
        if curr_type in valid_transitions.get(prev_type, []):
            return 1.0
        else:
            return 0.5  # 不太合理但不完全错误
    
    def _classify_action_type(self, action: str) -> str:
        """分类动作类型"""
        action_lower = action.lower()
        
        for action_type, keywords in self.action_patterns.items():
            if any(keyword in action_lower for keyword in keywords):
                return action_type
                
        return 'unknown'
    
    def _evaluate_action_transitions(self, actions: List[str]) -> float:
        """评估动作转换的合理性"""
        if len(actions) < 2:
            return 1.0
            
        transition_scores = []
        
        for i in range(1, len(actions)):
            prev_type = self._classify_action_type(actions[i-1])
            curr_type = self._classify_action_type(actions[i])
            
            # 基于动作类型转换的合理性评分
            if prev_type == curr_type:
                score = 0.8  # 同类型动作连续执行，一般合理
            elif (prev_type == 'navigation' and curr_type == 'manipulation') or \
                 (prev_type == 'manipulation' and curr_type == 'navigation'):
                score = 1.0  # 导航和操作交替，非常合理
            elif prev_type == 'observation' and curr_type in ['navigation', 'manipulation']:
                score = 0.9  # 观察后执行动作，合理
            else:
                score = 0.6  # 其他转换，基本合理
                
            transition_scores.append(score)
        
        return np.mean(transition_scores)
    
    def _evaluate_miqp_consistency(self, 
                                 actions: List[str],
                                 miqp_assignments: List[np.ndarray]) -> float:
        """评估与MIQP分配的一致性"""
        if not miqp_assignments or len(actions) != len(miqp_assignments):
            return 1.0  # 如果没有MIQP数据，默认一致
            
        consistency_scores = []
        
        for i, (action, assignment) in enumerate(zip(actions, miqp_assignments)):
            # 检查动作是否与MIQP分配的任务一致
            # 这里需要根据具体的MIQP输出格式进行调整
            
            # 简化的一致性检查：
            # 如果MIQP分配了任务（assignment中有非零元素），
            # 那么应该有对应的动作执行
            has_assignment = np.any(assignment > 0.1) if assignment is not None else False
            has_action = action and action.strip() != "Wait[]"
            
            if has_assignment == has_action:
                consistency_scores.append(1.0)
            else:
                consistency_scores.append(0.5)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0 