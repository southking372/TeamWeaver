 #!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import re
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import json

@dataclass
class TruthfulnessMetrics:
    """事实一致性评估指标"""
    factual_accuracy: float      # 事实准确性 [0,1]
    world_consistency: float     # 世界状态一致性 [0,1]
    object_existence: float      # 对象存在性准确率 [0,1]
    spatial_accuracy: float      # 空间关系准确性 [0,1]
    temporal_consistency: float  # 时序一致性 [0,1]
    overall_truthfulness: float  # 总体真实性 [0,1]

class TruthfulnessEvaluator:
    """
    事实一致性评估器
    基于Habitat-Sim的World Graph作为事实来源，评估LLM生成内容的真实性
    """
    
    def __init__(self):
        """初始化评估器"""
        self.spatial_relations = {
            'on': ['on_top', 'placed_on', 'sitting_on'],
            'in': ['inside', 'within', 'contained_in'],
            'near': ['next_to', 'close_to', 'beside'],
            'under': ['underneath', 'below'],
            'above': ['over', 'on_top_of']
        }
        
        # 定义可接受的误差范围
        self.position_tolerance = 0.5  # 位置误差容忍度（米）
        self.angle_tolerance = 0.2     # 角度误差容忍度（弧度）
        
    def evaluate_truthfulness(self,
                            planning_trace: List[str],
                            world_graph_sequence: List[Dict[str, Any]],
                            ground_truth_states: List[Dict[str, Any]]) -> TruthfulnessMetrics:
        """
        评估规划轨迹的事实一致性
        
        Args:
            planning_trace: LLM生成的规划步骤序列
            world_graph_sequence: 对应的世界图状态序列
            ground_truth_states: 仿真环境的真实状态序列
            
        Returns:
            TruthfulnessMetrics: 事实一致性评估结果
        """
        # 1. 事实准确性评估
        factual_score = self._evaluate_factual_accuracy(
            planning_trace, world_graph_sequence
        )
        
        # 2. 世界状态一致性评估
        world_consistency = self._evaluate_world_consistency(
            world_graph_sequence, ground_truth_states
        )
        
        # 3. 对象存在性评估
        object_existence = self._evaluate_object_existence(
            planning_trace, world_graph_sequence
        )
        
        # 4. 空间关系准确性评估
        spatial_accuracy = self._evaluate_spatial_accuracy(
            planning_trace, world_graph_sequence
        )
        
        # 5. 时序一致性评估
        temporal_consistency = self._evaluate_temporal_consistency(
            planning_trace, world_graph_sequence
        )
        
        # 6. 计算总体真实性（加权平均）
        overall_score = (
            0.25 * factual_score +
            0.20 * world_consistency +
            0.20 * object_existence +
            0.20 * spatial_accuracy +
            0.15 * temporal_consistency
        )
        
        return TruthfulnessMetrics(
            factual_accuracy=factual_score,
            world_consistency=world_consistency,
            object_existence=object_existence,
            spatial_accuracy=spatial_accuracy,
            temporal_consistency=temporal_consistency,
            overall_truthfulness=overall_score
        )
    
    def _evaluate_factual_accuracy(self,
                                 planning_trace: List[str],
                                 world_graph_sequence: List[Dict[str, Any]]) -> float:
        """
        评估LLM生成内容中事实陈述的准确性
        """
        if not planning_trace or not world_graph_sequence:
            return 1.0
            
        accuracy_scores = []
        
        for i, step in enumerate(planning_trace):
            world_state = world_graph_sequence[i] if i < len(world_graph_sequence) else {}
            
            # 提取步骤中的事实陈述
            factual_claims = self._extract_factual_claims(step)
            
            # 验证每个事实陈述
            step_accuracy = self._verify_factual_claims(factual_claims, world_state)
            accuracy_scores.append(step_accuracy)
        
        return np.mean(accuracy_scores) if accuracy_scores else 1.0
    
    def _evaluate_world_consistency(self,
                                  world_graph_sequence: List[Dict[str, Any]],
                                  ground_truth_states: List[Dict[str, Any]]) -> float:
        """
        评估世界图与真实状态的一致性
        """
        if not world_graph_sequence or not ground_truth_states:
            return 1.0
            
        consistency_scores = []
        
        for wg_state, gt_state in zip(world_graph_sequence, ground_truth_states):
            # 比较对象位置
            position_consistency = self._compare_object_positions(wg_state, gt_state)
            
            # 比较对象状态
            state_consistency = self._compare_object_states(wg_state, gt_state)
            
            # 比较空间关系
            relation_consistency = self._compare_spatial_relations(wg_state, gt_state)
            
            step_consistency = (
                0.4 * position_consistency +
                0.3 * state_consistency +
                0.3 * relation_consistency
            )
            consistency_scores.append(step_consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _evaluate_object_existence(self,
                                 planning_trace: List[str],
                                 world_graph_sequence: List[Dict[str, Any]]) -> float:
        """
        评估LLM提到的对象是否真实存在
        """
        if not planning_trace or not world_graph_sequence:
            return 1.0
            
        existence_scores = []
        
        for i, step in enumerate(planning_trace):
            world_state = world_graph_sequence[i] if i < len(world_graph_sequence) else {}
            
            # 提取步骤中提到的对象
            mentioned_objects = self._extract_mentioned_objects(step)
            
            # 检查对象是否存在
            if mentioned_objects:
                existing_objects = self._get_existing_objects(world_state)
                correct_count = sum(1 for obj in mentioned_objects 
                                  if self._object_exists(obj, existing_objects))
                step_score = correct_count / len(mentioned_objects)
            else:
                step_score = 1.0  # 没有提到对象，默认正确
                
            existence_scores.append(step_score)
        
        return np.mean(existence_scores) if existence_scores else 1.0
    
    def _evaluate_spatial_accuracy(self,
                                 planning_trace: List[str],
                                 world_graph_sequence: List[Dict[str, Any]]) -> float:
        """
        评估空间关系描述的准确性
        """
        if not planning_trace or not world_graph_sequence:
            return 1.0
            
        spatial_scores = []
        
        for i, step in enumerate(planning_trace):
            world_state = world_graph_sequence[i] if i < len(world_graph_sequence) else {}
            
            # 提取空间关系陈述
            spatial_claims = self._extract_spatial_claims(step)
            
            # 验证空间关系
            if spatial_claims:
                correct_count = sum(1 for claim in spatial_claims
                                  if self._verify_spatial_claim(claim, world_state))
                step_score = correct_count / len(spatial_claims)
            else:
                step_score = 1.0  # 没有空间关系陈述，默认正确
                
            spatial_scores.append(step_score)
        
        return np.mean(spatial_scores) if spatial_scores else 1.0
    
    def _evaluate_temporal_consistency(self,
                                     planning_trace: List[str],
                                     world_graph_sequence: List[Dict[str, Any]]) -> float:
        """
        评估时序描述的一致性
        """
        if len(planning_trace) < 2 or len(world_graph_sequence) < 2:
            return 1.0
            
        temporal_scores = []
        
        for i in range(1, len(planning_trace)):
            prev_step = planning_trace[i-1]
            curr_step = planning_trace[i]
            prev_state = world_graph_sequence[i-1] if i-1 < len(world_graph_sequence) else {}
            curr_state = world_graph_sequence[i] if i < len(world_graph_sequence) else {}
            
            # 检查时序一致性
            temporal_score = self._check_temporal_consistency(
                prev_step, curr_step, prev_state, curr_state
            )
            temporal_scores.append(temporal_score)
        
        return np.mean(temporal_scores) if temporal_scores else 1.0
    
    def _extract_factual_claims(self, planning_step: str) -> List[Dict[str, Any]]:
        """从规划步骤中提取事实陈述"""
        claims = []
        
        # 提取思考过程中的陈述
        thought_pattern = r"Thought:\s*(.*?)(?=\n|Agent_|$)"
        thought_match = re.search(thought_pattern, planning_step, re.DOTALL)
        
        if thought_match:
            thought_text = thought_match.group(1)
            
            # 查找对象位置陈述
            location_patterns = [
                r"(\w+)\s+(?:is|are)\s+(?:on|in|at|near)\s+(\w+)",
                r"(\w+)\s+(?:located|placed|positioned)\s+(?:on|in|at|near)\s+(\w+)",
                r"I can see\s+(\w+)\s+(?:on|in|at|near)\s+(\w+)"
            ]
            
            for pattern in location_patterns:
                matches = re.findall(pattern, thought_text, re.IGNORECASE)
                for match in matches:
                    claims.append({
                        'type': 'location',
                        'object': match[0],
                        'location': match[1],
                        'text': f"{match[0]} is at {match[1]}"
                    })
        
        return claims
    
    def _verify_factual_claims(self, 
                             claims: List[Dict[str, Any]], 
                             world_state: Dict[str, Any]) -> float:
        """验证事实陈述的准确性"""
        if not claims:
            return 1.0
            
        correct_count = 0
        
        for claim in claims:
            if claim['type'] == 'location':
                if self._verify_location_claim(claim, world_state):
                    correct_count += 1
        
        return correct_count / len(claims)
    
    def _verify_location_claim(self, 
                             claim: Dict[str, Any], 
                             world_state: Dict[str, Any]) -> bool:
        """验证位置陈述的准确性"""
        obj_name = claim['object'].lower()
        location_name = claim['location'].lower()
        
        # 从世界状态中查找对象和位置
        objects = world_state.get('objects', {})
        furniture = world_state.get('furniture', {})
        
        # 检查对象是否在指定位置
        for obj_id, obj_info in objects.items():
            if obj_name in obj_info.get('name', '').lower():
                obj_location = obj_info.get('parent', '')
                
                # 检查位置匹配
                for furn_id, furn_info in furniture.items():
                    if (location_name in furn_info.get('name', '').lower() and 
                        furn_id == obj_location):
                        return True
        
        return False
    
    def _compare_object_positions(self, 
                                wg_state: Dict[str, Any], 
                                gt_state: Dict[str, Any]) -> float:
        """比较对象位置的一致性"""
        if not wg_state or not gt_state:
            return 1.0
            
        wg_objects = wg_state.get('objects', {})
        gt_objects = gt_state.get('objects', {})
        
        if not wg_objects or not gt_objects:
            return 1.0
            
        position_scores = []
        
        for obj_id in wg_objects:
            if obj_id in gt_objects:
                wg_pos = wg_objects[obj_id].get('position', [0, 0, 0])
                gt_pos = gt_objects[obj_id].get('position', [0, 0, 0])
                
                # 计算位置误差
                distance = np.linalg.norm(np.array(wg_pos) - np.array(gt_pos))
                
                # 基于误差计算得分
                if distance <= self.position_tolerance:
                    score = 1.0
                else:
                    score = max(0.0, 1.0 - (distance - self.position_tolerance) / self.position_tolerance)
                
                position_scores.append(score)
        
        return np.mean(position_scores) if position_scores else 1.0
    
    def _compare_object_states(self, 
                             wg_state: Dict[str, Any], 
                             gt_state: Dict[str, Any]) -> float:
        """比较对象状态的一致性"""
        if not wg_state or not gt_state:
            return 1.0
            
        wg_objects = wg_state.get('objects', {})
        gt_objects = gt_state.get('objects', {})
        
        if not wg_objects or not gt_objects:
            return 1.0
            
        state_scores = []
        
        for obj_id in wg_objects:
            if obj_id in gt_objects:
                wg_states = wg_objects[obj_id].get('states', {})
                gt_states = gt_objects[obj_id].get('states', {})
                
                # 比较状态属性
                common_states = set(wg_states.keys()) & set(gt_states.keys())
                
                if common_states:
                    correct_states = sum(1 for state in common_states
                                       if wg_states[state] == gt_states[state])
                    score = correct_states / len(common_states)
                else:
                    score = 1.0  # 没有共同状态，默认一致
                
                state_scores.append(score)
        
        return np.mean(state_scores) if state_scores else 1.0
    
    def _compare_spatial_relations(self, 
                                 wg_state: Dict[str, Any], 
                                 gt_state: Dict[str, Any]) -> float:
        """比较空间关系的一致性"""
        if not wg_state or not gt_state:
            return 1.0
            
        wg_relations = wg_state.get('spatial_relations', [])
        gt_relations = gt_state.get('spatial_relations', [])
        
        if not wg_relations and not gt_relations:
            return 1.0
            
        # 计算关系匹配度
        matched_relations = 0
        total_relations = max(len(wg_relations), len(gt_relations))
        
        for wg_rel in wg_relations:
            for gt_rel in gt_relations:
                if self._relations_match(wg_rel, gt_rel):
                    matched_relations += 1
                    break
        
        return matched_relations / total_relations if total_relations > 0 else 1.0
    
    def _extract_mentioned_objects(self, planning_step: str) -> List[str]:
        """提取规划步骤中提到的对象"""
        objects = []
        
        # 从动作中提取对象
        action_pattern = r"Agent_\d+_Action:\s*(.*?)(?=\n|$)"
        action_match = re.search(action_pattern, planning_step)
        
        if action_match:
            action_text = action_match.group(1)
            
            # 查找动作参数中的对象
            object_patterns = [
                r"Pick\[([^\]]+)\]",
                r"Place\[([^,]+),",
                r"Navigate\[([^\]]+)\]",
                r"Find\[([^\]]+)\]"
            ]
            
            for pattern in object_patterns:
                matches = re.findall(pattern, action_text)
                objects.extend(matches)
        
        return [obj.strip() for obj in objects]
    
    def _get_existing_objects(self, world_state: Dict[str, Any]) -> Set[str]:
        """获取世界状态中存在的对象"""
        existing_objects = set()
        
        objects = world_state.get('objects', {})
        for obj_info in objects.values():
            obj_name = obj_info.get('name', '')
            if obj_name:
                existing_objects.add(obj_name.lower())
        
        return existing_objects
    
    def _object_exists(self, mentioned_obj: str, existing_objects: Set[str]) -> bool:
        """检查提到的对象是否存在"""
        mentioned_obj = mentioned_obj.lower().strip()
        
        # 精确匹配
        if mentioned_obj in existing_objects:
            return True
            
        # 模糊匹配（包含关系）
        for existing_obj in existing_objects:
            if mentioned_obj in existing_obj or existing_obj in mentioned_obj:
                return True
                
        return False
    
    def _extract_spatial_claims(self, planning_step: str) -> List[Dict[str, Any]]:
        """提取空间关系陈述"""
        spatial_claims = []
        
        # 从思考过程中提取空间关系
        thought_pattern = r"Thought:\s*(.*?)(?=\n|Agent_|$)"
        thought_match = re.search(thought_pattern, planning_step, re.DOTALL)
        
        if thought_match:
            thought_text = thought_match.group(1)
            
            # 查找空间关系描述
            spatial_patterns = [
                r"(\w+)\s+is\s+(on|in|near|under|above)\s+(\w+)",
                r"(\w+)\s+(?:placed|located|positioned)\s+(on|in|near|under|above)\s+(\w+)"
            ]
            
            for pattern in spatial_patterns:
                matches = re.findall(pattern, thought_text, re.IGNORECASE)
                for match in matches:
                    spatial_claims.append({
                        'object1': match[0],
                        'relation': match[1],
                        'object2': match[2]
                    })
        
        return spatial_claims
    
    def _verify_spatial_claim(self, 
                            claim: Dict[str, Any], 
                            world_state: Dict[str, Any]) -> bool:
        """验证空间关系陈述"""
        obj1 = claim['object1'].lower()
        relation = claim['relation'].lower()
        obj2 = claim['object2'].lower()
        
        # 从世界状态中获取空间关系
        spatial_relations = world_state.get('spatial_relations', [])
        
        for rel in spatial_relations:
            if (obj1 in rel.get('object1', '').lower() and
                obj2 in rel.get('object2', '').lower() and
                relation in rel.get('relation', '').lower()):
                return True
        
        return False
    
    def _check_temporal_consistency(self, 
                                  prev_step: str, 
                                  curr_step: str,
                                  prev_state: Dict[str, Any], 
                                  curr_state: Dict[str, Any]) -> float:
        """检查时序一致性"""
        # 提取前后步骤的动作
        prev_action = self._extract_action(prev_step)
        curr_action = self._extract_action(curr_step)
        
        if not prev_action or not curr_action:
            return 1.0
            
        # 检查动作的因果关系是否合理
        # 例如：如果前一步是Pick，那么对象应该从原位置消失
        if 'pick' in prev_action.lower():
            # 检查对象是否被正确移动
            return self._verify_pick_effect(prev_action, prev_state, curr_state)
        elif 'place' in prev_action.lower():
            # 检查对象是否被正确放置
            return self._verify_place_effect(prev_action, prev_state, curr_state)
        
        return 1.0  # 其他情况默认一致
    
    def _extract_action(self, planning_step: str) -> str:
        """提取动作描述"""
        action_pattern = r"Agent_\d+_Action:\s*(.*?)(?=\n|$)"
        match = re.search(action_pattern, planning_step)
        return match.group(1).strip() if match else ""
    
    def _verify_pick_effect(self, 
                          action: str, 
                          prev_state: Dict[str, Any], 
                          curr_state: Dict[str, Any]) -> float:
        """验证Pick动作的效果"""
        # 简化的验证逻辑
        # 实际应用中需要更详细的状态比较
        return 0.8  # 默认评分
    
    def _verify_place_effect(self, 
                           action: str, 
                           prev_state: Dict[str, Any], 
                           curr_state: Dict[str, Any]) -> float:
        """验证Place动作的效果"""
        # 简化的验证逻辑
        # 实际应用中需要更详细的状态比较
        return 0.8  # 默认评分
    
    def _relations_match(self, rel1: Dict[str, Any], rel2: Dict[str, Any]) -> bool:
        """检查两个空间关系是否匹配"""
        return (rel1.get('object1', '').lower() == rel2.get('object1', '').lower() and
                rel1.get('object2', '').lower() == rel2.get('object2', '').lower() and
                rel1.get('relation', '').lower() == rel2.get('relation', '').lower()) 