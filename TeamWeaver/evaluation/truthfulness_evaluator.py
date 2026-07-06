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
    """factual consistency evaluation index"""
    factual_accuracy: float      #factual accuracy[0,1]
    world_consistency: float     #world state consistency[0,1]
    object_existence: float      #Object existence accuracy[0,1]
    spatial_accuracy: float      #Spatial relationship accuracy[0,1]
    temporal_consistency: float  #Timing consistency[0,1]
    overall_truthfulness: float  #overall authenticity[0,1]

class TruthfulnessEvaluator:
    """
fact consistency evaluator
based onHabitat-SimofWorld GraphAs a source of fact, evaluateLLMAuthenticity of generated content
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.spatial_relations = {
            'on': ['on_top', 'placed_on', 'sitting_on'],
            'in': ['inside', 'within', 'contained_in'],
            'near': ['next_to', 'close_to', 'beside'],
            'under': ['underneath', 'below'],
            'above': ['over', 'on_top_of']
        }
        
        #Define acceptable error range
        self.position_tolerance = 0.5  #Position error tolerance (meters)
        self.angle_tolerance = 0.2     #Angle error tolerance (radians)
        
    def evaluate_truthfulness(self,
                            planning_trace: List[str],
                            world_graph_sequence: List[Dict[str, Any]],
                            ground_truth_states: List[Dict[str, Any]]) -> TruthfulnessMetrics:
        """
Assess the factual consistency of planning trajectories
        
        Args:
            planning_trace: LLMGenerated sequence of planning steps
            world_graph_sequence:The corresponding world graph state sequence
            ground_truth_states:Real state sequence of simulation environment
            
        Returns:
            TruthfulnessMetrics:factual consistency assessment results
        """
        #1. Assessment of factual accuracy
        factual_score = self._evaluate_factual_accuracy(
            planning_trace, world_graph_sequence
        )
        
        #2. World state consistency assessment
        world_consistency = self._evaluate_world_consistency(
            world_graph_sequence, ground_truth_states
        )
        
        #3. Object existence assessment
        object_existence = self._evaluate_object_existence(
            planning_trace, world_graph_sequence
        )
        
        #4. Assessment of spatial relationship accuracy
        spatial_accuracy = self._evaluate_spatial_accuracy(
            planning_trace, world_graph_sequence
        )
        
        #5. Timing consistency assessment
        temporal_consistency = self._evaluate_temporal_consistency(
            planning_trace, world_graph_sequence
        )
        
        #6. Calculate overall authenticity (weighted average)
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
EvaluateLLMAccuracy of factual statements in generated content
        """
        if not planning_trace or not world_graph_sequence:
            return 1.0
            
        accuracy_scores = []
        
        for i, step in enumerate(planning_trace):
            world_state = world_graph_sequence[i] if i < len(world_graph_sequence) else {}
            
            #Statement of facts in the extraction step
            factual_claims = self._extract_factual_claims(step)
            
            #Verify each factual statement
            step_accuracy = self._verify_factual_claims(factual_claims, world_state)
            accuracy_scores.append(step_accuracy)
        
        return np.mean(accuracy_scores) if accuracy_scores else 1.0
    
    def _evaluate_world_consistency(self,
                                  world_graph_sequence: List[Dict[str, Any]],
                                  ground_truth_states: List[Dict[str, Any]]) -> float:
        """
Evaluate the consistency of the world graph with the real state
        """
        if not world_graph_sequence or not ground_truth_states:
            return 1.0
            
        consistency_scores = []
        
        for wg_state, gt_state in zip(world_graph_sequence, ground_truth_states):
            #Compare object position
            position_consistency = self._compare_object_positions(wg_state, gt_state)
            
            #Compare object status
            state_consistency = self._compare_object_states(wg_state, gt_state)
            
            #compare spatial relationships
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
EvaluateLLMWhether the object mentioned actually exists
        """
        if not planning_trace or not world_graph_sequence:
            return 1.0
            
        existence_scores = []
        
        for i, step in enumerate(planning_trace):
            world_state = world_graph_sequence[i] if i < len(world_graph_sequence) else {}
            
            #Objects mentioned in the extraction step
            mentioned_objects = self._extract_mentioned_objects(step)
            
            #Check if object exists
            if mentioned_objects:
                existing_objects = self._get_existing_objects(world_state)
                correct_count = sum(1 for obj in mentioned_objects 
                                  if self._object_exists(obj, existing_objects))
                step_score = correct_count / len(mentioned_objects)
            else:
                step_score = 1.0  #No object mentioned, default is correct
                
            existence_scores.append(step_score)
        
        return np.mean(existence_scores) if existence_scores else 1.0
    
    def _evaluate_spatial_accuracy(self,
                                 planning_trace: List[str],
                                 world_graph_sequence: List[Dict[str, Any]]) -> float:
        """
Evaluate the accuracy of spatial relationship descriptions
        """
        if not planning_trace or not world_graph_sequence:
            return 1.0
            
        spatial_scores = []
        
        for i, step in enumerate(planning_trace):
            world_state = world_graph_sequence[i] if i < len(world_graph_sequence) else {}
            
            #Extract spatial relationship statements
            spatial_claims = self._extract_spatial_claims(step)
            
            #Verify spatial relationships
            if spatial_claims:
                correct_count = sum(1 for claim in spatial_claims
                                  if self._verify_spatial_claim(claim, world_state))
                step_score = correct_count / len(spatial_claims)
            else:
                step_score = 1.0  #If there is no spatial relationship statement, the default is correct.
                
            spatial_scores.append(step_score)
        
        return np.mean(spatial_scores) if spatial_scores else 1.0
    
    def _evaluate_temporal_consistency(self,
                                     planning_trace: List[str],
                                     world_graph_sequence: List[Dict[str, Any]]) -> float:
        """
Evaluate the consistency of timing descriptions
        """
        if len(planning_trace) < 2 or len(world_graph_sequence) < 2:
            return 1.0
            
        temporal_scores = []
        
        for i in range(1, len(planning_trace)):
            prev_step = planning_trace[i-1]
            curr_step = planning_trace[i]
            prev_state = world_graph_sequence[i-1] if i-1 < len(world_graph_sequence) else {}
            curr_state = world_graph_sequence[i] if i < len(world_graph_sequence) else {}
            
            #Check timing consistency
            temporal_score = self._check_temporal_consistency(
                prev_step, curr_step, prev_state, curr_state
            )
            temporal_scores.append(temporal_score)
        
        return np.mean(temporal_scores) if temporal_scores else 1.0
    
    def _extract_factual_claims(self, planning_step: str) -> List[Dict[str, Any]]:
        """Extract factual statements from planning steps"""
        claims = []
        
        #Extract statements from the thinking process
        thought_pattern = r"Thought:\s*(.*?)(?=\n|Agent_|$)"
        thought_match = re.search(thought_pattern, planning_step, re.DOTALL)
        
        if thought_match:
            thought_text = thought_match.group(1)
            
            #Find object location statement
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
        """Verify the accuracy of factual statements"""
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
        """Verify the accuracy of location statements"""
        obj_name = claim['object'].lower()
        location_name = claim['location'].lower()
        
        #Find objects and locations from the world state
        objects = world_state.get('objects', {})
        furniture = world_state.get('furniture', {})
        
        #Check if the object is at the specified location
        for obj_id, obj_info in objects.items():
            if obj_name in obj_info.get('name', '').lower():
                obj_location = obj_info.get('parent', '')
                
                #Check location match
                for furn_id, furn_info in furniture.items():
                    if (location_name in furn_info.get('name', '').lower() and 
                        furn_id == obj_location):
                        return True
        
        return False
    
    def _compare_object_positions(self, 
                                wg_state: Dict[str, Any], 
                                gt_state: Dict[str, Any]) -> float:
        """Compare object positions for consistency"""
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
                
                #Calculate position error
                distance = np.linalg.norm(np.array(wg_pos) - np.array(gt_pos))
                
                #Calculate score based on error
                if distance <= self.position_tolerance:
                    score = 1.0
                else:
                    score = max(0.0, 1.0 - (distance - self.position_tolerance) / self.position_tolerance)
                
                position_scores.append(score)
        
        return np.mean(position_scores) if position_scores else 1.0
    
    def _compare_object_states(self, 
                             wg_state: Dict[str, Any], 
                             gt_state: Dict[str, Any]) -> float:
        """Compare object states for consistency"""
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
                
                #Compare status attributes
                common_states = set(wg_states.keys()) & set(gt_states.keys())
                
                if common_states:
                    correct_states = sum(1 for state in common_states
                                       if wg_states[state] == gt_states[state])
                    score = correct_states / len(common_states)
                else:
                    score = 1.0  #There is no common state and the default is the same.
                
                state_scores.append(score)
        
        return np.mean(state_scores) if state_scores else 1.0
    
    def _compare_spatial_relations(self, 
                                 wg_state: Dict[str, Any], 
                                 gt_state: Dict[str, Any]) -> float:
        """Compare spatial relationships for consistency"""
        if not wg_state or not gt_state:
            return 1.0
            
        wg_relations = wg_state.get('spatial_relations', [])
        gt_relations = gt_state.get('spatial_relations', [])
        
        if not wg_relations and not gt_relations:
            return 1.0
            
        #Calculate relationship matching degree
        matched_relations = 0
        total_relations = max(len(wg_relations), len(gt_relations))
        
        for wg_rel in wg_relations:
            for gt_rel in gt_relations:
                if self._relations_match(wg_rel, gt_rel):
                    matched_relations += 1
                    break
        
        return matched_relations / total_relations if total_relations > 0 else 1.0
    
    def _extract_mentioned_objects(self, planning_step: str) -> List[str]:
        """Extract objects mentioned in the planning step"""
        objects = []
        
        #Extract objects from actions
        action_pattern = r"Agent_\d+_Action:\s*(.*?)(?=\n|$)"
        action_match = re.search(action_pattern, planning_step)
        
        if action_match:
            action_text = action_match.group(1)
            
            #Find objects in action parameters
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
        """Get the objects present in the world state"""
        existing_objects = set()
        
        objects = world_state.get('objects', {})
        for obj_info in objects.values():
            obj_name = obj_info.get('name', '')
            if obj_name:
                existing_objects.add(obj_name.lower())
        
        return existing_objects
    
    def _object_exists(self, mentioned_obj: str, existing_objects: Set[str]) -> bool:
        """Check if the mentioned object exists"""
        mentioned_obj = mentioned_obj.lower().strip()
        
        #exact match
        if mentioned_obj in existing_objects:
            return True
            
        #Fuzzy matching (inclusion relationship)
        for existing_obj in existing_objects:
            if mentioned_obj in existing_obj or existing_obj in mentioned_obj:
                return True
                
        return False
    
    def _extract_spatial_claims(self, planning_step: str) -> List[Dict[str, Any]]:
        """Extract spatial relationship statements"""
        spatial_claims = []
        
        #Extract spatial relationships from the thinking process
        thought_pattern = r"Thought:\s*(.*?)(?=\n|Agent_|$)"
        thought_match = re.search(thought_pattern, planning_step, re.DOTALL)
        
        if thought_match:
            thought_text = thought_match.group(1)
            
            #Find spatial relationship descriptions
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
        """Verify spatial relationship statements"""
        obj1 = claim['object1'].lower()
        relation = claim['relation'].lower()
        obj2 = claim['object2'].lower()
        
        #Get spatial relationships from world state
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
        """Check timing consistency"""
        #Extract the actions of the previous and next steps
        prev_action = self._extract_action(prev_step)
        curr_action = self._extract_action(curr_step)
        
        if not prev_action or not curr_action:
            return 1.0
            
        #Check whether the cause and effect relationship of the action is reasonable
        #For example: if the previous step wasPick, then the object should disappear from its original position
        if 'pick' in prev_action.lower():
            #Check if the object is correctlymovement
            return self._verify_pick_effect(prev_action, prev_state, curr_state)
        elif 'place' in prev_action.lower():
            #Check if the object is placed correctly
            return self._verify_place_effect(prev_action, prev_state, curr_state)
        
        return 1.0  #In other cases, the default is the same
    
    def _extract_action(self, planning_step: str) -> str:
        """Extract action description"""
        action_pattern = r"Agent_\d+_Action:\s*(.*?)(?=\n|$)"
        match = re.search(action_pattern, planning_step)
        return match.group(1).strip() if match else ""
    
    def _verify_pick_effect(self, 
                          action: str, 
                          prev_state: Dict[str, Any], 
                          curr_state: Dict[str, Any]) -> float:
        """verifyPickaction effect"""
        #Simplified validation logic
        #Practical applications require more detailed status comparisons
        return 0.8  #Default rating
    
    def _verify_place_effect(self, 
                           action: str, 
                           prev_state: Dict[str, Any], 
                           curr_state: Dict[str, Any]) -> float:
        """verifyPlaceaction effect"""
        #Simplified validation logic
        #Practical applications require more detailed status comparisons
        return 0.8  #Default rating
    
    def _relations_match(self, rel1: Dict[str, Any], rel2: Dict[str, Any]) -> bool:
        """Check if two spatial relationships match"""
        return (rel1.get('object1', '').lower() == rel2.get('object1', '').lower() and
                rel1.get('object2', '').lower() == rel2.get('object2', '').lower() and
                rel1.get('relation', '').lower() == rel2.get('relation', '').lower()) 
