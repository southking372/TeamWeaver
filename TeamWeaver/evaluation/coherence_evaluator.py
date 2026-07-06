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
    """Contextual consistency evaluation metrics"""
    semantic_coherence: float  #semantic coherence[0,1]
    temporal_coherence: float  #temporal coherence[0,1] 
    action_coherence: float    #action continuity[0,1]
    overall_coherence: float   #overall coherence[0,1]
    
class CoherenceEvaluator:
    """
context consistency evaluator
for evaluationLLMCoherence of the generated planning sequence at the semantic, temporal and action levels
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
Initialize evaluator
        
        Args:
            model_name:Sentence embedding model for semantic similarity calculation
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
Assessing the contextual consistency of planning trajectories
        
        Args:
            planning_trace: LLMGenerated sequence of planning steps
            world_state_sequence:The corresponding world state sequence
            miqp_assignments: MIQPOptimizer's task allocation result sequence
            
        Returns:
            CoherenceMetrics:Consistency assessment results
        """
        #1. Semantic coherence assessment
        semantic_score = self._evaluate_semantic_coherence(planning_trace)
        
        #2. Timing continuity assessment
        temporal_score = self._evaluate_temporal_coherence(
            planning_trace, world_state_sequence
        )
        
        #3. Action coherence assessment
        action_score = self._evaluate_action_coherence(
            planning_trace, miqp_assignments
        )
        
        #4. Calculate overall coherence (weighted average)
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
Assessing semantic coherence
By computing semantic similarity between adjacent planning steps
        """
        if len(planning_trace) < 2:
            return 1.0
            
        #Extract thought process and action descriptions
        thoughts = []
        actions = []
        
        for step in planning_trace:
            thought = self._extract_thought(step)
            action = self._extract_action(step)
            if thought:
                thoughts.append(thought)
            if action:
                actions.append(action)
        
        #Semantic coherence of computational thinking processes
        thought_coherence = self._calculate_sequence_coherence(thoughts)
        
        #Compute semantic coherence of action sequences
        action_coherence = self._calculate_sequence_coherence(actions)
        
        #weighted average
        return 0.6 * thought_coherence + 0.4 * action_coherence
    
    def _evaluate_temporal_coherence(self, 
                                   planning_trace: List[str],
                                   world_state_sequence: List[Dict[str, Any]]) -> float:
        """
Assessing temporal coherence
Check whether planning steps comply with temporal logic and cause-and-effect relationships
        """
        if len(planning_trace) < 2:
            return 1.0
            
        coherence_scores = []
        
        for i in range(1, len(planning_trace)):
            prev_step = planning_trace[i-1]
            curr_step = planning_trace[i]
            
            #Check whether the preconditions are met
            precondition_score = self._check_preconditions(
                prev_step, curr_step, 
                world_state_sequence[i-1] if i-1 < len(world_state_sequence) else {}
            )
            
            #Check the logic of action sequences
            logical_score = self._check_action_logic(prev_step, curr_step)
            
            step_score = 0.5 * precondition_score + 0.5 * logical_score
            coherence_scores.append(step_score)
        
        return np.mean(coherence_scores) if coherence_scores else 1.0
    
    def _evaluate_action_coherence(self, 
                                 planning_trace: List[str],
                                 miqp_assignments: Optional[List[np.ndarray]]) -> float:
        """
Evaluate movement coherence
Check whether the action sequence is reasonable and related toMIQPAllocation consistency
        """
        if len(planning_trace) < 2:
            return 1.0
            
        #Extract action sequence
        actions = [self._extract_action(step) for step in planning_trace]
        actions = [a for a in actions if a]  #filter null values
        
        if len(actions) < 2:
            return 1.0
        
        #1. Reasonability of action type conversion
        transition_score = self._evaluate_action_transitions(actions)
        
        #2. withMIQPAllocation consistency (if provided)
        miqp_consistency = 1.0
        if miqp_assignments:
            miqp_consistency = self._evaluate_miqp_consistency(
                actions, miqp_assignments
            )
        
        return 0.7 * transition_score + 0.3 * miqp_consistency
    
    def _calculate_sequence_coherence(self, text_sequence: List[str]) -> float:
        """Compute semantic coherence of text sequences"""
        if len(text_sequence) < 2:
            return 1.0
            
        #Generate sentence embeddings
        embeddings = self.sentence_model.encode(text_sequence)
        
        #Calculate cosine similarity of adjacent sentences
        similarities = []
        for i in range(1, len(embeddings)):
            sim = cosine_similarity(
                embeddings[i-1].reshape(1, -1),
                embeddings[i].reshape(1, -1)
            )[0][0]
            similarities.append(sim)
        
        #Returns the average similarity
        return np.mean(similarities) if similarities else 1.0
    
    def _extract_thought(self, planning_step: str) -> str:
        """Extract the thought process from the planning steps"""
        #match"Thought:"content after
        thought_pattern = r"Thought:\s*(.*?)(?=\n|Agent_|$)"
        match = re.search(thought_pattern, planning_step, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _extract_action(self, planning_step: str) -> str:
        """Extract action descriptions from planning steps"""
        #match"Agent_X_Action:"content after
        action_pattern = r"Agent_\d+_Action:\s*(.*?)(?=\n|$)"
        match = re.search(action_pattern, planning_step)
        return match.group(1).strip() if match else ""
    
    def _check_preconditions(self, 
                           prev_step: str, 
                           curr_step: str,
                           world_state: Dict[str, Any]) -> float:
        """Check whether the preconditions of the action are met"""
        curr_action = self._extract_action(curr_step)
        if not curr_action:
            return 1.0
            
        #Check preconditions for navigation actions
        if any(nav in curr_action.lower() for nav in self.action_patterns['navigation']):
            #Navigation actions usually do not require special preconditions
            return 1.0
            
        #examinemanipulationAction preconditions
        if any(manip in curr_action.lower() for manip in self.action_patterns['manipulation']):
            # manipulationThe action requires the target object to exist and be reachable
            return 0.8  #Simplified scoring
            
        return 1.0
    
    def _check_action_logic(self, prev_step: str, curr_step: str) -> float:
        """Check the logic of action sequences"""
        prev_action = self._extract_action(prev_step)
        curr_action = self._extract_action(curr_step)
        
        if not prev_action or not curr_action:
            return 1.0
            
        #Check the plausibility of action sequences
        #For example:pickAfter that it should beplaceornavigate, instead of againpicksame object
        
        prev_type = self._classify_action_type(prev_action)
        curr_type = self._classify_action_type(curr_action)
        
        #Define reasonable action transitions
        valid_transitions = {
            'navigation': ['manipulation', 'observation', 'navigation'],
            'manipulation': ['navigation', 'manipulation', 'communication'],
            'observation': ['navigation', 'manipulation', 'observation'],
            'communication': ['navigation', 'observation', 'communication']
        }
        
        if curr_type in valid_transitions.get(prev_type, []):
            return 1.0
        else:
            return 0.5  #Unreasonable but not completely wrong
    
    def _classify_action_type(self, action: str) -> str:
        """Classification action types"""
        action_lower = action.lower()
        
        for action_type, keywords in self.action_patterns.items():
            if any(keyword in action_lower for keyword in keywords):
                return action_type
                
        return 'unknown'
    
    def _evaluate_action_transitions(self, actions: List[str]) -> float:
        """Evaluate the plausibility of action transitions"""
        if len(actions) < 2:
            return 1.0
            
        transition_scores = []
        
        for i in range(1, len(actions)):
            prev_type = self._classify_action_type(actions[i-1])
            curr_type = self._classify_action_type(actions[i])
            
            #Plausibility scoring based on action type conversion
            if prev_type == curr_type:
                score = 0.8  #Continuous execution of the same type of actions is generally reasonable
            elif (prev_type == 'navigation' and curr_type == 'manipulation') or \
                 (prev_type == 'manipulation' and curr_type == 'navigation'):
                score = 1.0  #navigation andmanipulationAlternate, very reasonable
            elif prev_type == 'observation' and curr_type in ['navigation', 'manipulation']:
                score = 0.9  #Perform actions after observation, reasonable
            else:
                score = 0.6  #Other conversions are basically reasonable
                
            transition_scores.append(score)
        
        return np.mean(transition_scores)
    
    def _evaluate_miqp_consistency(self, 
                                 actions: List[str],
                                 miqp_assignments: List[np.ndarray]) -> float:
        """Assessment andMIQPAllocation consistency"""
        if not miqp_assignments or len(actions) != len(miqp_assignments):
            return 1.0  #if notMIQPData, consistent by default
            
        consistency_scores = []
        
        for i, (action, assignment) in enumerate(zip(actions, miqp_assignments)):
            #Check if the action matchesMIQPThe assigned tasks are consistent
            #Here it is necessary to base on specificMIQPAdjust the output format
            
            #Simplified consistency check:
            #ifMIQPTask assigned (assignmentcontains non-zero elements),
            #Then there should be corresponding actions to execute
            has_assignment = np.any(assignment > 0.1) if assignment is not None else False
            has_action = action and action.strip() != "Wait[]"
            
            if has_assignment == has_action:
                consistency_scores.append(1.0)
            else:
                consistency_scores.append(0.5)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0 
