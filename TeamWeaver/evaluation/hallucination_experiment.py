# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

from habitat_llm.evaluation.coherence_evaluator import CoherenceEvaluator, CoherenceMetrics
from habitat_llm.evaluation.truthfulness_evaluator import TruthfulnessEvaluator, TruthfulnessMetrics

@dataclass
class HallucinationMetrics:
    """Comprehensive evaluation index for large model hallucinations"""
    coherence_metrics: CoherenceMetrics
    truthfulness_metrics: TruthfulnessMetrics
    
    #Comprehensive indicators
    hallucination_score: float  #degree of hallucination[0,1], the lower the better
    reliability_score: float    #reliability score[0,1], the higher the better
    
    #Soft constraint optimization effect
    miqp_improvement: float     # MIQPImprovements brought about by optimization[0,1]
    constraint_satisfaction: float  #constraint satisfaction[0,1]

@dataclass
class ExperimentConfig:
    """Experimental configuration"""
    #Dataset configuration
    dataset_path: str
    num_episodes: int = 100
    
    #Evaluate configuration
    enable_coherence: bool = True
    enable_truthfulness: bool = True
    enable_miqp_analysis: bool = True
    
    #Compare experimental configurations
    baseline_methods: List[str] = None  # ['vanilla_llm', 'cot', 'rag']
    
    #Output configuration
    output_dir: str = "hallucination_results"
    save_detailed_logs: bool = True
    generate_plots: bool = True

class HallucinationExperiment:
    """
Experimental framework for large model hallucination assessment
Integrate contextual consistency and factual consistency assessments
    """
    
    def __init__(self, config: ExperimentConfig):
        """
Initialize the experimental framework
        
        Args:
            config:Experimental configuration
        """
        self.config = config
        self.coherence_evaluator = CoherenceEvaluator() if config.enable_coherence else None
        self.truthfulness_evaluator = TruthfulnessEvaluator() if config.enable_truthfulness else None
        
        #Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        #Setup log
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'experiment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        #Store experimental results
        self.results = []
        self.baseline_results = {}
        
    def run_experiment(self, 
                      planning_traces: Dict[str, List[str]],
                      world_graph_sequences: Dict[str, List[Dict[str, Any]]],
                      ground_truth_sequences: Dict[str, List[Dict[str, Any]]],
                      miqp_data: Optional[Dict[str, List[np.ndarray]]] = None) -> Dict[str, Any]:
        """
Run a complete hallucination assessment experiment
        
        Args:
            planning_traces:Planning trajectories for each method{method_name: [trace_steps]}
            world_graph_sequences:world graph sequence{method_name: [world_states]}
            ground_truth_sequences:true state sequence{method_name: [gt_states]}
            miqp_data: MIQPOptimize data{method_name: [assignments]}
            
        Returns:
            Dict:Summary of experimental results
        """
        self.logger.info("Begin large-model illusion evaluation experiment")
        
        #1. Evaluate the degree of hallucination of each method
        for method_name in planning_traces.keys():
            self.logger.info(f"Assessment method: {method_name}")
            
            method_results = self._evaluate_method(
                method_name=method_name,
                planning_trace=planning_traces[method_name],
                world_graph_sequence=world_graph_sequences[method_name],
                ground_truth_sequence=ground_truth_sequences[method_name],
                miqp_assignments=miqp_data.get(method_name) if miqp_data else None
            )
            
            self.results.append(method_results)
        
        #2. Comparative analysis
        comparison_results = self._compare_methods()
        
        # 3. MIQPOptimization effect analysis
        miqp_analysis = {}
        if self.config.enable_miqp_analysis and miqp_data:
            miqp_analysis = self._analyze_miqp_effects(miqp_data)
        
        #4. Generate reports
        final_results = {
            'individual_results': self.results,
            'comparison_analysis': comparison_results,
            'miqp_analysis': miqp_analysis,
            'summary_statistics': self._generate_summary_statistics()
        }
        
        #5. Save results and generate visualizations
        self._save_results(final_results)
        
        if self.config.generate_plots:
            self._generate_visualizations(final_results)
        
        self.logger.info("Experiment completed")
        return final_results
    
    def _evaluate_method(self,
                        method_name: str,
                        planning_trace: List[str],
                        world_graph_sequence: List[Dict[str, Any]],
                        ground_truth_sequence: List[Dict[str, Any]],
                        miqp_assignments: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """Evaluate the degree of hallucination of an individual method"""
        
        #1. Contextual consistency assessment
        coherence_metrics = None
        if self.coherence_evaluator:
            coherence_metrics = self.coherence_evaluator.evaluate_coherence(
                planning_trace, world_graph_sequence, miqp_assignments
            )
        
        #2. Factual consistency assessment
        truthfulness_metrics = None
        if self.truthfulness_evaluator:
            truthfulness_metrics = self.truthfulness_evaluator.evaluate_truthfulness(
                planning_trace, world_graph_sequence, ground_truth_sequence
            )
        
        #3. Calculate the comprehensive hallucination index
        hallucination_metrics = self._calculate_hallucination_metrics(
            coherence_metrics, truthfulness_metrics, miqp_assignments
        )
        
        return {
            'method_name': method_name,
            'hallucination_metrics': hallucination_metrics,
            'detailed_analysis': self._detailed_analysis(
                planning_trace, coherence_metrics, truthfulness_metrics
            )
        }
    
    def _calculate_hallucination_metrics(self,
                                       coherence_metrics: Optional[CoherenceMetrics],
                                       truthfulness_metrics: Optional[TruthfulnessMetrics],
                                       miqp_assignments: Optional[List[np.ndarray]]) -> HallucinationMetrics:
        """Calculating the Comprehensive Illusion Index"""
        
        #Calculate the degree of hallucination (lower is better)
        hallucination_components = []
        
        if coherence_metrics:
            #degree of incoherence=1 - Continuity
            incoherence = 1.0 - coherence_metrics.overall_coherence
            hallucination_components.append(incoherence)
        
        if truthfulness_metrics:
            #degree of unreality=1 - Authenticity
            untruthfulness = 1.0 - truthfulness_metrics.overall_truthfulness
            hallucination_components.append(untruthfulness)
        
        hallucination_score = np.mean(hallucination_components) if hallucination_components else 0.0
        
        #Calculate reliability score (higher is better)
        reliability_components = []
        
        if coherence_metrics:
            reliability_components.append(coherence_metrics.overall_coherence)
        
        if truthfulness_metrics:
            reliability_components.append(truthfulness_metrics.overall_truthfulness)
        
        reliability_score = np.mean(reliability_components) if reliability_components else 1.0
        
        # MIQPOptimization effect (if any)
        miqp_improvement = 0.0
        constraint_satisfaction = 1.0
        
        if miqp_assignments:
            #calculateMIQPimprovements brought about
            miqp_improvement = self._calculate_miqp_improvement(miqp_assignments)
            constraint_satisfaction = self._calculate_constraint_satisfaction(miqp_assignments)
        
        return HallucinationMetrics(
            coherence_metrics=coherence_metrics,
            truthfulness_metrics=truthfulness_metrics,
            hallucination_score=hallucination_score,
            reliability_score=reliability_score,
            miqp_improvement=miqp_improvement,
            constraint_satisfaction=constraint_satisfaction
        )
    
    def _detailed_analysis(self,
                          planning_trace: List[str],
                          coherence_metrics: Optional[CoherenceMetrics],
                          truthfulness_metrics: Optional[TruthfulnessMetrics]) -> Dict[str, Any]:
        """Detailed analysis"""
        
        analysis = {
            'trace_length': len(planning_trace),
            'avg_step_length': np.mean([len(step) for step in planning_trace]),
        }
        
        #Add coherence analysis
        if coherence_metrics:
            analysis.update({
                'semantic_coherence_breakdown': self._analyze_semantic_coherence(planning_trace),
                'temporal_coherence_issues': self._identify_temporal_issues(planning_trace),
                'action_coherence_patterns': self._analyze_action_patterns(planning_trace)
            })
        
        #Add authenticity analysis
        if truthfulness_metrics:
            analysis.update({
                'factual_error_types': self._categorize_factual_errors(planning_trace),
                'object_hallucination_rate': self._calculate_object_hallucination_rate(planning_trace),
                'spatial_error_patterns': self._analyze_spatial_errors(planning_trace)
            })
        
        return analysis
    
    def _compare_methods(self) -> Dict[str, Any]:
        """Compare the performance of different methods"""
        
        if len(self.results) < 2:
            return {"message": "At least two methods are needed for comparison"}
        
        comparison = {
            'hallucination_ranking': [],
            'reliability_ranking': [],
            'coherence_comparison': {},
            'truthfulness_comparison': {},
            'statistical_significance': {}
        }
        
        #Sorting method
        methods_by_hallucination = sorted(
            self.results, 
            key=lambda x: x['hallucination_metrics'].hallucination_score
        )
        
        methods_by_reliability = sorted(
            self.results, 
            key=lambda x: x['hallucination_metrics'].reliability_score, 
            reverse=True
        )
        
        comparison['hallucination_ranking'] = [
            (r['method_name'], r['hallucination_metrics'].hallucination_score) 
            for r in methods_by_hallucination
        ]
        
        comparison['reliability_ranking'] = [
            (r['method_name'], r['hallucination_metrics'].reliability_score) 
            for r in methods_by_reliability
        ]
        
        #Detailed comparison
        if self.config.enable_coherence:
            comparison['coherence_comparison'] = self._compare_coherence_metrics()
        
        if self.config.enable_truthfulness:
            comparison['truthfulness_comparison'] = self._compare_truthfulness_metrics()
        
        return comparison
    
    def _analyze_miqp_effects(self, miqp_data: Dict[str, List[np.ndarray]]) -> Dict[str, Any]:
        """analyzeMIQPOptimization effect"""
        
        analysis = {
            'optimization_effectiveness': {},
            'constraint_satisfaction_rates': {},
            'task_allocation_efficiency': {},
            'convergence_analysis': {}
        }
        
        for method_name, assignments in miqp_data.items():
            #Analyze optimization effects
            analysis['optimization_effectiveness'][method_name] = {
                'avg_assignment_entropy': self._calculate_assignment_entropy(assignments),
                'allocation_balance': self._calculate_allocation_balance(assignments),
                'temporal_consistency': self._analyze_temporal_assignment_consistency(assignments)
            }
            
            #constraint satisfaction rate
            analysis['constraint_satisfaction_rates'][method_name] = \
                self._calculate_constraint_satisfaction(assignments)
            
            #task allocation efficiency
            analysis['task_allocation_efficiency'][method_name] = \
                self._calculate_allocation_efficiency(assignments)
        
        return analysis
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        
        if not self.results:
            return {}
        
        #Extract all indicators
        hallucination_scores = [r['hallucination_metrics'].hallucination_score for r in self.results]
        reliability_scores = [r['hallucination_metrics'].reliability_score for r in self.results]
        
        coherence_scores = []
        truthfulness_scores = []
        
        for r in self.results:
            if r['hallucination_metrics'].coherence_metrics:
                coherence_scores.append(r['hallucination_metrics'].coherence_metrics.overall_coherence)
            if r['hallucination_metrics'].truthfulness_metrics:
                truthfulness_scores.append(r['hallucination_metrics'].truthfulness_metrics.overall_truthfulness)
        
        summary = {
            'hallucination_statistics': {
                'mean': np.mean(hallucination_scores),
                'std': np.std(hallucination_scores),
                'min': np.min(hallucination_scores),
                'max': np.max(hallucination_scores)
            },
            'reliability_statistics': {
                'mean': np.mean(reliability_scores),
                'std': np.std(reliability_scores),
                'min': np.min(reliability_scores),
                'max': np.max(reliability_scores)
            }
        }
        
        if coherence_scores:
            summary['coherence_statistics'] = {
                'mean': np.mean(coherence_scores),
                'std': np.std(coherence_scores),
                'min': np.min(coherence_scores),
                'max': np.max(coherence_scores)
            }
        
        if truthfulness_scores:
            summary['truthfulness_statistics'] = {
                'mean': np.mean(truthfulness_scores),
                'std': np.std(truthfulness_scores),
                'min': np.min(truthfulness_scores),
                'max': np.max(truthfulness_scores)
            }
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results"""
        
        #Save complete results
        with open(self.output_dir / 'full_results.json', 'w', encoding='utf-8') as f:
            #Convert to serializable format
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        #Save summary table
        summary_df = self._create_summary_dataframe()
        summary_df.to_csv(self.output_dir / 'summary_table.csv', index=False)
        
        #Save detailed log
        if self.config.save_detailed_logs:
            self._save_detailed_logs()
        
        self.logger.info(f"Results have been saved to: {self.output_dir}")
    
    def _generate_visualizations(self, results: Dict[str, Any]):
        """Generate visualization charts"""
        
        #1. Comparison chart of hallucination levels
        self._plot_hallucination_comparison()
        
        #2. Reliability score comparison chart
        self._plot_reliability_comparison()
        
        #3. Coherence and Authenticity Scatter Plot
        self._plot_coherence_vs_truthfulness()
        
        # 4. MIQPOptimization renderings
        if self.config.enable_miqp_analysis:
            self._plot_miqp_effects()
        
        #5. Radar chart comparison
        self._plot_radar_comparison()
        
        self.logger.info(f"Visualization saved to: {self.output_dir}")
    
    #Auxiliary method implementation
    def _calculate_miqp_improvement(self, assignments: List[np.ndarray]) -> float:
        """calculateMIQPimprovements brought about"""
        #Simplified implementation should actually be based on specific optimization goals
        if not assignments:
            return 0.0
        
        #Calculate allocation stability and efficiency
        assignment_variance = np.var([np.sum(a) for a in assignments if a is not None])
        improvement = max(0.0, 1.0 - assignment_variance / 10.0)  #normalization
        return improvement
    
    def _calculate_constraint_satisfaction(self, assignments: List[np.ndarray]) -> float:
        """Calculate constraint satisfaction"""
        if not assignments:
            return 1.0
        
        #Simplified implementation: check whether the allocation is reasonable
        valid_assignments = sum(1 for a in assignments if a is not None and np.sum(a) > 0)
        return valid_assignments / len(assignments)
    
    def _make_serializable(self, obj):
        """Convert object to serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return asdict(obj) if hasattr(obj, '__dataclass_fields__') else str(obj)
        else:
            return obj
    
    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Create summary data table"""
        data = []
        
        for result in self.results:
            metrics = result['hallucination_metrics']
            row = {
                'Method': result['method_name'],
                'Hallucination_Score': metrics.hallucination_score,
                'Reliability_Score': metrics.reliability_score,
            }
            
            if metrics.coherence_metrics:
                row.update({
                    'Semantic_Coherence': metrics.coherence_metrics.semantic_coherence,
                    'Temporal_Coherence': metrics.coherence_metrics.temporal_coherence,
                    'Action_Coherence': metrics.coherence_metrics.action_coherence,
                    'Overall_Coherence': metrics.coherence_metrics.overall_coherence
                })
            
            if metrics.truthfulness_metrics:
                row.update({
                    'Factual_Accuracy': metrics.truthfulness_metrics.factual_accuracy,
                    'World_Consistency': metrics.truthfulness_metrics.world_consistency,
                    'Object_Existence': metrics.truthfulness_metrics.object_existence,
                    'Spatial_Accuracy': metrics.truthfulness_metrics.spatial_accuracy,
                    'Temporal_Consistency': metrics.truthfulness_metrics.temporal_consistency,
                    'Overall_Truthfulness': metrics.truthfulness_metrics.overall_truthfulness
                })
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    #Placeholder method (needs to be implemented according to specific needs)
    def _analyze_semantic_coherence(self, planning_trace: List[str]) -> Dict[str, Any]:
        return {"placeholder": "semantic_analysis"}
    
    def _identify_temporal_issues(self, planning_trace: List[str]) -> List[str]:
        return ["placeholder_issue"]
    
    def _analyze_action_patterns(self, planning_trace: List[str]) -> Dict[str, Any]:
        return {"placeholder": "action_patterns"}
    
    def _categorize_factual_errors(self, planning_trace: List[str]) -> Dict[str, int]:
        return {"placeholder_error": 0}
    
    def _calculate_object_hallucination_rate(self, planning_trace: List[str]) -> float:
        return 0.0
    
    def _analyze_spatial_errors(self, planning_trace: List[str]) -> Dict[str, Any]:
        return {"placeholder": "spatial_errors"}
    
    def _compare_coherence_metrics(self) -> Dict[str, Any]:
        return {"placeholder": "coherence_comparison"}
    
    def _compare_truthfulness_metrics(self) -> Dict[str, Any]:
        return {"placeholder": "truthfulness_comparison"}
    
    def _calculate_assignment_entropy(self, assignments: List[np.ndarray]) -> float:
        return 0.0
    
    def _calculate_allocation_balance(self, assignments: List[np.ndarray]) -> float:
        return 0.0
    
    def _analyze_temporal_assignment_consistency(self, assignments: List[np.ndarray]) -> float:
        return 0.0
    
    def _calculate_allocation_efficiency(self, assignments: List[np.ndarray]) -> float:
        return 0.0
    
    def _save_detailed_logs(self):
        pass
    
    def _plot_hallucination_comparison(self):
        pass
    
    def _plot_reliability_comparison(self):
        pass
    
    def _plot_coherence_vs_truthfulness(self):
        pass
    
    def _plot_miqp_effects(self):
        pass
    
    def _plot_radar_comparison(self):
        pass 
