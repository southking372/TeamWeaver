#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

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
    """大模型幻觉综合评估指标"""
    coherence_metrics: CoherenceMetrics
    truthfulness_metrics: TruthfulnessMetrics
    
    # 综合指标
    hallucination_score: float  # 幻觉程度 [0,1]，越低越好
    reliability_score: float    # 可靠性评分 [0,1]，越高越好
    
    # 软约束优化效果
    miqp_improvement: float     # MIQP优化带来的改善 [0,1]
    constraint_satisfaction: float  # 约束满足度 [0,1]

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 数据集配置
    dataset_path: str
    num_episodes: int = 100
    
    # 评估配置
    enable_coherence: bool = True
    enable_truthfulness: bool = True
    enable_miqp_analysis: bool = True
    
    # 对比实验配置
    baseline_methods: List[str] = None  # ['vanilla_llm', 'cot', 'rag']
    
    # 输出配置
    output_dir: str = "hallucination_results"
    save_detailed_logs: bool = True
    generate_plots: bool = True

class HallucinationExperiment:
    """
    大模型幻觉评估实验框架
    整合上下文一致性和事实一致性评估
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        初始化实验框架
        
        Args:
            config: 实验配置
        """
        self.config = config
        self.coherence_evaluator = CoherenceEvaluator() if config.enable_coherence else None
        self.truthfulness_evaluator = TruthfulnessEvaluator() if config.enable_truthfulness else None
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'experiment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 存储实验结果
        self.results = []
        self.baseline_results = {}
        
    def run_experiment(self, 
                      planning_traces: Dict[str, List[str]],
                      world_graph_sequences: Dict[str, List[Dict[str, Any]]],
                      ground_truth_sequences: Dict[str, List[Dict[str, Any]]],
                      miqp_data: Optional[Dict[str, List[np.ndarray]]] = None) -> Dict[str, Any]:
        """
        运行完整的幻觉评估实验
        
        Args:
            planning_traces: 各方法的规划轨迹 {method_name: [trace_steps]}
            world_graph_sequences: 世界图序列 {method_name: [world_states]}
            ground_truth_sequences: 真实状态序列 {method_name: [gt_states]}
            miqp_data: MIQP优化数据 {method_name: [assignments]}
            
        Returns:
            Dict: 实验结果汇总
        """
        self.logger.info("开始大模型幻觉评估实验")
        
        # 1. 评估各方法的幻觉程度
        for method_name in planning_traces.keys():
            self.logger.info(f"评估方法: {method_name}")
            
            method_results = self._evaluate_method(
                method_name=method_name,
                planning_trace=planning_traces[method_name],
                world_graph_sequence=world_graph_sequences[method_name],
                ground_truth_sequence=ground_truth_sequences[method_name],
                miqp_assignments=miqp_data.get(method_name) if miqp_data else None
            )
            
            self.results.append(method_results)
        
        # 2. 对比分析
        comparison_results = self._compare_methods()
        
        # 3. MIQP优化效果分析
        miqp_analysis = {}
        if self.config.enable_miqp_analysis and miqp_data:
            miqp_analysis = self._analyze_miqp_effects(miqp_data)
        
        # 4. 生成报告
        final_results = {
            'individual_results': self.results,
            'comparison_analysis': comparison_results,
            'miqp_analysis': miqp_analysis,
            'summary_statistics': self._generate_summary_statistics()
        }
        
        # 5. 保存结果和生成可视化
        self._save_results(final_results)
        
        if self.config.generate_plots:
            self._generate_visualizations(final_results)
        
        self.logger.info("实验完成")
        return final_results
    
    def _evaluate_method(self,
                        method_name: str,
                        planning_trace: List[str],
                        world_graph_sequence: List[Dict[str, Any]],
                        ground_truth_sequence: List[Dict[str, Any]],
                        miqp_assignments: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """评估单个方法的幻觉程度"""
        
        # 1. 上下文一致性评估
        coherence_metrics = None
        if self.coherence_evaluator:
            coherence_metrics = self.coherence_evaluator.evaluate_coherence(
                planning_trace, world_graph_sequence, miqp_assignments
            )
        
        # 2. 事实一致性评估
        truthfulness_metrics = None
        if self.truthfulness_evaluator:
            truthfulness_metrics = self.truthfulness_evaluator.evaluate_truthfulness(
                planning_trace, world_graph_sequence, ground_truth_sequence
            )
        
        # 3. 计算综合幻觉指标
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
        """计算综合幻觉指标"""
        
        # 计算幻觉程度（越低越好）
        hallucination_components = []
        
        if coherence_metrics:
            # 不连贯程度 = 1 - 连贯性
            incoherence = 1.0 - coherence_metrics.overall_coherence
            hallucination_components.append(incoherence)
        
        if truthfulness_metrics:
            # 不真实程度 = 1 - 真实性
            untruthfulness = 1.0 - truthfulness_metrics.overall_truthfulness
            hallucination_components.append(untruthfulness)
        
        hallucination_score = np.mean(hallucination_components) if hallucination_components else 0.0
        
        # 计算可靠性评分（越高越好）
        reliability_components = []
        
        if coherence_metrics:
            reliability_components.append(coherence_metrics.overall_coherence)
        
        if truthfulness_metrics:
            reliability_components.append(truthfulness_metrics.overall_truthfulness)
        
        reliability_score = np.mean(reliability_components) if reliability_components else 1.0
        
        # MIQP优化效果（如果有的话）
        miqp_improvement = 0.0
        constraint_satisfaction = 1.0
        
        if miqp_assignments:
            # 计算MIQP带来的改善
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
        """详细分析"""
        
        analysis = {
            'trace_length': len(planning_trace),
            'avg_step_length': np.mean([len(step) for step in planning_trace]),
        }
        
        # 添加连贯性分析
        if coherence_metrics:
            analysis.update({
                'semantic_coherence_breakdown': self._analyze_semantic_coherence(planning_trace),
                'temporal_coherence_issues': self._identify_temporal_issues(planning_trace),
                'action_coherence_patterns': self._analyze_action_patterns(planning_trace)
            })
        
        # 添加真实性分析
        if truthfulness_metrics:
            analysis.update({
                'factual_error_types': self._categorize_factual_errors(planning_trace),
                'object_hallucination_rate': self._calculate_object_hallucination_rate(planning_trace),
                'spatial_error_patterns': self._analyze_spatial_errors(planning_trace)
            })
        
        return analysis
    
    def _compare_methods(self) -> Dict[str, Any]:
        """对比不同方法的表现"""
        
        if len(self.results) < 2:
            return {"message": "需要至少两种方法进行对比"}
        
        comparison = {
            'hallucination_ranking': [],
            'reliability_ranking': [],
            'coherence_comparison': {},
            'truthfulness_comparison': {},
            'statistical_significance': {}
        }
        
        # 排序方法
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
        
        # 详细对比
        if self.config.enable_coherence:
            comparison['coherence_comparison'] = self._compare_coherence_metrics()
        
        if self.config.enable_truthfulness:
            comparison['truthfulness_comparison'] = self._compare_truthfulness_metrics()
        
        return comparison
    
    def _analyze_miqp_effects(self, miqp_data: Dict[str, List[np.ndarray]]) -> Dict[str, Any]:
        """分析MIQP优化的效果"""
        
        analysis = {
            'optimization_effectiveness': {},
            'constraint_satisfaction_rates': {},
            'task_allocation_efficiency': {},
            'convergence_analysis': {}
        }
        
        for method_name, assignments in miqp_data.items():
            # 分析优化效果
            analysis['optimization_effectiveness'][method_name] = {
                'avg_assignment_entropy': self._calculate_assignment_entropy(assignments),
                'allocation_balance': self._calculate_allocation_balance(assignments),
                'temporal_consistency': self._analyze_temporal_assignment_consistency(assignments)
            }
            
            # 约束满足率
            analysis['constraint_satisfaction_rates'][method_name] = \
                self._calculate_constraint_satisfaction(assignments)
            
            # 任务分配效率
            analysis['task_allocation_efficiency'][method_name] = \
                self._calculate_allocation_efficiency(assignments)
        
        return analysis
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """生成汇总统计"""
        
        if not self.results:
            return {}
        
        # 提取所有指标
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
        """保存实验结果"""
        
        # 保存完整结果
        with open(self.output_dir / 'full_results.json', 'w', encoding='utf-8') as f:
            # 转换为可序列化格式
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 保存汇总表格
        summary_df = self._create_summary_dataframe()
        summary_df.to_csv(self.output_dir / 'summary_table.csv', index=False)
        
        # 保存详细日志
        if self.config.save_detailed_logs:
            self._save_detailed_logs()
        
        self.logger.info(f"结果已保存到: {self.output_dir}")
    
    def _generate_visualizations(self, results: Dict[str, Any]):
        """生成可视化图表"""
        
        # 1. 幻觉程度对比图
        self._plot_hallucination_comparison()
        
        # 2. 可靠性评分对比图
        self._plot_reliability_comparison()
        
        # 3. 连贯性和真实性散点图
        self._plot_coherence_vs_truthfulness()
        
        # 4. MIQP优化效果图
        if self.config.enable_miqp_analysis:
            self._plot_miqp_effects()
        
        # 5. 雷达图对比
        self._plot_radar_comparison()
        
        self.logger.info(f"可视化图表已保存到: {self.output_dir}")
    
    # 辅助方法实现
    def _calculate_miqp_improvement(self, assignments: List[np.ndarray]) -> float:
        """计算MIQP带来的改善"""
        # 简化实现，实际应该基于具体的优化目标
        if not assignments:
            return 0.0
        
        # 计算分配的稳定性和效率
        assignment_variance = np.var([np.sum(a) for a in assignments if a is not None])
        improvement = max(0.0, 1.0 - assignment_variance / 10.0)  # 归一化
        return improvement
    
    def _calculate_constraint_satisfaction(self, assignments: List[np.ndarray]) -> float:
        """计算约束满足度"""
        if not assignments:
            return 1.0
        
        # 简化实现：检查分配是否合理
        valid_assignments = sum(1 for a in assignments if a is not None and np.sum(a) > 0)
        return valid_assignments / len(assignments)
    
    def _make_serializable(self, obj):
        """将对象转换为可序列化格式"""
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
        """创建汇总数据表"""
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
    
    # 占位符方法（需要根据具体需求实现）
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