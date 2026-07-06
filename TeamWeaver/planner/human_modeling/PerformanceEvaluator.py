# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

class PerformanceEvaluator:
    def __init__(self, human_models):
        """
        Initialize performance evaluator
        
        Parameters:
            human_models: list of human models
        """
        self.human_models = human_models
        
    def evaluate_performance(self, human_idx, task_type, execution_data):
        """
        Evaluate task execution performance
        
        Parameters:
            human_idx: human model index
            task_type: task type
            execution_data: execution data
            
        Returns:
            performance score
        """
        # validate index
        if human_idx < 0 or human_idx >= len(self.human_models):
            raise ValueError(f"Invalid human model index: {human_idx}")
            
        # get corresponding human model
        human_model = self.human_models[human_idx]
        
        # extract key performance metrics
        metrics = self._extract_metrics(execution_data)
        
        # compute aggregate performance score
        performance_score = self._calculate_performance_score(metrics)
        
        # update human model
        human_model.adapt_capability(task_type, performance_score)
        
        return performance_score
        
    def _extract_metrics(self, execution_data):
        """Extract performance metrics."""
        # extract different metrics depending on task type
        metrics = {}
        
        if 'transport' in execution_data:
            metrics['accuracy'] = self._calculate_accuracy(execution_data)
            metrics['efficiency'] = self._calculate_efficiency(execution_data)
            metrics['safety'] = self._calculate_safety(execution_data)
            
        elif 'coverage' in execution_data:
            metrics['coverage_area'] = self._calculate_coverage_area(execution_data)
            metrics['uniformity'] = self._calculate_uniformity(execution_data)
            metrics['response_time'] = self._calculate_response_time(execution_data)
            
        elif 'precision' in execution_data:
            metrics['accuracy'] = self._calculate_accuracy(execution_data)
            metrics['stability'] = self._calculate_stability(execution_data)
            metrics['adaptability'] = self._calculate_adaptability(execution_data)
            
        return metrics
        
    def _calculate_performance_score(self, metrics):
        """Compute aggregate performance score."""
        # weighted sum over metrics
        weights = {
            'accuracy': 0.4,
            'efficiency': 0.3,
            'safety': 0.3,
            'coverage_area': 0.4,
            'uniformity': 0.3,
            'response_time': 0.3,
            'stability': 0.4,
            'adaptability': 0.3
        }
        
        # compute weighted average
        total_weight = 0
        weighted_sum = 0
        
        for metric, value in metrics.items():
            if metric in weights:
                weighted_sum += value * weights[metric]
                total_weight += weights[metric]
        
        # no valid metrics
        if total_weight == 0:
            return 0
            
        # final score
        performance_score = weighted_sum / total_weight
        
        # clamp to [0, 1]
        return max(0, min(1, performance_score))
            
    def _calculate_accuracy(self, execution_data):
        """Compute accuracy metric."""
        # extract accuracy-related metrics from execution data
        if 'target_position' in execution_data and 'actual_position' in execution_data:
            # error between target and actual position
            target = execution_data['target_position']
            actual = execution_data['actual_position']
            error = sum((t - a) ** 2 for t, a in zip(target, actual)) ** 0.5
            
            # convert error to [0, 1] score; lower error is better
            max_error = execution_data.get('max_error', 10.0)
            accuracy = max(0, 1 - error / max_error)
            return accuracy
        else:
            # default when position data is missing
            return 0.7
            
    def _calculate_efficiency(self, execution_data):
        """Compute efficiency metric."""
        # extract efficiency-related metrics from execution data
        if 'completion_time' in execution_data and 'expected_time' in execution_data:
            # ratio of completion time to expected time
            actual_time = execution_data['completion_time']
            expected_time = execution_data['expected_time']
            
            # shorter time is better, with a lower bound
            efficiency = min(1.0, expected_time / max(actual_time, 0.1))
            return efficiency
        else:
            # default when timing data is missing
            return 0.6
            
    def _calculate_safety(self, execution_data):
        """Compute safety metric."""
        # extract safety-related metrics from execution data
        if 'collisions' in execution_data and 'total_operations' in execution_data:
            # collision rate
            collisions = execution_data['collisions']
            total_ops = execution_data['total_operations']
            
            # fewer collisions means higher safety
            collision_rate = collisions / max(total_ops, 1)
            safety = max(0, 1 - collision_rate)
            return safety
        else:
            # default when collision data is missing
            return 0.8
            
    def _calculate_coverage_area(self, execution_data):
        """Compute coverage area metric."""
        # extract coverage area metrics from execution data
        if 'covered_area' in execution_data and 'total_area' in execution_data:
            # covered area ratio
            covered = execution_data['covered_area']
            total = execution_data['total_area']
            
            coverage = min(1.0, covered / max(total, 0.1))
            return coverage
        else:
            # default when area data is missing
            return 0.75
            
    def _calculate_uniformity(self, execution_data):
        """Compute uniformity metric."""
        # extract uniformity metrics from execution data
        if 'coverage_density' in execution_data:
            # std dev of coverage density; lower std dev means higher uniformity
            densities = execution_data['coverage_density']
            if len(densities) > 1:
                mean_density = sum(densities) / len(densities)
                variance = sum((d - mean_density) ** 2 for d in densities) / len(densities)
                std_dev = variance ** 0.5
                
                # convert std dev to [0, 1] score; lower std dev is better
                max_std = execution_data.get('max_std', 0.5)
                uniformity = max(0, 1 - std_dev / max_std)
                return uniformity
            else:
                return 0.7
        else:
            # default when density data is missing
            return 0.7
            
    def _calculate_response_time(self, execution_data):
        """Compute response time metric."""
        # extract response time metrics from execution data
        if 'response_times' in execution_data:
            # average response time
            times = execution_data['response_times']
            if times:
                avg_time = sum(times) / len(times)
                
                # shorter response time is better, with a lower bound
                max_time = execution_data.get('max_response_time', 5.0)
                response_score = max(0, 1 - avg_time / max_time)
                return response_score
            else:
                return 0.6
        else:
            # default when response time data is missing
            return 0.6
            
    def _calculate_stability(self, execution_data):
        """Compute stability metric."""
        # extract stability metrics from execution data
        if 'control_variations' in execution_data:
            # magnitude of control variation; smaller variation means higher stability
            variations = execution_data['control_variations']
            if variations:
                avg_variation = sum(variations) / len(variations)
                
                # convert variation magnitude to [0, 1] score; smaller variation is better
                max_variation = execution_data.get('max_variation', 2.0)
                stability = max(0, 1 - avg_variation / max_variation)
                return stability
            else:
                return 0.7
        else:
            # default when variation data is missing
            return 0.7
            
    def _calculate_adaptability(self, execution_data):
        """Compute adaptability metric."""
        # extract adaptability metrics from execution data
        if 'environment_changes' in execution_data and 'adaptation_success' in execution_data:
            # success rate after environment changes
            changes = execution_data['environment_changes']
            successes = execution_data['adaptation_success']
            
            if changes > 0:
                adaptability = successes / changes
                return adaptability
            else:
                return 0.7
        else:
            # default when adaptability data is missing
            return 0.7
