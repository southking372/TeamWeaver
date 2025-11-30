class PerformanceEvaluator:
    def __init__(self, human_models):
        """
        初始化性能评估器
        
        参数:
            human_models: 人类模型列表
        """
        self.human_models = human_models
        
    def evaluate_performance(self, human_idx, task_type, execution_data):
        """
        评估任务执行性能
        
        参数:
            human_idx: 人类模型索引
            task_type: 任务类型
            execution_data: 执行数据
            
        返回:
            性能分数
        """
        # 确保索引有效
        if human_idx < 0 or human_idx >= len(self.human_models):
            raise ValueError(f"无效的人类模型索引: {human_idx}")
            
        # 获取对应的人类模型
        human_model = self.human_models[human_idx]
        
        # 提取关键性能指标
        metrics = self._extract_metrics(execution_data)
        
        # 计算综合性能分数
        performance_score = self._calculate_performance_score(metrics)
        
        # 更新人类模型
        human_model.adapt_capability(task_type, performance_score)
        
        return performance_score
        
    def _extract_metrics(self, execution_data):
        """提取性能指标"""
        # 根据任务类型提取不同的指标
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
        """计算综合性能分数"""
        # 根据指标权重计算综合分数
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
        
        # 计算加权平均分数
        total_weight = 0
        weighted_sum = 0
        
        for metric, value in metrics.items():
            if metric in weights:
                weighted_sum += value * weights[metric]
                total_weight += weights[metric]
        
        # 如果没有有效指标，返回0
        if total_weight == 0:
            return 0
            
        # 计算最终分数
        performance_score = weighted_sum / total_weight
        
        # 确保分数在0-1范围内
        return max(0, min(1, performance_score))
            
    def _calculate_accuracy(self, execution_data):
        """计算准确性指标"""
        # 从执行数据中提取准确性相关指标
        if 'target_position' in execution_data and 'actual_position' in execution_data:
            # 计算目标位置与实际位置的误差
            target = execution_data['target_position']
            actual = execution_data['actual_position']
            error = sum((t - a) ** 2 for t, a in zip(target, actual)) ** 0.5
            
            # 将误差转换为0-1范围的分数，误差越小分数越高
            max_error = execution_data.get('max_error', 10.0)
            accuracy = max(0, 1 - error / max_error)
            return accuracy
        else:
            # 如果没有位置数据，返回默认值
            return 0.7
            
    def _calculate_efficiency(self, execution_data):
        """计算效率指标"""
        # 从执行数据中提取效率相关指标
        if 'completion_time' in execution_data and 'expected_time' in execution_data:
            # 计算完成时间与预期时间的比率
            actual_time = execution_data['completion_time']
            expected_time = execution_data['expected_time']
            
            # 时间越短效率越高，但设置一个下限
            efficiency = min(1.0, expected_time / max(actual_time, 0.1))
            return efficiency
        else:
            # 如果没有时间数据，返回默认值
            return 0.6
            
    def _calculate_safety(self, execution_data):
        """计算安全性指标"""
        # 从执行数据中提取安全性相关指标
        if 'collisions' in execution_data and 'total_operations' in execution_data:
            # 计算碰撞率
            collisions = execution_data['collisions']
            total_ops = execution_data['total_operations']
            
            # 碰撞越少安全性越高
            collision_rate = collisions / max(total_ops, 1)
            safety = max(0, 1 - collision_rate)
            return safety
        else:
            # 如果没有碰撞数据，返回默认值
            return 0.8
            
    def _calculate_coverage_area(self, execution_data):
        """计算覆盖面积指标"""
        # 从执行数据中提取覆盖面积相关指标
        if 'covered_area' in execution_data and 'total_area' in execution_data:
            # 计算覆盖面积比例
            covered = execution_data['covered_area']
            total = execution_data['total_area']
            
            coverage = min(1.0, covered / max(total, 0.1))
            return coverage
        else:
            # 如果没有面积数据，返回默认值
            return 0.75
            
    def _calculate_uniformity(self, execution_data):
        """计算均匀性指标"""
        # 从执行数据中提取均匀性相关指标
        if 'coverage_density' in execution_data:
            # 计算覆盖密度的标准差，标准差越小均匀性越高
            densities = execution_data['coverage_density']
            if len(densities) > 1:
                mean_density = sum(densities) / len(densities)
                variance = sum((d - mean_density) ** 2 for d in densities) / len(densities)
                std_dev = variance ** 0.5
                
                # 将标准差转换为0-1范围的分数，标准差越小分数越高
                max_std = execution_data.get('max_std', 0.5)
                uniformity = max(0, 1 - std_dev / max_std)
                return uniformity
            else:
                return 0.7
        else:
            # 如果没有密度数据，返回默认值
            return 0.7
            
    def _calculate_response_time(self, execution_data):
        """计算响应时间指标"""
        # 从执行数据中提取响应时间相关指标
        if 'response_times' in execution_data:
            # 计算平均响应时间
            times = execution_data['response_times']
            if times:
                avg_time = sum(times) / len(times)
                
                # 响应时间越短分数越高，但设置一个下限
                max_time = execution_data.get('max_response_time', 5.0)
                response_score = max(0, 1 - avg_time / max_time)
                return response_score
            else:
                return 0.6
        else:
            # 如果没有响应时间数据，返回默认值
            return 0.6
            
    def _calculate_stability(self, execution_data):
        """计算稳定性指标"""
        # 从执行数据中提取稳定性相关指标
        if 'control_variations' in execution_data:
            # 计算控制变化的幅度，变化越小稳定性越高
            variations = execution_data['control_variations']
            if variations:
                avg_variation = sum(variations) / len(variations)
                
                # 将变化幅度转换为0-1范围的分数，变化越小分数越高
                max_variation = execution_data.get('max_variation', 2.0)
                stability = max(0, 1 - avg_variation / max_variation)
                return stability
            else:
                return 0.7
        else:
            # 如果没有变化数据，返回默认值
            return 0.7
            
    def _calculate_adaptability(self, execution_data):
        """计算适应性指标"""
        # 从执行数据中提取适应性相关指标
        if 'environment_changes' in execution_data and 'adaptation_success' in execution_data:
            # 计算环境变化后的适应成功率
            changes = execution_data['environment_changes']
            successes = execution_data['adaptation_success']
            
            if changes > 0:
                adaptability = successes / changes
                return adaptability
            else:
                return 0.7
        else:
            # 如果没有适应数据，返回默认值
            return 0.7