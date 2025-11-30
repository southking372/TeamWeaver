import time

class HumanCareerModel:
    def __init__(self, profession_type, initial_capabilities):
        self.profession_type = profession_type  # 职业类型
        self.capabilities = initial_capabilities  # 初始能力值
        self.adaptation_history = []  # 适应历史记录
        
    def get_capability(self, task_type):
        """获取特定任务类型的能力值"""
        return self.capabilities.get(task_type, 0.5)  # 默认中等能力
        
    def adapt_capability(self, task_type, performance_data):
        """根据性能数据调整能力值"""
        # 小幅度调整能力值
        current_value = self.capabilities.get(task_type, 0.5)
        adaptation_factor = self._calculate_adaptation_factor(performance_data)
        new_value = current_value + 0.1 * adaptation_factor  # 最大调整幅度为0.1
        
        # 确保能力值在[0,1]范围内
        new_value = max(0, min(1, new_value))
        self.capabilities[task_type] = new_value
        
        # 记录适应历史
        self.adaptation_history.append({
            'task_type': task_type,
            'old_value': current_value,
            'new_value': new_value,
            'performance_data': performance_data,
            'timestamp': time.time()
        })
        
    def _calculate_adaptation_factor(self, performance_data):
        """计算适应因子"""
        # 基于性能数据计算适应因子
        # 可以是简单的线性关系，也可以是更复杂的函数
        return performance_data - 0.5  # 假设性能数据在[0,1]范围内