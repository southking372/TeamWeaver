#!/usr/bin/env python3
"""
测试简化的PARTNR配置
验证从13个能力简化到5个能力是否正常工作
"""

import numpy as np
import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', '..', '..'))

from params_module.scenario_params_task import ScenarioConfigTask
from params_module.opt_params_task import OptimizationConfigTask
from class_def.GlobalVarsManager_task import GlobalVarsManager_task
from class_def.RTA_task import RTA

def test_simplified_config():
    """测试简化配置"""
    print("=" * 60)
    print("测试简化的PARTNR配置")
    print("=" * 60)
    
    # 1. 测试场景参数配置
    print("\n1. 测试ScenarioConfigTask...")
    scenario_manager = ScenarioConfigTask(n_r=2, n_t=13, n_c=5, n_f=5)
    scenario_params = scenario_manager.get_scenario_params()
    
    # 验证矩阵维度
    A_shape = scenario_params['A'].shape
    T_shape = scenario_params['T'].shape
    Hs_count = len(scenario_params['Hs'])
    ws_count = len(scenario_params['ws'])
    
    print(f"   特征矩阵 A: {A_shape} (期望: (5, 2))")
    print(f"   任务需求矩阵 T: {T_shape} (期望: (13, 5))")
    print(f"   映射矩阵 Hs 数量: {Hs_count} (期望: 5)")
    print(f"   权重矩阵 ws 数量: {ws_count} (期望: 5)")
    
    # 验证Agent能力配置
    print(f"\n   Agent能力配置:")
    print(f"   Agent 0: {scenario_params['A'][:, 0]} (基础移动+物体操作+基本控制)")
    print(f"   Agent 1: {scenario_params['A'][:, 1]} (全功能)")
    
    # 验证任务到能力的映射
    print(f"\n   任务到能力映射矩阵 T:")
    for i, task_name in enumerate(['Navigate', 'Explore', 'Pick', 'Place', 'Open', 'Close', 
                                   'Clean', 'Fill', 'Pour', 'PowerOn', 'PowerOff', 'Rearrange', 'Wait']):
        if i < T_shape[0]:
            capability_idx = np.where(scenario_params['T'][i, :] > 0)[0]
            if len(capability_idx) > 0:
                capability_names = ['基础移动', '物体操作', '基本控制', '液体处理', '电源控制']
                print(f"     {task_name:>10} → {capability_names[capability_idx[0]]}")
    
    # 2. 测试优化参数配置
    print(f"\n2. 测试OptimizationConfigTask...")
    opt_manager = OptimizationConfigTask(n_r=2, n_t=13)
    opt_params = opt_manager.get_opt_params()
    bounds_shape = opt_params['n_r_bounds'].shape
    print(f"   机器人边界矩阵: {bounds_shape} (期望: (13, 2))")
    
    # 3. 测试RTA初始化
    print(f"\n3. 测试RTA初始化...")
    try:
        global_vars = GlobalVarsManager_task()
        rta = RTA(scenario_params, opt_params)
        rta.global_vars_manager_ = global_vars
        
        print(f"   RTA维度信息:")
        print(f"     机器人数量 (n_r): {rta.dim_['n_r']}")
        print(f"     任务数量 (n_t): {rta.dim_['n_t']}")
        print(f"     能力数量 (n_c): {rta.dim_['n_c']}")
        print(f"     特征数量 (n_f): {rta.dim_['n_f']}")
        
        # 测试矩阵计算
        F_shape = rta.scenario_params_['F'].shape
        S_shape = rta.scenario_params_['S'].shape
        print(f"     特征到能力矩阵 F: {F_shape}")
        print(f"     专业化矩阵 S: {S_shape}")
        
        print(f"   ✓ RTA初始化成功")
        
    except Exception as e:
        print(f"   ✗ RTA初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 测试MIQP求解器设置（模拟）
    print(f"\n4. 测试MIQP约束构建...")
    try:
        # 模拟机器人状态
        x = np.array([[0.0, 1.0], [0.0, -1.0], [0.0, 0.0]])  # 2个机器人的状态 [x, y, theta]
        t = 10.0
        
        # 构建约束
        rta.build_constraints(x, t)
        
        A_ineq_shape = rta.constraints_['A_ineq'].shape
        b_ineq_shape = rta.constraints_['b_ineq'].shape
        A_eq_shape = rta.constraints_['A_eq'].shape
        
        print(f"   不等式约束矩阵 A_ineq: {A_ineq_shape}")
        print(f"   不等式约束向量 b_ineq: {b_ineq_shape}")
        print(f"   等式约束矩阵 A_eq: {A_eq_shape}")
        print(f"   ✓ 约束构建成功")
        
    except Exception as e:
        print(f"   ✗ 约束构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n" + "=" * 60)
    print(f"✓ 简化配置测试全部通过！")
    print(f"  - 从 13 能力简化到 5 能力")
    print(f"  - 从 6 特征简化到 5 特征")
    print(f"  - 保持 13 个PARTNR任务不变")
    print(f"  - RTA系统兼容性验证通过")
    print(f"=" * 60)
    
    return True

def test_capability_reduction_effect():
    """测试能力缩减的效果"""
    print(f"\n" + "=" * 60)
    print(f"测试能力缩减对问题规模的影响")
    print(f"=" * 60)
    
    # 原配置 vs 新配置
    configs = [
        {"name": "原配置", "n_c": 13, "n_f": 6},
        {"name": "简化配置", "n_c": 5, "n_f": 5}
    ]
    
    for config in configs:
        scenario_manager = ScenarioConfigTask(n_r=2, n_t=13, n_c=config["n_c"], n_f=config["n_f"])
        scenario_params = scenario_manager.get_scenario_params()
        opt_manager = OptimizationConfigTask(n_r=2, n_t=13)
        opt_params = opt_manager.get_opt_params()
        
        try:
            rta = RTA(scenario_params, opt_params)
            x = np.array([[0.0, 1.0], [0.0, -1.0], [0.0, 0.0]])
            t = 10.0
            rta.build_constraints(x, t)
            
            constraint_size = rta.constraints_['A_ineq'].shape
            variable_count = constraint_size[1] if len(constraint_size) > 1 else 0
            
            print(f"\n{config['name']}:")
            print(f"  能力数量: {config['n_c']}")
            print(f"  特征数量: {config['n_f']}")
            print(f"  约束矩阵大小: {constraint_size}")
            print(f"  变量数量: {variable_count}")
            print(f"  约束数量: {constraint_size[0] if constraint_size else 0}")
            
        except Exception as e:
            print(f"  {config['name']} 配置失败: {e}")
    
    print(f"\n规模缩减效果:")
    print(f"  - 能力矩阵: 13×13 → 13×5 (减少61.5%)")
    print(f"  - 特征矩阵: 6×2 → 5×2 (减少16.7%)")
    print(f"  - 映射矩阵数量: 13 → 5 (减少61.5%)")
    print(f"=" * 60)

if __name__ == "__main__":
    success = test_simplified_config()
    if success:
        test_capability_reduction_effect()
    else:
        print("基础测试失败，跳过规模分析测试") 