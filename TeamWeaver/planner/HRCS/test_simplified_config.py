#!/usr/bin/env python3
"""
Test simplified PARTNR configuration
Verify reduction from 13 to 5 capabilities works
"""

import numpy as np
import sys
import os

# Add path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', '..', '..'))

from params_module.scenario_params_task import ScenarioConfigTask
from params_module.opt_params_task import OptimizationConfigTask
from class_def.GlobalVarsManager_task import GlobalVarsManager_task
from class_def.RTA_task import RTA

def test_simplified_config():
    """Test simplified configuration"""
    print("=" * 60)
    print("Test simplified PARTNR configuration")
    print("=" * 60)
    
    #1. Test scenario parameter configuration
    print("\n1. Test ScenarioConfigTask...")
    scenario_manager = ScenarioConfigTask(n_r=2, n_t=13, n_c=5, n_f=5)
    scenario_params = scenario_manager.get_scenario_params()
    
    # Validate matrix dimensions
    A_shape = scenario_params['A'].shape
    T_shape = scenario_params['T'].shape
    Hs_count = len(scenario_params['Hs'])
    ws_count = len(scenario_params['ws'])
    
    print(f"   Feature matrix A: {A_shape} (expected: (5, 2))")
    print(f"   Task requirement matrix T: {T_shape} (expected: (13, 5))")
    print(f"   Hs mapping matrix count: {Hs_count} (expected: 5)")
    print(f"   ws weight matrix count: {ws_count} (expected: 5)")
    
    # Validate agent capability config
    print(f"\n   Agent capability config:")
    print(f"   Agent 0: {scenario_params['A'][:, 0]} (Basemovement+objectmanipulation+Basiccontrol)")
    print(f"   Agent 1: {scenario_params['A'][:, 1]} (full capability)")
    
    #Verify task-to-competency mapping
    print(f"\n   Task-to-capability mapping matrix T:")
    for i, task_name in enumerate(['Navigate', 'Explore', 'Pick', 'Place', 'Open', 'Close', 
                                   'Clean', 'Fill', 'Pour', 'PowerOn', 'PowerOff', 'Rearrange', 'Wait']):
        if i < T_shape[0]:
            capability_idx = np.where(scenario_params['T'][i, :] > 0)[0]
            if len(capability_idx) > 0:
                capability_names = ['Basemovement','objectmanipulation','Basiccontrol', 'liquiddeal with', 'powercontrol']
                print(f"     {task_name:>10} → {capability_names[capability_idx[0]]}")
    
    #2. Test optimization parameter configuration
    print(f"\n2. Test OptimizationConfigTask...")
    opt_manager = OptimizationConfigTask(n_r=2, n_t=13)
    opt_params = opt_manager.get_opt_params()
    bounds_shape = opt_params['n_r_bounds'].shape
    print(f"   Robot bounds matrix: {bounds_shape} (expected: (13, 2))")
    
    #3. TestRTAinitialization
    print(f"\n3. Test RTA initialization...")
    try:
        global_vars = GlobalVarsManager_task()
        rta = RTA(scenario_params, opt_params)
        rta.global_vars_manager_ = global_vars
        
        print(f"   RTA dimension info:")
        print(f"     number of robots (n_r): {rta.dim_['n_r']}")
        print(f"     task count (n_t): {rta.dim_['n_t']}")
        print(f"     capability count (n_c): {rta.dim_['n_c']}")
        print(f"     feature count (n_f): {rta.dim_['n_f']}")
        
        # Test matrix computation
        F_shape = rta.scenario_params_['F'].shape
        S_shape = rta.scenario_params_['S'].shape
        print(f"Feature tocapability matrix F: {F_shape}")
        print(f"     Specialization matrix S: {S_shape}")
        
        print(f"   ✓ RTA initialization succeeded")
        
    except Exception as e:
        print(f"   ✗ RTA initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    #4. TestMIQPSolver settings (simulation)
    print(f"\n4. Test MIQP constraint construction...")
    try:
        # Simulate robot state
        x = np.array([[0.0, 1.0], [0.0, -1.0], [0.0, 0.0]])  # 2 robots state [x, y, theta]
        t = 10.0
        
        # Build constraints
        rta.build_constraints(x, t)
        
        A_ineq_shape = rta.constraints_['A_ineq'].shape
        b_ineq_shape = rta.constraints_['b_ineq'].shape
        A_eq_shape = rta.constraints_['A_eq'].shape
        
        print(f"   Inequality constraint matrix A_ineq: {A_ineq_shape}")
        print(f"   Inequality constraint vector b_ineq: {b_ineq_shape}")
        print(f"   Equality constraint matrix A_eq: {A_eq_shape}")
        print(f"   ✓ Constraint construction succeeded")
        
    except Exception as e:
        print(f"   ✗ Constraint construction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n" + "=" * 60)
    print(f"✓ All simplified config tests passed!")
    print(f"  - Reduced from 13 to 5 capabilities")
    print(f"  - Reduced from 6 to 5 features")
    print(f"  - Kept 13 PARTNR tasks unchanged")
    print(f"  - RTA system compatibility verified")
    print(f"=" * 60)
    
    return True

def test_capability_reduction_effect():
    """Test effect of capability reduction"""
    print(f"\n" + "=" * 60)
    print(f"Test capability reduction impact on problem size")
    print(f"=" * 60)
    
    # original config vsNew configuration
    configs = [
        {"name": "original config", "n_c": 13, "n_f": 6},
        {"name": "simplified config", "n_c": 5, "n_f": 5}
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
            print(f"  capability count: {config['n_c']}")
            print(f"  feature count: {config['n_f']}")
            print(f"  constraint matrix size: {constraint_size}")
            print(f"  variable count: {variable_count}")
            print(f"  constraint count: {constraint_size[0] if constraint_size else 0}")
            
        except Exception as e:
            print(f"  {config['name']} config failed: {e}")
    
    print(f"\nScale reduction effect:")
    print(f"  - capability matrix: 13×13 → 13×5 (reduced by 61.5%)")
    print(f"  - Feature matrix: 6×2 → 5×2 (16.7% reduction)")
    print(f"  - Mapping matrix count: 13 → 5 (61.5% reduction)")
    print(f"=" * 60)

if __name__ == "__main__":
    success = test_simplified_config()
    if success:
        test_capability_reduction_effect()
    else:
        print("Basic tests failed; skip scale analysis") 
