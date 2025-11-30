import argparse
import json
import os
import sys
import time
from datetime import datetime
import numpy as np

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# 基础配置与任务模块
from class_def.task_module.GlobalVarsManager_task import GlobalVarsManager_task
from task_utils.task_module.scenario_params_task import ScenarioConfigTask
from task_utils.task_module.opt_params_task import OptimizationConfigTask
from task_utils.task_module.simulation_module import SimulationConfig
from class_def.task_module.baseline_planners import create_planner

# 任务控制（用于可选的位姿推进）
from task_utils.task_module.navi_module import NaviTask
from task_utils.task_module.explore_module import ExploreTask
from task_utils.task_module.manipulation_module import ManipulationTask
from task_utils.task_module.wait_module import WaitTask

from metrics import (
    compute_F_matrix,
    compute_capability_utilization_metrics,
    compute_parallelism_and_switching,
    estimate_minimal_robot_count_per_task,
)
from scenarios import (
    apply_dynamic_capability_drop,
    append_exploration_targets,
)


def reshape_alpha(alpha_vec, n_t, n_r):
    """将扁平 alpha 按列主序重塑为 (n_t, n_r)。"""
    return np.reshape(alpha_vec, (n_t, n_r), order="F")


def assignment_from_alpha(alpha_mat):
    """将 (n_t, n_r) 的 alpha 转换为每个机器人的任务编号列表（0-based）。"""
    return np.argmax(alpha_mat, axis=0).tolist()


def maybe_step_states(x, task_assignment, t, vars_dict, dt):
    """可选: 用任务控制做一次简单的位姿推进（默认不开启）。"""
    n_r = x.shape[1]
    for i in range(n_r):
        task_id = task_assignment[i]
        try:
            if task_id == 0:
                x[:, i] = NaviTask.apply_motion_control(x[:, i], t, i, vars_dict, dt)
            elif task_id == 1:
                x[:, i] = ExploreTask.apply_motion_control(x[:, i], t, i, vars_dict, dt)
            elif task_id == 2:
                x[:, i] = ManipulationTask.apply_motion_control(x[:, i], t, i, vars_dict, dt)
            elif task_id == 3:
                x[:, i] = WaitTask.apply_motion_control(x[:, i], t, i, vars_dict, dt)
        except Exception:
            pass
    return x


def run_single_experiment(planner_type: str,
                          total_steps: int = 60,
                          dt: float = 0.1,
                          replan_every: int = 1,
                          dynamic_change_step: int = 30,
                          enable_state_step: bool = False,
                          seed: int = 0):
    np.random.seed(seed)

    # 初始化全局变量与场景/优化配置
    global_vars = GlobalVarsManager_task()
    scenario_cfg = ScenarioConfigTask(n_u=2)
    opt_cfg = OptimizationConfigTask()
    scenario_params = scenario_cfg.get_scenario_params()
    opt_params = opt_cfg.get_opt_params()

    # 模拟器（仅用于初始位姿与 dt 参数）
    sim_cfg = SimulationConfig()
    x = sim_cfg.get_initial_states().copy()
    n_x, n_r = x.shape
    n_t = scenario_params['T'].shape[0]

    # 注册全局任务变量
    global_task_vars = scenario_cfg.get_global_task_vars()
    global_vars.register_vars_from_dict(global_task_vars)
    global_vars.x = x

    # 规划器实例
    planner = create_planner(planner_type, scenario_params, opt_params, global_vars)

    # 存储度量
    metrics_log = {
        'planner': planner_type,
        'dt': dt,
        'total_steps': total_steps,
        'replan_every': replan_every,
        'dynamic_change_step': dynamic_change_step,
        'seed': seed,
        'per_step': []
    }

    prev_assignment = None
    recovered_after_change_step = None
    changed_applied = False

    for step in range(total_steps):
        t = step * dt
        global_vars.set_var('t', t)

        # 每 replan_every 步重规划一次
        if step % replan_every == 0:
            alpha_vec, u_vec, delta_vec, solve_time, _ = planner.solve_miqp(x, t)
            alpha_mat = reshape_alpha(alpha_vec, n_t, n_r)
            assignment = assignment_from_alpha(alpha_mat)
        else:
            assignment = assignment_from_alpha(alpha_mat)
            solve_time = 0.0

        # 可选推进位姿
        if enable_state_step:
            x = maybe_step_states(x, assignment, t, global_vars.get_all_vars(), dt)
            global_vars.x = x

        # 计算 F 与能力相关度量
        F = compute_F_matrix(scenario_params)
        cap_metrics = compute_capability_utilization_metrics(alpha_mat, F, scenario_params['T'])
        min_robots = estimate_minimal_robot_count_per_task(F, scenario_params['T'])

        # 并行度与切换
        par_sw = compute_parallelism_and_switching(assignment, prev_assignment, n_t)

        # 动态变化: 在设定步数注入变化，然后重新创建规划器以反映 A/T 变化
        if (not changed_applied) and (dynamic_change_step is not None) and (step == dynamic_change_step):
            apply_dynamic_capability_drop(scenario_params, drop_ratio=0.5, feature_index_for_manipulation=2)
            append_exploration_targets(global_vars, num_new=2, radius=1.0)
            # 重新创建规划器以刷新映射
            planner = create_planner(planner_type, scenario_params, opt_params, global_vars)
            changed_applied = True

        # 记录是否恢复（所有任务的覆盖率 >= 0.99 视为恢复）
        if changed_applied and recovered_after_change_step is None:
            if np.all(np.array(cap_metrics['coverage_per_task']) >= 0.99):
                recovered_after_change_step = step

        # 记录一步的结果
        metrics_log['per_step'].append({
            'step': step,
            'time': t,
            'solve_time': solve_time,
            'assignment': assignment,
            'coverage_per_task': cap_metrics['coverage_per_task'],
            'avg_coverage': cap_metrics['avg_coverage'],
            'redundancy_ratio_per_task': cap_metrics['redundancy_ratio_per_task'],
            'avg_redundancy_ratio': cap_metrics['avg_redundancy_ratio'],
            'parallel_tasks': par_sw['parallel_tasks'],
            'parallel_ratio': par_sw['parallel_ratio'],
            'switch_count': par_sw['switch_count'],
            'switch_ratio': par_sw['switch_ratio'],
            'minimal_robot_count_per_task': min_robots,
        })

        prev_assignment = assignment

    # 汇总
    per_step = metrics_log['per_step']
    avg_coverage = float(np.mean([s['avg_coverage'] for s in per_step]))
    avg_parallel_ratio = float(np.mean([s['parallel_ratio'] for s in per_step]))
    avg_switch_ratio = float(np.mean([s['switch_ratio'] for s in per_step if s['switch_ratio'] is not None]))
    avg_redundancy_ratio = float(np.mean([s['avg_redundancy_ratio'] for s in per_step]))

    metrics_log['summary'] = {
        'avg_coverage': avg_coverage,
        'avg_parallel_ratio': avg_parallel_ratio,
        'avg_switch_ratio': avg_switch_ratio,
        'avg_redundancy_ratio': avg_redundancy_ratio,
        'recovered_after_change_step': recovered_after_change_step,
    }

    return metrics_log


def main():
    parser = argparse.ArgumentParser(description='MRTA 消融实验')
    parser.add_argument('--planner', type=str, default='all', choices=['all', 'miqp', 'greedy', 'linear_programming'])
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--replan_every', type=int, default=1)
    parser.add_argument('--dynamic_step', type=int, default=30)
    parser.add_argument('--no_state_step', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    planners = ['miqp', 'greedy', 'linear_programming'] if args.planner == 'all' else [args.planner]
    results = {}

    for p in planners:
        log = run_single_experiment(
            planner_type=p,
            total_steps=args.steps,
            dt=args.dt,
            replan_every=args.replan_every,
            dynamic_change_step=args.dynamic_step,
            enable_state_step=not args.no_state_step,
            seed=args.seed,
        )
        results[p] = log

    # 写入结果
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(out_dir, f'ablation_results_{ts}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f'结果已保存: {out_path}')


if __name__ == '__main__':
    main()


