import numpy as np
import time
from habitat_llm.agent.agent import Agent
from habitat_llm.planner.HRCS.params_module.scenario_params_task import ScenarioConfigTask
from habitat_llm.planner.HRCS.params_module.opt_params_task import OptimizationConfigTask
from habitat_llm.planner.HRCS.class_def.RTA_task import RTA
from typing import List

class MIQPSolverWrapper:
    def __init__(self):
        self.scenario_params = None
        self.opt_params = None
        self.rta = None

    def reset(self):
        """
        Reset the MIQP solver wrapper state.
        Clear all cached parameters and solver instances.
        """
        self.scenario_params = None
        self.opt_params = None
        self.rta = None
        print("[DEBUG] MIQPSolverWrapper reset completed")

    def task_plan_MIQP_set(self, agents: List["Agent"]):
        """设置MIQP参数"""
        try:
            n_agents = len(agents)
            self.scenario_params = ScenarioConfigTask(n_r=n_agents)
            opt_manager = OptimizationConfigTask(n_r=n_agents, n_t=13)
            self.opt_params = opt_manager
            
            self.rta = RTA(self.scenario_params.get_scenario_params(), self.opt_params.get_opt_params())
            
        except ImportError:
            print("MIQP modules not available, using placeholder")
            self.scenario_params = None
            self.opt_params = {}
            self.rta = None

    def task_plan_MIQP_solve_phase_aware(self, x, t, phase_task_info, agents):
        """
        Solve the MIQP problem for a specific phase with dynamic tasks.
        """
        if not self.rta:
            print("[ERROR] RTA not initialized. Call task_plan_MIQP_set first.")
            return None, None, None, 0, "RTA Uninitialized"

        if not self.scenario_params or not self.opt_params:
            print("[ERROR] Scenario or Optimization parameters not set.")
            return None, None, None, 0, "Params Unset"
        
        # 从 scenario_params 获取最新的全局变量
        global_vars = self.scenario_params.get_global_task_vars()

        n_instances = phase_task_info.get('n_phase_tasks', 0)
        if n_instances == 0:
            print("[WARNING] No task instances in the current phase to solve.")
            return np.array([]), np.zeros(len(agents) * 2), np.array([]), 0, "No Tasks"

        # 动态构建用于RTA的场景参数
        phase_scenario = {
            'T': phase_task_info.get('matrix'),
            'n_r_bounds': phase_task_info.get('n_r_bounds'),
            'tasks': phase_task_info.get('tasks', []),
            'aptitude_matrix': phase_task_info.get('aptitude_matrix')
        }

        # 调用RTA求解器，并传入最新的全局变量
        alpha, u, delta, time_to_solve, opt_sol_info = self.rta.solve_miqp_phase_aware(
            x, t, phase_scenario, n_instances, global_vars
        )

        # 调整alpha矩阵的形状以便于后续处理
        if alpha is not None:
            num_agents = len(agents)
            # The shape should be (num_agents, num_instances)
            # alpha is flattened, so we reshape it.
            try:
                # Reshape alpha to be [agents, tasks]
                alpha = alpha.reshape((num_agents, n_instances))
            except ValueError:
                # If reshape fails, it might be an issue with solver output.
                print(f"[ERROR] MIQP wrapper: Could not reshape alpha matrix. Expected size {num_agents * n_instances}, got {alpha.size}")

        return alpha, u, delta, time_to_solve, opt_sol_info

    def task_plan_MIQP_solve(self, x, t, agents):
        """
        Legacy MIQP solver for backward compatibility. This should not be used for phase-based planning.
        """
        print("[WARNING] Legacy task_plan_MIQP_solve called. This method is deprecated for phase-aware planning.")
        # Fallback to a heuristic or default behavior if called unexpectedly
        n_agents = len(agents)
        alpha = np.zeros((n_agents, 13))  # Assume 13 fixed tasks for legacy mode
        u = np.zeros((3, n_agents))
        delta = np.zeros(13)
        return alpha, u, delta, 0.0, "LEGACY_FALLBACK" 

    def solve_lp_phase_aware(self, x, t, phase_task_info, agents):
        """
        基于线性规划松弛+匈牙利算法的实例感知求解（Phase-aware LP）。
        - 动态按当前phase的任务实例数构建效用/成本矩阵
        - 使用匈牙利算法得到离散分配解
        - 支持可选的aptitude偏好代价（与RTA实例感知版本类似）
        返回: (alpha, u, delta, time_to_solve, opt_sol_info)
        其中alpha为形状 [n_agents, n_instances]
        """
        start_time = time.time()

        # 1) 校验可用性
        if not self.scenario_params or not self.opt_params:
            print("[ERROR] Scenario or Optimization parameters not set.")
            return None, None, None, 0, "Params Unset"

        # 2) 读取全局变量与phase实例信息
        try:
            global_vars = self.scenario_params.get_global_task_vars()
        except Exception:
            global_vars = None

        n_agents = len(agents)
        n_instances = phase_task_info.get('n_phase_tasks', 0)
        if n_instances == 0:
            print("[WARNING] No task instances in the current phase to solve (LP).")
            return np.array([]), np.zeros(len(agents) * 2), np.array([]), 0, "No Tasks"

        # 3) 获取基础场景与任务定义；构造实例->任务类型映射
        base_scenario = self.scenario_params.get_scenario_params()
        base_tasks = base_scenario.get('tasks', [])
        robot_dyn = base_scenario.get('robot_dyn', {})
        n_u = int(robot_dyn.get('n_u', 0)) if isinstance(robot_dyn, dict) else 0

        task_name_to_index = {}
        for idx, task in enumerate(base_tasks):
            if isinstance(task, dict) and task.get('name'):
                task_name_to_index[task['name']] = idx

        def resolve_task_type_index(task_entry, fallback_j):
            # 支持{'task_type': name} 或 {'name': name} 或 直接使用索引回退
            if isinstance(task_entry, dict):
                name = task_entry.get('task_type') or task_entry.get('name')
                if name in task_name_to_index:
                    return task_name_to_index[name]
                # 允许直接给出索引
                if 'type_idx' in task_entry and isinstance(task_entry['type_idx'], int):
                    return task_entry['type_idx']
            # 回退：按位置索引
            return fallback_j

        tasks_phase = phase_task_info.get('tasks', [])

        # 4) 构建成本矩阵（负效用 + aptitude惩罚）
        #    匈牙利算法求解最小成本，故将效用取负作为成本
        import numpy as _np
        cost_matrix = _np.zeros((n_agents, n_instances), dtype=float)

        # aptitude可选项（形状应为 n_agents x n_instances）
        aptitude_matrix = phase_task_info.get('aptitude_matrix', None)
        use_aptitude = (
            aptitude_matrix is not None and
            isinstance(aptitude_matrix, _np.ndarray) and
            aptitude_matrix.shape == (n_agents, n_instances)
        )
        aptitude_weight = 1.0

        for i in range(n_agents):
            for j in range(n_instances):
                try:
                    type_idx = resolve_task_type_index(
                        tasks_phase[j] if j < len(tasks_phase) else {}, j
                    )
                    # 边界检查
                    if type_idx < 0 or type_idx >= len(base_tasks):
                        raise IndexError(f"task type index {type_idx} out of range")
                    task_def = base_tasks[type_idx]
                    # 统一传参：vars_dict允许为None
                    if global_vars is not None:
                        util = task_def['function'](x[:, i], t, i, vars_dict=global_vars)
                    else:
                        util = task_def['function'](x[:, i], t, i, vars_dict=None)

                    # 成本为负效用
                    cost = -float(util) if util is not None else 0.0

                    # 叠加aptitude偏好（越不擅长成本越高）
                    if use_aptitude:
                        inv_ap = 1.0 / (float(aptitude_matrix[i, j]) + 1e-6)
                        cost += aptitude_weight * inv_ap

                    cost_matrix[i, j] = cost
                except Exception as e:
                    print(f"[LP-PhaseAware] 计算成本出错 Robot {i}, Inst {j}: {e}")
                    cost_matrix[i, j] = 0.0

        # 5) 使用匈牙利算法进行分配（必要时进行方阵扩展）
        try:
            from scipy.optimize import linear_sum_assignment
            extended = False
            if n_agents > n_instances:
                # 扩展列（添加虚拟任务）
                pad = _np.zeros((n_agents, n_agents - n_instances))
                cost_ext = _np.hstack([cost_matrix, pad])
                extended = True
            elif n_instances > n_agents:
                # 扩展行（添加虚拟机器人）
                pad = _np.zeros((n_instances - n_agents, n_instances))
                cost_ext = _np.vstack([cost_matrix, pad])
                extended = True
            else:
                cost_ext = cost_matrix

            rows, cols = linear_sum_assignment(cost_ext)

            # 6) 构建输出alpha（[n_agents, n_instances]），u与delta设为0
            alpha_mat = _np.zeros((n_agents, n_instances))
            for r, c in zip(rows, cols):
                if r < n_agents and c < n_instances:
                    alpha_mat[r, c] = 1.0

            u = _np.zeros(n_agents * n_u)
            delta = _np.zeros(n_agents * n_instances)

            solve_time = time.time() - start_time
            # 返回与MIQP包装器风格一致的二维alpha
            return alpha_mat, u, delta, solve_time, "LP + Hungarian (Phase-aware)"

        except ImportError:
            # 回退：简单优先/轮询分配
            print("[WARNING] scipy 不可用，LP实例感知求解回退到简单分配。")
            alpha_mat = _np.zeros((n_agents, n_instances))
            # 简单策略：按每个机器人选择当前列成本最低的未占用任务
            assigned = set()
            for i in range(n_agents):
                avail = [j for j in range(n_instances) if j not in assigned]
                if not avail:
                    break
                j_star = min(avail, key=lambda jj: cost_matrix[i, jj])
                alpha_mat[i, j_star] = 1.0
                assigned.add(j_star)
            u = _np.zeros(n_agents * n_u)
            delta = _np.zeros(n_agents * n_instances)
            solve_time = time.time() - start_time
            return alpha_mat, u, delta, solve_time, "LP Fallback (No SciPy)"

    def task_plan_LP_solve_phase_aware(self, x, t, phase_task_info, agents):
        """
        使用LP+匈牙利算法的实例感知任务分配封装方法。
        行为与`task_plan_MIQP_solve_phase_aware`一致，但内部调用`solve_lp_phase_aware`。
        """
        # 仅需检查参数是否初始化；LP不依赖RTA对象
        if not self.scenario_params or not self.opt_params:
            print("[ERROR] Scenario or Optimization parameters not set.")
            return None, None, None, 0, "Params Unset"

        n_instances = phase_task_info.get('n_phase_tasks', 0)
        if n_instances == 0:
            print("[WARNING] No task instances in the current phase to solve (LP wrapper).")
            return np.array([]), np.zeros(len(agents) * 2), np.array([]), 0, "No Tasks"

        # 调用LP实例感知求解
        alpha, u, delta, time_to_solve, opt_sol_info = self.solve_lp_phase_aware(
            x, t, phase_task_info, agents
        )

        # 形状统一处理：确保alpha为 [n_agents, n_instances]
        if alpha is not None:
            num_agents = len(agents)
            try:
                if alpha.ndim == 1:
                    alpha = alpha.reshape((num_agents, n_instances))
                elif alpha.ndim == 2 and alpha.shape != (num_agents, n_instances):
                    alpha = alpha[:num_agents, :n_instances]
            except Exception as e:
                print(f"[ERROR] LP wrapper: Could not reshape alpha matrix: {e}")

        return alpha, u, delta, time_to_solve, opt_sol_info