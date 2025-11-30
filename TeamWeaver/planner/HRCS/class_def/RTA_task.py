import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import cvxpy as cp
import sys
# import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# task_module_dir = os.path.join(current_dir, '..', '..', '..', 'task-plan', 'class_def', 'task_module')
# if task_module_dir not in sys.path:
#     sys.path.append(task_module_dir)

from habitat_llm.planner.HRCS.class_def.task_utility_normalizer import TaskUtilityNormalizer, TaskPriorityConfig
from habitat_llm.planner.HRCS.class_def.RTA_task_analyzer import RTA_task_analyzer

class RTA:
    def __init__(self, scenario_params, opt_params, task_priority_config=None):
        """
        初始化RTA类
        """
        assert all(field in scenario_params for field in ['A', 'Hs', 'T', 'ws', 'robot_dyn', 'tasks']), '缺少场景参数'
        assert all(field in opt_params for field in ['l', 'kappa', 'gamma', 'n_r_bounds', 'delta_max']), '缺少优化参数'
        
        self.scenario_params_ = scenario_params
        self.opt_params_ = opt_params
        
        # 维度信息
        self.dim_ = {}
        self.dim_['n_r'] = scenario_params['A'].shape[1]  # 机器人数量
        self.dim_['n_t'] = scenario_params['T'].shape[0]  # 任务数量
        self.dim_['n_c'] = scenario_params['T'].shape[1]  # 能力数量
        self.dim_['n_f'] = scenario_params['A'].shape[0]  # 特征数量
        self.dim_['n_x'] = scenario_params['robot_dyn']['n_x']  # 状态维度
        self.dim_['n_u'] = scenario_params['robot_dyn']['n_u']  # 输入维度
        
        # 计算映射和专业化
        self.evaluate_mappings_and_specializations()
        self.check_tasks()
        
        # 初始化约束字典
        self.constraints_ = {}
        self.global_vars_manager_ = None
        self.task_utility_normalizer_ = TaskUtilityNormalizer(
            self.dim_, 
            self.scenario_params_['tasks'],
            task_priority_config
        )
        self.analyzer = RTA_task_analyzer(self)
        self.task_name_to_index_ = {
            task['name']: i for i, task in enumerate(self.scenario_params_['tasks']) if task and 'name' in task
        }
    
    def reset(self):
        """
        Resets the RTA state.
        This method can be expanded to clear caches or re-initialize components if needed.
        """
        # Currently, RTA is mostly stateless between solves, but this provides a hook.
        self.constraints_ = {}
        # Re-evaluating mappings might be necessary if scenario params can change mid-episode
        # self.evaluate_mappings_and_specializations()
        # print("[DEBUG] RTA reset completed")
    
    def get_global_vars_manager(self):
        """获取全局变量管理器"""
        if self.global_vars_manager_ is not None:
            return self.global_vars_manager_
            
        try:
            for module_name, module in sys.modules.items():
                if hasattr(module, 'global_vars'):
                    self.global_vars_manager_ = getattr(module, 'global_vars')
                    return self.global_vars_manager_
            return None
        except Exception as e:
            print(f"Error: RTA - get_global_vars_manager: {e}")
            return None
    
    def get_global_vars_dict(self):
        manager = self.get_global_vars_manager()
        if manager is not None and hasattr(manager, 'get_all_vars'):
            return manager.get_all_vars()
        return None
    
    def normalize_task_utilities(self, x, t, global_vars_dict=None, llm_response=""):
        if global_vars_dict is None:
            global_vars_dict = self.get_global_vars_dict()
        return self.task_utility_normalizer_.calculate_scaling_factors(x, t, global_vars_dict, llm_response)
        
    def solve_miqp(self, x, t):
        """使用CVXPY/Gurobi解决MIQP优化问题"""
        
        # 如果开启调试模式，使用详细分析版本
        if True:
            alpha, u, delta, solve_time, status, constraints_info = self.analyzer.solve_miqp_with_detailed_analysis(x, t)
            return alpha, u, delta, solve_time, status
        
        # 标准版本
        self.build_constraints(x, t)
        
        alpha_dim = self.dim_['n_r'] * self.dim_['n_t']
        u_dim = self.dim_['n_r'] * self.dim_['n_u']
        delta_dim = self.dim_['n_r'] * self.dim_['n_t']
        start_time = time.time()
        
        try:
            alpha_var = cp.Variable(alpha_dim, boolean=True)
            u_var = cp.Variable(u_dim)
            delta_var = cp.Variable(delta_dim)
            
            P_squared = self.P_.T @ self.P_
            S_diag = np.diag(np.reshape(self.scenario_params_['S'], (-1)))
            
            alpha_cost = 1e6 * max(1, self.opt_params_['l']) * cp.quad_form(alpha_var, P_squared)
            u_cost = cp.quad_form(u_var, np.eye(u_dim))
            delta_cost = self.opt_params_['l'] * cp.quad_form(delta_var, S_diag)
            objective = cp.Minimize(alpha_cost + u_cost + delta_cost)
            
            constraints = []
            all_vars_h = cp.hstack([alpha_var, u_var, delta_var])
            all_vars = all_vars_h.T
            constraints.append(self.constraints_['A_ineq'] @ all_vars <= self.constraints_['b_ineq']) # 添加线性不等式约束
            constraints.append(self.constraints_['A_eq'] @ all_vars == self.constraints_['b_eq']) # 添加等式约束
            constraints.append(alpha_var >= self.constraints_['lb'][:alpha_dim]) # 添加变量边界约束
            constraints.append(alpha_var <= self.constraints_['ub'][:alpha_dim])
            
            lb_idx = alpha_dim + u_dim
            constraints.append(delta_var >= self.constraints_['lb'][lb_idx:lb_idx+delta_dim])
            constraints.append(delta_var <= self.constraints_['ub'][lb_idx:lb_idx+delta_dim])
            
            # 使用更宽松的求解参数以提高数值稳定性
            solve_params = {
                'NumericFocus': 2,  # 降低数值精度要求
                'FeasibilityTol': 1e-6,  # 放宽可行性容差
                'OptimalityTol': 1e-6,   # 放宽最优性容差
                'IntFeasTol': 1e-6,      # 放宽整数可行性容差
                'MIPGap': 1e-4          # 允许一定的MIP gap
            }
            problem = cp.Problem(objective, constraints)
            
            problem.solve(solver=cp.GUROBI, verbose=False, **solve_params)
            time_to_solve_miqp = time.time() - start_time
            
            if problem.status in ["infeasible_or_unbounded", "unknown"]:
                print(f"[DEBUG] Problem status unclear: {problem.status}, re-solving for precise diagnosis...")
                try:
                    problem.solve(solver=cp.GUROBI, verbose=True, reoptimize=True)
                except Exception as e:
                    print(f"[DEBUG] Re-solve failed: {e}")
                    problem.solve(solver=cp.GUROBI, verbose=True, 
                                MIPGap=1e-4, 
                                MIPGapAbs=1e-4,
                                NumericFocus=2)
            
            if problem.status == cp.OPTIMAL:
                alpha = alpha_var.value
                u = u_var.value
                delta = delta_var.value
                opt_sol_info = "Optimal"
            elif problem.status == cp.INFEASIBLE or problem.status == "infeasible":
                print(f"[ERROR] 优化问题不可行！检查约束条件...")
                self.analyzer._diagnose_infeasible_constraints()
                alpha = np.zeros(alpha_dim)
                u = np.zeros(u_dim)
                delta = np.zeros(delta_dim)
                opt_sol_info = f"Infeasible: {problem.status}"
            elif problem.status == cp.UNBOUNDED or problem.status == "unbounded":
                print(f"[ERROR] 优化问题无界！检查目标函数...")
                alpha = np.zeros(alpha_dim)
                u = np.zeros(u_dim)
                delta = np.zeros(delta_dim)
                opt_sol_info = f"Unbounded: {problem.status}"
            else:
                print(f"优化未收敛，状态: {problem.status}")
                print(f"[DEBUG] 开始诊断约束条件...")
                self.analyzer._diagnose_infeasible_constraints()
                alpha = np.zeros(alpha_dim)
                u = np.zeros(u_dim)
                delta = np.zeros(delta_dim)
                opt_sol_info = f"Not optimal: {problem.status}"
            
            print("MIQP Status:", opt_sol_info)
            
            return alpha, u, delta, time_to_solve_miqp, opt_sol_info
            
        except cp.error.SolverError as e:
            print(f"CVXPY求解器错误: {e}")
            return np.zeros(alpha_dim), np.zeros(u_dim), np.zeros(delta_dim), 0, "Error"
        
    def solve_reduced_qp(self, x, alpha, t):
        """求解固定alpha下的简化QP问题"""
        self.build_reduced_constraints(x, t)
        
        alpha_dim = self.dim_['n_r'] * self.dim_['n_t']
        u_dim = self.dim_['n_r'] * self.dim_['n_u']
        delta_dim = self.dim_['n_r'] * self.dim_['n_t']
        
        start_time = time.time()
        
        model = gp.Model("RTA_QP")
        model.setParam('OutputFlag', 0)
        
        u_var = model.addVars(u_dim, lb=-float('inf'), ub=float('inf'), name="u")
        delta_var = model.addVars(delta_dim, lb=0, ub=self.opt_params_['delta_max'], name="delta")
        u_flat = [u_var[i] for i in range(u_dim)]
        delta_flat = [delta_var[i] for i in range(delta_dim)]
        
        all_vars = u_flat + delta_flat
        
        S_diag = np.reshape(self.scenario_params_['S'], (-1))
        
        for i in range(u_dim):
            model.addQConstr(2 * u_var[i] * u_var[i], GRB.EQUAL, model.addVar(name=f"q_u_{i}"))
            
        for i in range(delta_dim):
            model.addQConstr(2 * self.opt_params_['l'] * S_diag[i] * delta_var[i] * delta_var[i], 
                             GRB.EQUAL, 
                             model.addVar(name=f"q_delta_{i}"))
        
        A = self.constraints_['A_ineq'][:, alpha_dim:]
        b = self.constraints_['b_ineq'] - self.constraints_['A_ineq'][:, :alpha_dim] @ alpha
        
        for i in range(A.shape[0]):
            row = A[i, :]
            expr = 0
            for j in range(len(all_vars)):
                if row[j] != 0:
                    expr += row[j] * all_vars[j]
            model.addConstr(expr <= b[i])
        
        for i in range(u_dim + delta_dim):
            idx = alpha_dim + i
            model.addConstr(all_vars[i] >= self.constraints_['lb'][idx])
            model.addConstr(all_vars[i] <= self.constraints_['ub'][idx])
        
        model.optimize()
        
        time_to_solve_qp = time.time() - start_time
        
        if model.status == GRB.OPTIMAL:
            u = np.array([u_var[i].X for i in range(u_dim)])
            delta = np.array([delta_var[i].X for i in range(delta_dim)])
            opt_sol_info = "Optimal"
        else:
            u = np.zeros(u_dim)
            delta = np.zeros(delta_dim)
            opt_sol_info = f"Not optimal: {model.status}"
            
        return u, delta, time_to_solve_qp, opt_sol_info
    
    def get_scaled_task_values(self, x, i, j, **kwargs):
        """
        获取缩放后的任务函数值、梯度和时间导数。
        """
        task_types_in_phase = kwargs.get('task_types_in_phase')

        if task_types_in_phase:
            # Phase-aware mode: j is the instance index.
            task_name = task_types_in_phase[j]
            if not task_name or task_name not in self.task_name_to_index_:
                print(f"[ERROR] Invalid or missing task type '{task_name}' for instance {j}. Skipping CBF.")
                return 0.0, np.zeros(self.dim_['n_x']), 0.0
            task_type_index = self.task_name_to_index_[task_name]
        else:
            # Legacy mode: j is the task type index.
            task_type_index = j

        if task_type_index >= len(self.scenario_params_['tasks']) or not self.scenario_params_['tasks'][task_type_index]:
            print(f"[ERROR] Task index {task_type_index} is out of bounds or task definition is null.")
            return 0.0, np.zeros(self.dim_['n_x']), 0.0
        
        task = self.scenario_params_['tasks'][task_type_index]
        
        # 从 kwargs 获取可选参数，提供默认值
        t = kwargs.get('t', 0.0)
        global_vars_dict = kwargs.get('global_vars_dict', self.get_global_vars_dict())
        scaling_factors = kwargs.get('scaling_factors', None)

        # 准备传递给任务函数的参数
        task_kwargs = {}
        if global_vars_dict is not None:
            task_kwargs['vars_dict'] = global_vars_dict

        # 调用底层任务函数
        task_func_value = task['function'](x[:, i], t, i, **task_kwargs)
        task_grad_value = task['gradient'](x[:, i], t, i, **task_kwargs)
        task_time_deriv_value = task['time_derivative'](x[:, i], t, i, **task_kwargs)
        
        # 如果提供了缩放因子，则应用缩放 (当前逻辑为注释状态)
        if scaling_factors is not None and j < len(scaling_factors):
            task_func_value *= scaling_factors[j]
            task_grad_value *= scaling_factors[j]
            task_time_deriv_value *= scaling_factors[j]
            # pass

        return task_func_value, task_grad_value, task_time_deriv_value
        
    def build_constraints(self, x, t):
        """构建MIQP约束"""
        n_r = self.dim_['n_r']
        n_t = self.dim_['n_t']
        n_c = self.dim_['n_c']
        n_u = self.dim_['n_u']
        
        print(f"[DEBUG] Building constraints for n_r={n_r}, n_t={n_t}, n_c={n_c}, n_u={n_u}")
        
        # cbf_constraints = n_r * n_t
        # simplified_delta_alpha_constraints = n_r * min(n_t, 3)
        # cbf_slack_constraints = n_r * n_t
        capability_constraints = n_t * n_c
        robot_bound_constraints = 2 * n_t
        
        # total_ineq = cbf_constraints + cbf_slack_constraints + capability_constraints + robot_bound_constraints
        total_ineq = capability_constraints + robot_bound_constraints
        total_vars = 2*n_r*n_t + n_r*n_u
        
        print(f"[DEBUG] Simplified constraint matrix dimensions: {total_ineq} x {total_vars}")
        # print(f"[DEBUG] Delta-Alpha constraints reduced from {n_r*n_t*(n_t-1)} to {simplified_delta_alpha_constraints}")
        
        A_ineq = np.zeros((total_ineq, total_vars))
        b_ineq = np.zeros(total_ineq)
        A_eq = np.zeros((n_r, total_vars))
        b_eq = np.ones(n_r)
        lb = -np.inf * np.ones(total_vars)
        ub = np.inf * np.ones(total_vars)
        
        # 设置变量边界
        lb[:n_r*n_t] = np.zeros(n_r*n_t)
        ub[:n_r*n_t] = np.ones(n_r*n_t)
        lb[n_r*n_t+n_r*n_u:] = np.zeros(n_r*n_t)
        ub[n_r*n_t+n_r*n_u:] = self.opt_params_['delta_max'] * np.ones(n_r*n_t)
        
        global_vars_dict = self.get_global_vars_dict()
        
        # Get LLM Response
        llm_response = ""
        if global_vars_dict and 'current_llm_response' in global_vars_dict:
            llm_response = global_vars_dict['current_llm_response']
        
        # 归一化与任务对应化调整后续统一
        # scaling_factors = self.normalize_task_utilities(x, t, global_vars_dict, llm_response)
        # print(f"[DEBUG-LYP] scaling_factors: {scaling_factors}")
        
        # constraint_idx = 0
        # === 1. CBF约束 (Control Barrier Functions) ===
        # print(f"[DEBUG] Adding CBF constraints...")
        # for i in range(n_r):
        #     for j in range(n_t):
        #         task = self.scenario_params_['tasks'][j]
        #         robot_dyn = self.scenario_params_['robot_dyn']
                
        #         # 获取缩放后的任务函数值、梯度和时间导数
        #         task_func_value, task_grad_value, task_time_deriv_value = self.get_scaled_task_values(
        #             x, i, j, t=t, global_vars_dict=global_vars_dict, scaling_factors=scaling_factors
        #         )
                
        #         # CBF约束: dot(h) + gamma(h) >= 0
        #         A_ineq[constraint_idx, n_r*n_t+i*n_u:n_r*n_t+(i+1)*n_u] = -task_grad_value @ robot_dyn['g'](x[:, i])
        #         b_ineq[constraint_idx] = (task_grad_value @ robot_dyn['f'](x[:, i]) + 
        #                                  task_time_deriv_value + 
        #                                  self.opt_params_['gamma'](task_func_value))
        #         constraint_idx += 1
        
        # === 2. 简化的Delta-Alpha约束 (大幅减少的任务切换约束) ===
        # print(f"[DEBUG] Adding simplified delta-alpha constraints...")
        # # 定义关键任务索引：Navigate(0), Pick(2), Place(3) - 最常用的任务
        # critical_tasks = [0, 2, 3] if n_t > 3 else list(range(min(n_t, 3)))
        
        # for i in range(n_r):
        #     constraint_count = 0
        #     for j_idx, j in enumerate(critical_tasks):
        #         for k_idx, k in enumerate(critical_tasks):
        #             if j != k and constraint_count < min(n_t, 3):
        #                 # delta_ij >= (alpha_ij - alpha_ik) * delta_max / kappa
        #                 A_ineq[constraint_idx, i*n_t + j] = self.opt_params_['delta_max']  # alpha_ij
        #                 A_ineq[constraint_idx, i*n_t + k] = -self.opt_params_['delta_max']  # alpha_ik
        #                 A_ineq[constraint_idx, n_r*n_t+n_r*n_u+i*n_t+j] = -1/self.opt_params_['kappa']  # delta_ij
        #                 b_ineq[constraint_idx] = 0
        #                 constraint_idx += 1
        #                 constraint_count += 1
        
        # === 3. CBF slack variable约束 ===
        # print(f"[DEBUG] Adding CBF slack constraints...")
        # slack_start_idx = cbf_constraints
        # for i in range(n_r):
        #     for j in range(n_t):
        #         slack_idx = slack_start_idx + i*n_t + j
        #         if slack_idx < total_ineq:  # 边界检查
        #             A_ineq[slack_idx, n_r*n_t+n_r*n_u+i*n_t+j] = -1
        #             b_ineq[slack_idx] = 0
        
        # === 4. 能力约束 (Feature capability constraints) ===
        print(f"[DEBUG] Adding capability constraints...")
        # cap_start_idx = cbf_constraints + cbf_slack_constraints
        cap_start_idx = 0
        for j in range(n_t):
            for c in range(n_c):
                cap_idx = cap_start_idx + j*n_c + c
                if cap_idx < total_ineq:  # 边界检查
                    # F * alpha >= T: 确保分配的机器人具备执行任务j所需的能力c
                    for r in range(n_r):
                        A_ineq[cap_idx, r*n_t+j] = -self.scenario_params_['F'][c, r]
                    b_ineq[cap_idx] = -self.scenario_params_['T'][j, c]
        
        # === 5. 机器人数量约束 (Robot count bounds) ===
        print(f"[DEBUG] Adding robot count constraints...")
        bound_start_idx = cap_start_idx + n_t*n_c
        for j in range(n_t):
            # 最大机器人数约束: sum(alpha_rj) <= max_robots_j
            max_idx = bound_start_idx + j
            if max_idx < total_ineq:  # 边界检查
                for r in range(n_r):
                    A_ineq[max_idx, r*n_t+j] = 1
                b_ineq[max_idx] = self.opt_params_['n_r_bounds'][j, 1]
            
            # 最小机器人数约束: sum(alpha_rj) >= min_robots_j
            min_idx = bound_start_idx + n_t + j
            if min_idx < total_ineq:  # 边界检查
                for r in range(n_r):
                    A_ineq[min_idx, r*n_t+j] = -1
                b_ineq[min_idx] = -self.opt_params_['n_r_bounds'][j, 0]
        
        # print(f"[DEBUG] Constraints building completed. Final constraint_idx: {constraint_idx}")
        print(f"[DEBUG] Used constraint indices: {capability_constraints + robot_bound_constraints}")
        # print(f"[DEBUG] Used constraint indices: {cbf_constraints + cbf_slack_constraints + capability_constraints + robot_bound_constraints}")
        
        # 验证约束矩阵一致性
        expected_total = capability_constraints + robot_bound_constraints
        if expected_total != total_ineq:
            print(f"[ERROR] Constraint matrix size calculation error!")
            # print(f"  CBF constraints: {cbf_constraints}")
            # print(f"  Simplified Delta-alpha constraints: {simplified_delta_alpha_constraints}")
            # print(f"  CBF slack constraints: {cbf_slack_constraints}")
            print(f"  Capability constraints: {capability_constraints}")
            print(f"  Robot bound constraints: {robot_bound_constraints}")
            print(f"  Expected total: {expected_total}")
            print(f"  Actual matrix rows: {total_ineq}")
            
            # 调整矩阵大小以匹配实际需要
            if expected_total > total_ineq:
                additional_rows = expected_total - total_ineq
                A_ineq = np.vstack([A_ineq, np.zeros((additional_rows, total_vars))])
                b_ineq = np.hstack([b_ineq, np.zeros(additional_rows)])
                print(f"[DEBUG] Extended constraint matrix to {A_ineq.shape}")
            elif expected_total < total_ineq:
                # 截断矩阵
                A_ineq = A_ineq[:expected_total, :]
                b_ineq = b_ineq[:expected_total]
                print(f"[DEBUG] Truncated constraint matrix to {A_ineq.shape}")
                
        # 检查矩阵有效性
        if np.any(np.isnan(A_ineq)) or np.any(np.isinf(A_ineq)):
            print(f"[ERROR] A_ineq contains NaN or Inf values!")
        if np.any(np.isnan(b_ineq)) or np.any(np.isinf(b_ineq)):
            print(f"[ERROR] b_ineq contains NaN or Inf values!")
        if np.any(np.isnan(A_eq)) or np.any(np.isinf(A_eq)):
            print(f"[ERROR] A_eq contains NaN or Inf values!")
        if np.any(np.isnan(b_eq)) or np.any(np.isinf(b_eq)):
            print(f"[ERROR] b_eq contains NaN or Inf values!")
        
        # 添加等式约束：每个机器人必须分配到至少一个任务
        print(f"[DEBUG] Adding equality constraints (each robot assigned to exactly one task)...")
        for i in range(n_r):
            for j in range(n_t):
                A_eq[i, i*n_t + j] = 1
        
        print(f"[DEBUG] Equality constraint matrix A_eq shape: {A_eq.shape}")
        print(f"[DEBUG] Equality constraint RHS b_eq: {b_eq}")
        print(f"[DEBUG] This enforces: each robot must be assigned to exactly one task")
        
        self.constraints_['A_ineq'] = A_ineq
        self.constraints_['b_ineq'] = b_ineq
        self.constraints_['A_eq'] = A_eq
        self.constraints_['b_eq'] = b_eq
        self.constraints_['lb'] = lb
        self.constraints_['ub'] = ub
        
    def build_reduced_constraints(self, x, t):
        """构建简化约束（固定alpha的QP）"""
        n_r = self.dim_['n_r']
        n_t = self.dim_['n_t']
        n_u = self.dim_['n_u']
        
        # 初始化约束矩阵
        total_ineq = n_r*n_t + n_r*n_t**2
        total_vars = 2*n_r*n_t + n_r*n_u
        
        A_ineq = np.zeros((total_ineq, total_vars))
        b_ineq = np.zeros(total_ineq)
        A_eq = np.zeros((n_r, total_vars))
        b_eq = np.zeros(n_r)
        lb = -np.inf * np.ones(total_vars)
        ub = np.inf * np.ones(total_vars)
        
        lb[n_r*n_t+n_r*n_u:] = np.zeros(n_r*n_t)
        ub[n_r*n_t+n_r*n_u:] = self.opt_params_['delta_max'] * np.ones(n_r*n_t)
        global_vars_dict = self.get_global_vars_dict()
        
        # 获取LLM响应信息
        llm_response = ""
        if global_vars_dict and 'current_llm_response' in global_vars_dict:
            llm_response = global_vars_dict['current_llm_response']
        
        scaling_factors = self.normalize_task_utilities(x, t, global_vars_dict, llm_response)
        
        # Task CBFs and delta-alpha constraints
        for i in range(n_r):
            for j in range(n_t):
                # CBFs for tasks
                idx = (i*n_t) + j
                
                # 获取缩放后的任务函数值、梯度和时间导数
                task_func_value, task_grad_value, task_time_deriv_value = self.get_scaled_task_values(
                    x, i, j, t=t, global_vars_dict=global_vars_dict, scaling_factors=scaling_factors
                )
                
                robot_dyn = self.scenario_params_['robot_dyn']
                
                A_ineq[idx, n_r*n_t+(i*n_u):(n_r*n_t+(i+1)*n_u)] = -task_grad_value @ robot_dyn['g'](x[:, i])
                b_ineq[idx] = (task_grad_value @ robot_dyn['f'](x[:, i]) + 
                               task_time_deriv_value + 
                               self.opt_params_['gamma'](task_func_value))
                
                # delta-alpha constraints
                base_idx = n_r*n_t + (i*n_t**2) + (j*n_t)
                for k in range(n_t):
                    if j != k:  # Skip constraints between a task and itself
                        A_ineq[base_idx+k, (i*n_t):(i+1)*n_t] = self.opt_params_['delta_max'] * self.onec(n_t, j)
                        A_ineq[base_idx+k, n_r*n_t+n_r*n_u+(i*n_t):n_r*n_t+n_r*n_u+(i+1)*n_t] = -1/self.opt_params_['kappa'] * np.eye(n_t)[k] + self.onec(n_t, j)[k]
        
        # CBFs for tasks - additional constraints
        A_ineq[:n_r*n_t, n_r*n_t+n_r*n_u:] = -np.eye(n_r*n_t)
        
        # delta-alpha constraints - right-hand side
        b_ineq[n_r*n_t:n_r*n_t+n_r*n_t**2] = self.opt_params_['delta_max'] * np.ones(n_r*n_t**2)
        
        # Remove constraints between a task and itself
        to_remove = []
        for i in range(n_r):
            for j in range(n_t):
                base_idx = n_r*n_t + (i*n_t**2) + (j*n_t) + j
                to_remove.append(base_idx)
        
        # Sort and remove from the end to avoid index shifting
        to_remove.sort(reverse=True)
        for idx in to_remove:
            A_ineq = np.delete(A_ineq, idx, axis=0)
            b_ineq = np.delete(b_ineq, idx)
        
        # Store constraints
        self.constraints_['A_ineq'] = A_ineq
        self.constraints_['b_ineq'] = b_ineq
        self.constraints_['A_eq'] = A_eq
        self.constraints_['b_eq'] = b_eq
        self.constraints_['lb'] = lb
        self.constraints_['ub'] = ub
        
    def set_scenario_params(self, scenario_params):
        """更新场景参数"""
        for field_name, value in scenario_params.items():
            if field_name in self.scenario_params_:
                assert field_name not in ['F', 'S'], '矩阵F和S无法设置（自动计算）'
                self.scenario_params_[field_name] = value
        
        self.evaluate_mappings_and_specializations()
        
    def set_opt_params(self, opt_params):
        for field_name, value in opt_params.items():
            if field_name in self.opt_params_:
                self.opt_params_[field_name] = value
                
    def set_specializations(self, S):
        self.scenario_params_['S'] = S
        self.build_projector()
    def get_specializations(self):
        return self.scenario_params_['S']
        
    def evaluate_mappings_and_specializations(self):
        """计算特征到能力(F)和任务到机器人(S)的映射"""
        n_c = self.dim_['n_c']
        n_r = self.dim_['n_r']
        
        # 初始化F矩阵
        self.scenario_params_['F'] = np.zeros((n_c, n_r))
        
        # 计算F
        for k in range(n_c):
            if self.scenario_params_['ws'] is not None and len(self.scenario_params_['ws']) > 0:
                W_k = np.diag(self.scenario_params_['ws'][k])
                self.scenario_params_['F'][k, :] = W_k @ ((self.scenario_params_['Hs'][k] @ self.scenario_params_['A']) > 0.999)
            else:
                self.scenario_params_['F'][k, :] = ((self.scenario_params_['Hs'][k] @ self.scenario_params_['A']) > 0.999)
        
        # 计算S
        self.scenario_params_['S'] = ((self.scenario_params_['T'] @ self.scenario_params_['F']) > 0.999).astype(float)
        
        # 构建投影矩阵
        self.build_projector()

    def build_projector(self):
        """构建投影矩阵P"""
        n_r = self.dim_['n_r']
        n_t = self.dim_['n_t']
        
        # 初始化P矩阵
        self.P_ = np.zeros((n_t, n_t*n_r))
        for i in range(n_r):
            self.P_[:, i*n_t:(i+1)*n_t] = np.eye(n_t)
        
        # 根据专业化矩阵更新P
        for i in range(n_r):
            Si = np.diag(self.scenario_params_['S'][:, i])
            self.P_[:, i*n_t:(i+1)*n_t] = self.P_[:, i*n_t:(i+1)*n_t] - Si @ np.linalg.pinv(Si)
        self.P_ = np.where(np.abs(self.P_) < 1e-10, 0, self.P_)
        
    def check_tasks(self):
        for i in range(self.dim_['n_t']):
            if 'gradient' not in self.scenario_params_['tasks'][i] or self.scenario_params_['tasks'][i]['gradient'] is None:
                self.scenario_params_['tasks'][i]['gradient'] = self.get_dh_dx_handle(i)
            
            if 'time_derivative' not in self.scenario_params_['tasks'][i] or self.scenario_params_['tasks'][i]['time_derivative'] is None:
                self.scenario_params_['tasks'][i]['time_derivative'] = self.get_dh_dt_handle(i)
    
    def get_dh_dx_handle(self, task_idx):
        def dh_dx(x_value, t_value, i):
            n = x_value.shape[0]
            dh_dx_value = np.zeros(n)
            for j in range(n):
                ej = np.zeros(n)
                ej[j] = 1
                dh_dx_value[j] = (self.scenario_params_['tasks'][task_idx]['function'](x_value + 1e-3*ej, t_value, i) - 
                                  self.scenario_params_['tasks'][task_idx]['function'](x_value - 1e-3*ej, t_value, i)) / (2e-3)
            return dh_dx_value
        return dh_dx
    
    def get_dh_dt_handle(self, task_idx):
        def dh_dt(x_value, t_value, i):
            return (self.scenario_params_['tasks'][task_idx]['function'](x_value, t_value + 1e-3, i) - 
                    self.scenario_params_['tasks'][task_idx]['function'](x_value, t_value - 1e-3, i)) / (2e-3)
        return dh_dt
    
    @staticmethod
    def onec(dim, col_idx):
        """创建一个特定维度的列向量，只有一个元素为1"""
        m = np.zeros(dim)
        m[col_idx] = 1
        return m
    
    def update_task_priority_config(self, **kwargs):
        self.task_utility_normalizer_.update_config(**kwargs)
    def set_custom_priority_function(self, priority_func):
        self.task_utility_normalizer_.config.custom_priority_func = priority_func
    def set_task_specific_weights(self, weights_dict):
        self.task_utility_normalizer_.config.task_specific_weights = weights_dict
    def get_task_priority_config(self):
        return self.task_utility_normalizer_.config
    
    def solve_miqp_phase_aware(self, x, t, phase_scenario_params, n_tasks, global_vars=None):
        """
        Solves the MIQP optimization problem for a dynamic number of task instances within a specific phase.
        This is the instance-aware version of the solver, designed to replace the fixed-task-type approach.

        Args:
            x: Current state of the agents.
            t: Current time.
            phase_scenario_params: A dictionary containing phase-specific parameters, including:
                - 'T': The instance-based capability requirement matrix [n_instances x n_capabilities].
                - 'n_r_bounds': The robot count bounds for each task instance [n_instances x 2].
            n_tasks: The number of task instances in the current phase.

        Returns:
            A tuple (alpha, u, delta, time_to_solve, opt_sol_info) containing the solution.
        """
        start_time = time.time()

        # 1. Get dynamic dimensions for this specific solve operation
        n_r = self.dim_['n_r']
        n_t = n_tasks  # n_t is now the number of task *instances*
        n_c = self.dim_['n_c']
        n_u = self.dim_['n_u']

        # 2. Build dynamic specialization (S) and projector (P) matrices for the current phase
        T_phase = phase_scenario_params['T']
        F_static = self.scenario_params_['F']
        S_phase = ((T_phase @ F_static) > 0.999).astype(float)

        P_phase = np.zeros((n_t, n_t * n_r))
        for i in range(n_r):
            P_phase[:, i * n_t:(i + 1) * n_t] = np.eye(n_t)
            Si = np.diag(S_phase[:, i])
            P_phase[:, i * n_t:(i + 1) * n_t] = P_phase[:, i * n_t:(i + 1) * n_t] - Si @ np.linalg.pinv(Si)
        P_phase = np.where(np.abs(P_phase) < 1e-10, 0, P_phase)

        # 3. Build dynamic constraints based on the current phase's instances
        dynamic_constraints = self.build_constraints_phase_aware(x, t, n_t, phase_scenario_params, global_vars)

        # 4. Set up and solve the MIQP problem using CVXPY
        alpha_dim = n_r * n_t
        u_dim = n_r * n_u
        delta_dim = n_r * n_t

        try:
            alpha_var = cp.Variable(alpha_dim, boolean=True)
            u_var = cp.Variable(u_dim)
            delta_var = cp.Variable(delta_dim)

            P_squared = P_phase.T @ P_phase
            S_diag = np.diag(np.reshape(S_phase, (-1)))

            # *** NEW: Add Preference Cost based on Aptitude ***
            aptitude_matrix = phase_scenario_params.get('aptitude_matrix')
            preference_cost = 0
            if aptitude_matrix is not None and aptitude_matrix.shape == (n_r, n_t):
                # Flatten the matrix to match alpha_var shape (n_r * n_t)
                inv_aptitude = 1.0 / (aptitude_matrix.flatten('C') + 1e-6)
                # A weight to balance this cost against other costs.
                preference_weight = 1.0
                preference_cost = preference_weight * (inv_aptitude @ alpha_var)
                # print("[DEBUG] RTA: Added preference cost to objective.")
            alpha_cost = 1e6 * max(1, self.opt_params_['l']) * cp.quad_form(alpha_var, P_squared)
            u_cost = cp.quad_form(u_var, np.eye(u_dim))
            delta_cost = self.opt_params_['l'] * cp.quad_form(delta_var, S_diag)
            objective = cp.Minimize(alpha_cost + u_cost + delta_cost + preference_cost)

            # Add all constraints
            constraints = []
            all_vars_h = cp.hstack([alpha_var, u_var, delta_var])
            all_vars = all_vars_h.T
            constraints.append(dynamic_constraints['A_ineq'] @ all_vars <= dynamic_constraints['b_ineq'])
            constraints.append(dynamic_constraints['A_eq'] @ all_vars == dynamic_constraints['b_eq'])

            # Add variable bounds
            constraints.append(alpha_var >= dynamic_constraints['lb'][:alpha_dim])
            constraints.append(alpha_var <= dynamic_constraints['ub'][:alpha_dim])
            lb_idx_delta = alpha_dim + u_dim
            constraints.append(delta_var >= dynamic_constraints['lb'][lb_idx_delta:lb_idx_delta + delta_dim])
            constraints.append(delta_var <= dynamic_constraints['ub'][lb_idx_delta:lb_idx_delta + delta_dim])

            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.GUROBI, verbose=False)
            time_to_solve = time.time() - start_time

            # 5. Process and return the solution
            if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
                alpha = alpha_var.value
                u = u_var.value
                delta = delta_var.value
                opt_sol_info = "Optimal"
            else:
                print(f"[ERROR] Instance-aware MIQP optimization failed with status: {problem.status}")
                self.analyzer._diagnose_infeasible_constraints()  # Run diagnostics on failure
                alpha, u, delta = np.zeros(alpha_dim), np.zeros(u_dim), np.zeros(delta_dim)
                opt_sol_info = f"Failed: {problem.status}"

            return alpha, u, delta, time_to_solve, opt_sol_info

        except cp.error.SolverError as e:
            print(f"CVXPY solver error in instance-aware MIQP: {e}")
            return np.zeros(alpha_dim), np.zeros(u_dim), np.zeros(delta_dim), 0, "SolverError"
        except Exception as e:
            print(f"An unexpected error occurred in instance-aware MIQP solve: {e}")
            return np.zeros(alpha_dim), np.zeros(u_dim), np.zeros(delta_dim), 0, "Exception"

    def build_constraints_phase_aware(self, x, t, n_t, phase_scenario_params, global_vars=None):
        """
        Builds constraints for a dynamic number of task instances. This is a counterpart
        to `build_constraints` but designed for instance-aware planning.
        """
        n_r = self.dim_['n_r']
        n_c = self.dim_['n_c']
        n_u = self.dim_['n_u']

        T_phase = phase_scenario_params['T']
        n_r_bounds_phase = phase_scenario_params.get('n_r_bounds')
        if n_r_bounds_phase is None:
            # Fallback if not provided
            n_r_bounds_phase = np.array([[0, n_r]] * n_t)
            print(f"[WARNING] 'n_r_bounds' not found in phase_scenario_params. Falling back to default [0, {n_r}] for all tasks.")
            
        tasks_with_pos = phase_scenario_params.get('tasks', [])
        task_types_in_phase = [task.get('task_type') for task in tasks_with_pos]

        # Determine matrix sizes based on the number of instances (n_t)
        cbf_constraints = n_r * n_t
        capability_constraints = n_t * n_c
        robot_bound_constraints = 2 * n_t
        total_ineq = cbf_constraints + capability_constraints + robot_bound_constraints
        total_vars = 2 * n_r * n_t + n_r * n_u

        A_ineq = np.zeros((total_ineq, total_vars))
        b_ineq = np.zeros(total_ineq)
        A_eq = np.zeros((n_r, total_vars))
        b_eq = np.ones(n_r)
        lb = -np.inf * np.ones(total_vars)
        ub = np.inf * np.ones(total_vars)

        # Set variable bounds
        lb[:n_r * n_t] = 0
        ub[:n_r * n_t] = 1
        lb[n_r * n_t + n_r * n_u:] = 0
        ub[n_r * n_t + n_r * n_u:] = self.opt_params_['delta_max']

        # === 1. CBF Constraints (Leveraging instance-specific data) ===
        constraint_idx = 0
        if global_vars is None:
            global_vars = self.get_global_vars_dict() or {}

        for r in range(n_r):
            for j in range(n_t):
                # Get scaled task values using the correct instance data
                try:
                    task_func, task_grad, task_deriv = self.get_scaled_task_values(
                        x, r, j, t=t, global_vars_dict=global_vars, scaling_factors=None,
                        task_types_in_phase=task_types_in_phase
                    )
                except KeyError as e:
                    task_type_name = task_types_in_phase[j] if j < len(task_types_in_phase) else "Unknown"
                    print(f"[ERROR] Missing required parameter for task {task_type_name}: {e}")
                    # Fallback to a zero-valued function to avoid crashing the optimization
                    task_func, task_grad, task_deriv = 0.0, np.zeros(self.dim_['n_x']), 0.0

                robot_dyn = self.scenario_params_['robot_dyn']
                cbf_idx = r * n_t + j
                
                A_ineq[cbf_idx, n_r * n_t + r * n_u : n_r * n_t + (r + 1) * n_u] = -task_grad @ robot_dyn['g'](x[:, r])
                b_ineq[cbf_idx] = (task_grad @ robot_dyn['f'](x[:, r]) + 
                                   task_deriv + 
                                   self.opt_params_['gamma'](task_func))
        constraint_idx += cbf_constraints

        # === 2. Capability Constraints (F·α ≥ T) ===
        F_static = self.scenario_params_['F']
        for j in range(n_t):  # For each task instance
            for c in range(n_c):  # For each capability
                cap_idx = constraint_idx + j * n_c + c
                for r in range(n_r):
                    A_ineq[cap_idx, r * n_t + j] = -F_static[c, r]
                b_ineq[cap_idx] = -T_phase[j, c]
        constraint_idx += capability_constraints

        # === 3. Robot Count Bounds ===
        for j in range(n_t):  # For each task instance
            # Max robots constraint: sum(α_rj for r) <= max_robots_j
            max_idx = constraint_idx + j
            for r in range(n_r):
                A_ineq[max_idx, r * n_t + j] = 1
            b_ineq[max_idx] = n_r_bounds_phase[j, 1]

            # Min robots constraint: sum(α_rj for r) >= min_robots_j
            min_idx = constraint_idx + n_t + j
            for r in range(n_r):
                A_ineq[min_idx, r * n_t + j] = -1
            b_ineq[min_idx] = -n_r_bounds_phase[j, 0]
        constraint_idx += robot_bound_constraints

        # === 4. Equality Constraints (Each robot gets one task) ===
        for i in range(n_r):
            for j in range(n_t):
                A_eq[i, i * n_t + j] = 1

        return {
            'A_ineq': A_ineq, 'b_ineq': b_ineq,
            'A_eq': A_eq, 'b_eq': b_eq,
            'lb': lb, 'ub': ub
        }