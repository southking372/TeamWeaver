import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import cvxpy as cp
import sys
import os

# 添加task_module路径以便导入TaskUtilityNormalizer
current_dir = os.path.dirname(os.path.abspath(__file__))
task_module_dir = os.path.join(current_dir, '..', '..', '..', 'task-plan', 'class_def', 'task_module')
if task_module_dir not in sys.path:
    sys.path.append(task_module_dir)

from habitat_llm.planner.HRCS.class_def.task_utility_normalizer import TaskUtilityNormalizer, TaskPriorityConfig

def print_matrix_similar_to_matlab(matrix, name="Matrix"):
    print(f"{name}:\n{'*' * 40}\n")
    num_rows, num_cols = matrix.shape
    col_groups = [(s, min(s + 12, num_cols)) for s in range(0, num_cols, 12)]
    zero = 0
    for group_idx, (start, end) in enumerate(col_groups):
        print(f"列 {start+1} 至 {end}")
        for i in range(num_rows):
            for j in range(start, end):
                if j == end - 1:
                    if matrix[i, j] == 0:
                        print(f"{zero:10}", end="\n")
                    else:
                        print(f"{matrix[i, j]:10.4f}", end="\n")
                else:
                    if matrix[i, j] == 0:
                        print(f"{zero:10}", end=" ")
                    else:
                        print(f"{matrix[i, j]:10.4f}", end=" ")
        print("\n" + "=" * 40)
    print("\n" + '*' * 40)

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
            alpha, u, delta, solve_time, status, constraints_info = self.solve_miqp_with_detailed_analysis(x, t)
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
                self._diagnose_infeasible_constraints()
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
                self._diagnose_infeasible_constraints()
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
    
    def get_scaled_task_values(self, x, t, i, j, global_vars_dict, scaling_factors):
        """获取缩放后的任务函数值、梯度和时间导数"""
        task = self.scenario_params_['tasks'][j]
        
        if global_vars_dict is not None:
            task_func_value = task['function'](x[:, i], t, i, vars_dict=global_vars_dict)
            task_grad_value = task['gradient'](x[:, i], t, i, vars_dict=global_vars_dict)
            task_time_deriv_value = task['time_derivative'](x[:, i], t, i, vars_dict=global_vars_dict)
        else:
            task_func_value = task['function'](x[:, i], t, i)
            task_grad_value = task['gradient'](x[:, i], t, i) 
            task_time_deriv_value = task['time_derivative'](x[:, i], t, i)
        
        # 应用缩放因子
        # task_func_value = task_func_value * scaling_factors[j]
        # task_grad_value = task_grad_value * scaling_factors[j]
        # task_time_deriv_value = task_time_deriv_value * scaling_factors[j]
        
        # [TODO] 暂时取消缩放因子
        task_func_value = task_func_value
        task_grad_value = task_grad_value
        task_time_deriv_value = task_time_deriv_value
        
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
        
        constraint_idx = 0
        # === 1. CBF约束 (Control Barrier Functions) ===
        # print(f"[DEBUG] Adding CBF constraints...")
        # for i in range(n_r):
        #     for j in range(n_t):
        #         task = self.scenario_params_['tasks'][j]
        #         robot_dyn = self.scenario_params_['robot_dyn']
                
        #         # 获取缩放后的任务函数值、梯度和时间导数
        #         task_func_value, task_grad_value, task_time_deriv_value = self.get_scaled_task_values(
        #             x, t, i, j, global_vars_dict, scaling_factors
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
                    x, t, i, j, global_vars_dict, scaling_factors
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
    
    def _diagnose_infeasible_constraints(self):
        """诊断导致不可行的约束条件"""
        print(f"[DEBUG] 诊断不可行约束...")
        
        # 检查约束矩阵的基本信息
        A_ineq = self.constraints_['A_ineq']
        b_ineq = self.constraints_['b_ineq']
        A_eq = self.constraints_['A_eq']
        b_eq = self.constraints_['b_eq']
        lb = self.constraints_['lb']
        ub = self.constraints_['ub']
        
        print(f"  不等式约束矩阵: {A_ineq.shape}")
        print(f"  等式约束矩阵: {A_eq.shape}")
        print(f"  变量边界: lb={lb.shape}, ub={ub.shape}")
        
        # 检查是否有明显冲突的约束
        n_r = self.dim_['n_r']
        n_t = self.dim_['n_t']
        n_c = self.dim_['n_c']
        
        # 1. 检查变量边界的一致性
        print(f"  1. 检查变量边界:")
        inconsistent_bounds = np.where(lb > ub)[0]
        if len(inconsistent_bounds) > 0:
            print(f"    [ERROR] 发现 {len(inconsistent_bounds)} 个变量的下界 > 上界!")
            for idx in inconsistent_bounds[:5]:  # 只显示前5个
                print(f"      变量 {idx}: lb={lb[idx]}, ub={ub[idx]}")
        
        # 2. 检查机器人数量约束
        print(f"  2. 机器人数量边界:")
        total_min_robots = 0
        total_max_robots = 0
        for j in range(n_t):
            min_robots = self.opt_params_['n_r_bounds'][j, 0]
            max_robots = self.opt_params_['n_r_bounds'][j, 1]
            total_min_robots += min_robots
            total_max_robots += max_robots
            print(f"    任务 {j}: 最小={min_robots}, 最大={max_robots}")
            if min_robots > max_robots:
                print(f"    [ERROR] 任务 {j}: 最小机器人数 > 最大机器人数!")
            if min_robots > n_r:
                print(f"    [ERROR] 任务 {j}: 最小机器人数 > 总机器人数!")
        
        print(f"    总计: 最小需求={total_min_robots}, 最大需求={total_max_robots}, 可用机器人={n_r}")
        if total_min_robots > n_r:
            print(f"    [ERROR] 所有任务的最小机器人需求({total_min_robots}) > 可用机器人数({n_r})!")
        
        # 3. 检查能力矩阵和任务需求的兼容性
        F = self.scenario_params_['F']
        T = self.scenario_params_['T']
        print(f"  3. 能力兼容性检查:")
        print(f"    能力矩阵 F: {F.shape}")
        print(f"    任务需求矩阵 T: {T.shape}")
        
        for j in range(n_t):
            required_capabilities = np.where(T[j, :] > 0.5)[0]
            if len(required_capabilities) > 0:
                print(f"    任务 {j} 需要能力: {required_capabilities}")
                for c in required_capabilities:
                    capable_robots = np.where(F[c, :] > 0.5)[0]
                    print(f"      能力 {c}: 具备的机器人 = {capable_robots} (共{len(capable_robots)}个)")
                    if len(capable_robots) == 0:
                        print(f"      [ERROR] 没有机器人具备能力 {c}!")
                    elif len(capable_robots) < self.opt_params_['n_r_bounds'][j, 0]:
                        print(f"      [ERROR] 具备能力 {c} 的机器人数({len(capable_robots)}) < 任务 {j} 的最小需求({self.opt_params_['n_r_bounds'][j, 0]})")
        
        # 4. 检查等式约束的一致性
        print(f"  4. 等式约束检查:")
        print(f"    等式约束右端向量 b_eq: {b_eq}")
        print(f"    要求: 每个机器人分配到恰好一个任务")
        if not np.allclose(b_eq, 1.0):
            print(f"    [ERROR] 等式约束要求每个机器人分配值不等于1!")
        
        # 5. 检查约束矩阵的数值稳定性
        print(f"  5. 数值稳定性检查:")
        if np.any(np.isnan(A_ineq)) or np.any(np.isinf(A_ineq)):
            print(f"    [ERROR] A_ineq 包含 NaN 或 Inf 值!")
        if np.any(np.isnan(b_ineq)) or np.any(np.isinf(b_ineq)):
            print(f"    [ERROR] b_ineq 包含 NaN 或 Inf 值!")
        if np.any(np.isnan(A_eq)) or np.any(np.isinf(A_eq)):
            print(f"    [ERROR] A_eq 包含 NaN 或 Inf 值!")
        if np.any(np.isnan(b_eq)) or np.any(np.isinf(b_eq)):
            print(f"    [ERROR] b_eq 包含 NaN 或 Inf 值!")
        
        # 6. 检查约束矩阵的条件数
        try:
            cond_ineq = np.linalg.cond(A_ineq @ A_ineq.T + 1e-10 * np.eye(A_ineq.shape[0]))
            cond_eq = np.linalg.cond(A_eq @ A_eq.T + 1e-10 * np.eye(A_eq.shape[0]))
            print(f"    不等式约束矩阵条件数: {cond_ineq:.2e}")
            print(f"    等式约束矩阵条件数: {cond_eq:.2e}")
            if cond_ineq > 1e12:
                print(f"    [WARNING] 不等式约束矩阵条件数过大，可能数值不稳定!")
            if cond_eq > 1e12:
                print(f"    [WARNING] 等式约束矩阵条件数过大，可能数值不稳定!")
        except Exception as e:
            print(f"    [WARNING] 无法计算条件数: {e}")
        
        print(f"[DEBUG] 约束诊断完成。")

    def analyze_constraints_detailed(self, x, t, alpha_var, u_var, delta_var):
        """
        详细分析所有约束条件及其对目标函数和控制变量的作用
        
        Args:
            x: 当前状态 [n_x, n_r]
            t: 当前时间
            alpha_var, u_var, delta_var: CVXPY变量
            
        Returns:
            constraints_info: 详细的约束信息字典
        """
        print("\n" + "="*80)
        print("🔍 MIQP约束条件详细分析")
        print("="*80)
        
        n_r = self.dim_['n_r']
        n_t = self.dim_['n_t']
        n_c = self.dim_['n_c']
        n_u = self.dim_['n_u']
        
        constraints_info = {
            'constraint_types': [],
            'matrix_info': {},
            'variable_bounds': {},
            'constraint_violations': {},
            'feasibility_analysis': {}
        }
        
        # === 1. 变量边界分析 ===
        print("\n📊 变量边界分析:")
        print("-" * 50)
        
        alpha_bounds = {
            'name': 'Task Assignment Variables (α)',
            'dimension': f"{n_r}×{n_t} = {n_r*n_t}",
            'type': 'Binary (0 or 1)',
            'lower_bound': 0,
            'upper_bound': 1,
            'meaning': '机器人i是否分配给任务j'
        }
        
        u_bounds = {
            'name': 'Control Input Variables (u)', 
            'dimension': f"{n_r}×{n_u} = {n_r*n_u}",
            'type': 'Continuous',
            'lower_bound': '-∞',
            'upper_bound': '+∞',
            'meaning': '机器人i的控制输入向量'
        }
        
        delta_bounds = {
            'name': 'Slack Variables (δ)',
            'dimension': f"{n_r}×{n_t} = {n_r*n_t}",
            'type': 'Continuous', 
            'lower_bound': 0,
            'upper_bound': f"δ_max = {self.opt_params_['delta_max']}",
            'meaning': '约束松弛变量，允许轻微违反CBF约束'
        }
        
        for var_info in [alpha_bounds, u_bounds, delta_bounds]:
            print(f"  • {var_info['name']}")
            print(f"    维度: {var_info['dimension']}")
            print(f"    类型: {var_info['type']}")
            print(f"    范围: [{var_info['lower_bound']}, {var_info['upper_bound']}]")
            print(f"    含义: {var_info['meaning']}")
            print()
        
        constraints_info['variable_bounds'] = {
            'alpha': alpha_bounds,
            'u': u_bounds, 
            'delta': delta_bounds
        }
        
        # === 2. 能力约束分析 ===
        print("\n🎯 能力约束分析 (F·α ≥ T):")
        print("-" * 50)
        
        F_matrix = self.scenario_params_['F']
        T_matrix = self.scenario_params_['T']
        
        print(f"能力映射矩阵 F: {F_matrix.shape} (能力×机器人)")
        print(f"任务需求矩阵 T: {T_matrix.shape} (任务×能力)")
        
        capability_constraints = []
        for j in range(n_t):
            task_name = f"Task_{j}"
            for c in range(n_c):
                capability_name = f"Capability_{c}"
                
                # 分析哪些机器人具备此能力
                capable_robots = [r for r in range(n_r) if F_matrix[c, r] > 0.5]
                task_requirement = T_matrix[j, c]
                
                if task_requirement > 0.01:  # 只显示有实际需求的约束
                    constraint_info = {
                        'constraint_id': f"Cap_{j}_{c}",
                        'task': task_name,
                        'capability': capability_name, 
                        'requirement': f"{task_requirement:.3f}",
                        'capable_robots': capable_robots,
                        'constraint_form': f"Σ(F[{c},r] * α[r,{j}]) ≥ {task_requirement:.3f}",
                        'physical_meaning': f"执行{task_name}需要{capability_name}，只有机器人{capable_robots}具备此能力"
                    }
                    
                    capability_constraints.append(constraint_info)
                    
                    print(f"  📌 {constraint_info['constraint_id']}: {constraint_info['physical_meaning']}")
                    print(f"     约束式: {constraint_info['constraint_form']}")
                    print(f"     具备能力的机器人: {capable_robots}")
                    
                    if len(capable_robots) == 0:
                        print(f"     ⚠️  警告: 没有机器人具备{capability_name}!")
                    elif len(capable_robots) < task_requirement:
                        print(f"     ⚠️  警告: 具备能力的机器人数({len(capable_robots)}) < 需求({task_requirement})")
                    print()
        
        constraints_info['capability_constraints'] = capability_constraints
        
        # === 3. 机器人数量约束分析 ===
        print("\n👥 机器人数量约束分析:")
        print("-" * 50)
        
        robot_count_constraints = []
        for j in range(n_t):
            min_robots = self.opt_params_['n_r_bounds'][j, 0]
            max_robots = self.opt_params_['n_r_bounds'][j, 1] 
            
            constraint_info = {
                'task_id': j,
                'min_constraint': f"{min_robots} ≤ Σ(α[r,{j}]) for r=0..{n_r-1}",
                'max_constraint': f"Σ(α[r,{j}]) ≤ {max_robots} for r=0..{n_r-1}",
                'meaning': f"任务{j}需要{min_robots}-{max_robots}个机器人",
                'feasibility': 'OK' if min_robots <= max_robots <= n_r else 'INFEASIBLE'
            }
            
            robot_count_constraints.append(constraint_info)
            
            print(f"  📋 Task_{j}: {constraint_info['meaning']}")
            print(f"     最小约束: {constraint_info['min_constraint']}")
            print(f"     最大约束: {constraint_info['max_constraint']}")
            print(f"     可行性: {constraint_info['feasibility']}")
            
            if constraint_info['feasibility'] == 'INFEASIBLE':
                if min_robots > max_robots:
                    print(f"     ❌ 错误: 最小需求 > 最大需求!")
                if max_robots > n_r:
                    print(f"     ❌ 错误: 最大需求 > 可用机器人数!")
            print()
        
        constraints_info['robot_count_constraints'] = robot_count_constraints
        
        # === 4. 等式约束分析 ===  
        print("\n⚖️  等式约束分析 (机器人唯一分配):")
        print("-" * 50)
        
        equality_constraints = []
        for i in range(n_r):
            constraint_info = {
                'robot_id': i,
                'constraint': f"Σ(α[{i},j]) = 1 for j=0..{n_t-1}",
                'meaning': f"机器人{i}必须且只能分配给一个任务"
            }
            equality_constraints.append(constraint_info)
            print(f"  🤖 Robot_{i}: {constraint_info['meaning']}")
            print(f"     约束式: {constraint_info['constraint']}")
        
        constraints_info['equality_constraints'] = equality_constraints
        
        # === 5. 目标函数分析 ===
        print(f"\n🎯 目标函数分析:")
        print("-" * 50)
        
        P_matrix = self.P_
        S_matrix = self.scenario_params_['S']
        l_param = self.opt_params_['l']
        
        objective_info = {
            'total_form': 'minimize: α_cost + u_cost + δ_cost',
            'alpha_cost': {
                'form': f"1e6 × max(1, {l_param}) × α^T × P^T × P × α",
                'weight': f"{1e6 * max(1, l_param):.0e}",
                'purpose': '任务分配稳定性，避免频繁切换',
                'matrix_P_shape': f"{P_matrix.shape}",
                'matrix_P_property': '基于任务专业化构建的投影矩阵'
            },
            'u_cost': {
                'form': "u^T × I × u",
                'weight': "1.0",
                'purpose': '控制输入最小化，降低能耗',
                'matrix_shape': f"{n_r*n_u}×{n_r*n_u} 单位矩阵"
            },
            'delta_cost': {
                'form': f"{l_param} × δ^T × S × δ",
                'weight': f"{l_param}",
                'purpose': '松弛变量惩罚，软约束违反',
                'matrix_S_shape': f"{S_matrix.shape}",
                'matrix_S_meaning': '任务专业化矩阵，权重不同任务的违反代价'
            }
        }
        
        print(f"  📈 总目标函数: {objective_info['total_form']}")
        print()
        print(f"  1️⃣ 任务分配代价 (α_cost):")
        print(f"     形式: {objective_info['alpha_cost']['form']}")
        print(f"     权重: {objective_info['alpha_cost']['weight']}")
        print(f"     目的: {objective_info['alpha_cost']['purpose']}")
        print(f"     矩阵P: {objective_info['alpha_cost']['matrix_P_shape']} - {objective_info['alpha_cost']['matrix_P_property']}")
        print()
        print(f"  2️⃣ 控制输入代价 (u_cost):")
        print(f"     形式: {objective_info['u_cost']['form']}")
        print(f"     权重: {objective_info['u_cost']['weight']}")
        print(f"     目的: {objective_info['u_cost']['purpose']}")
        print(f"     矩阵: {objective_info['u_cost']['matrix_shape']}")
        print()
        print(f"  3️⃣ 松弛变量代价 (δ_cost):")
        print(f"     形式: {objective_info['delta_cost']['form']}")
        print(f"     权重: {objective_info['delta_cost']['weight']}")
        print(f"     目的: {objective_info['delta_cost']['purpose']}")
        print(f"     矩阵S: {objective_info['delta_cost']['matrix_S_shape']} - {objective_info['delta_cost']['matrix_S_meaning']}")
        
        constraints_info['objective_function'] = objective_info
        
        # === 6. 约束矩阵统计信息 ===
        print(f"\n📊 约束矩阵统计信息:")
        print("-" * 50)
        
        A_ineq = self.constraints_['A_ineq']
        b_ineq = self.constraints_['b_ineq']
        A_eq = self.constraints_['A_eq']
        b_eq = self.constraints_['b_eq']
        
        matrix_stats = {
            'inequality_constraints': {
                'matrix_A_shape': A_ineq.shape,
                'vector_b_shape': b_ineq.shape,
                'non_zero_elements': np.count_nonzero(A_ineq),
                'sparsity': f"{(1 - np.count_nonzero(A_ineq) / A_ineq.size) * 100:.1f}%",
                'condition_number': np.linalg.cond(A_ineq @ A_ineq.T + 1e-10 * np.eye(A_ineq.shape[0]))
            },
            'equality_constraints': {
                'matrix_A_shape': A_eq.shape,
                'vector_b_shape': b_eq.shape,
                'non_zero_elements': np.count_nonzero(A_eq),
                'rank': np.linalg.matrix_rank(A_eq)
            }
        }
        
        print(f"  📋 不等式约束:")
        print(f"     矩阵 A_ineq: {matrix_stats['inequality_constraints']['matrix_A_shape']}")
        print(f"     向量 b_ineq: {matrix_stats['inequality_constraints']['vector_b_shape']}")  
        print(f"     非零元素: {matrix_stats['inequality_constraints']['non_zero_elements']}")
        print(f"     稀疏度: {matrix_stats['inequality_constraints']['sparsity']}")
        print(f"     条件数: {matrix_stats['inequality_constraints']['condition_number']:.2e}")
        print()
        print(f"  ⚖️  等式约束:")
        print(f"     矩阵 A_eq: {matrix_stats['equality_constraints']['matrix_A_shape']}")
        print(f"     向量 b_eq: {matrix_stats['equality_constraints']['vector_b_shape']}")
        print(f"     非零元素: {matrix_stats['equality_constraints']['non_zero_elements']}")
        print(f"     矩阵秩: {matrix_stats['equality_constraints']['rank']}")
        
        constraints_info['matrix_info'] = matrix_stats
        
        # === 7. 可行性预检查 ===
        print(f"\n🔬 可行性预检查:")
        print("-" * 50)
        
        feasibility_issues = []
        
        # 检查能力匹配
        for j in range(n_t):
            for c in range(n_c):
                if T_matrix[j, c] > 0.01:
                    capable_robots = np.sum(F_matrix[c, :] > 0.5)
                    if capable_robots == 0:
                        issue = f"任务{j}需要能力{c}，但没有机器人具备此能力"
                        feasibility_issues.append(issue)
                        print(f"     ❌ {issue}")
        
        # 检查机器人数量
        total_min_demand = np.sum([self.opt_params_['n_r_bounds'][j, 0] for j in range(n_t)])
        if total_min_demand > n_r:
            issue = f"所有任务最小需求({total_min_demand}) > 可用机器人数({n_r})"
            feasibility_issues.append(issue)
            print(f"     ❌ {issue}")
        
        # 检查矩阵数值稳定性
        if matrix_stats['inequality_constraints']['condition_number'] > 1e12:
            issue = f"不等式约束矩阵条件数过大 ({matrix_stats['inequality_constraints']['condition_number']:.2e})"
            feasibility_issues.append(issue)
            print(f"     ⚠️  {issue}")
        
        if len(feasibility_issues) == 0:
            print(f"     ✅ 预检查通过，约束系统看起来是可行的")
        
        constraints_info['feasibility_analysis'] = {
            'issues': feasibility_issues,
            'is_likely_feasible': len(feasibility_issues) == 0
        }
        
        print("\n" + "="*80)
        print("🏁 约束分析完成")
        print("="*80)
        
        return constraints_info

    def _display_capability_constraint_calculations(self, F_matrix, T_matrix, alpha_var=None, alpha_solution=None):
        """
        详细展示能力约束的计算过程：F·α ≥ T
        
        Args:
            F_matrix: 能力映射矩阵 [n_c, n_r]
            T_matrix: 任务需求矩阵 [n_t, n_c]  
            alpha_var: CVXPY变量 (可选)
            alpha_solution: 求解后的α值 (可选)
        """
        print(f"\n🧮 能力约束计算过程详解 (F·α ≥ T):")
        print("="*80)
        
        n_r = self.dim_['n_r'] 
        n_t = self.dim_['n_t']
        n_c = self.dim_['n_c']
        
        # 1. 展示矩阵结构
        print(f"\n📋 矩阵维度信息:")
        print(f"   F (能力映射): {F_matrix.shape} - [能力×机器人]")
        print(f"   T (任务需求): {T_matrix.shape} - [任务×能力]") 
        print(f"   α (分配变量): [{n_r}×{n_t}] = [{n_r*n_t}] - 展开为向量")
        
        # 2. 展示F矩阵详细内容
        print(f"\n🤖 能力映射矩阵 F:")
        print(f"   行：能力 [Movement, Object_Manip, Basic_Control, Liquid_Handle, Power_Control]")
        print(f"   列：机器人 [Robot_0, Robot_1]")
        capability_names = ["Movement", "Object_Manip", "Basic_Control", "Liquid_Handle", "Power_Control"]
        robot_names = [f"Robot_{i}" for i in range(n_r)]
        
        print(f"\n   {'能力':<15} ", end="")
        for robot_name in robot_names:
            print(f"{robot_name:>10}", end="")
        print()
        print(f"   {'-'*15} ", end="")
        for _ in robot_names:
            print(f"{'-'*10}", end="")
        print()
        
        for c in range(n_c):
            print(f"   {capability_names[c]:<15} ", end="")
            for r in range(n_r):
                print(f"{F_matrix[c, r]:>10.1f}", end="")
            print()
        
        # 3. 展示T矩阵详细内容
        print(f"\n📋 任务需求矩阵 T:")
        print(f"   行：任务 [Navigate, Explore, Pick, Place, Open, Close, Clean, Fill, Pour, PowerOn, PowerOff, Rearrange, Wait]")
        print(f"   列：能力 [Movement, Object_Manip, Basic_Control, Liquid_Handle, Power_Control]")
        
        task_names = ["Navigate", "Explore", "Pick", "Place", "Open", "Close", 
                     "Clean", "Fill", "Pour", "PowerOn", "PowerOff", "Rearrange", "Wait"]
        
        print(f"\n   {'任务':<12} ", end="")
        for cap_name in capability_names:
            print(f"{cap_name[:8]:>9}", end="")
        print()
        print(f"   {'-'*12} ", end="")
        for _ in capability_names:
            print(f"{'-'*9}", end="")
        print()
        
        for j in range(n_t):
            task_name = task_names[j] if j < len(task_names) else f"Task_{j}"
            print(f"   {task_name:<12} ", end="")
            for c in range(n_c):
                print(f"{T_matrix[j, c]:>9.1f}", end="")
            print()
        
        # 4. 展示具体的约束计算
        print(f"\n🎯 具体约束计算过程:")
        print(f"   约束形式: 对于每个任务j和能力c, Σ(F[c,r] × α[r,j]) ≥ T[j,c]")
        print(f"   意义: 分配给任务j的机器人在能力c上的总和必须满足任务需求")
        
        constraint_count = 0
        for j in range(n_t):
            task_name = task_names[j] if j < len(task_names) else f"Task_{j}"
            for c in range(n_c):
                if T_matrix[j, c] > 0.01:  # 只显示有实际需求的约束
                    constraint_count += 1
                    print(f"\n   📌 约束 #{constraint_count}: {task_name} 需要 {capability_names[c]}")
                    print(f"      数学表达式: ", end="")
                    
                    # 构建约束表达式
                    terms = []
                    for r in range(n_r):
                        if F_matrix[c, r] > 0.01:
                            terms.append(f"{F_matrix[c, r]:.1f}*α[{r},{j}]")
                    
                    constraint_expr = " + ".join(terms) if terms else "0"
                    print(f"{constraint_expr} ≥ {T_matrix[j, c]:.1f}")
                    
                    # 显示哪些机器人能满足这个能力
                    capable_robots = [r for r in range(n_r) if F_matrix[c, r] > 0.5]
                    print(f"      具备能力的机器人: {capable_robots}")
                    
                    if len(capable_robots) == 0:
                        print(f"      ⚠️  警告: 没有机器人具备此能力！")
                    elif len(capable_robots) < T_matrix[j, c]:
                        print(f"      ⚠️  注意: 具备能力的机器人数({len(capable_robots)}) < 需求({T_matrix[j, c]:.1f})")
                        
        # 5. 如果有求解结果，展示约束满足情况
        if alpha_solution is not None:
            print(f"\n🎯 约束满足情况检查 (基于求解结果):")
            alpha_matrix = alpha_solution.reshape(n_r, n_t)
            
            all_satisfied = True
            for j in range(n_t):
                task_name = task_names[j] if j < len(task_names) else f"Task_{j}"
                for c in range(n_c):
                    if T_matrix[j, c] > 0.01:
                        # 计算 F[c,:] · α[:,j] 
                        assigned_capability = np.dot(F_matrix[c, :], alpha_matrix[:, j])
                        required_capability = T_matrix[j, c]
                        satisfied = assigned_capability >= required_capability - 1e-6
                        
                        status = "✅" if satisfied else "❌"
                        print(f"      {status} {task_name}-{capability_names[c]}: {assigned_capability:.3f} ≥ {required_capability:.1f}")
                        
                        if not satisfied:
                            all_satisfied = False
                            print(f"         违反量: {required_capability - assigned_capability:.3f}")
            
            if all_satisfied:
                print(f"\n   🎉 所有能力约束都得到满足！")
            else:
                print(f"\n   ⚠️  存在未满足的能力约束")
        
        print("="*80)

    def solve_miqp_with_detailed_analysis(self, x, t):
        """
        带详细分析的MIQP求解方法
        
        Args:
            x: 当前状态
            t: 当前时间
            
        Returns:
            alpha, u, delta, solve_time, status, constraints_info
        """
        print("\n🚀 开始MIQP求解 (详细分析模式)")
        
        # 1. 构建约束
        self.build_constraints(x, t)
        
        # 2. 设置变量
        alpha_dim = self.dim_['n_r'] * self.dim_['n_t']
        u_dim = self.dim_['n_r'] * self.dim_['n_u']
        delta_dim = self.dim_['n_r'] * self.dim_['n_t']
        
        alpha_var = cp.Variable(alpha_dim, boolean=True)
        u_var = cp.Variable(u_dim)
        delta_var = cp.Variable(delta_dim)
        
        # 3. 详细分析约束
        constraints_info = self.analyze_constraints_detailed(x, t, alpha_var, u_var, delta_var)
        
        # 4. 详细展示能力约束计算过程
        F_matrix = self.scenario_params_['F']
        T_matrix = self.scenario_params_['T']
        self._display_capability_constraint_calculations(F_matrix, T_matrix, alpha_var)
        
        # 5. 构建目标函数
        P_squared = self.P_.T @ self.P_
        S_diag = np.diag(np.reshape(self.scenario_params_['S'], (-1)))
        
        alpha_cost = 1e6 * max(1, self.opt_params_['l']) * cp.quad_form(alpha_var, P_squared)
        u_cost = cp.quad_form(u_var, np.eye(u_dim))
        delta_cost = self.opt_params_['l'] * cp.quad_form(delta_var, S_diag)
        objective = cp.Minimize(alpha_cost + u_cost + delta_cost)
        
        # 6. 添加约束
        constraints = []
        all_vars_h = cp.hstack([alpha_var, u_var, delta_var])
        all_vars = all_vars_h.T
        constraints.append(self.constraints_['A_ineq'] @ all_vars <= self.constraints_['b_ineq'])
        constraints.append(self.constraints_['A_eq'] @ all_vars == self.constraints_['b_eq'])  # 等式约束
        constraints.append(alpha_var >= self.constraints_['lb'][:alpha_dim])
        constraints.append(alpha_var <= self.constraints_['ub'][:alpha_dim])
        
        lb_idx = alpha_dim + u_dim
        constraints.append(delta_var >= self.constraints_['lb'][lb_idx:lb_idx+delta_dim])
        constraints.append(delta_var <= self.constraints_['ub'][lb_idx:lb_idx+delta_dim])
        
        # 7. 创建问题并导出.lp文件
        problem = cp.Problem(objective, constraints)
        
        print(f"\n📄 导出.lp文件进行模型检查...")
        try:
            lp_file_path = f"MIQP_model_{int(t*1000)}.lp"
            
            # 使用Gurobi后端生成.lp文件
            problem.solve(solver=cp.GUROBI, verbose=False, save_file=lp_file_path)
            
            print(f"   ✅ .lp文件已保存: {lp_file_path}")
            print(f"   📖 您可以用文本编辑器打开查看完整的数学模型")
            
            # 显示.lp文件的关键信息
            self._display_lp_file_summary(lp_file_path)
            
        except Exception as e:
            print(f"   ⚠️ .lp文件导出失败: {e}")
        
        # 8. 求解
        print(f"\n🔧 开始CVXPY求解...")
        start_time = time.time()
        
        solve_params = {
            'NumericFocus': 2,
            'FeasibilityTol': 1e-6,
            'OptimalityTol': 1e-6,
            'IntFeasTol': 1e-6,
            'MIPGap': 1e-4
        }
        
        problem.solve(solver=cp.GUROBI, verbose=True, **solve_params)
        solve_time = time.time() - start_time
        
        # 9. 分析求解结果
        print(f"\n📋 求解结果分析:")
        print(f"   状态: {problem.status}")
        print(f"   求解时间: {solve_time:.4f}秒")
        
        if problem.status == cp.OPTIMAL:
            print(f"   目标函数值: {problem.value:.6f}")
            alpha = alpha_var.value
            u = u_var.value  
            delta = delta_var.value
            
            # 分析解的质量
            alpha_cost_val = 1e6 * max(1, self.opt_params_['l']) * np.dot(alpha, P_squared @ alpha)
            u_cost_val = np.dot(u, u)
            delta_cost_val = self.opt_params_['l'] * np.dot(delta, S_diag @ delta)
            
            print(f"   任务分配代价: {alpha_cost_val:.6f}")
            print(f"   控制输入代价: {u_cost_val:.6f}")
            print(f"   松弛变量代价: {delta_cost_val:.6f}")
            
            # 分析任务分配结果
            print(f"\n📊 任务分配结果:")
            alpha_matrix = alpha.reshape(self.dim_['n_r'], self.dim_['n_t'])
            for i in range(self.dim_['n_r']):
                assigned_tasks = [j for j in range(self.dim_['n_t']) if alpha_matrix[i, j] > 0.5]
                print(f"   机器人{i}: 分配给任务{assigned_tasks}")
            
            # 重新展示能力约束满足情况
            self._display_capability_constraint_calculations(F_matrix, T_matrix, alpha_var, alpha)
            
            status = "Optimal"
        else:
            print(f"   ❌ 优化失败: {problem.status}")
            alpha = np.zeros(alpha_dim)
            u = np.zeros(u_dim)
            delta = np.zeros(delta_dim)
            status = f"Failed: {problem.status}"
            
            if problem.status in [cp.INFEASIBLE, "infeasible", "infeasible_or_unbounded"]:
                print(f"\n🔍 不可行分析:")
                
                # 首先运行传统诊断
                self._diagnose_infeasible_constraints()
                
                # 然后运行IIS分析
                self._analyze_infeasible_constraints_with_iis(x, t)
        
        return alpha, u, delta, solve_time, status, constraints_info

    def _display_lp_file_summary(self, lp_file_path):
        """
        展示.lp文件的关键信息摘要
        """
        try:
            print(f"\n📖 .lp文件内容摘要:")
            print("-" * 50)
            
            with open(lp_file_path, 'r') as f:
                lines = f.readlines()
            
            # 统计信息
            obj_lines = [l for l in lines if l.strip().startswith('Minimize') or l.strip().startswith('Maximize')]
            constraint_lines = [l for l in lines if ':' in l and not l.strip().startswith('\\') and not l.strip().startswith('Minimize') and not l.strip().startswith('Maximize')]
            bound_lines = [l for l in lines if l.strip().startswith('Bounds')]
            binary_lines = [l for l in lines if l.strip().startswith('Binary') or l.strip().startswith('Binaries')]
            
            print(f"   📈 目标函数行数: {len(obj_lines)}")
            print(f"   📋 约束条件行数: {len(constraint_lines)}")
            print(f"   🔢 变量边界行数: {len(bound_lines)}")
            print(f"   🎯 二进制变量行数: {len(binary_lines)}")
            
            # 显示目标函数（前几行）
            if obj_lines:
                print(f"\n   🎯 目标函数 (前3行):")
                for i, line in enumerate(obj_lines[:3]):
                    print(f"      {line.strip()}")
                if len(obj_lines) > 3:
                    print(f"      ... (还有{len(obj_lines)-3}行)")
            
            # 显示约束条件示例（前几个）
            if constraint_lines:
                print(f"\n   📋 约束条件示例 (前5个):")
                for i, line in enumerate(constraint_lines[:5]):
                    print(f"      {line.strip()}")
                if len(constraint_lines) > 5:
                    print(f"      ... (还有{len(constraint_lines)-5}个约束)")
            
            print(f"\n   💡 提示: 打开 {lp_file_path} 查看完整的数学模型")
            
        except Exception as e:
            print(f"   ⚠️ 无法读取.lp文件: {e}")

    def _analyze_infeasible_constraints_with_iis(self, x, t):
        """
        使用Gurobi的IIS (Irreducible Inconsistent Subsystem) 方法
        精确识别导致不可行的最小约束集合
        
        Args:
            x: 当前状态
            t: 当前时间
        """
        print(f"\n🔍 开始IIS分析 - 寻找不可行约束的最小集合")
        print("="*80)
        
        try:
            # 1. 创建Gurobi模型
            model = gp.Model("MIQP_IIS_Analysis")
            model.setParam('OutputFlag', 0)  # 静默模式
            
            # 2. 设置变量维度
            n_r = self.dim_['n_r']
            n_t = self.dim_['n_t']
            n_c = self.dim_['n_c']
            n_u = self.dim_['n_u']
            
            alpha_dim = n_r * n_t
            u_dim = n_r * n_u
            delta_dim = n_r * n_t
            
            print(f"📊 模型规模: {n_r}机器人, {n_t}任务, {n_c}能力, {n_u}控制维度")
            
            # 3. 添加变量
            alpha_vars = model.addVars(alpha_dim, vtype=GRB.BINARY, name="alpha")
            u_vars = model.addVars(u_dim, lb=-GRB.INFINITY, name="u")
            delta_vars = model.addVars(delta_dim, lb=0, ub=self.opt_params_['delta_max'], name="delta")
            
            # 4. 构建约束 (重用已构建的约束矩阵)
            self.build_constraints(x, t)
            A_ineq = self.constraints_['A_ineq']
            b_ineq = self.constraints_['b_ineq']
            A_eq = self.constraints_['A_eq']
            b_eq = self.constraints_['b_eq']
            
            print(f"📋 约束规模: {A_ineq.shape[0]}个不等式, {A_eq.shape[0]}个等式")
            
            # 5. 添加不等式约束并标记
            ineq_constraints = {}
            ineq_constraint_names = {}
            
            # 5.1 能力约束
            constraint_idx = 0
            for j in range(n_t):
                for c in range(n_c):
                    if self.scenario_params_['T'][j, c] > 0.01:  # 只有有需求的才添加
                        cap_idx = constraint_idx
                        if cap_idx < A_ineq.shape[0]:
                            # 构建约束表达式
                            expr = gp.LinExpr()
                            for r in range(n_r):
                                alpha_idx = r * n_t + j
                                coeff = A_ineq[cap_idx, alpha_idx]
                                if abs(coeff) > 1e-10:
                                    expr.addTerms(coeff, alpha_vars[alpha_idx])
                            
                            # 添加u和delta变量
                            for var_idx in range(alpha_dim, A_ineq.shape[1]):
                                coeff = A_ineq[cap_idx, var_idx]
                                if abs(coeff) > 1e-10:
                                    if var_idx < alpha_dim + u_dim:
                                        u_idx = var_idx - alpha_dim
                                        expr.addTerms(coeff, u_vars[u_idx])
                                    else:
                                        delta_idx = var_idx - alpha_dim - u_dim
                                        if delta_idx < delta_dim:
                                            expr.addTerms(coeff, delta_vars[delta_idx])
                            
                            # 添加约束
                            constr_name = f"Capability_Task{j}_Cap{c}"
                            constraint = model.addConstr(expr <= b_ineq[cap_idx], name=constr_name)
                            ineq_constraints[cap_idx] = constraint
                            ineq_constraint_names[cap_idx] = constr_name
                            constraint_idx += 1
            
            # 5.2 机器人数量约束
            for j in range(n_t):
                # 最大约束
                max_idx = constraint_idx
                if max_idx < A_ineq.shape[0]:
                    expr = gp.LinExpr()
                    for r in range(n_r):
                        alpha_idx = r * n_t + j
                        expr.addTerms(1.0, alpha_vars[alpha_idx])
                    
                    constr_name = f"MaxRobots_Task{j}"
                    constraint = model.addConstr(expr <= self.opt_params_['n_r_bounds'][j, 1], name=constr_name)
                    ineq_constraints[max_idx] = constraint
                    ineq_constraint_names[max_idx] = constr_name
                    constraint_idx += 1
                
                # 最小约束
                min_idx = constraint_idx
                if min_idx < A_ineq.shape[0]:
                    expr = gp.LinExpr()
                    for r in range(n_r):
                        alpha_idx = r * n_t + j
                        expr.addTerms(1.0, alpha_vars[alpha_idx])
                    
                    constr_name = f"MinRobots_Task{j}"
                    constraint = model.addConstr(expr >= self.opt_params_['n_r_bounds'][j, 0], name=constr_name)
                    ineq_constraints[min_idx] = constraint
                    ineq_constraint_names[min_idx] = constr_name
                    constraint_idx += 1
            
            # 6. 添加等式约束 (机器人分配约束)
            eq_constraints = {}
            eq_constraint_names = {}
            
            for i in range(n_r):
                expr = gp.LinExpr()
                for j in range(n_t):
                    alpha_idx = i * n_t + j
                    expr.addTerms(1.0, alpha_vars[alpha_idx])
                
                constr_name = f"RobotAssignment_Robot{i}"
                constraint = model.addConstr(expr == 1.0, name=constr_name)
                eq_constraints[i] = constraint
                eq_constraint_names[i] = constr_name
            
            print(f"✅ 约束添加完成: {len(ineq_constraints)}个不等式, {len(eq_constraints)}个等式")
            
            # 7. 求解并检查可行性
            print(f"\n🔧 开始求解检查...")
            model.optimize()
            
            if model.status == GRB.INFEASIBLE:
                print(f"❌ 模型不可行，开始IIS分析...")
                
                # 8. 计算IIS
                model.computeIIS()
                
                print(f"\n🎯 IIS分析结果 - 导致不可行的最小约束集合:")
                print("-" * 60)
                
                # 9. 分析IIS中的不等式约束
                infeasible_ineq_constraints = []
                for idx, constraint in ineq_constraints.items():
                    if constraint.IISConstr:
                        constraint_name = ineq_constraint_names[idx]
                        infeasible_ineq_constraints.append((idx, constraint_name, constraint))
                        
                        # 详细分析这个约束
                        self._analyze_specific_constraint(idx, constraint_name, A_ineq, b_ineq)
                
                # 10. 分析IIS中的等式约束  
                infeasible_eq_constraints = []
                for idx, constraint in eq_constraints.items():
                    if constraint.IISConstr:
                        constraint_name = eq_constraint_names[idx]
                        infeasible_eq_constraints.append((idx, constraint_name, constraint))
                        print(f"🔴 等式约束冲突: {constraint_name}")
                        print(f"   约束内容: 机器人{idx}必须分配到恰好1个任务")
                        print(f"   可能原因: 与其他约束冲突，使得无法满足分配要求")
                
                # 11. 生成修复建议
                self._generate_iis_fix_suggestions(infeasible_ineq_constraints, infeasible_eq_constraints)
                
            elif model.status == GRB.OPTIMAL:
                print(f"✅ 模型可行! 最优值: {model.objVal:.6f}")
                
                # 显示解
                print(f"\n📊 最优解:")
                for i in range(n_r):
                    assigned_tasks = []
                    for j in range(n_t):
                        alpha_idx = i * n_t + j
                        if alpha_vars[alpha_idx].X > 0.5:
                            assigned_tasks.append(j)
                    print(f"   机器人{i}: 分配任务{assigned_tasks}")
                    
            else:
                print(f"⚠️ 求解状态: {model.status}")
                
        except Exception as e:
            print(f"❌ IIS分析失败: {e}")
            import traceback
            traceback.print_exc()
            
        print("="*80)

    def _analyze_specific_constraint(self, constraint_idx, constraint_name, A_ineq, b_ineq):
        """
        分析特定约束的详细信息
        """
        print(f"🔴 不等式约束冲突: {constraint_name}")
        
        # 解析约束名称获取任务和能力信息
        if "Capability" in constraint_name:
            parts = constraint_name.split("_")
            if len(parts) >= 3:
                task_part = parts[1]  # Task0
                cap_part = parts[2]   # Cap0
                
                task_id = int(task_part.replace("Task", ""))
                cap_id = int(cap_part.replace("Cap", ""))
                
                capability_names = ["Movement", "Object_Manip", "Basic_Control", "Liquid_Handle", "Power_Control"]
                task_names = ["Navigate", "Explore", "Pick", "Place", "Open", "Close", 
                             "Clean", "Fill", "Pour", "PowerOn", "PowerOff", "Rearrange", "Wait"]
                
                task_name = task_names[task_id] if task_id < len(task_names) else f"Task_{task_id}"
                cap_name = capability_names[cap_id] if cap_id < len(capability_names) else f"Cap_{cap_id}"
                
                print(f"   约束类型: 能力约束")
                print(f"   任务: {task_name} (ID: {task_id})")
                print(f"   能力: {cap_name} (ID: {cap_id})")
                
                # 显示具体的约束系数
                constraint_row = A_ineq[constraint_idx, :]
                rhs = b_ineq[constraint_idx]
                
                print(f"   约束右端值: {rhs:.3f}")
                print(f"   需求量: {self.scenario_params_['T'][task_id, cap_id]:.3f}")
                
                # 分析哪些机器人具备这个能力
                F_matrix = self.scenario_params_['F']
                capable_robots = [r for r in range(self.dim_['n_r']) if F_matrix[cap_id, r] > 0.5]
                print(f"   具备该能力的机器人: {capable_robots}")
                
                if len(capable_robots) == 0:
                    print(f"   ❌ 根本原因: 没有机器人具备{cap_name}能力!")
                    print(f"   🔧 修复建议: 为至少一个机器人添加{cap_name}能力，或移除需要此能力的任务")
                elif len(capable_robots) < self.scenario_params_['T'][task_id, cap_id]:
                    print(f"   ❌ 根本原因: 具备能力的机器人数({len(capable_robots)}) < 需求({self.scenario_params_['T'][task_id, cap_id]:.1f})")
                    print(f"   🔧 修复建议: 增加具备{cap_name}能力的机器人，或降低任务需求")
                    
        elif "Robots" in constraint_name:
            parts = constraint_name.split("_")
            if len(parts) >= 2:
                task_part = parts[1]  # Task0
                task_id = int(task_part.replace("Task", ""))
                
                task_names = ["Navigate", "Explore", "Pick", "Place", "Open", "Close", 
                             "Clean", "Fill", "Pour", "PowerOn", "PowerOff", "Rearrange", "Wait"]
                task_name = task_names[task_id] if task_id < len(task_names) else f"Task_{task_id}"
                
                print(f"   约束类型: 机器人数量约束")
                print(f"   任务: {task_name} (ID: {task_id})")
                
                min_robots = self.opt_params_['n_r_bounds'][task_id, 0]
                max_robots = self.opt_params_['n_r_bounds'][task_id, 1]
                
                if "Max" in constraint_name:
                    print(f"   约束: 最多{max_robots}个机器人")
                    if max_robots > self.dim_['n_r']:
                        print(f"   ❌ 根本原因: 最大需求({max_robots}) > 总机器人数({self.dim_['n_r']})")
                        print(f"   🔧 修复建议: 降低最大机器人需求至{self.dim_['n_r']}以下")
                else:
                    print(f"   约束: 至少{min_robots}个机器人")
                    if min_robots > self.dim_['n_r']:
                        print(f"   ❌ 根本原因: 最小需求({min_robots}) > 总机器人数({self.dim_['n_r']})")
                        print(f"   🔧 修复建议: 降低最小机器人需求或增加机器人数量")
        
        print()

    def _generate_iis_fix_suggestions(self, infeasible_ineq_constraints, infeasible_eq_constraints):
        """
        基于IIS分析结果生成修复建议
        """
        print(f"\n💡 修复建议总结:")
        print("-" * 50)
        
        if not infeasible_ineq_constraints and not infeasible_eq_constraints:
            print(f"   🎉 没有发现冲突约束!")
            return
        
        suggestions = []
        
        # 分析能力约束冲突
        capability_issues = [c for c in infeasible_ineq_constraints if "Capability" in c[1]]
        if capability_issues:
            suggestions.append("🔧 能力矩阵调整:")
            suggestions.append("   - 检查机器人能力矩阵A，确保有足够机器人具备所需能力")
            suggestions.append("   - 或者降低任务需求矩阵T中的能力要求")
        
        # 分析机器人数量约束冲突
        robot_count_issues = [c for c in infeasible_ineq_constraints if "Robots" in c[1]]
        if robot_count_issues:
            suggestions.append("🔧 机器人数量调整:")
            suggestions.append("   - 检查n_r_bounds参数，确保需求不超过可用机器人数")
            suggestions.append("   - 或者增加机器人数量")
        
        # 分析等式约束冲突
        if infeasible_eq_constraints:
            suggestions.append("🔧 分配约束调整:")
            suggestions.append("   - 机器人分配约束与其他约束冲突")
            suggestions.append("   - 考虑放宽某些任务的能力要求")
            suggestions.append("   - 或者允许机器人不分配任务(修改等式约束为不等式)")
        
        # 通用建议
        suggestions.extend([
            "",
            "🎯 通用修复策略:",
            "   1. 降低目标函数中alpha_cost的权重(当前1e6)",
            "   2. 增加松弛变量的使用范围",
            "   3. 检查矩阵F和T的数值是否合理",
            "   4. 考虑使用更宽松的求解参数"
        ])
        
        for suggestion in suggestions:
            print(suggestion)