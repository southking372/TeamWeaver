# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import cvxpy as cp
import sys
import os

#Add totask_modulePath to importTaskUtilityNormalizer
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
        print(f"List{start+1}to{end}")
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
initializationRTAkind
        """
        assert all(field in scenario_params for field in ['A', 'Hs', 'T', 'ws', 'robot_dyn', 'tasks']),'Missing scene parameter'
        assert all(field in opt_params for field in ['l', 'kappa', 'gamma', 'n_r_bounds', 'delta_max']),'Missing optimization parameters'
        
        self.scenario_params_ = scenario_params
        self.opt_params_ = opt_params
        
        #Dimension information
        self.dim_ = {}
        self.dim_['n_r'] = scenario_params['A'].shape[1]  # number of robots
        self.dim_['n_t'] = scenario_params['T'].shape[0]  #Number of tasks
        self.dim_['n_c'] = scenario_params['T'].shape[1]  #Ability quantity
        self.dim_['n_f'] = scenario_params['A'].shape[0]  #Number of features
        self.dim_['n_x'] = scenario_params['robot_dyn']['n_x']  #status dimension
        self.dim_['n_u'] = scenario_params['robot_dyn']['n_u']  #input dimensions
        
        #Computational mapping and specialization
        self.evaluate_mappings_and_specializations()
        self.check_tasks()
        
        #Initialize constraint dictionary
        self.constraints_ = {}
        self.global_vars_manager_ = None
        self.task_utility_normalizer_ = TaskUtilityNormalizer(
            self.dim_, 
            self.scenario_params_['tasks'],
            task_priority_config
        )
    
    def get_global_vars_manager(self):
        """Get global variable manager"""
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
        """useCVXPY/GurobisolveMIQPoptimization problem"""
        
        #If debug mode is enabled, use the detailed analysis version
        if True:
            alpha, u, delta, solve_time, status, constraints_info = self.solve_miqp_with_detailed_analysis(x, t)
            return alpha, u, delta, solve_time, status
        
        #Standard version
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
            constraints.append(self.constraints_['A_ineq'] @ all_vars <= self.constraints_['b_ineq']) #Add linear inequality constraints
            constraints.append(self.constraints_['A_eq'] @ all_vars == self.constraints_['b_eq']) #Add equality constraints
            constraints.append(alpha_var >= self.constraints_['lb'][:alpha_dim]) #Add variable boundary constraints
            constraints.append(alpha_var <= self.constraints_['ub'][:alpha_dim])
            
            lb_idx = alpha_dim + u_dim
            constraints.append(delta_var >= self.constraints_['lb'][lb_idx:lb_idx+delta_dim])
            constraints.append(delta_var <= self.constraints_['ub'][lb_idx:lb_idx+delta_dim])
            
            #Use more relaxed solution parameters to improve numerical stability
            solve_params = {
                'NumericFocus': 2,  #Reduce numerical accuracy requirements
                'FeasibilityTol': 1e-6,  #Relax feasibility tolerance
                'OptimalityTol': 1e-6,   #Relax optimality tolerance
                'IntFeasTol': 1e-6,      #Relax integer feasibility tolerance
                'MIPGap': 1e-4          #allow certainMIP gap
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
                print(f"[ERROR]The optimization problem is not feasible! Check constraints...")
                self._diagnose_infeasible_constraints()
                alpha = np.zeros(alpha_dim)
                u = np.zeros(u_dim)
                delta = np.zeros(delta_dim)
                opt_sol_info = f"Infeasible: {problem.status}"
            elif problem.status == cp.UNBOUNDED or problem.status == "unbounded":
                print(f"[ERROR]Optimization problems are unbounded! Check the objective function...")
                alpha = np.zeros(alpha_dim)
                u = np.zeros(u_dim)
                delta = np.zeros(delta_dim)
                opt_sol_info = f"Unbounded: {problem.status}"
            else:
                print(f"Optimization has not converged, status: {problem.status}")
                print(f"[DEBUG]Start diagnosing constraints...")
                self._diagnose_infeasible_constraints()
                alpha = np.zeros(alpha_dim)
                u = np.zeros(u_dim)
                delta = np.zeros(delta_dim)
                opt_sol_info = f"Not optimal: {problem.status}"
            
            print("MIQP Status:", opt_sol_info)
            
            return alpha, u, delta, time_to_solve_miqp, opt_sol_info
            
        except cp.error.SolverError as e:
            print(f"CVXPYSolver error: {e}")
            return np.zeros(alpha_dim), np.zeros(u_dim), np.zeros(delta_dim), 0, "Error"
        
    def solve_reduced_qp(self, x, alpha, t):
        """Solution fixedalphaSimplification belowQPquestion"""
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
        """Get scaled task function values, gradients, and time derivatives"""
        task = self.scenario_params_['tasks'][j]
        
        if global_vars_dict is not None:
            task_func_value = task['function'](x[:, i], t, i, vars_dict=global_vars_dict)
            task_grad_value = task['gradient'](x[:, i], t, i, vars_dict=global_vars_dict)
            task_time_deriv_value = task['time_derivative'](x[:, i], t, i, vars_dict=global_vars_dict)
        else:
            task_func_value = task['function'](x[:, i], t, i)
            task_grad_value = task['gradient'](x[:, i], t, i) 
            task_time_deriv_value = task['time_derivative'](x[:, i], t, i)
        
        #Apply scaling factor
        # task_func_value = task_func_value * scaling_factors[j]
        # task_grad_value = task_grad_value * scaling_factors[j]
        # task_time_deriv_value = task_time_deriv_value * scaling_factors[j]
        
        # [TODO]Temporarily cancel the zoom factor
        task_func_value = task_func_value
        task_grad_value = task_grad_value
        task_time_deriv_value = task_time_deriv_value
        
        return task_func_value, task_grad_value, task_time_deriv_value
        
    def build_constraints(self, x, t):
        """buildMIQPconstraint"""
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
        
        #Set variable boundaries
        lb[:n_r*n_t] = np.zeros(n_r*n_t)
        ub[:n_r*n_t] = np.ones(n_r*n_t)
        lb[n_r*n_t+n_r*n_u:] = np.zeros(n_r*n_t)
        ub[n_r*n_t+n_r*n_u:] = self.opt_params_['delta_max'] * np.ones(n_r*n_t)
        
        global_vars_dict = self.get_global_vars_dict()
        
        # Get LLM Response
        llm_response = ""
        if global_vars_dict and 'current_llm_response' in global_vars_dict:
            llm_response = global_vars_dict['current_llm_response']
        
        #Normalization and task corresponding adjustment and subsequent unification
        # scaling_factors = self.normalize_task_utilities(x, t, global_vars_dict, llm_response)
        # print(f"[DEBUG-LYP] scaling_factors: {scaling_factors}")
        
        constraint_idx = 0
        # === 1. CBFconstraint(Control Barrier Functions) ===
        # print(f"[DEBUG] Adding CBF constraints...")
        # for i in range(n_r):
        #     for j in range(n_t):
        #         task = self.scenario_params_['tasks'][j]
        #         robot_dyn = self.scenario_params_['robot_dyn']
                
        #         #Get scaled task function values, gradients, and time derivatives
        #         task_func_value, task_grad_value, task_time_deriv_value = self.get_scaled_task_values(
        #             x, t, i, j, global_vars_dict, scaling_factors
        #         )
                
        #         # CBFconstraint: dot(h) + gamma(h) >= 0
        #         A_ineq[constraint_idx, n_r*n_t+i*n_u:n_r*n_t+(i+1)*n_u] = -task_grad_value @ robot_dyn['g'](x[:, i])
        #         b_ineq[constraint_idx] = (task_grad_value @ robot_dyn['f'](x[:, i]) + 
        #                                  task_time_deriv_value + 
        #                                  self.opt_params_['gamma'](task_func_value))
        #         constraint_idx += 1
        
        # ===2. SimplifiedDelta-Alphaconstraint(Dramatically reduced task switching constraints) ===
        # print(f"[DEBUG] Adding simplified delta-alpha constraints...")
        # #Define mission-critical indexes:Navigate(0), Pick(2), Place(3)- Most commonly used tasks
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
        
        # === 3. CBF slack variableconstraint===
        # print(f"[DEBUG] Adding CBF slack constraints...")
        # slack_start_idx = cbf_constraints
        # for i in range(n_r):
        #     for j in range(n_t):
        #         slack_idx = slack_start_idx + i*n_t + j
        #         if slack_idx < total_ineq:  #Boundary checking
        #             A_ineq[slack_idx, n_r*n_t+n_r*n_u+i*n_t+j] = -1
        #             b_ineq[slack_idx] = 0
        
        # ===4. Capability constraints(Feature capability constraints) ===
        print(f"[DEBUG] Adding capability constraints...")
        # cap_start_idx = cbf_constraints + cbf_slack_constraints
        cap_start_idx = 0
        for j in range(n_t):
            for c in range(n_c):
                cap_idx = cap_start_idx + j*n_c + c
                if cap_idx < total_ineq:  #Boundary checking
                    # F * alpha >= T:Ensure assigned robots are equipped to perform tasksjrequired capabilitiesc
                    for r in range(n_r):
                        A_ineq[cap_idx, r*n_t+j] = -self.scenario_params_['F'][c, r]
                    b_ineq[cap_idx] = -self.scenario_params_['T'][j, c]
        
        # === 5. number of robotsconstraint(Robot count bounds) ===
        print(f"[DEBUG] Adding robot count constraints...")
        bound_start_idx = cap_start_idx + n_t*n_c
        for j in range(n_t):
            #Maximum number of robots constraint: sum(alpha_rj) <= max_robots_j
            max_idx = bound_start_idx + j
            if max_idx < total_ineq:  #Boundary checking
                for r in range(n_r):
                    A_ineq[max_idx, r*n_t+j] = 1
                b_ineq[max_idx] = self.opt_params_['n_r_bounds'][j, 1]
            
            #Minimum number of robots constraint: sum(alpha_rj) >= min_robots_j
            min_idx = bound_start_idx + n_t + j
            if min_idx < total_ineq:  #Boundary checking
                for r in range(n_r):
                    A_ineq[min_idx, r*n_t+j] = -1
                b_ineq[min_idx] = -self.opt_params_['n_r_bounds'][j, 0]
        
        # print(f"[DEBUG] Constraints building completed. Final constraint_idx: {constraint_idx}")
        print(f"[DEBUG] Used constraint indices: {capability_constraints + robot_bound_constraints}")
        # print(f"[DEBUG] Used constraint indices: {cbf_constraints + cbf_slack_constraints + capability_constraints + robot_bound_constraints}")
        
        #Verify constraint matrix consistency
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
            
            #Adjust matrix size to match actual needs
            if expected_total > total_ineq:
                additional_rows = expected_total - total_ineq
                A_ineq = np.vstack([A_ineq, np.zeros((additional_rows, total_vars))])
                b_ineq = np.hstack([b_ineq, np.zeros(additional_rows)])
                print(f"[DEBUG] Extended constraint matrix to {A_ineq.shape}")
            elif expected_total < total_ineq:
                #truncation matrix
                A_ineq = A_ineq[:expected_total, :]
                b_ineq = b_ineq[:expected_total]
                print(f"[DEBUG] Truncated constraint matrix to {A_ineq.shape}")
                
        #Check matrix validity
        if np.any(np.isnan(A_ineq)) or np.any(np.isinf(A_ineq)):
            print(f"[ERROR] A_ineq contains NaN or Inf values!")
        if np.any(np.isnan(b_ineq)) or np.any(np.isinf(b_ineq)):
            print(f"[ERROR] b_ineq contains NaN or Inf values!")
        if np.any(np.isnan(A_eq)) or np.any(np.isinf(A_eq)):
            print(f"[ERROR] A_eq contains NaN or Inf values!")
        if np.any(np.isnan(b_eq)) or np.any(np.isinf(b_eq)):
            print(f"[ERROR] b_eq contains NaN or Inf values!")
        
        #Add an equality constraint: each robot must be assigned to at least one task
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
        """Build simplified constraints (fixedalphaofQP）"""
        n_r = self.dim_['n_r']
        n_t = self.dim_['n_t']
        n_u = self.dim_['n_u']
        
        #Initialize constraint matrix
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
        
        #getLLMresponse message
        llm_response = ""
        if global_vars_dict and 'current_llm_response' in global_vars_dict:
            llm_response = global_vars_dict['current_llm_response']
        
        scaling_factors = self.normalize_task_utilities(x, t, global_vars_dict, llm_response)
        
        # Task CBFs and delta-alpha constraints
        for i in range(n_r):
            for j in range(n_t):
                # CBFs for tasks
                idx = (i*n_t) + j
                
                #Get scaled task function values, gradients, and time derivatives
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
        """Update scene parameters"""
        for field_name, value in scenario_params.items():
            if field_name in self.scenario_params_:
                assert field_name not in ['F', 'S'],'matrixFandSUnable to set (automatically calculated)'
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
        """Compute characteristics to capabilities(F)and task to robot(S)mapping"""
        n_c = self.dim_['n_c']
        n_r = self.dim_['n_r']
        
        #initializationFmatrix
        self.scenario_params_['F'] = np.zeros((n_c, n_r))
        
        #calculateF
        for k in range(n_c):
            if self.scenario_params_['ws'] is not None and len(self.scenario_params_['ws']) > 0:
                W_k = np.diag(self.scenario_params_['ws'][k])
                self.scenario_params_['F'][k, :] = W_k @ ((self.scenario_params_['Hs'][k] @ self.scenario_params_['A']) > 0.999)
            else:
                self.scenario_params_['F'][k, :] = ((self.scenario_params_['Hs'][k] @ self.scenario_params_['A']) > 0.999)
        
        #calculateS
        self.scenario_params_['S'] = ((self.scenario_params_['T'] @ self.scenario_params_['F']) > 0.999).astype(float)
        
        #Build projection matrix
        self.build_projector()

    def build_projector(self):
        """Build projection matrixP"""
        n_r = self.dim_['n_r']
        n_t = self.dim_['n_t']
        
        #initializationPmatrix
        self.P_ = np.zeros((n_t, n_t*n_r))
        for i in range(n_r):
            self.P_[:, i*n_t:(i+1)*n_t] = np.eye(n_t)
        
        #Updated based on specialization matrixP
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
        """Create a column vector of a specific dimension with only one element being 1"""
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
        """Diagnose constraints that lead to infeasibility"""
        print(f"[DEBUG]Diagnosing infeasible constraints...")
        
        #Check the basic information of the constraint matrix
        A_ineq = self.constraints_['A_ineq']
        b_ineq = self.constraints_['b_ineq']
        A_eq = self.constraints_['A_eq']
        b_eq = self.constraints_['b_eq']
        lb = self.constraints_['lb']
        ub = self.constraints_['ub']
        
        print(f"Inequality constraint matrix: {A_ineq.shape}")
        print(f"Equality constraint matrix: {A_eq.shape}")
        print(f"variable bounds: lb={lb.shape}, ub={ub.shape}")
        
        #Check for obviously conflicting constraints
        n_r = self.dim_['n_r']
        n_t = self.dim_['n_t']
        n_c = self.dim_['n_c']
        
        #1. Check the consistency of variable boundaries
        print(f"1. Check variable bounds:")
        inconsistent_bounds = np.where(lb > ub)[0]
        if len(inconsistent_bounds) > 0:
            print(f"    [ERROR]Discover{len(inconsistent_bounds)}lower bound of a variable>upper bound!")
            for idx in inconsistent_bounds[:5]:  #Only show the first 5
                print(f"variable{idx}: lb={lb[idx]}, ub={ub[idx]}")
        
        #2. Checknumber of robotsconstraint
        print(f"  2. number of robotsboundary:")
        total_min_robots = 0
        total_max_robots = 0
        for j in range(n_t):
            min_robots = self.opt_params_['n_r_bounds'][j, 0]
            max_robots = self.opt_params_['n_r_bounds'][j, 1]
            total_min_robots += min_robots
            total_max_robots += max_robots
            print(f"Task{j}:smallest={min_robots},maximum={max_robots}")
            if min_robots > max_robots:
                print(f"    [ERROR]Task{j}:Minimum number of robots>Maximum number of robots!")
            if min_robots > n_r:
                print(f"    [ERROR]Task{j}:Minimum number of robots>Total number of robots!")
        
        print(f"total:minimum requirements={total_min_robots},maximum demand={total_max_robots},Available robots={n_r}")
        if total_min_robots > n_r:
            print(f"    [ERROR]Minimum robotic requirements for all tasks({total_min_robots}) >Number of robots available({n_r})!")
        
        #3. Checkcapability matrixCompatibility with mission requirements
        F = self.scenario_params_['F']
        T = self.scenario_params_['T']
        print(f"3. Capability compatibility check:")
        print(f"    capability matrix F: {F.shape}")
        print(f"    Task requirement matrix T: {T.shape}")
        
        for j in range(n_t):
            required_capabilities = np.where(T[j, :] > 0.5)[0]
            if len(required_capabilities) > 0:
                print(f"Task{j}Requires ability: {required_capabilities}")
                for c in required_capabilities:
                    capable_robots = np.where(F[c, :] > 0.5)[0]
                    print(f"ability{c}:Robots with= {capable_robots} (common{len(capable_robots)}indivual)")
                    if len(capable_robots) == 0:
                        print(f"      [ERROR]No robot has the ability{c}!")
                    elif len(capable_robots) < self.opt_params_['n_r_bounds'][j, 0]:
                        print(f"      [ERROR]Have the ability{c}number of robots({len(capable_robots)}) <Task{j}minimum requirements({self.opt_params_['n_r_bounds'][j, 0]})")
        
        #4. Check the consistency of equality constraints
        print(f"4. Equality constraint checking:")
        print(f"Equality constraint right-hand vectorb_eq: {b_eq}")
        print(f"Require:Each robot is assigned exactly one task")
        if not np.allclose(b_eq, 1.0):
            print(f"    [ERROR]The equality constraint requires that each robot be assigned a value not equal to 1!")
        
        #5. Check the numerical stability of the constraint matrix
        print(f"5. Numerical stability check:")
        if np.any(np.isnan(A_ineq)) or np.any(np.isinf(A_ineq)):
            print(f"    [ERROR] A_ineqIncludeNaNorInfvalue!")
        if np.any(np.isnan(b_ineq)) or np.any(np.isinf(b_ineq)):
            print(f"    [ERROR] b_ineqIncludeNaNorInfvalue!")
        if np.any(np.isnan(A_eq)) or np.any(np.isinf(A_eq)):
            print(f"    [ERROR] A_eqIncludeNaNorInfvalue!")
        if np.any(np.isnan(b_eq)) or np.any(np.isinf(b_eq)):
            print(f"    [ERROR] b_eqIncludeNaNorInfvalue!")
        
        #6. Check the condition number of the constraint matrix
        try:
            cond_ineq = np.linalg.cond(A_ineq @ A_ineq.T + 1e-10 * np.eye(A_ineq.shape[0]))
            cond_eq = np.linalg.cond(A_eq @ A_eq.T + 1e-10 * np.eye(A_eq.shape[0]))
            print(f"Inequality constraint matrix condition number: {cond_ineq:.2e}")
            print(f"Equality constraint matrix condition number: {cond_eq:.2e}")
            if cond_ineq > 1e12:
                print(f"    [WARNING]The condition number of the inequality constraint matrix is ​​too large and may be numerically unstable.!")
            if cond_eq > 1e12:
                print(f"    [WARNING]The condition number of the equality constraint matrix is ​​too large and may be numerically unstable.!")
        except Exception as e:
            print(f"    [WARNING]Unable to calculate condition number: {e}")
        
        print(f"[DEBUG]Constraint diagnostics completed.")

    def analyze_constraints_detailed(self, x, t, alpha_var, u_var, delta_var):
        """
Detailed analysis of all constraints and their impact on the objective function andcontrolThe role of variables
        
        Args:
            x:Current status[n_x, n_r]
            t: current time
            alpha_var, u_var, delta_var: CVXPYvariable
            
        Returns:
            constraints_info:Detailed constraint information dictionary
        """
        print("\n" + "="*80)
        print("🔍 MIQPDetailed analysis of constraints")
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
        
        # ===1. Variable boundary analysis===
        print("\n📊variable boundary analysis:")
        print("-" * 50)
        
        alpha_bounds = {
            'name': 'Task Assignment Variables (α)',
            'dimension': f"{n_r}×{n_t} = {n_r*n_t}",
            'type': 'Binary (0 or 1)',
            'lower_bound': 0,
            'upper_bound': 1,
            'meaning':'robotiWhether to assign to taskj'
        }
        
        u_bounds = {
            'name': 'Control Input Variables (u)', 
            'dimension': f"{n_r}×{n_u} = {n_r*n_u}",
            'type': 'Continuous',
            'lower_bound': '-∞',
            'upper_bound': '+∞',
            'meaning':'robotiofcontrolinput vector'
        }
        
        delta_bounds = {
            'name': 'Slack Variables (δ)',
            'dimension': f"{n_r}×{n_t} = {n_r*n_t}",
            'type': 'Continuous', 
            'lower_bound': 0,
            'upper_bound': f"δ_max = {self.opt_params_['delta_max']}",
            'meaning':'Constrain slack variables, allowing minor violationsCBFconstraint'
        }
        
        for var_info in [alpha_bounds, u_bounds, delta_bounds]:
            print(f"  • {var_info['name']}")
            print(f"Dimensions: {var_info['dimension']}")
            print(f"type: {var_info['type']}")
            print(f"scope: [{var_info['lower_bound']}, {var_info['upper_bound']}]")
            print(f"meaning: {var_info['meaning']}")
            print()
        
        constraints_info['variable_bounds'] = {
            'alpha': alpha_bounds,
            'u': u_bounds, 
            'delta': delta_bounds
        }
        
        # ===2. Analysis of Capability Constraints===
        print("\n🎯Capability Constraint Analysis(F·α ≥ T):")
        print("-" * 50)
        
        F_matrix = self.scenario_params_['F']
        T_matrix = self.scenario_params_['T']
        
        print(f"Capability mapping matrixF: {F_matrix.shape} (Ability × Robot)")
        print(f"Task requirement matrix T: {T_matrix.shape} (Task×Ability)")
        
        capability_constraints = []
        for j in range(n_t):
            task_name = f"Task_{j}"
            for c in range(n_c):
                capability_name = f"Capability_{c}"
                
                #Analyze which robots have this capability
                capable_robots = [r for r in range(n_r) if F_matrix[c, r] > 0.5]
                task_requirement = T_matrix[j, c]
                
                if task_requirement > 0.01:  #Only show constraints with actual requirements
                    constraint_info = {
                        'constraint_id': f"Cap_{j}_{c}",
                        'task': task_name,
                        'capability': capability_name, 
                        'requirement': f"{task_requirement:.3f}",
                        'capable_robots': capable_robots,
                        'constraint_form': f"Σ(F[{c},r] * α[r,{j}]) ≥ {task_requirement:.3f}",
                        'physical_meaning': f"implement{task_name}need{capability_name}, only robots{capable_robots}Have this ability"
                    }
                    
                    capability_constraints.append(constraint_info)
                    
                    print(f"  📌 {constraint_info['constraint_id']}: {constraint_info['physical_meaning']}")
                    print(f"constrained: {constraint_info['constraint_form']}")
                    print(f"capable robots: {capable_robots}")
                    
                    if len(capable_robots) == 0:
                        print(f"     ⚠️warn:No robot has{capability_name}!")
                    elif len(capable_robots) < task_requirement:
                        print(f"     ⚠️warn:Number of capable robots({len(capable_robots)}) <need({task_requirement})")
                    print()
        
        constraints_info['capability_constraints'] = capability_constraints
        
        # === 3. number of robotsconstraint analysis===
        print("\n👥 number of robotsconstraint analysis:")
        print("-" * 50)
        
        robot_count_constraints = []
        for j in range(n_t):
            min_robots = self.opt_params_['n_r_bounds'][j, 0]
            max_robots = self.opt_params_['n_r_bounds'][j, 1] 
            
            constraint_info = {
                'task_id': j,
                'min_constraint': f"{min_robots} ≤ Σ(α[r,{j}]) for r=0..{n_r-1}",
                'max_constraint': f"Σ(α[r,{j}]) ≤ {max_robots} for r=0..{n_r-1}",
                'meaning': f"Task{j}need{min_robots}-{max_robots}robot",
                'feasibility': 'OK' if min_robots <= max_robots <= n_r else 'INFEASIBLE'
            }
            
            robot_count_constraints.append(constraint_info)
            
            print(f"  📋 Task_{j}: {constraint_info['meaning']}")
            print(f"minimum constraint: {constraint_info['min_constraint']}")
            print(f"maximum constraint: {constraint_info['max_constraint']}")
            print(f"feasibility: {constraint_info['feasibility']}")
            
            if constraint_info['feasibility'] == 'INFEASIBLE':
                if min_robots > max_robots:
                    print(f"     ❌mistake:minimum requirements>maximum demand!")
                if max_robots > n_r:
                    print(f"     ❌mistake:maximum demand>Number of robots available!")
            print()
        
        constraints_info['robot_count_constraints'] = robot_count_constraints
        
        # ===4. Equality constraint analysis===  
        print("\n⚖️Equality constraint analysis(Robot unique allocation):")
        print("-" * 50)
        
        equality_constraints = []
        for i in range(n_r):
            constraint_info = {
                'robot_id': i,
                'constraint': f"Σ(α[{i},j]) = 1 for j=0..{n_t-1}",
                'meaning': f"robot{i}Must and can only be assigned to one task"
            }
            equality_constraints.append(constraint_info)
            print(f"  🤖 Robot_{i}: {constraint_info['meaning']}")
            print(f"constrained: {constraint_info['constraint']}")
        
        constraints_info['equality_constraints'] = equality_constraints
        
        # ===5. Objective function analysis===
        print(f"\n🎯Objective function analysis:")
        print("-" * 50)
        
        P_matrix = self.P_
        S_matrix = self.scenario_params_['S']
        l_param = self.opt_params_['l']
        
        objective_info = {
            'total_form': 'minimize: α_cost + u_cost + δ_cost',
            'alpha_cost': {
                'form': f"1e6 × max(1, {l_param}) × α^T × P^T × P × α",
                'weight': f"{1e6 * max(1, l_param):.0e}",
                'purpose':'Task allocation stability and avoid frequent switching',
                'matrix_P_shape': f"{P_matrix.shape}",
                'matrix_P_property':'Projection matrices constructed based on task specialization'
            },
            'u_cost': {
                'form': "u^T × I × u",
                'weight': "1.0",
                'purpose': 'controlMinimize input and reduce energy consumption',
                'matrix_shape': f"{n_r*n_u}×{n_r*n_u}Identity matrix"
            },
            'delta_cost': {
                'form': f"{l_param} × δ^T × S × δ",
                'weight': f"{l_param}",
                'purpose':'Slack variable penalty, soft constraint violation',
                'matrix_S_shape': f"{S_matrix.shape}",
                'matrix_S_meaning':'Task specialization matrix, violation costs of tasks with different weights'
            }
        }
        
        print(f"  📈overall objective function: {objective_info['total_form']}")
        print()
        print(f"  1️⃣Task allocation cost(α_cost):")
        print(f"form: {objective_info['alpha_cost']['form']}")
        print(f"weight: {objective_info['alpha_cost']['weight']}")
        print(f"Purpose: {objective_info['alpha_cost']['purpose']}")
        print(f"matrixP: {objective_info['alpha_cost']['matrix_P_shape']} - {objective_info['alpha_cost']['matrix_P_property']}")
        print()
        print(f"  2️⃣ controlinput cost(u_cost):")
        print(f"form: {objective_info['u_cost']['form']}")
        print(f"weight: {objective_info['u_cost']['weight']}")
        print(f"Purpose: {objective_info['u_cost']['purpose']}")
        print(f"matrix: {objective_info['u_cost']['matrix_shape']}")
        print()
        print(f"  3️⃣slack variable cost(δ_cost):")
        print(f"form: {objective_info['delta_cost']['form']}")
        print(f"weight: {objective_info['delta_cost']['weight']}")
        print(f"Purpose: {objective_info['delta_cost']['purpose']}")
        print(f"matrixS: {objective_info['delta_cost']['matrix_S_shape']} - {objective_info['delta_cost']['matrix_S_meaning']}")
        
        constraints_info['objective_function'] = objective_info
        
        # ===6. Constraint matrix statistics===
        print(f"\n📊Constraint matrix statistics:")
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
        
        print(f"  📋inequality constraints:")
        print(f"matrixA_ineq: {matrix_stats['inequality_constraints']['matrix_A_shape']}")
        print(f"vectorb_ineq: {matrix_stats['inequality_constraints']['vector_b_shape']}")  
        print(f"non-zero elements: {matrix_stats['inequality_constraints']['non_zero_elements']}")
        print(f"sparsity: {matrix_stats['inequality_constraints']['sparsity']}")
        print(f"condition number: {matrix_stats['inequality_constraints']['condition_number']:.2e}")
        print()
        print(f"  ⚖️Equality constraints:")
        print(f"matrixA_eq: {matrix_stats['equality_constraints']['matrix_A_shape']}")
        print(f"vectorb_eq: {matrix_stats['equality_constraints']['vector_b_shape']}")
        print(f"non-zero elements: {matrix_stats['equality_constraints']['non_zero_elements']}")
        print(f"Matrix rank: {matrix_stats['equality_constraints']['rank']}")
        
        constraints_info['matrix_info'] = matrix_stats
        
        # ===7. Feasibility pre-check===
        print(f"\n🔬Feasibility pre-check:")
        print("-" * 50)
        
        feasibility_issues = []
        
        #Check capability match
        for j in range(n_t):
            for c in range(n_c):
                if T_matrix[j, c] > 0.01:
                    capable_robots = np.sum(F_matrix[c, :] > 0.5)
                    if capable_robots == 0:
                        issue = f"Task{j}Requires ability{c}, but no robot has this ability"
                        feasibility_issues.append(issue)
                        print(f"     ❌ {issue}")
        
        #examinenumber of robots
        total_min_demand = np.sum([self.opt_params_['n_r_bounds'][j, 0] for j in range(n_t)])
        if total_min_demand > n_r:
            issue = f"Minimum requirements for all tasks({total_min_demand}) >Number of robots available({n_r})"
            feasibility_issues.append(issue)
            print(f"     ❌ {issue}")
        
        #Check matrix numerical stability
        if matrix_stats['inequality_constraints']['condition_number'] > 1e12:
            issue = f"The condition number of the inequality constraint matrix is ​​too large({matrix_stats['inequality_constraints']['condition_number']:.2e})"
            feasibility_issues.append(issue)
            print(f"     ⚠️  {issue}")
        
        if len(feasibility_issues) == 0:
            print(f"     ✅Pre-check passed, restraint system looks feasible")
        
        constraints_info['feasibility_analysis'] = {
            'issues': feasibility_issues,
            'is_likely_feasible': len(feasibility_issues) == 0
        }
        
        print("\n" + "="*80)
        print("🏁Constraint analysis completed")
        print("="*80)
        
        return constraints_info

    def _display_capability_constraint_calculations(self, F_matrix, T_matrix, alpha_var=None, alpha_solution=None):
        """
Show the calculation process of capacity constraints in detail:F·α ≥ T
        
        Args:
            F_matrix:Capability mapping matrix[n_c, n_r]
            T_matrix:task requirements matrix[n_t, n_c]  
            alpha_var: CVXPYvariable(Optional)
            alpha_solution:After solvingαvalue(Optional)
        """
        print(f"\n🧮Detailed explanation of the capability constraint calculation process(F·α ≥ T):")
        print("="*80)
        
        n_r = self.dim_['n_r'] 
        n_t = self.dim_['n_t']
        n_c = self.dim_['n_c']
        
        #1. Show matrix structure
        print(f"\n📋Matrix dimension information:")
        print(f"   F (Capability mapping): {F_matrix.shape} - [Ability × Robot]")
        print(f"   T (Mission requirements): {T_matrix.shape} - [Task×Ability]") 
        print(f"   α (Assign variables): [{n_r}×{n_t}] = [{n_r*n_t}]- Expand to vector")
        
        #2. DisplayFMatrix details
        print(f"\n🤖Capability mapping matrixF:")
        print(f"Line: ability[Movement, Object_Manip, Basic_Control, Liquid_Handle, Power_Control]")
        print(f"Column: Robot[Robot_0, Robot_1]")
        capability_names = ["Movement", "Object_Manip", "Basic_Control", "Liquid_Handle", "Power_Control"]
        robot_names = [f"Robot_{i}" for i in range(n_r)]
        
        print(f"\n   {'ability':<15} ", end="")
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
        
        #3. DisplayTMatrix details
        print(f"\n📋 Task requirement matrix T:")
        print(f"row: task[Navigate, Explore, Pick, Place, Open, Close, Clean, Fill, Pour, PowerOn, PowerOff, Rearrange, Wait]")
        print(f"Column: ability[Movement, Object_Manip, Basic_Control, Liquid_Handle, Power_Control]")
        
        task_names = ["Navigate", "Explore", "Pick", "Place", "Open", "Close", 
                     "Clean", "Fill", "Pour", "PowerOn", "PowerOff", "Rearrange", "Wait"]
        
        print(f"\n   {'Task':<12} ", end="")
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
        
        #4. Show specific constraint calculations
        print(f"\n🎯Specific constraint calculation process:")
        print(f"constraint form:for each taskjand abilityc, Σ(F[c,r] × α[r,j]) ≥ T[j,c]")
        print(f"significance:assigned to tasksjof robots in capabilitiescThe sum must meet the task requirements")
        
        constraint_count = 0
        for j in range(n_t):
            task_name = task_names[j] if j < len(task_names) else f"Task_{j}"
            for c in range(n_c):
                if T_matrix[j, c] > 0.01:  #Only show constraints with actual requirements
                    constraint_count += 1
                    print(f"\n   📌constraint#{constraint_count}: {task_name}need{capability_names[c]}")
                    print(f"mathematical expression: ", end="")
                    
                    # Build constraintsexpression
                    terms = []
                    for r in range(n_r):
                        if F_matrix[c, r] > 0.01:
                            terms.append(f"{F_matrix[c, r]:.1f}*α[{r},{j}]")
                    
                    constraint_expr = " + ".join(terms) if terms else "0"
                    print(f"{constraint_expr} ≥ {T_matrix[j, c]:.1f}")
                    
                    #Show which robots meet this capability
                    capable_robots = [r for r in range(n_r) if F_matrix[c, r] > 0.5]
                    print(f"capable robots: {capable_robots}")
                    
                    if len(capable_robots) == 0:
                        print(f"      ⚠️warn:No robot has this ability!")
                    elif len(capable_robots) < T_matrix[j, c]:
                        print(f"      ⚠️Notice:Number of capable robots({len(capable_robots)}) <need({T_matrix[j, c]:.1f})")
                        
        #5. If there are solution results, show how the constraints are satisfied.
        if alpha_solution is not None:
            print(f"\n🎯Constraint satisfaction check(Based on solution results):")
            alpha_matrix = alpha_solution.reshape(n_r, n_t)
            
            all_satisfied = True
            for j in range(n_t):
                task_name = task_names[j] if j < len(task_names) else f"Task_{j}"
                for c in range(n_c):
                    if T_matrix[j, c] > 0.01:
                        #calculateF[c,:] · α[:,j] 
                        assigned_capability = np.dot(F_matrix[c, :], alpha_matrix[:, j])
                        required_capability = T_matrix[j, c]
                        satisfied = assigned_capability >= required_capability - 1e-6
                        
                        status = "✅" if satisfied else "❌"
                        print(f"      {status} {task_name}-{capability_names[c]}: {assigned_capability:.3f} ≥ {required_capability:.1f}")
                        
                        if not satisfied:
                            all_satisfied = False
                            print(f"amount of violation: {required_capability - assigned_capability:.3f}")
            
            if all_satisfied:
                print(f"\n   🎉All capability constraints are met!")
            else:
                print(f"\n   ⚠️There are unmet capability constraints")
        
        print("="*80)

    def solve_miqp_with_detailed_analysis(self, x, t):
        """
with detailed analysisMIQPSolution method
        
        Args:
            x:Current status
            t: current time
            
        Returns:
            alpha, u, delta, solve_time, status, constraints_info
        """
        print("\n🚀startMIQPSolve(Detailed analysis mode)")
        
        # 1. Build constraints
        self.build_constraints(x, t)
        
        #2. Set variables
        alpha_dim = self.dim_['n_r'] * self.dim_['n_t']
        u_dim = self.dim_['n_r'] * self.dim_['n_u']
        delta_dim = self.dim_['n_r'] * self.dim_['n_t']
        
        alpha_var = cp.Variable(alpha_dim, boolean=True)
        u_var = cp.Variable(u_dim)
        delta_var = cp.Variable(delta_dim)
        
        #3. Analyze constraints in detail
        constraints_info = self.analyze_constraints_detailed(x, t, alpha_var, u_var, delta_var)
        
        #4. Show the capability constraint calculation process in detail
        F_matrix = self.scenario_params_['F']
        T_matrix = self.scenario_params_['T']
        self._display_capability_constraint_calculations(F_matrix, T_matrix, alpha_var)
        
        #5. Construct objective function
        P_squared = self.P_.T @ self.P_
        S_diag = np.diag(np.reshape(self.scenario_params_['S'], (-1)))
        
        alpha_cost = 1e6 * max(1, self.opt_params_['l']) * cp.quad_form(alpha_var, P_squared)
        u_cost = cp.quad_form(u_var, np.eye(u_dim))
        delta_cost = self.opt_params_['l'] * cp.quad_form(delta_var, S_diag)
        objective = cp.Minimize(alpha_cost + u_cost + delta_cost)
        
        #6. Add constraints
        constraints = []
        all_vars_h = cp.hstack([alpha_var, u_var, delta_var])
        all_vars = all_vars_h.T
        constraints.append(self.constraints_['A_ineq'] @ all_vars <= self.constraints_['b_ineq'])
        constraints.append(self.constraints_['A_eq'] @ all_vars == self.constraints_['b_eq'])  #Equality constraints
        constraints.append(alpha_var >= self.constraints_['lb'][:alpha_dim])
        constraints.append(alpha_var <= self.constraints_['ub'][:alpha_dim])
        
        lb_idx = alpha_dim + u_dim
        constraints.append(delta_var >= self.constraints_['lb'][lb_idx:lb_idx+delta_dim])
        constraints.append(delta_var <= self.constraints_['ub'][lb_idx:lb_idx+delta_dim])
        
        #7. Create an issue and export it.lpdocument
        problem = cp.Problem(objective, constraints)
        
        print(f"\n📄Export.lpFile for model checking...")
        try:
            lp_file_path = f"MIQP_model_{int(t*1000)}.lp"
            
            #useGurobiBackend generation.lpdocument
            problem.solve(solver=cp.GUROBI, verbose=False, save_file=lp_file_path)
            
            print(f"   ✅ .lpFile saved: {lp_file_path}")
            print(f"   📖You can open it with a text editor to view the complete mathematical model")
            
            #show.lpKey information of the file
            self._display_lp_file_summary(lp_file_path)
            
        except Exception as e:
            print(f"   ⚠️ .lpFile export failed: {e}")
        
        #8. Solve
        print(f"\n🔧startCVXPYSolve...")
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
        
        #9. Analyze solution results
        print(f"\n📋Analysis of solution results:")
        print(f"state: {problem.status}")
        print(f"Solution time: {solve_time:.4f}Second")
        
        if problem.status == cp.OPTIMAL:
            print(f"objective function value: {problem.value:.6f}")
            alpha = alpha_var.value
            u = u_var.value  
            delta = delta_var.value
            
            #Analytical solution quality
            alpha_cost_val = 1e6 * max(1, self.opt_params_['l']) * np.dot(alpha, P_squared @ alpha)
            u_cost_val = np.dot(u, u)
            delta_cost_val = self.opt_params_['l'] * np.dot(delta, S_diag @ delta)
            
            print(f"Task allocation cost: {alpha_cost_val:.6f}")
            print(f"   controlinput cost: {u_cost_val:.6f}")
            print(f"slack variable cost: {delta_cost_val:.6f}")
            
            #Analyze task assignment results
            print(f"\n📊Task assignment results:")
            alpha_matrix = alpha.reshape(self.dim_['n_r'], self.dim_['n_t'])
            for i in range(self.dim_['n_r']):
                assigned_tasks = [j for j in range(self.dim_['n_t']) if alpha_matrix[i, j] > 0.5]
                print(f"robot{i}:assigned to tasks{assigned_tasks}")
            
            #Re-display capability constraint satisfaction
            self._display_capability_constraint_calculations(F_matrix, T_matrix, alpha_var, alpha)
            
            status = "Optimal"
        else:
            print(f"   ❌Optimization failed: {problem.status}")
            alpha = np.zeros(alpha_dim)
            u = np.zeros(u_dim)
            delta = np.zeros(delta_dim)
            status = f"Failed: {problem.status}"
            
            if problem.status in [cp.INFEASIBLE, "infeasible", "infeasible_or_unbounded"]:
                print(f"\n🔍infeasible analysis:")
                
                #First run traditional diagnostics
                self._diagnose_infeasible_constraints()
                
                #then runIISanalyze
                self._analyze_infeasible_constraints_with_iis(x, t)
        
        return alpha, u, delta, solve_time, status, constraints_info

    def _display_lp_file_summary(self, lp_file_path):
        """
exhibit.lpSummary of key information from the document
        """
        try:
            print(f"\n📖 .lpSummary of file content:")
            print("-" * 50)
            
            with open(lp_file_path, 'r') as f:
                lines = f.readlines()
            
            #Statistics
            obj_lines = [l for l in lines if l.strip().startswith('Minimize') or l.strip().startswith('Maximize')]
            constraint_lines = [l for l in lines if ':' in l and not l.strip().startswith('\\') and not l.strip().startswith('Minimize') and not l.strip().startswith('Maximize')]
            bound_lines = [l for l in lines if l.strip().startswith('Bounds')]
            binary_lines = [l for l in lines if l.strip().startswith('Binary') or l.strip().startswith('Binaries')]
            
            print(f"   📈Number of rows of objective function: {len(obj_lines)}")
            print(f"   📋Number of constraint rows: {len(constraint_lines)}")
            print(f"   🔢Variable boundary row number: {len(bound_lines)}")
            print(f"   🎯Binary variable row number: {len(binary_lines)}")
            
            #Show objective function (first few lines)
            if obj_lines:
                print(f"\n   🎯objective function(first 3 lines):")
                for i, line in enumerate(obj_lines[:3]):
                    print(f"      {line.strip()}")
                if len(obj_lines) > 3:
                    print(f"      ... (besides{len(obj_lines)-3}OK)")
            
            #Show constraint examples (first few)
            if constraint_lines:
                print(f"\n   📋Constraint example(Top 5):")
                for i, line in enumerate(constraint_lines[:5]):
                    print(f"      {line.strip()}")
                if len(constraint_lines) > 5:
                    print(f"      ... (besides{len(constraint_lines)-5}constraints)")
            
            print(f"\n   💡hint:Open{lp_file_path}View full mathematical model")
            
        except Exception as e:
            print(f"   ⚠️Unable to read.lpdocument: {e}")

    def _analyze_infeasible_constraints_with_iis(self, x, t):
        """
useGurobiofIIS (Irreducible Inconsistent Subsystem)method
Precisely identify the minimum set of constraints that result in infeasibility
        
        Args:
            x:Current status
            t: current time
        """
        print(f"\n🔍startIISAnalysis - Finding the Minimum Set of Infeasible Constraints")
        print("="*80)
        
        try:
            #1. CreateGurobiModel
            model = gp.Model("MIQP_IIS_Analysis")
            model.setParam('OutputFlag', 0)  #silent mode
            
            #2. Set variable dimensions
            n_r = self.dim_['n_r']
            n_t = self.dim_['n_t']
            n_c = self.dim_['n_c']
            n_u = self.dim_['n_u']
            
            alpha_dim = n_r * n_t
            u_dim = n_r * n_u
            delta_dim = n_r * n_t
            
            print(f"📊Model size: {n_r}robot, {n_t}Task, {n_c}ability, {n_u}controlDimensions")
            
            #3. Add variables
            alpha_vars = model.addVars(alpha_dim, vtype=GRB.BINARY, name="alpha")
            u_vars = model.addVars(u_dim, lb=-GRB.INFINITY, name="u")
            delta_vars = model.addVars(delta_dim, lb=0, ub=self.opt_params_['delta_max'], name="delta")
            
            # 4. Build constraints (Reuse an already constructed constraint matrix)
            self.build_constraints(x, t)
            A_ineq = self.constraints_['A_ineq']
            b_ineq = self.constraints_['b_ineq']
            A_eq = self.constraints_['A_eq']
            b_eq = self.constraints_['b_eq']
            
            print(f"📋Constraint scale: {A_ineq.shape[0]}inequalities, {A_eq.shape[0]}equation")
            
            #5. Add inequality constraints and label them
            ineq_constraints = {}
            ineq_constraint_names = {}
            
            #5.1 Capability constraints
            constraint_idx = 0
            for j in range(n_t):
                for c in range(n_c):
                    if self.scenario_params_['T'][j, c] > 0.01:  #Only add those who need it
                        cap_idx = constraint_idx
                        if cap_idx < A_ineq.shape[0]:
                            # Build constraintsexpression
                            expr = gp.LinExpr()
                            for r in range(n_r):
                                alpha_idx = r * n_t + j
                                coeff = A_ineq[cap_idx, alpha_idx]
                                if abs(coeff) > 1e-10:
                                    expr.addTerms(coeff, alpha_vars[alpha_idx])
                            
                            #Add touanddeltavariable
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
                            
                            #Add constraints
                            constr_name = f"Capability_Task{j}_Cap{c}"
                            constraint = model.addConstr(expr <= b_ineq[cap_idx], name=constr_name)
                            ineq_constraints[cap_idx] = constraint
                            ineq_constraint_names[cap_idx] = constr_name
                            constraint_idx += 1
            
            # 5.2 number of robotsconstraint
            for j in range(n_t):
                #maximum constraint
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
                
                #minimum constraint
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
            
            #6. Add equality constraints(Robot allocation constraints)
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
            
            print(f"✅Constraint addition completed: {len(ineq_constraints)}inequalities, {len(eq_constraints)}equation")
            
            #7. Solve and check feasibility
            print(f"\n🔧Start solving check...")
            model.optimize()
            
            if model.status == GRB.INFEASIBLE:
                print(f"❌The model is not feasible, startIISanalyze...")
                
                #8. CalculationIIS
                model.computeIIS()
                
                print(f"\n🎯 IISAnalysis results - the minimum set of constraints that results in infeasibility:")
                print("-" * 60)
                
                #9. AnalysisIISInequality constraints in
                infeasible_ineq_constraints = []
                for idx, constraint in ineq_constraints.items():
                    if constraint.IISConstr:
                        constraint_name = ineq_constraint_names[idx]
                        infeasible_ineq_constraints.append((idx, constraint_name, constraint))
                        
                        #Analyze this constraint in detail
                        self._analyze_specific_constraint(idx, constraint_name, A_ineq, b_ineq)
                
                #10. AnalysisIISEquality constraints in
                infeasible_eq_constraints = []
                for idx, constraint in eq_constraints.items():
                    if constraint.IISConstr:
                        constraint_name = eq_constraint_names[idx]
                        infeasible_eq_constraints.append((idx, constraint_name, constraint))
                        print(f"🔴Equality constraint conflict: {constraint_name}")
                        print(f"Constraint content:robot{idx}Must be assigned to exactly 1 task")
                        print(f"Possible reasons:Conflicts with other constraints, making allocation requirements unsatisfactory")
                
                #11. Generate repair suggestions
                self._generate_iis_fix_suggestions(infeasible_ineq_constraints, infeasible_eq_constraints)
                
            elif model.status == GRB.OPTIMAL:
                print(f"✅The model is feasible!optimal value: {model.objVal:.6f}")
                
                #show solution
                print(f"\n📊optimal solution:")
                for i in range(n_r):
                    assigned_tasks = []
                    for j in range(n_t):
                        alpha_idx = i * n_t + j
                        if alpha_vars[alpha_idx].X > 0.5:
                            assigned_tasks.append(j)
                    print(f"robot{i}:Assign tasks{assigned_tasks}")
                    
            else:
                print(f"⚠️Solution status: {model.status}")
                
        except Exception as e:
            print(f"❌ IISAnalysis failed: {e}")
            import traceback
            traceback.print_exc()
            
        print("="*80)

    def _analyze_specific_constraint(self, constraint_idx, constraint_name, A_ineq, b_ineq):
        """
Analyze the details of a specific constraint
        """
        print(f"🔴Inequality constraint conflict: {constraint_name}")
        
        #Parse constraint names to obtain task and capability information
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
                
                print(f"constraint type:Capability constraints")
                print(f"Task: {task_name} (ID: {task_id})")
                print(f"ability: {cap_name} (ID: {cap_id})")
                
                #Show specific constraint coefficients
                constraint_row = A_ineq[constraint_idx, :]
                rhs = b_ineq[constraint_idx]
                
                print(f"Constraint right end value: {rhs:.3f}")
                print(f"demand: {self.scenario_params_['T'][task_id, cap_id]:.3f}")
                
                #Analyze which robots have this ability
                F_matrix = self.scenario_params_['F']
                capable_robots = [r for r in range(self.dim_['n_r']) if F_matrix[cap_id, r] > 0.5]
                print(f"Robots with this ability: {capable_robots}")
                
                if len(capable_robots) == 0:
                    print(f"   ❌root cause:No robot has{cap_name}ability!")
                    print(f"   🔧Repair suggestions:Add for at least one bot{cap_name}ability, or remove tasks that require this ability")
                elif len(capable_robots) < self.scenario_params_['T'][task_id, cap_id]:
                    print(f"   ❌root cause:Number of capable robots({len(capable_robots)}) <need({self.scenario_params_['T'][task_id, cap_id]:.1f})")
                    print(f"   🔧Repair suggestions:Increase the ability{cap_name}capabilities of the robot, or reduce task demands")
                    
        elif "Robots" in constraint_name:
            parts = constraint_name.split("_")
            if len(parts) >= 2:
                task_part = parts[1]  # Task0
                task_id = int(task_part.replace("Task", ""))
                
                task_names = ["Navigate", "Explore", "Pick", "Place", "Open", "Close", 
                             "Clean", "Fill", "Pour", "PowerOn", "PowerOff", "Rearrange", "Wait"]
                task_name = task_names[task_id] if task_id < len(task_names) else f"Task_{task_id}"
                
                print(f"constraint type: number of robotsconstraint")
                print(f"Task: {task_name} (ID: {task_id})")
                
                min_robots = self.opt_params_['n_r_bounds'][task_id, 0]
                max_robots = self.opt_params_['n_r_bounds'][task_id, 1]
                
                if "Max" in constraint_name:
                    print(f"constraint:most{max_robots}robot")
                    if max_robots > self.dim_['n_r']:
                        print(f"   ❌root cause:maximum demand({max_robots}) >Total number of robots({self.dim_['n_r']})")
                        print(f"   🔧Repair suggestions:Reduce the maximum robot requirement to{self.dim_['n_r']}the following")
                else:
                    print(f"constraint:At least{min_robots}robot")
                    if min_robots > self.dim_['n_r']:
                        print(f"   ❌root cause:minimum requirements({min_robots}) >Total number of robots({self.dim_['n_r']})")
                        print(f"   🔧Repair suggestions:Reduce minimum robot requirements or increasenumber of robots")
        
        print()

    def _generate_iis_fix_suggestions(self, infeasible_ineq_constraints, infeasible_eq_constraints):
        """
based onIISAnalysis results generate repair recommendations
        """
        print(f"\n💡Summary of repair suggestions:")
        print("-" * 50)
        
        if not infeasible_ineq_constraints and not infeasible_eq_constraints:
            print(f"   🎉No conflicting constraints found!")
            return
        
        suggestions = []
        
        #Analyze capability constraint conflicts
        capability_issues = [c for c in infeasible_ineq_constraints if "Capability" in c[1]]
        if capability_issues:
            suggestions.append("🔧 capability matrixAdjustment:")
            suggestions.append("- Check the robotcapability matrixA, ensuring there are enough robots with the required capabilities")
            suggestions.append("- Or reduce the task demand matrixTability requirements in")
        
        #analyzenumber of robotsconstraint conflict
        robot_count_issues = [c for c in infeasible_ineq_constraints if "Robots" in c[1]]
        if robot_count_issues:
            suggestions.append("🔧 number of robotsAdjustment:")
            suggestions.append("- examinen_r_boundsparameters to ensure that demand does not exceed the number of available robots")
            suggestions.append("- or increasenumber of robots")
        
        #Analyze equality constraint conflicts
        if infeasible_eq_constraints:
            suggestions.append("🔧Allocation constraint adjustment:")
            suggestions.append("- Robot allocation constraints conflict with other constraints")
            suggestions.append("- Consider relaxing the ability requirements for certain tasks")
            suggestions.append("- Or allow the robot to not assign tasks(Modify equality constraints to inequalities)")
        
        #General advice
        suggestions.extend([
            "",
            "🎯Common repair strategies:",
            "1. Reduce the objective functionalpha_costweight(Current 1e6)",
            "2. Increase the usage range of slack variables",
            "3. Check the matrixFandTIs the value of",
            "4. Consider using more relaxed solution parameters"
        ])
        
        for suggestion in suggestions:
            print(suggestion)
