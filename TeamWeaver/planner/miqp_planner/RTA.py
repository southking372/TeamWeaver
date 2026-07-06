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



class RTA:
    def __init__(self, scenario_params, opt_params, global_vars_manager=None):
        """
        Constructor for RTA (Run-Time Assurance) class
        
        Parameters:
        -----------
        scenario_params : dict
            scenario_params['A'] : Matrix containing feature to robot mapping (size(A) = n_f * n_r)
            scenario_params['Hs'] : List where each element corresponds to a capability.
                                   Each element is a matrix of size [c_nhe x n_f]:
                                       c_nhe: the number of hyper-edges for capability k
                                       n_f  : the number of features.x
            scenario_params['T'] : Matrix containing task to capability mapping. (size(T) = n_t * n_c)
            scenario_params['ws'] : List where each element corresponds to a capability.
                                   Each element is a vector containing the weights for the hyper-edges.
            scenario_params['robot_dyn'] : Dictionary of functions for f and g in $\dot x = f(x) + g(x) u$,
                                         and n_x and n_u, where $x \in \mathbb{R}^{n_x}$ and $u \in \mathbb{R}^{n_u}$
            scenario_params['tasks'] : List of tasks. (size(tasks) = n_t)
                                      Each task is a dictionary with function, gradient, and time_derivative fields
        
        opt_params : dict
            opt_params['l'] : Scaling constant for the cost
            opt_params['kappa'] : Scaling constant for alpha's
            opt_params['gamma'] : Class K function
            opt_params['n_r_bounds'] : Matrix of min and max number of robots for each task (size(n_r_bounds) = n_t * 2)
            opt_params['delta_max'] : Max infinity-norm for each delta_i
        global_vars_manager : GlobalVarsManager, optional
            Instance of the global variable manager. Defaults to None.
        """
        # Check required fields
        assert all(field in scenario_params for field in ['A', 'Hs', 'T', 'ws', 'robot_dyn', 'tasks']), 'Missing scenario parameters.'
        assert all(field in opt_params for field in ['l', 'kappa', 'gamma', 'n_r_bounds', 'delta_max']), 'Missing optimization parameters.'
        
        self.scenario_params_ = scenario_params
        self.opt_params_ = opt_params
        
        # Dimensions
        self.dim_ = {}
        self.dim_['n_r'] = scenario_params['A'].shape[1]  # Number of Robots
        self.dim_['n_t'] = scenario_params['T'].shape[0]  # Number of Tasks
        self.dim_['n_c'] = scenario_params['T'].shape[1]  # Number of Capabilities
        self.dim_['n_f'] = scenario_params['A'].shape[0]  # Number of Features
        self.dim_['n_x'] = scenario_params['robot_dyn']['n_x']  # State dimension
        self.dim_['n_u'] = scenario_params['robot_dyn']['n_u']  # Input dimension
        
        # Initialize constraints dictionary
        self.constraints_ = {}
        
        self.global_vars_manager_ = global_vars_manager
    
    def add_global_vars_manager(self, global_vars_manager):
        """
        Adds or updates the GlobalVarsManager instance used by RTA.

        Parameters:
        -----------
        global_vars_manager : GlobalVarsManager
            The instance of the global variable manager.
        """
        # Optional: Add type checking if GlobalVarsManager class is accessible here
        # from habitat_llm.planner.miqp_planner.GlobalVarsManager import GlobalVarsManager
        # if not isinstance(global_vars_manager, GlobalVarsManager):
        #     raise TypeError("global_vars_manager must be an instance of GlobalVarsManager")
        print("Updating GlobalVarsManager for RTA instance.")
        self.global_vars_manager_ = global_vars_manager
        
    def solve_miqp(self, x, t):
        """
        Using CVXPY to solve MIQP optimization problems
        
        Parameters:
        -----------
        x : numpy.ndarray
            Current robot status
        t : float
            current time
            
        Returns:
        --------
        alpha : numpy.ndarray
            task assignment binary variable
        u : numpy.ndarray
            controlinput
        delta : numpy.ndarray
            slack variable
        time_to_solve_miqp : float
            Time required to solve MIQP
        opt_sol_info : str
            Optimization solution status information
        """
        # Construct constraints on current state and time
        self.build_constraints(x, t)
        
        # problem dimensions
        alpha_dim = self.dim_['n_r'] * self.dim_['n_t']
        u_dim = self.dim_['n_r'] * self.dim_['n_u']
        delta_dim = self.dim_['n_r'] * self.dim_['n_t']
        
        # Start timing
        start_time = time.time()
        
        try:
            # Define variables
            alpha_var = cp.Variable(alpha_dim, boolean=True)  # binary variable
            u_var = cp.Variable(u_dim)                        # controlinput variables
            delta_var = cp.Variable(delta_dim)                # slack variable
            
            # Calculate P_.T @ P_for secondary type
            P_squared = self.P_.T @ self.P_
            
            # Create a diagonal matrix of S
            S_diag = np.diag(np.reshape(self.scenario_params_['S'], (-1)))
            
            # objective function
            alpha_cost = 1e6 * max(1, self.opt_params_['l']) * cp.quad_form(alpha_var, P_squared)
            u_cost = cp.quad_form(u_var, np.eye(u_dim))  # u_var' * u_var
            delta_cost = self.opt_params_['l'] * cp.quad_form(delta_var, S_diag)
            
            # Combined objective function
            objective = cp.Minimize(alpha_cost + u_cost + delta_cost)
            
            # Constraints
            constraints = []
            
            # Add linear inequality constraints: A_ineq * [alpha; u; delta] <= b_ineq
            all_vars_h = cp.hstack([alpha_var, u_var, delta_var])
            all_vars = all_vars_h.T
            constraints.append(self.constraints_['A_ineq'] @ all_vars <= self.constraints_['b_ineq'])
            
            # Add equality constraints (relax to inequality constraints): A_eq * [alpha; u; delta] <= b_eq
            constraints.append(self.constraints_['A_eq'] @ all_vars <= self.constraints_['b_eq'])
            
            # Add variable boundary constraints
            # alphaboundary constraints
            constraints.append(alpha_var >= self.constraints_['lb'][:alpha_dim])
            constraints.append(alpha_var <= self.constraints_['ub'][:alpha_dim])
            
            # deltaboundary constraints
            lb_idx = alpha_dim + u_dim
            constraints.append(delta_var >= self.constraints_['lb'][lb_idx:lb_idx+delta_dim])
            constraints.append(delta_var <= self.constraints_['ub'][lb_idx:lb_idx+delta_dim])
            
            # Build and solve problems
            problem = cp.Problem(objective, constraints)
            
            # Set solver parameters
            solve_params = {
                'NumericFocus': 3,  # Improve numerical accuracy
                'FeasibilityTol': 1e-9,  # Improve feasibility tolerance
                'OptimalityTol': 1e-9,  # Increase optimality tolerance
                'IntFeasTol': 1e-9  # Improve integer feasibility tolerance
            }
            
            # solve problems
            problem.solve(solver=cp.GUROBI, verbose=False, **solve_params)
            
            # Calculate solution time
            time_to_solve_miqp = time.time() - start_time
            
            # Extract results
            if problem.status == cp.OPTIMAL:
                alpha = alpha_var.value
                u = u_var.value
                delta = delta_var.value
                opt_sol_info = "Optimal"
            else:
                print(f"Optimization has not converged, status: {problem.status}")
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
        """
        Solve the reduced QP optimization problem with fixed alpha
        
        Parameters:
        -----------
        x : numpy.ndarray
            Current state of the robots
        alpha : numpy.ndarray
            Fixed task assignment variables
        t : float
            Current time
            
        Returns:
        --------
        u : numpy.ndarray
            Control inputs
        delta : numpy.ndarray
            Slack variables
        time_to_solve_qp : float
            Time taken to solve the QP
        opt_sol_info : str
            Optimization solution status
        """
        # Build reduced constraints for fixed alpha
        self.build_reduced_constraints(x, t)
        
        # Problem dimensions
        alpha_dim = self.dim_['n_r'] * self.dim_['n_t']
        u_dim = self.dim_['n_r'] * self.dim_['n_u']
        delta_dim = self.dim_['n_r'] * self.dim_['n_t']
        
        # Start timing
        start_time = time.time()
        
        # Create new model
        model = gp.Model("RTA_QP")
        model.setParam('OutputFlag', 0)  # Suppress output
        
        # Create variables
        u_var = model.addVars(u_dim, lb=-float('inf'), ub=float('inf'), name="u")
        delta_var = model.addVars(delta_dim, lb=0, ub=self.opt_params_['delta_max'], name="delta")
        
        # Flatten variables for easier handling
        u_flat = [u_var[i] for i in range(u_dim)]
        delta_flat = [delta_var[i] for i in range(delta_dim)]
        
        # Create combined variable list
        all_vars = u_flat + delta_flat
        
        # Build objective function
        S_diag = np.reshape(self.scenario_params_['S'], (-1))
        
        # Add quadratic terms for u
        for i in range(u_dim):
            model.addQConstr(2 * u_var[i] * u_var[i], GRB.EQUAL, model.addVar(name=f"q_u_{i}"))
            
        # Add quadratic terms for delta
        for i in range(delta_dim):
            model.addQConstr(2 * self.opt_params_['l'] * S_diag[i] * delta_var[i] * delta_var[i], 
                             GRB.EQUAL, 
                             model.addVar(name=f"q_delta_{i}"))
        
        # Get constraint matrices
        A = self.constraints_['A_ineq'][:, alpha_dim:]
        b = self.constraints_['b_ineq'] - self.constraints_['A_ineq'][:, :alpha_dim] @ alpha
        
        # Add inequality constraints: A * [u; delta] <= b
        for i in range(A.shape[0]):
            row = A[i, :]
            expr = 0
            for j in range(len(all_vars)):
                if row[j] != 0:
                    expr += row[j] * all_vars[j]
            model.addConstr(expr <= b[i])
        
        # Add bounds constraints
        for i in range(u_dim + delta_dim):
            idx = alpha_dim + i
            model.addConstr(all_vars[i] >= self.constraints_['lb'][idx])
            model.addConstr(all_vars[i] <= self.constraints_['ub'][idx])
        
        # Optimize model
        model.optimize()
        
        # Calculate solve time
        time_to_solve_qp = time.time() - start_time
        
        # Extract solution
        if model.status == GRB.OPTIMAL:
            u = np.array([u_var[i].X for i in range(u_dim)])
            delta = np.array([delta_var[i].X for i in range(delta_dim)])
            opt_sol_info = "Optimal"
        else:
            u = np.zeros(u_dim)
            delta = np.zeros(delta_dim)
            opt_sol_info = f"Not optimal: {model.status}"
            
        return u, delta, time_to_solve_qp, opt_sol_info
        
    def set_scenario_params(self, scenario_params):
        """
        Update scenario parameters
        
        Parameters:
        -----------
        scenario_params : dict
            Updated scenario parameters
        """
        for field_name, value in scenario_params.items():
            if field_name in self.scenario_params_:
                assert field_name not in ['F', 'S'], 'Matrices F and S cannot be set (they are automatically evaluated). To update S, use set_specializations'
                self.scenario_params_[field_name] = value
        
        self.evaluate_mappings_and_specializations()
        
    def set_opt_params(self, opt_params):
        """
        Update optimization parameters
        
        Parameters:
        -----------
        opt_params : dict
            Updated optimization parameters
        """
        for field_name, value in opt_params.items():
            if field_name in self.opt_params_:
                self.opt_params_[field_name] = value
                
    def set_specializations(self, S):
        """
        Set specialization matrix
        
        Parameters:
        -----------
        S : numpy.ndarray
            Specialization matrix
        """
        self.scenario_params_['S'] = S
        self.build_projector()
        
    def get_specializations(self):
        """
        Get specialization matrix
        
        Returns:
        --------
        S : numpy.ndarray
            Specialization matrix
        """
        return self.scenario_params_['S']
        
    def build_constraints(self, x, t):
        """
        Build the constraints for the MIQP
        
        Parameters:
        -----------
        x : numpy.ndarray
            Current state of the robots
        t : float
            Current time
        """
        n_r = self.dim_['n_r']
        n_t = self.dim_['n_t']
        n_c = self.dim_['n_c']
        n_u = self.dim_['n_u']
        
        # Initialize constraint matrices
        total_ineq = n_r*n_t + n_r*n_t**2 + n_t*n_c + 2*n_t
        total_vars = 2*n_r*n_t + n_r*n_u
        
        A_ineq = np.zeros((total_ineq, total_vars))
        b_ineq = np.zeros(total_ineq)
        A_eq = np.zeros((n_r, total_vars))
        b_eq = np.ones(n_r) # It should be a matrix of all ones
        # b_eq = np.zeros(n_r) 
        lb = -np.inf * np.ones(total_vars)
        ub = np.inf * np.ones(total_vars)
        
        # Set bounds for alpha
        lb[:n_r*n_t] = np.zeros(n_r*n_t)
        ub[:n_r*n_t] = np.ones(n_r*n_t)
        
        # Set bounds for u (Add input constraints)
        # lb[n_r*n_t:n_r*n_t+n_r*n_u] = -10.0 * np.ones(n_r*n_u)
        # ub[n_r*n_t:n_r*n_t+n_r*n_u] = 10.0 * np.ones(n_r*n_u)
        
        # Set bounds for delta
        lb[n_r*n_t+n_r*n_u:] = np.zeros(n_r*n_t)
        ub[n_r*n_t+n_r*n_u:] = self.opt_params_['delta_max'] * np.ones(n_r*n_t)
        
        # Get global variables
        global_vars_dict = None
        if self.global_vars_manager_ is not None:
            try:
                global_vars_dict = self.global_vars_manager_.get_all_vars()
                # print(f"Successfully obtained the global variable manager, variable count: {len(global_vars_dict)}")
            except Exception as e:
                print(f"Error getting global variables from manager: {e}")
        else:
            print("warn: RTA GlobalVarsManager is not set.")
        
        # Task CBFs and delta-alpha constraints
        for i in range(n_r):
            for j in range(n_t):
                # CBFs for tasks
                idx = i*n_t + j  # Corresponds to MATLAB (i-1)*n_t+j
                task = self.scenario_params_['tasks'][j]
                robot_dyn = self.scenario_params_['robot_dyn']
                
                # Use global vars dict to call the task function
                if global_vars_dict is not None:
                    # Call the task function and pass the global vars dict
                    task_func_value = task['function'](x[:, i], t, i, vars_dict=global_vars_dict)
                    task_grad_value = task['gradient'](x[:, i], t, i, vars_dict=global_vars_dict)
                    task_time_deriv_value = task['time_derivative'](x[:, i], t, i, vars_dict=global_vars_dict)
                else:
                    # There is no global variable manager, and the function is called normally
                    task_func_value = task['function'](x[:, i], t, i)
                    task_grad_value = task['gradient'](x[:, i], t, i) 
                    task_time_deriv_value = task['time_derivative'](x[:, i], t, i)
                
                # Corresponds to MATLAB: n_r*n_t+(i-1)*n_u+1 : n_r*n_t+i*n_u
                A_ineq[idx, n_r*n_t+i*n_u:n_r*n_t+(i+1)*n_u] = -task_grad_value @ robot_dyn['g'](x[:, i])
                b_ineq[idx] = (task_grad_value @ robot_dyn['f'](x[:, i]) + 
                              task_time_deriv_value + 
                              self.opt_params_['gamma'](task_func_value))
                
                # delta-alpha constraints
                # Corresponds to MATLAB: n_r*n_t+(i-1)*n_t^2+(j-1)*n_t+1 : n_r*n_t+(i-1)*n_t^2+j*n_t
                base_idx = n_r*n_t + i*n_t**2 + j*n_t
                
                # Corresponds to the entire row assignment in MATLAB
                A_ineq[base_idx:base_idx+n_t, i*n_t:(i+1)*n_t] = self.opt_params_['delta_max'] * self.onec(n_t, j)
                A_ineq[base_idx:base_idx+n_t, n_r*n_t+n_r*n_u+i*n_t:n_r*n_t+n_r*n_u+(i+1)*n_t] = -1/self.opt_params_['kappa'] * np.eye(n_t) + self.onec(n_t, j)
            
            # alpha_i sum up to 1 (Equality constraints)
            # NOTE: In MATLAB version this is an equality constraint but here we use inequality constraint for better feasibility
            # Original MATLAB: this.constraints_.A_eq(i,(i-1)*this.dim_.n_t+1:i*this.dim_.n_t) = ones(1, this.dim_.n_t);
            A_eq[i, (i*n_t):((i+1)*n_t)] = np.ones(n_t)
        
        # CBFs for tasks - additional constraints
        A_ineq[:n_r*n_t, n_r*n_t+n_r*n_u:] = -np.eye(n_r*n_t)
        
        # delta-alpha constraints - right-hand side
        b_ineq[n_r*n_t:n_r*n_t+n_r*n_t**2] = self.opt_params_['delta_max'] * np.ones(n_r*n_t**2)
        
        # Feature capability constraints and robot number bounds for each task
        start_idx = n_r*n_t + n_r*n_t**2
        for j in range(n_t):
            # F alpha >= T (minimum amount of capabilities for each task)
            cap_idx = start_idx + j*n_c
            for c in range(n_c):
                for r in range(n_r):
                    A_ineq[cap_idx+c, r*n_t+j] = -self.scenario_params_['F'][c, r]
                b_ineq[cap_idx+c] = -self.scenario_params_['T'][j, c]
            
            # Maximum number of robots for each task
            max_idx = start_idx + n_t*n_c + j
            for r in range(n_r):
                A_ineq[max_idx, r*n_t+j] = 1
            b_ineq[max_idx] = self.opt_params_['n_r_bounds'][j, 1]
            
            # Minimum number of robots for each task
            min_idx = start_idx + n_t*n_c + n_t + j
            for r in range(n_r):
                A_ineq[min_idx, r*n_t+j] = -1
            b_ineq[min_idx] = -self.opt_params_['n_r_bounds'][j, 0]
        
        # Remove constraints between a task and itself
        to_remove = []
        for i in range(n_r):
            for j in range(n_t):
                base_idx = n_r*n_t + (i*n_t**2) + (j*n_t) + j
                to_remove.append(base_idx)
        
        # Sort and remove from the end to avoid index shifting
        to_remove.sort(reverse=True)
        for idx in to_remove:
            if idx < A_ineq.shape[0]:  # Make sure the index is valid
                A_ineq = np.delete(A_ineq, idx, axis=0)
                b_ineq = np.delete(b_ineq, idx)
        
        # Store constraints
        self.constraints_['A_ineq'] = A_ineq
        self.constraints_['b_ineq'] = b_ineq
        self.constraints_['A_eq'] = A_eq
        self.constraints_['b_eq'] = b_eq
        self.constraints_['lb'] = lb
        self.constraints_['ub'] = ub
        
    def build_reduced_constraints(self, x, t):
        """
        Build the reduced constraints for the QP (with fixed alpha)
        
        Parameters:
        -----------
        x : numpy.ndarray
            Current state of the robots
        t : float
            Current time
        """
        n_r = self.dim_['n_r']
        n_t = self.dim_['n_t']
        n_u = self.dim_['n_u']
        
        # Initialize constraint matrices
        total_ineq = n_r*n_t + n_r*n_t**2
        total_vars = 2*n_r*n_t + n_r*n_u
        
        A_ineq = np.zeros((total_ineq, total_vars))
        b_ineq = np.zeros(total_ineq)
        A_eq = np.zeros((n_r, total_vars))
        b_eq = np.zeros(n_r)
        lb = -np.inf * np.ones(total_vars)
        ub = np.inf * np.ones(total_vars)
        
        # Set bounds for delta
        lb[n_r*n_t+n_r*n_u:] = np.zeros(n_r*n_t)
        ub[n_r*n_t+n_r*n_u:] = self.opt_params_['delta_max'] * np.ones(n_r*n_t)
        
        # Get global variables
        global_vars_dict = None
        if self.global_vars_manager_ is not None:
            try:
                global_vars_dict = self.global_vars_manager_.get_all_vars()
                # print(f"build_reduced_constraints: Successfully obtained the global variable manager, variable count: {len(global_vars_dict)}")
            except Exception as e:
                print(f"build_reduced_constraints: Error getting global variables from manager: {e}")
        else:
            print("build_reduced_constraints: warn: RTA GlobalVarsManager is not set.")
        
        # Task CBFs and delta-alpha constraints
        for i in range(n_r):
            for j in range(n_t):
                # CBFs for tasks
                idx = (i*n_t) + j
                task = self.scenario_params_['tasks'][j]
                robot_dyn = self.scenario_params_['robot_dyn']
                
                # Use global vars dict to call the task function
                if global_vars_dict is not None:
                    # Call the task function and pass the global vars dict
                    task_func_value = task['function'](x[:, i], t, i, vars_dict=global_vars_dict)
                    task_grad_value = task['gradient'](x[:, i], t, i, vars_dict=global_vars_dict)
                    task_time_deriv_value = task['time_derivative'](x[:, i], t, i, vars_dict=global_vars_dict)
                else:
                    # There is no global variable manager, and the function is called normally
                    task_func_value = task['function'](x[:, i], t, i)
                    task_grad_value = task['gradient'](x[:, i], t, i) 
                    task_time_deriv_value = task['time_derivative'](x[:, i], t, i)
                
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
        
    def evaluate_mappings_and_specializations(self):
        """
        Evaluate the feature to capability (F) and task to robot (S) mappings
        """
        n_c = self.dim_['n_c']
        n_r = self.dim_['n_r']
        
        # Initialize F matrix
        self.scenario_params_['F'] = np.zeros((n_c, n_r))
        
        # Evaluate F
        for k in range(n_c):
            if self.scenario_params_['ws'] is not None and len(self.scenario_params_['ws']) > 0:
                # If weights are provided
                W_k = np.diag(self.scenario_params_['ws'][k])
                self.scenario_params_['F'][k, :] = np.max(W_k @ ((self.scenario_params_['Hs'][k] @ self.scenario_params_['A']) > 0.999), axis=0)
            else:
                # If no weight is provided
                self.scenario_params_['F'][k, :] = np.max((self.scenario_params_['Hs'][k] @ self.scenario_params_['A']) > 0.999, axis=0)
        
        # Evaluate S - Use double to ensure the result is a floating point number
        self.scenario_params_['S'] = ((self.scenario_params_['T'] @ self.scenario_params_['F']) > 0.999).astype(float)
        
        # Build projector
        self.build_projector()

    def build_projector(self):
        """
        Build the projection matrix P
        """
        n_r = self.dim_['n_r']
        n_t = self.dim_['n_t']
        
        # Initialize P matrix as repeated identity matrices
        # This replicates MATLAB's repmat(eye(n_t), 1, n_r)
        self.P_ = np.zeros((n_t, n_t*n_r))
        for i in range(n_r):
            self.P_[:, i*n_t:(i+1)*n_t] = np.eye(n_t)
        
        # Update P based on specializations
        for i in range(n_r):
            Si = np.diag(self.scenario_params_['S'][:, i])
            # In MATLAB, indices start from 1, so (i-1)*n_t+1:i*n_t
            # In Python, indices start from 0, so we use i*n_t:(i+1)*n_t
            self.P_[:, i*n_t:(i+1)*n_t] = self.P_[:, i*n_t:(i+1)*n_t] - Si @ np.linalg.pinv(Si)
        
        # Ensure numerical stability
        self.P_ = np.where(np.abs(self.P_) < 1e-10, 0, self.P_)
        
    def check_tasks(self):
        """
        Check task functions and create numerical derivatives if needed
        """
        for i in range(self.dim_['n_t']):
            # Add numerical gradient if not provided
            if 'gradient' not in self.scenario_params_['tasks'][i] or self.scenario_params_['tasks'][i]['gradient'] is None:
                self.scenario_params_['tasks'][i]['gradient'] = self.get_dh_dx_handle(i)
            
            # Add numerical time derivative if not provided
            if 'time_derivative' not in self.scenario_params_['tasks'][i] or self.scenario_params_['tasks'][i]['time_derivative'] is None:
                self.scenario_params_['tasks'][i]['time_derivative'] = self.get_dh_dt_handle(i)
    
    def get_dh_dx_handle(self, task_idx):
        """
        Create a numerical gradient function for task task_idx
        
        Parameters:
        -----------
        task_idx : int
            Task index
            
        Returns:
        --------
        function
            Numerical gradient function
        """
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
        """
        Create a numerical time derivative function for task task_idx
        
        Parameters:
        -----------
        task_idx : int
            Task index
            
        Returns:
        --------
        function
            Numerical time derivative function
        """
        def dh_dt(x_value, t_value, i):
            return (self.scenario_params_['tasks'][task_idx]['function'](x_value, t_value + 1e-3, i) - 
                    self.scenario_params_['tasks'][task_idx]['function'](x_value, t_value - 1e-3, i)) / (2e-3)
        return dh_dt
    
    @staticmethod
    def onec(dim, col_idx):
        m = np.zeros(dim)
        m[col_idx] = 1
        return m