import numpy as np
import gurobipy as gp
from gurobipy import GRB
import cvxpy as cp
import time

class RTA_task_analyzer:
    def __init__(self, rta_instance):
        self.rta = rta_instance

    def _diagnose_infeasible_constraints(self):
        """Diagnose constraints that lead to infeasibility"""
        print(f"[DEBUG]Diagnosing infeasible constraints...")
        
        #Check the basic information of the constraint matrix
        A_ineq = self.rta.constraints_['A_ineq']
        b_ineq = self.rta.constraints_['b_ineq']
        A_eq = self.rta.constraints_['A_eq']
        b_eq = self.rta.constraints_['b_eq']
        lb = self.rta.constraints_['lb']
        ub = self.rta.constraints_['ub']
        
        print(f"Inequality constraint matrix: {A_ineq.shape}")
        print(f"Equality constraint matrix: {A_eq.shape}")
        print(f"variable bounds: lb={lb.shape}, ub={ub.shape}")
        
        #Check for obviously conflicting constraints
        n_r = self.rta.dim_['n_r']
        n_t = self.rta.dim_['n_t']
        n_c = self.rta.dim_['n_c']
        
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
            min_robots = self.rta.opt_params_['n_r_bounds'][j, 0]
            max_robots = self.rta.opt_params_['n_r_bounds'][j, 1]
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
        F = self.rta.scenario_params_['F']
        T = self.rta.scenario_params_['T']
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
                    elif len(capable_robots) < self.rta.opt_params_['n_r_bounds'][j, 0]:
                        print(f"      [ERROR]Have the ability{c}number of robots({len(capable_robots)}) <Task{j}minimum requirements({self.rta.opt_params_['n_r_bounds'][j, 0]})")
        
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
        
        n_r = self.rta.dim_['n_r']
        n_t = self.rta.dim_['n_t']
        n_c = self.rta.dim_['n_c']
        n_u = self.rta.dim_['n_u']
        
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
            'upper_bound': f"δ_max = {self.rta.opt_params_['delta_max']}",
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
        
        F_matrix = self.rta.scenario_params_['F']
        T_matrix = self.rta.scenario_params_['T']
        
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
            min_robots = self.rta.opt_params_['n_r_bounds'][j, 0]
            max_robots = self.rta.opt_params_['n_r_bounds'][j, 1] 
            
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
        
        P_matrix = self.rta.P_
        S_matrix = self.rta.scenario_params_['S']
        l_param = self.rta.opt_params_['l']
        
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
        
        A_ineq = self.rta.constraints_['A_ineq']
        b_ineq = self.rta.constraints_['b_ineq']
        A_eq = self.rta.constraints_['A_eq']
        b_eq = self.rta.constraints_['b_eq']
        
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
        total_min_demand = np.sum([self.rta.opt_params_['n_r_bounds'][j, 0] for j in range(n_t)])
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
        
        n_r = self.rta.dim_['n_r'] 
        n_t = self.rta.dim_['n_t']
        n_c = self.rta.dim_['n_c']
        
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
        self.rta.build_constraints(x, t)
        
        #2. Set variables
        alpha_dim = self.rta.dim_['n_r'] * self.rta.dim_['n_t']
        u_dim = self.rta.dim_['n_r'] * self.rta.dim_['n_u']
        delta_dim = self.rta.dim_['n_r'] * self.rta.dim_['n_t']
        
        alpha_var = cp.Variable(alpha_dim, boolean=True)
        u_var = cp.Variable(u_dim)
        delta_var = cp.Variable(delta_dim)
        
        #3. Analyze constraints in detail
        constraints_info = self.analyze_constraints_detailed(x, t, alpha_var, u_var, delta_var)
        
        #4. Show the capability constraint calculation process in detail
        F_matrix = self.rta.scenario_params_['F']
        T_matrix = self.rta.scenario_params_['T']
        self._display_capability_constraint_calculations(F_matrix, T_matrix, alpha_var)
        
        #5. Construct objective function
        P_squared = self.rta.P_.T @ self.rta.P_
        S_diag = np.diag(np.reshape(self.rta.scenario_params_['S'], (-1)))
        
        alpha_cost = 1e6 * max(1, self.rta.opt_params_['l']) * cp.quad_form(alpha_var, P_squared)
        u_cost = cp.quad_form(u_var, np.eye(u_dim))
        delta_cost = self.rta.opt_params_['l'] * cp.quad_form(delta_var, S_diag)
        objective = cp.Minimize(alpha_cost + u_cost + delta_cost)
        
        #6. Add constraints
        constraints = []
        all_vars_h = cp.hstack([alpha_var, u_var, delta_var])
        all_vars = all_vars_h.T
        constraints.append(self.rta.constraints_['A_ineq'] @ all_vars <= self.rta.constraints_['b_ineq'])
        constraints.append(self.rta.constraints_['A_eq'] @ all_vars == self.rta.constraints_['b_eq'])  #Equality constraints
        constraints.append(alpha_var >= self.rta.constraints_['lb'][:alpha_dim])
        constraints.append(alpha_var <= self.rta.constraints_['ub'][:alpha_dim])
        
        lb_idx = alpha_dim + u_dim
        constraints.append(delta_var >= self.rta.constraints_['lb'][lb_idx:lb_idx+delta_dim])
        constraints.append(delta_var <= self.rta.constraints_['ub'][lb_idx:lb_idx+delta_dim])
        
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
            alpha_cost_val = 1e6 * max(1, self.rta.opt_params_['l']) * np.dot(alpha, P_squared @ alpha)
            u_cost_val = np.dot(u, u)
            delta_cost_val = self.rta.opt_params_['l'] * np.dot(delta, S_diag @ delta)
            
            print(f"Task allocation cost: {alpha_cost_val:.6f}")
            print(f"   controlinput cost: {u_cost_val:.6f}")
            print(f"slack variable cost: {delta_cost_val:.6f}")
            
            #Analyze task assignment results
            print(f"\n📊Task assignment results:")
            alpha_matrix = alpha.reshape(self.rta.dim_['n_r'], self.rta.dim_['n_t'])
            for i in range(self.rta.dim_['n_r']):
                assigned_tasks = [j for j in range(self.rta.dim_['n_t']) if alpha_matrix[i, j] > 0.5]
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
            n_r = self.rta.dim_['n_r']
            n_t = self.rta.dim_['n_t']
            n_c = self.rta.dim_['n_c']
            n_u = self.rta.dim_['n_u']
            
            alpha_dim = n_r * n_t
            u_dim = n_r * n_u
            delta_dim = n_r * n_t
            
            print(f"📊Model size: {n_r}robot, {n_t}Task, {n_c}ability, {n_u}controlDimensions")
            
            #3. Add variables
            alpha_vars = model.addVars(alpha_dim, vtype=GRB.BINARY, name="alpha")
            u_vars = model.addVars(u_dim, lb=-GRB.INFINITY, name="u")
            delta_vars = model.addVars(delta_dim, lb=0, ub=self.rta.opt_params_['delta_max'], name="delta")
            
            # 4. Build constraints (Reuse an already constructed constraint matrix)
            self.rta.build_constraints(x, t)
            A_ineq = self.rta.constraints_['A_ineq']
            b_ineq = self.rta.constraints_['b_ineq']
            A_eq = self.rta.constraints_['A_eq']
            b_eq = self.rta.constraints_['b_eq']
            
            print(f"📋Constraint scale: {A_ineq.shape[0]}inequalities, {A_eq.shape[0]}equation")
            
            #5. Add inequality constraints and label them
            ineq_constraints = {}
            ineq_constraint_names = {}
            
            #5.1 Capability constraints
            constraint_idx = 0
            for j in range(n_t):
                for c in range(n_c):
                    if self.rta.scenario_params_['T'][j, c] > 0.01:  #Only add those who need it
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
                    constraint = model.addConstr(expr <= self.rta.opt_params_['n_r_bounds'][j, 1], name=constr_name)
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
                    constraint = model.addConstr(expr >= self.rta.opt_params_['n_r_bounds'][j, 0], name=constr_name)
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
                print(f"demand: {self.rta.scenario_params_['T'][task_id, cap_id]:.3f}")
                
                #Analyze which robots have this ability
                F_matrix = self.rta.scenario_params_['F']
                capable_robots = [r for r in range(self.rta.dim_['n_r']) if F_matrix[cap_id, r] > 0.5]
                print(f"Robots with this ability: {capable_robots}")
                
                if len(capable_robots) == 0:
                    print(f"   ❌root cause:No robot has{cap_name}ability!")
                    print(f"   🔧Repair suggestions:Add for at least one bot{cap_name}ability, or remove tasks that require this ability")
                elif len(capable_robots) < self.rta.scenario_params_['T'][task_id, cap_id]:
                    print(f"   ❌root cause:Number of capable robots({len(capable_robots)}) <need({self.rta.scenario_params_['T'][task_id, cap_id]:.1f})")
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
                
                min_robots = self.rta.opt_params_['n_r_bounds'][task_id, 0]
                max_robots = self.rta.opt_params_['n_r_bounds'][task_id, 1]
                
                if "Max" in constraint_name:
                    print(f"constraint:most{max_robots}robot")
                    if max_robots > self.rta.dim_['n_r']:
                        print(f"   ❌root cause:maximum demand({max_robots}) >Total number of robots({self.rta.dim_['n_r']})")
                        print(f"   🔧Repair suggestions:Reduce the maximum robot requirement to{self.rta.dim_['n_r']}the following")
                else:
                    print(f"constraint:At least{min_robots}robot")
                    if min_robots > self.rta.dim_['n_r']:
                        print(f"   ❌root cause:minimum requirements({min_robots}) >Total number of robots({self.rta.dim_['n_r']})")
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
