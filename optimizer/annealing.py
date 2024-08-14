from amplify import decode_solution, Solver
import numpy as np
 
from problems.topo_opt_level_set_fluid_flow import TopologyOptimizationProblem, TopologyOptimizationParameterTuning
from .optimizer import Optimizer

class AnnealingSolver():

    def __init__(self, client):
        self.client = client

    def solve_qubo_problem(self, problem):
        solver = Solver(self.client) 
        result = solver.solve(problem.binary_quadratic_model)
        solution = decode_solution(problem.q, result[0].values)
        return solution
     
     def solve_timeout_analysis(self, problem):
        solver = Solver(self.client) 
        solver.sort_solution = False 
        solver.client.parameters.outputs.sort = False 
        solver.client.parameters.outputs.num_outputs = 0 
        d = {"sampling_time":[],"energy":[]} 
        result = solver.solve(problem.binary_quadratic_model)
        for t, s in zip(solver.client_result.timing.time_stamps, result.solutions):
            if s.is_feasible:
                d["sampling_time"].append(t)
                d["energy"].append(s.energy)
        result = solver.solve(problem.binary_quadratic_model)
        solution =[]
        for i in range(len(d["energy"])):
            solution.append( decode_solution(problem.q, result[i].values))
        return solution, d
 

class Annealing(Optimizer):

    def __init__(self, fem):
        self.fem = fem

    def optimize(self, annealing_solver,
                 density_initial, density_min, density_max, volume_fraction_max):
        
        max_iterations_annealing = 3

        n_qubits_per_variable = 9

        objective_function_list = []
        volume_fraction_list = []

        density = density_initial
       
        self.fem.update_element_density(density)
        _, u, v, _, _, f = self.fem.solve()

        problem = TopologyOptimizationProblem(self.fem.ne, n_qubits_per_variable)
        problem.generate_discretizaton()

        for i_opt in range(max_iterations_annealing):
            # if i_opt==0:
            #     xc = 2
            # else:
            #     xc = max(np.sqrt(u**2+v**2))
            xc = 2
            density_old = density
            problem.generate_qubo_formulation(u, v, xc, volume_fraction_max, self.fem.mesh_v.neighbor_elements)
            solution = annealing_solver.solve_qubo_problem(problem)

            sol = []
            heaviside = []
            for t in solution:
                pred_t_d = np.sum(t[:-1])/n_qubits_per_variable
                sol.append(pred_t_d)
                pred_t_h = t[-1]
                heaviside.append(pred_t_h)

            density = np.array(sol)     #continuous
            check = np.array(heaviside)  # takes 0 or 1
            self.fem.update_element_density(density)
            ###
            epsilon = 8*(10**-2)
            E_eva = 1./epsilon*(1-check)*1/(1+check)
            _, _, _, _, _, f_eva = self.fem.solve(E_eva)
            
            ###
            _, u, v, _, _, f = self.fem.solve()

            volume_fraction = sum(sol)/self.fem.mesh_p.area
            objective_function_list.append(f_eva)
            volume_fraction_list.append(volume_fraction)
            print(f'Iteration: {i_opt}, Objective Function: {f_eva}, Volume Fraction: {volume_fraction}')
            # self.fem.plot_density()
            self.fem.plot_eva(check)
            if np.max(np.abs(density_old-density))<0.01:
                break

        self.objective_function_list = objective_function_list
        self.volume_fraction_list = volume_fraction_list

class AnnealingParameterTuning(Optimizer):

    def __init__(self, fem):
        self.fem = fem

    def optimize(self, annealing_solver,
                 density_initial, density_min, density_max, volume_fraction_max, el, ge, hev, dc_list, iterations):
        
        max_iterations_annealing = iterations

        n_qubits_per_variable = 9

        objective_function_list = []
        volume_fraction_list = []

        density = density_initial
       
        self.fem.update_element_density(density)
        _, u, v, _, _, f = self.fem.solve()

        problem = TopologyOptimizationParameterTuning(self.fem.ne, n_qubits_per_variable)
        problem.generate_discretizaton()

        for i_opt in range(max_iterations_annealing):
            dc = dc_list[i_opt]
            density_old = density
            problem.generate_qubo_formulation(u, v, el, ge, hev, dc, volume_fraction_max, self.fem.mesh_v.neighbor_elements)
            solution = annealing_solver.solve_qubo_problem(problem)

            sol = []
            heaviside = []
            for t in solution:
                pred_t_d = np.sum(t[:-1])/n_qubits_per_variable
                sol.append(pred_t_d)
                pred_t_h = t[-1]
                heaviside.append(pred_t_h)

            density = np.array(sol)     #continuous
            check = np.array(heaviside)  # takes 0 or 1
            self.fem.update_element_density(check) 
            
            _, u, v, _, _, f_eva = self.fem.solve()  

            volume_fraction = sum(sol)/self.fem.mesh_p.area
            objective_function_list.append(f_eva)
            volume_fraction_list.append(volume_fraction)
            print(f'Iteration: {i_opt}, Objective Function: {f_eva}, Volume Fraction: {volume_fraction}')
            print(max(u),max(v))
            # self.fem.plot_density()
            self.fem.plot_eva(check)
            self.fem.plot_eva(np.sqrt(u**2+v**2))
            if np.max(np.abs(density_old-density))<0.01:
                break

        self.objective_function_list = objective_function_list
        self.volume_fraction_list = volume_fraction_list
