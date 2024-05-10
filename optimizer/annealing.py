from amplify import decode_solution, Solver
import numpy as np

from flow_solver.Q2Q1FEM import Plot_patch 
from problems.topo_opt_level_set_fluid_flow import TopologyOptimizationProblem
from .optimizer import Optimizer

class AnnealingSolver():

    def __init__(self, client):
        self.client = client

    def solve_qubo_problem(self, problem):
        solver = Solver(self.client) 
        result = solver.solve(problem.binary_quadratic_model)
        solution = decode_solution(problem.q, result[0].values)
        return solution

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
            if i_opt==0:
                xc = 2
            else:
                xc = max(np.sqrt(u**2+v**2))
            density_old = density
            problem.generate_qubo_formulation(u, v, xc, volume_fraction_max, self.fem.mesh_v.neighbor_elements)
            solution = annealing_solver.solve_qubo_problem(problem)

            sol = []
            for t in solution:
                pred_t_d = np.sum(t[:-1])/n_qubits_per_variable
                sol.append(pred_t_d)

            density = np.array(sol)           
            self.fem.update_element_density(density)
            _, u, v, _, _, f = self.fem.solve()

            volume_fraction = sum(sol)/self.fem.mesh_p.area
            objective_function_list.append(f)
            volume_fraction_list.append(volume_fraction)
            print(f'Iteration: {i_opt}, Objective Function: {f}, Volume Fraction: {volume_fraction}')
            self.fem.plot_density()
            if np.max(np.abs(density_old-density))<0.01:
                break

        self.objective_function_list = objective_function_list
        self.volume_fraction_list = volume_fraction_list