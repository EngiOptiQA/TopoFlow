from amplify import decode_solution, Solver
import numpy as np
 
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

    def optimize(self, annealing_solver, density_initial, volume_fraction_max, hyperparameters, max_opt_steps=10, tol=1e-2):
        
        n_qubits_per_variable = 9

        objective_function_list = []
        volume_fraction_list = []

        resistance_coeff_solid = self.fem.viscosity/self.fem.epsilon

        # Note that density <-> level-set scaled (both from 0 to 1)
        level_set_scaled = density_initial
        self.fem.update_element_density(level_set_scaled)
        _, u, v, _, _, f = self.fem.solve()

        problem = TopologyOptimizationProblem(self.fem.ne, n_qubits_per_variable)
        problem.generate_discretizaton()

        for i_opt in range(max_opt_steps):

            level_set_scaled_old = level_set_scaled
            problem.generate_qubo_formulation(hyperparameters, u, v, volume_fraction_max, resistance_coeff_solid, self.fem.mesh_v.neighbor_elements)
            binary_solution = annealing_solver.solve_qubo_problem(problem)
            # Level-set in [-1,1], level-set scaled in [0,1], characteristic function in {0,1}
            level_set, level_set_scaled, char_func = problem.get_functions_from_binary_solution(binary_solution)
            # TODO Difference between the following evaluations for level-set scaled/characteristic function?
            self.fem.update_element_density(level_set_scaled)
            ###
            E_eva = resistance_coeff_solid*(1-char_func)
            _, _, _, _, _, f_eva = self.fem.solve(E_eva)
            
            ###
            _, u, v, _, _, f = self.fem.solve()

            volume_fraction = sum(level_set_scaled)/self.fem.mesh_p.area
            objective_function_list.append(f_eva)
            volume_fraction_list.append(volume_fraction)
            print(f'Iteration: {i_opt}, Objective Function: {f_eva}, Volume Fraction: {volume_fraction}')

            self.fem.plot_eva(char_func)
            if np.max(np.abs(level_set_scaled_old-level_set))<tol:
                break

        self.objective_function_list = objective_function_list
        self.volume_fraction_list = volume_fraction_list
