from amplify import decode_solution, Solver
import numpy as np
 
from .optimizer import Optimizer

class AnnealingSolver():

    def __init__(self, client):
        self.client = client

    def solve_qubo_problem(self, problem, track_history=False):
        solver = Solver(self.client)

        if track_history:
            # Let the order in which multiple solutions are stored depend on the output order of the client machine.
            solver.sort_solution = False
            # Do not sort spin arrays and energy values in ascending order of energy values.
            solver.client.parameters.outputs.sort = False
            # Output all spin arrays and energy values.
            solver.client.parameters.outputs.num_outputs = 0

        result = solver.solve(problem.binary_quadratic_model)

        if track_history:
            history = {"sampling_time":[], "energy":[]}
            for t, s in zip(solver.client_result.timing.time_stamps, result.solutions):
                if s.is_feasible:
                    history["sampling_time"].append(t)
                    history["energy"].append(s.energy)
            solution =[]
            for i in range(len(history["energy"])):
                solution.append(decode_solution(problem.q, result[i].values))
            return solution, history
        else:
            solution = decode_solution(problem.q, result[0].values)
            return solution

class Annealing(Optimizer):

    def __init__(self, fem):
        self.fem = fem

    def optimize(self, annealing_solver, problem, level_set_scaled_initial, max_opt_steps=10, tol=1e-2, plot_steps=False):
        
        objective_function_list = []
        volume_fraction_list = []

        resistance_coeff_solid = self.fem.viscosity/self.fem.epsilon

        # Compute initial velocity field.
        level_set_scaled = level_set_scaled_initial
        char_func = np.where(level_set_scaled >= 0.5, 1, 0)
        E = resistance_coeff_solid*(1-char_func)
        _, u, v, _, _, f = self.fem.solve(E)

        for i_opt in range(max_opt_steps):

            level_set_scaled_old = level_set_scaled
            char_func_old = char_func
            f_old = f

            problem.generate_qubo_formulation(u, v, resistance_coeff_solid, self.fem.mesh_v.neighbor_elements)
            binary_solutions = annealing_solver.solve_qubo_problem(problem)
            self.binary_solutions_optimum = binary_solutions

            # Level-set in [-1,1], level-set scaled in [0,1], characteristic function in {0,1}
            level_set, level_set_scaled, char_func = problem.get_functions_from_binary_solutions(binary_solutions)

            # Evaluate velocity field and objective function.
            E = resistance_coeff_solid*(1-char_func)
            _, u, v, _, _, f = self.fem.solve(E)
            # Evaluate volume fraction.
            volume_fraction = sum(char_func)/self.fem.mesh_p.area
            # Detect inconsistencies between level-set and characteristic functons.
            inconsistencies = problem.get_inconsistencies_from_solutions(binary_solutions)
            n_inconsistencies = np.sum(inconsistencies)

            objective_function_list.append(f)
            volume_fraction_list.append(volume_fraction)

            print(f'Iteration: {i_opt}, Objective Function: {f}, Volume Fraction: {volume_fraction}, Inconsistencies: {n_inconsistencies}')

            if plot_steps:
                self.fem.plot_eva(char_func, title='Characteristic Function')
                if problem.hyperparameters['regularization'] > 0:
                    self.fem.plot_eva(level_set, title='Level-Set')
                    if n_inconsistencies > 0:
                        self.fem.plot_eva(inconsistencies, title='Inconsistencies')

            char_func_abs_change = np.sum(np.abs(char_func_old-char_func))
            char_func_rel_change = char_func_abs_change/np.sum(char_func_old)
            objective_function_rel_change = abs(f-f_old)/f_old
            print(f'Abs. change in\n'+
                  f'\tchar. func.:{char_func_abs_change}')
            print(f'Rel. change in\n'+
                  f'\tchar. func.:{char_func_rel_change}\n'+
                  f'\tObj. func.: {objective_function_rel_change}')
            if objective_function_rel_change < tol:
                break

        if not plot_steps:
            self.fem.plot_eva(char_func, title='Characteristic Function')
            if problem.hyperparameters['regularization'] > 0:
                self.fem.plot_eva(level_set, title='Level-Set')
                if n_inconsistencies > 0:
                    self.fem.plot_eva(inconsistencies, title='Inconsistencies')

        self.objective_function_list = objective_function_list
        self.volume_fraction_list = volume_fraction_list

        self.objective_function = self.objective_function_list[-1]
        self.volume_fraction = self.volume_fraction_list[-1]
        self.n_inconsistencies = n_inconsistencies
