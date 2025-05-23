from amplify import BinaryPoly, gen_symbols, sum_poly
import numpy as np

class TopologyOptimizationProblem:

    def __init__(self, n_elem, n_qubits_per_variable, hyperparameters, volume_fraction_max):
        self.n_elem = n_elem
        self.n_qubits_per_variable = n_qubits_per_variable
        self.hyperparameters = hyperparameters
        self.volume_fraction_max = volume_fraction_max

    def generate_discretizaton(self):
        
        # Create qubits.
        # For each element, create a vector of qubits q that contains:
        #    - N qubits for the level-set function
        #    - 1 qubit for the characteristic function
        self.q = gen_symbols(BinaryPoly, self.n_elem, self.n_qubits_per_variable+1)

    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def get_level_set_from_binary_solution(self, q_k):

        level_set_scaled_elem = np.sum(q_k[:-1])/self.n_qubits_per_variable
        level_set_elem = 2. * level_set_scaled_elem - 1.

        return level_set_elem, level_set_scaled_elem

    def get_char_func_from_binary_solution(self, q_k):

        char_func_elem = q_k[-1]

        return char_func_elem

    def get_functions_from_binary_solutions(self, solutions):

        level_set = []
        level_set_scaled = []
        char_func = []
        for q_k in solutions:
            level_set_elem, level_set_scaled_elem = self.get_level_set_from_binary_solution(q_k)
            char_func_elem = self.get_char_func_from_binary_solution(q_k)

            level_set.append(level_set_elem)
            level_set_scaled.append(level_set_scaled_elem)
            char_func.append(char_func_elem)

        return np.array(level_set), np.array(level_set_scaled), np.array(char_func)

    def get_inconsistencies_from_solutions(self, solutions):

        inconsistencies = np.zeros(len(solutions), dtype=int)
        for k, q_k in enumerate(solutions):
            level_set_elem, level_set_scaled_elem = self.get_level_set_from_binary_solution(q_k)
            char_func_elem = self.get_char_func_from_binary_solution(q_k)

            if ((level_set_elem > 0. and char_func_elem == 0) or 
                (level_set_elem < 0. and char_func_elem == 1)):
                inconsistencies[k] =1
        return inconsistencies


    def generate_qubo_formulation(self, u, v, resistance_coeff_solid, neighbor_elements_Q1):
        volume_max = self.volume_fraction_max * self.n_elem

        # Coefficients for...
        lambda_dis  = self.hyperparameters['energy_dissipation'] # energy dissipation
        lambda_reg  = self.hyperparameters['regularization']     # regularization term
        lambda_vol  = self.hyperparameters['volume_constraint']  # volume constraint
        lambda_char = self.hyperparameters['char_func']          # consistency between level-set and characteristic functions

        # Initialize objective function.
        objective_function = BinaryPoly()

        # Add different contributions to objective function.
        # Energy Dissipation.
        for k in range(self.n_elem):
            char_func_elem = self.q[k][-1]
            resistance_coeff = (1-char_func_elem)*resistance_coeff_solid
            objective_function += lambda_dis*(resistance_coeff*(u[k]**2+v[k]**2) )
        # Regularization.
        for k, q_k in enumerate(self.q):
            for i, direction in enumerate(['vertical', 'horizontal']):
                neighbors = neighbor_elements_Q1[k][i]
                if len(neighbors) == 1: # one-sided gradient approximation
                    l = neighbors[0]
                    q_l = self.q[l]
                    level_set_elem = (sum_poly(q_k[:-1])/self.n_qubits_per_variable*2)-1
                    level_set_elem_neighbor = (sum_poly(q_l[:-1])/self.n_qubits_per_variable*2)-1
                    gradient_approx = level_set_elem-level_set_elem_neighbor/2
                    if direction == 'horizontal':
                        gradient_approx_x = gradient_approx
                    elif direction == 'vertical':
                        gradient_approx_y = gradient_approx
                elif len(neighbors) == 2: # central gradient approximation
                    l = neighbors[0]; m = neighbors[1]
                    q_l = self.q[l]; q_m = self.q[m]
                    level_set_elem_neighbor_0 = (sum_poly(q_l[:-1])/self.n_qubits_per_variable*2)-1
                    level_set_elem_neighbor_1 = (sum_poly(q_m[:-1])/self.n_qubits_per_variable*2)-1
                    gradient_approx = level_set_elem_neighbor_0-level_set_elem_neighbor_1
                    if direction == 'horizontal':
                        gradient_approx_x = gradient_approx
                    elif direction == 'vertical':
                        gradient_approx_y = gradient_approx
            objective_function += lambda_reg*(gradient_approx_x**2+gradient_approx_y**2)

        # Volume Constraint.
        volume_fluid = sum_poly([q_k[-1] for q_k in self.q]) # Sum up element-wise characteristic functions.
        objective_function += lambda_vol*(volume_fluid - volume_max)**2
        # Consistency between level-set and characteristic functions.
        for q_k in self.q:
            char_func_elem = q_k[-1]
            level_set_elem_scaled = sum_poly(q_k[:-1])/self.n_qubits_per_variable
            objective_function += lambda_char*(char_func_elem - level_set_elem_scaled)**2

        self.binary_quadratic_model = objective_function