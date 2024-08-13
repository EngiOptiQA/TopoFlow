from amplify import BinaryPoly, gen_symbols, sum_poly

class TopologyOptimizationProblem:

    def __init__(self, n_elem, n_qubits_per_variable):
        self.n_elem = n_elem
        self.n_qubits_per_variable = n_qubits_per_variable

    def generate_discretizaton(self):
        
        # Create qubits.
        # For each element, create a vector of qubits q that contains:
        #    - N qubits for the level-set function
        #    - 1 qubit for the characteristic function
        self.q = gen_symbols(BinaryPoly, self.n_elem, self.n_qubits_per_variable+1)

    def generate_qubo_formulation(self, u, v, xc, volume_fraction_max, neighbor_elements_Q1):
        volume_max = volume_fraction_max * self.n_elem

        # Resistance coefficient for porous medium (solid).
        resistance_coeff_solid= 10**2/4.

        # Coefficients for...
        lambda_dis  = 4*100*5.  # energy dissipation
        lambda_reg  = 10.       # regularization term
        lambda_vol  = 600.0     # volume constraint
        lambda_char = 300.0     # consistency between level-set and characteristic functions

        # Initialize objective function.
        objective_function = BinaryPoly()

        # Add different contributions to objective function.
        # Energy Dissipation.
        for k in range(self.n_elem):
            char_func_elem = self.q[k][-1]
            resistance_coeff = (1-char_func_elem)*resistance_coeff_solid
            t_uvec = u[k]/xc*2
            t_vvec = v[k]/xc*2
            objective_function += lambda_dis*(resistance_coeff*(t_uvec*t_uvec+t_vvec*t_vvec) )
        # Regularization.
        for t_id,t_q in enumerate(self.q):
            for tt_id in neighbor_elements_Q1[t_id]:
                for t_coord in range(2): # 2 if it's two-dimensional; in general, n if it's n-dimensional
                    if len(tt_id) == 1:
                        phi_i = (sum_poly(t_q[:-1])/self.n_qubits_per_variable*2)-1
                        phi_j = (sum_poly(self.q[tt_id[0]][:-1])/self.n_qubits_per_variable*2)-1
                        t_obj = alpha_ge/2*((phi_i-phi_j)/2)**2
                    elif len(tt_id) == 2:
                        phi_i = (sum_poly(self.q[tt_id[0]][:-1])/self.n_qubits_per_variable*2)-1
                        phi_j = (sum_poly(self.q[tt_id[1]][:-1])/self.n_qubits_per_variable*2)-1
                        t_obj = alpha_ge/2*(phi_i-phi_j)**2
                    objective_function += t_obj
        # Volume Constraint.
        volume_fluid = sum_poly([q_k[-1] for q_k in self.q]) # Sum up element-wise characteristic functions.
        objective_function += lambda_vol*(volume_fluid - volume_max)**2
        # Consistency between level-set and characteristic functions.
        for q_k in self.q:
            char_func_elem = q_k[-1]
            level_set_elem_scaled = sum_poly(q_k[:-1])/self.n_qubits_per_variable
            objective_function += lambda_char*(char_func_elem - level_set_elem_scaled)**2

        self.binary_quadratic_model = objective_function

class TopologyOptimizationParameterTuning:

    def __init__(self, n_elem, n_qubits_per_variable):
        self.n_elem = n_elem
        self.n_qubits_per_variable = n_qubits_per_variable

    def generate_discretizaton(self):
        
        # Create qubits.
        self.q = gen_symbols(BinaryPoly, self.n_elem, self.n_qubits_per_variable+1) # TODO +1 for ancilla qubit?

    def generate_qubo_formulation(self, u, v, el, ge, hev, dc volume_fraction_max, neighbor_elements_Q1):
        volume_max = volume_fraction_max * self.n_elem

        # Coefficient for porous resistance.
        coef_pr = 10**2/4.

        # Coefficients for...
        alpha_el = el*100.*4. # energy loss     🐳
        alpha_eqn_c = 1. # equation constraint
        alpha_ge = ge # gradient energy
        alpha_hev = hev # heaviside function
        #alpha_bc = 1.0 # boundary constraint
        alpha_dc = dc # domain constraint

        # Create qubits.
        # q = gen_symbols(BinaryPoly, n_elem, n_qubits_per_variable+1)

        # Initialize objective function.
        objective_function = BinaryPoly()

        # Add different contributions to objective function.
        # Energy loss.
        for t_id in range(self.n_elem):
            coef_pr_e = (1-self.q[t_id][-1])*coef_pr
            t_uvec = u[t_id]/xc*2
            t_vvec = v[t_id]/xc*2
            objective_function += alpha_el*(coef_pr_e*(t_uvec*t_uvec+t_vvec*t_vvec) )
        # Gradient energy.  🐳
        for t_id,t_q in enumerate(self.q):
            for tt_id in neighbor_elements_Q1[t_id]:
                for t_coord in range(2): # 2 if it's two-dimensional; in general, n if it's n-dimensional
                    if len(tt_id) == 1:
                        phi_i = (sum_poly(t_q[:-1])/self.n_qubits_per_variable*2)-1
                        phi_j = (sum_poly(self.q[tt_id[0]][:-1])/self.n_qubits_per_variable*2)-1
                        t_obj = alpha_ge/2*((phi_i-phi_j)/2)**2
                    elif len(tt_id) == 2:
                        phi_i = (sum_poly(self.q[tt_id[0]][:-1])/self.n_qubits_per_variable*2)-1
                        phi_j = (sum_poly(self.q[tt_id[1]][:-1])/self.n_qubits_per_variable*2)-1
                        t_obj = alpha_ge/2*(phi_i-phi_j)**2
                    objective_function += t_obj
        # Heaviside function.
        for t_id, t_q in enumerate(self.q):
            objective_function += alpha_hev*(sum_poly(t_q[:-1])/self.n_qubits_per_variable - t_q[-1])**2
        # Domain constraints.
        objective_function += alpha_dc*(sum_poly([t[-1] for t in self.q]) - volume_max)**2

        self.binary_quadratic_model = objective_function
