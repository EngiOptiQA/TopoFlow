from amplify import BinaryPoly, gen_symbols, sum_poly

class TopologyOptimizationProblem:

    def __init__(self, n_elem, n_qubits_per_variable):
        self.n_elem = n_elem
        self.n_qubits_per_variable = n_qubits_per_variable

    def generate_discretizaton(self):
        
        # Create qubits.
        self.q = gen_symbols(BinaryPoly, self.n_elem, self.n_qubits_per_variable+1) # TODO +1 for ancilla qubit?

    def generate_qubo_formulation(self, u, v, xc, volume_fraction_max, neighbor_elements_Q1):
        volume_max = volume_fraction_max * self.n_elem

        # Coefficient for porous resistance.
        coef_pr = 10**2/4.

        # Coefficients for...
        alpha_el = 4*100*5. # energy loss
        #alpha_eqn_c = 1. # equation constraint
        alpha_ge = 10. # gradient energy
        alpha_hev = 300.0 # heaviside function
        #alpha_bc = 1.0 # boundary constraint
        alpha_dc = 600.0 # domain constraint

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
        # Gradient energy.
        for t_id,t_q in enumerate(self.q):
            for tt_id in neighbor_elements_Q1[t_id]:
                phi_i = (sum_poly(t_q[:-1])/self.n_qubits_per_variable*2)-1
                phi_j = (sum_poly(self.q[tt_id][:-1])/self.n_qubits_per_variable*2)-1
                objective_function += alpha_ge/2*(phi_i-phi_j)**2
        # Heaviside function.
        for t_id, t_q in enumerate(self.q):
            objective_function += alpha_hev*(sum_poly(t_q[:-1])/self.n_qubits_per_variable - t_q[-1])**2
        # Domain constraints.
        objective_function += alpha_dc*(sum_poly([t[-1] for t in self.q]) - volume_max)**2

        self.binary_quadratic_model = objective_function