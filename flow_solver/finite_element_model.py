import numpy as np
import numpy.matlib
from scipy.sparse import csr_matrix, linalg, hstack, vstack

class QuadratureRule:
    def __init__(self, order):
        if order == 1:
            s = np.sqrt(1/3)
            self.x = np.array([[-s, -s], [s, -s], [s, s], [-s, s]]) # Coordinates of quadrature points
            self.w = np.array([1., 1., 1., 1.])  # Weights of quadrature points
            self.n = len(self.x)
        elif order == 2:
            s = np.sqrt(3/5)
            self.x = np.array([[-s, -s], [0, -s], [s, -s],
                        [-s,  0], [0,  0], [s,  0],
                        [-s,  s], [0,  s], [s, s]])  # Coordinates of quadrature points
            self.w = np.array([25., 40., 25., 40., 64., 40., 25., 40., 25.]) / 81.0  # Weights of quadrature points
            self.n = len(self.x)
class FEM:
    def __init__(self, mesh_v, mesh_p, viscosity, density_uniform):
        self.mesh_v = mesh_v
        self.mesh_p = mesh_p
        
        self.ne_v  = mesh_v.elements.shape[0] # Number of elements for velocity.
        self.ne_p  = mesh_p.elements.shape[0] # Number of elements for pressure.
        if self.ne_v != self.ne_p:
            raise Exception('Meshes for velocity and pressure have different number of elements.')
        self.ne = self.ne_v

        self.nen_v = mesh_v.elements.shape[1]
        self.nen_p = mesh_p.elements.shape[1]

        self.nn_v = mesh_v.coords.shape[0]
        self.nn_p = mesh_p.coords.shape[0]

        self.viscosity = viscosity
        self.density = density_uniform*np.ones(mesh_v.n_elem)

        self.boundary_values_v = None

    def update_element_density(self, density):
        self.density = density

    def set_boudary_values_v(self, boundary_values_v):
        # [i_node, i_dof, i_value]
        self.boundary_values_v = boundary_values_v

    def assemble(self):

        elements_v = self.mesh_v.elements
        elements_p = self.mesh_p.elements

        ndf_per_elem_v = 2 * self.nen_v * np.ones(self.ne_v, dtype=int) # Two DOFs (velocity components) for each element node per element.
        ndf_per_elem_p = self.nen_p * np.ones(self.ne_p, dtype=int) # One DOF (pressure) for each element node per element.

        i_A       = np.zeros(sum(ndf_per_elem_v**2), dtype=int)
        j_A       = np.zeros(sum(ndf_per_elem_v**2), dtype=int)
        k_A_mu    = np.zeros(sum(ndf_per_elem_v**2)) 
        k_A_alpha = np.zeros(sum(ndf_per_elem_v**2))

        e         = np.zeros(sum(ndf_per_elem_v**2), dtype=int)

        i_B = np.zeros(sum(ndf_per_elem_p)*sum(ndf_per_elem_v), dtype=int)
        j_B = np.zeros(sum(ndf_per_elem_p)*sum(ndf_per_elem_v), dtype=int)
        k_B = np.zeros(sum(ndf_per_elem_p)*sum(ndf_per_elem_v))

        # Quadrature rule.
        quad_rule = QuadratureRule(order=2)

        # Precompute shape functions at quadrature points.
        self.shape_functions_v = self.mesh_v.evaluate_shape_functions(quad_rule.x)
        self.shape_function_derivatives_v = self.mesh_v.evaluate_shape_function_derivatives(quad_rule.x)
        self.shape_functions_p = self.mesh_p.evaluate_shape_functions(quad_rule.x)

        # Assembly.
        index_A = 0; index_B = 0
        for ie in range(self.ne):
            elem_nodes_v = elements_v[ie,:]
            elem_ndf_v   = ndf_per_elem_v[ie]
            elem_nodes_p = elements_p[ie,:]
            elem_ndf_p   = ndf_per_elem_p[ie]

            # Compute (constant) element matrix.
            if ie == 0:
                A_mu_e, A_alpha_e = self.local_K_v(ie, quad_rule)
                B_e = self.local_K_p(ie, quad_rule)

            i_dofs_v = np.concatenate([(2*elem_nodes_v).reshape(self.nen_v,1),
                                       (2*elem_nodes_v+1).reshape(self.nen_v,1)],
                                       axis=1).reshape(elem_ndf_v,1)
            i_dofs_p = elem_nodes_p.reshape(elem_ndf_p,1)

            I_vv = np.matlib.repmat(i_dofs_v, 1, elem_ndf_v)
            J_vv = I_vv.T
            I_vp = np.matlib.repmat(i_dofs_v ,1,elem_ndf_p)
            J_pv = np.matlib.repmat(i_dofs_p ,1,elem_ndf_v)

            i_A[index_A:index_A+elem_ndf_v**2] = I_vv.T.flatten()
            j_A[index_A:index_A+elem_ndf_v**2] = J_vv.T.flatten()

            k_A_mu[index_A:index_A+elem_ndf_v**2]    = A_mu_e.flatten()
            k_A_alpha[index_A:index_A+elem_ndf_v**2] = A_alpha_e.flatten()
            e[index_A:index_A+elem_ndf_v**2] = ie

            i_B[index_B:index_B+elem_ndf_v*elem_ndf_p] = I_vp.T.flatten()
            j_B[index_B:index_B+elem_ndf_v*elem_ndf_p] = J_pv.T.flatten()
            k_B[index_B:index_B+elem_ndf_v*elem_ndf_p] = B_e.T.flatten()

            index_A += elem_ndf_v**2
            index_B += elem_ndf_v*elem_ndf_p

        return i_A, j_A, k_A_mu, k_A_alpha, e, i_B, j_B, k_B

    def solve(self, E=None):

        if self.boundary_values_v is None:
            raise Exception('Boundary values for velocity need to be set before solving.')
        
        # Assemble matrices.
        i_A, j_A, k_A_mu, k_A_alpha, e, i_B, j_B, k_B = self.assemble()

        # Restrict DOFs due to boundary conditions.
        fixed_dofs = np.zeros((1, len(self.boundary_values_v)), dtype=int)
        G = np.zeros((len(self.boundary_values_v), 1))
        for i in range(len(self.boundary_values_v)):
            fixed_dofs[0,i] = 2 * int(self.boundary_values_v[i,0]) + int(self.boundary_values_v[i,1])
            G[i] = self.boundary_values_v[i,2]
        n_dofs = 2 * self.nn_v + self.nn_p + 1 # TODO Why + 1?
        free_dofs = np.setdiff1d(np.arange(n_dofs), fixed_dofs)
        
        # Set inverse permeability.
        if E is None:
            epsilon = 8*(10**-2)
            E = 1./epsilon*(1-self.density)*0.01/(0.01+self.density)

        # Setup system of equations.
        A = csr_matrix((k_A_mu + E[e]*k_A_alpha,(i_A,j_A)))
        B = csr_matrix((k_B,(i_B,j_B)))
        Z = np.zeros((2*self.nn_v,1))
        O = csr_matrix((self.nn_p, self.nn_p), dtype='int')
        K_1 = hstack([A,B,Z])
        K_2 = hstack(([B.T,O,csr_matrix(self.mesh_v.element_areas[0]*np.ones((self.nn_p,1)))]))
        K_3 = hstack([Z.T,csr_matrix(self.mesh_v.element_areas[0]*np.ones(self.nn_p)),0])
        K = vstack([K_1,K_2,K_3])
        K = (K+K.transpose())/2
        S = np.zeros((n_dofs,1)) 
        S[fixed_dofs,:] = G

        # Solve the system of equations.
        S[free_dofs,:] = linalg.spsolve(K[free_dofs,:][:,free_dofs],
                                        -K[free_dofs.flatten(),:][:,fixed_dofs.flatten()]*S[fixed_dofs,:].flatten())[:,np.newaxis]
        
        # Extract velocity and pressure from solution vector.
        U = S[:2*self.nn_v]
        p = S[2*self.nn_v:]

        # Compute energy dissipation.
        F = A*U
        f = 1./2.*np.dot(F.flatten(),U.flatten())

        # Average velocity and pressure on elements.
        u_e = np.zeros(self.ne)
        v_e = np.zeros(self.ne)
        p_e = np.zeros(self.ne)
        u = U[::2]; v = U[1::2]
        for ie in range(self.ne):
            u_e[ie] = np.mean(u[self.mesh_v.elements[ie,:]])
            v_e[ie] = np.mean(v[self.mesh_v.elements[ie,:]])
            p_e[ie] = np.mean(p[self.mesh_p.elements[ie,:]])

        return U, u_e, v_e, p_e, F, f 

    def local_K_p(self, ie, quad_rule):
        nen_v = self.nen_v
        nen_p = self.nen_p

        element_nodes = self.mesh_v.elements[ie,:]

        B_e = np.zeros((2*nen_v, nen_p))

        # Loop over quadrature points.
        for q in range(quad_rule.n):
            G_e = np.zeros((2*nen_v, nen_p))

            dNdxi = self.shape_function_derivatives_v[:,:,q]
            # Compute Jacobian at quadrature point. 
            J = np.dot(self.mesh_v.coords[element_nodes,:].T, dNdxi)
            # Compute absolute value of the Jacobian determinant.
            abs_det_J = abs(np.linalg.det(J))
            abs_det_J = np.linalg.det(J)

            # Compute effective weighting.
            integrator = quad_rule.w[q] * abs_det_J

            gradient_operator_1 = np.zeros((nen_v, nen_p))
            gradient_operator_2 = np.zeros((nen_v, nen_p))

            for i in range(nen_v):
                dNdxi1 = dNdxi[i,0]
                dNdxi2 = dNdxi[i,1]
                for j in range(nen_p):
                    N_p = self.shape_functions_p[j,q]
                    gradient_operator_1[i,j] += N_p * dNdxi1
                    gradient_operator_2[i,j] += N_p * dNdxi2
            G_e[ ::2,:nen_p] -= gradient_operator_1
            G_e[1::2,:nen_p] -= gradient_operator_2
            B_e[:,:] += G_e*integrator
        return B_e

    def local_K_v(self, ie, quad_rule):

        nen = self.nen_v
        element_nodes = self.mesh_v.elements[ie,:]

        C_mu = self.viscosity * np.array([[2,0,0],[0,2,0],[0,0,1]])

        A_mu_e    = np.zeros((2*nen, 2*nen))
        A_alpha_e = np.zeros((2*nen, 2*nen))
        
        # Loop over quadrature points.
        for q in range(quad_rule.n):

            # Compute Jacobian at quadrature point.
            dNdxi = self.shape_function_derivatives_v[:,:,q] 
            J = np.dot(self.mesh_v.coords[element_nodes,:].T, dNdxi)
            # Compute absolute value of the Jacobian determinant.
            abs_det_J = abs(np.linalg.det(J))
            abs_det_J = np.linalg.det(J)

            # Compute effective weighting.
            integrator = quad_rule.w[q] * abs_det_J

            # Compute A_mu_e.
            B = np.zeros((3, 2*nen))
            B[0,  ::2] = dNdxi[:,0] 
            B[1, 1::2] = dNdxi[:,1] 
            B[2,  ::2] = dNdxi[:,1] 
            B[2, 1::2] = dNdxi[:,0]
            A_mu_e = A_mu_e + np.dot(np.dot(B.T, C_mu),B)*integrator

            # Compute A_alpha_e.
            N = self.shape_functions_v[:,q]
            Nu = np.zeros((2, 2*nen))
            Nu[0,  ::2] = N.reshape(nen)
            Nu[1, 1::2] = N.reshape(nen)
            A_alpha_e = A_alpha_e + np.dot(Nu.T,Nu)*integrator

        return A_mu_e, A_alpha_e
    
    def plot_velocity_field(self, velocity_field):
        self.mesh_v.plot_vector_field(velocity_field)

    def plot_velocity_magnitude(self, velocity_field):
        velocity_magnitude = np.zeros(self.ne)
        for e in range(self.ne):
            velocity_magnitude[e] = np.sqrt(velocity_field[0][e]**2+velocity_field[1][e]**2)
        self.mesh_v.plot_element_quantity(velocity_magnitude, min(velocity_magnitude), max(velocity_magnitude))
    
    def plot_pressure(self, pressure):
        self.mesh_p.plot_element_quantity(pressure, min(pressure), max(pressure))

    def plot_density(self):
        self.mesh_v.plot_element_quantity(self.density, min(self.density), max(self.density), cmap='cool')
        
#################        
    def plot_eva(self, density):
        self.mesh_v.plot_element_quantity(density, min(density), max(density), cmap='cool')

    
