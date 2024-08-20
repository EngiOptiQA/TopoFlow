import numpy as np
from scipy.sparse import csr_matrix

from .optimizer import Optimizer

class PolyTop(Optimizer):
    def __init__(self, fem):
        self.fem = fem

    def optimize(self, density_initial, density_min, density_max, volume_fraction_max,
                 design_tolerance, max_iterations, q_values,
                 opt_OCMove, opt_OCEta):

        # Perform Optimization.
        objective_function_list = []
        volume_fraction_list = []

        density = density_initial

        P = csr_matrix(np.identity(self.fem.mesh_v.n_elem))

        for q in q_values:
            print(str(f' q = {q} ').center(40, '#'))
            tolerance = design_tolerance*(density_max-density_min)
            design_change = 2*tolerance
            E, dEdy, V, dVdy = self.mat_int_fnc(P*density, q)
            i_opt = 0
            while i_opt < max_iterations and design_change > tolerance:
                i_opt += 1
                g, dgdE, dgdV = self.constraint_fnc(E, V, volume_fraction_max)
                f, dfdE, dfdV = self.objective_fnc(E, V)
                dfdz = P.T*(np.multiply(dEdy,dfdE)+np.multiply(dVdy,dfdV))
                dgdz = P.T*(np.multiply(dEdy,dgdE)+np.multiply(dVdy,dgdV))
                density, design_change = self.update_scheme(dfdz,g,dgdz,density,density_min,density_max,opt_OCMove,opt_OCEta)
                self.fem.update_element_density(density)
                E, dEdy, V, dVdy = self.mat_int_fnc(P*density, q)
                volume_fraction = density.sum()/self.fem.mesh_v.area
                print(f'Iteration: {i_opt}, Objective Function: {f}, Volume Fraction: {volume_fraction}')
                objective_function_list.append(f)
                volume_fraction_list.append(volume_fraction)

                if i_opt%5==0:
                    self.fem.plot_density()
            self.fem.plot_density()

        self.objective_function_list = objective_function_list
        self.volume_fraction_list = volume_fraction_list

        return density

    def constraint_fnc(self, E, V, volume_fraction_max):
        element_areas = self.fem.mesh_v.element_areas

        g = sum(np.multiply(element_areas,V))/sum(element_areas) - volume_fraction_max
        dgdE = np.zeros(E.shape)
        dgdV = element_areas/sum(element_areas)

        return g, dgdE, dgdV,

    def objective_fnc(self, E, V):
        
        i_A, j_A, _, k_A_alpha, _, _, _, _ = self.fem.assemble()
        U, u_e, v_e, p_e, F, f  = self.fem.solve(E)
        ndf_per_elem_v = 2 * self.fem.nen_v * np.ones(self.fem.ne_v, dtype=int)

        temp = np.cumsum((np.multiply(np.multiply(U[i_A].flatten(), k_A_alpha), U[j_A].flatten())))
        temp = temp[(np.cumsum(ndf_per_elem_v**2))-1]
        dfdE = np.zeros(self.fem.ne)
        dfdE[0] = 1/2*(temp[0])
        dfdE[1:] = 1/2*(temp[1:]-temp[:-1])
        dfdV = np.zeros(V.shape)
        return f, dfdE, dfdV

    def update_scheme(self, dfdz, g, dgdz, z0, zMin, zMax, opt_OCMove, eta):
        move=opt_OCMove*(zMax-zMin)
        l1=0
        l2=1e6
        while l2-l1 > 1e-4:
            lmid = 0.5*(l1+l2)
            B = -(dfdz/dgdz)/lmid
            zCnd = zMin+(np.multiply((z0-zMin),B**eta))
            zNew = np.fmax(np.fmax(np.fmin(np.fmin(zCnd,z0+move),zMax),z0-move),zMin)
            if ((g+dgdz.reshape((1,dgdz.shape[0]))@(zNew-z0))>0):
                l1 = lmid
            else:
                l2 = lmid
        Change = max(abs(zNew-z0))/(zMax-zMin)
        return zNew,Change

    # Material interpolation function.
    def mat_int_fnc(self, y, q):
        E = (self.fem.viscosity/self.fem.epsilon)*q*(1-y)/(y+q)
        dEdy = -(self.fem.viscosity/self.fem.epsilon)*(1+q)*q/((y+q)**2)
        V = y
        dVdy = np.ones(y.shape[0])
        return E, dEdy, V, dVdy

    

