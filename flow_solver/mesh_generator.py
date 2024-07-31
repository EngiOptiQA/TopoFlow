from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import numpy as np

class Mesh(ABC):
    def __init__(self):
        self.n_elem = len(self.elements)
        self.element_areas = self.compute_element_areas()

    @abstractmethod
    def create_mesh_Q1(self):
        pass

    @abstractmethod
    def create_mesh_Q2(self):
        pass

    @abstractmethod
    def create_nodes(self):
        pass

    @abstractmethod
    def create_neighbor_elements(self):
        pass

    def compute_element_areas(self):
        element_areas = np.zeros(self.n_elem)
        if self.elem_type in ['Q1','Q2']:
            for i_elem in range(self.n_elem):
                x = self.coords[self.elements[i_elem,:],0][:4]
                y = self.coords[self.elements[i_elem,:],1][:4]
                [x_s, y_s] = [x[[1,2,3,0]], y[[1,2,3,0]]]
                element_areas[i_elem] = 0.5*sum(x*y_s-y*x_s)
        else:
            raise Exception(f"Computing element areas not implemented for element type '{self.elem_type}'")
        return element_areas

    def evaluate_shape_functions(self, eval_points):
        N_evaluated = None
        if self.elem_type == 'Q1':
            N_evaluated = np.zeros((4, len(eval_points)))
            for i in range(len(eval_points)):
                [xi, eta] = eval_points[i,:]
                # Calculate shape function values for nodes 1 to 4
                N = [(1 - xi) * (1 - eta) * 0.25,  # Element node 1
                    (1 + xi) * (1 - eta) * 0.25,  # Element node 2
                    (1 + xi) * (1 + eta) * 0.25,  # Element node 3
                    (1 - xi) * (1 + eta) * 0.25]  # Element node 4
                N_evaluated[:,i] = N

        elif self.elem_type == 'Q2':
            N_evaluated = np.zeros((9, len(eval_points)))
            for i in range(len(eval_points)):
                [xi, eta] = eval_points[i,:]
                # Calculate shape function values for nodes 1 to 9
                N = [ (1 - xi) * (1 - eta) * xi * eta * 0.25,  # Element node 1
                     -(1 + xi) * (1 - eta) * xi * eta * 0.25,  # Element node 2
                      (1 + xi) * (1 + eta) * xi * eta * 0.25,  # Element node 3
                     -(1 - xi) * (1 + eta) * xi * eta * 0.25,  # Element node 4
                     -(1-xi**2) * (1-eta) * eta * 0.5,         # Element node 5
                      (1+xi)    * (1-eta**2) * xi * 0.5,       # Element node 6
                      (1-xi**2) * (1+eta) * eta * 0.5,         # Element node 7
                     -(1-xi)    * (1-eta**2) * xi * 0.5,       # Element node 8
                      (1-xi**2) * (1-eta**2)]                  # Element node 9
                N_evaluated[:,i] = N
        else:
            raise Exception(f'Shape functions not implemented for element type {self.elem_type}.')
        return N_evaluated  

    def evaluate_shape_function_derivatives(self, eval_points):
        dN_evaluated = None
        if self.elem_type == 'Q1':
            dN_evaluated = np.zeros((4, 2, len(eval_points)))
            for i in range(len(eval_points)):
                [xi, eta] = eval_points[i,:]
                # Derivatives of shape function with respect to the reference coordinates
                dN = [[-1 * (1 - eta) * 0.25, -1 * (1 - xi) * 0.25],  # Element node 1
                      [(1 - eta) * 0.25, -1 * (1 + xi) * 0.25],       # Element node 2
                      [(1 + eta) * 0.25, (1 + xi) * 0.25],            # Element node 3
                      [-1 * (1 + eta) * 0.25, (1 - xi) * 0.25]]       # Element node 4
                dN_evaluated[:,:,i] = np.asarray(dN)

        elif self.elem_type == 'Q2':
            dN_evaluated = np.zeros((9, 2, len(eval_points)))
            for i in range(len(eval_points)):
                [xi, eta] = eval_points[i,:]
                # Derivatives of shape function with respect to the reference coordinates
                dN = [[(xi-1/2)*eta*(eta-1)/2, xi*(xi-1)*(eta-1/2)/2],  # Element node 1
                      [(xi+1/2)*eta*(eta-1)/2, xi*(xi+1)*(eta-1/2)/2],  # Element node 2
                      [(xi+1/2)*eta*(eta+1)/2, xi*(xi+1)*(eta+1/2)/2],  # Element node 3
                      [(xi-1/2)*eta*(eta+1)/2, xi*(xi-1)*(eta+1/2)/2],  # Element node 4
                      [-xi*eta*(eta-1)       , (1-xi**2)*(eta-1/2)],    # Element node 5
                      [(xi+1/2)*(1-eta**2)   , xi*(xi+1)*(-eta)],       # Element node 6
                      [-xi*eta*(eta+1)       , (1-xi**2)*(eta+1/2)],    # Element node 7
                      [(xi-1/2)*(1-eta**2)   , xi*(xi-1)*(-eta)],       # Element node 8
                      [-2*xi*(1-eta**2)      , (1-xi**2)*(-2*eta)]]     # Element node 9
                dN_evaluated[:,:,i] = np.asarray(dN)
        else:
            raise Exception(f'Shape function derivatives not implemented for element type {self.elem_type}.')
        return dN_evaluated             
                 
    def draw_elements(self, annotate=False):
        coords = self.coords
        elements = self.elements
        for i_elem, e in enumerate(elements):
            plt.plot([coords[e[0],0],coords[e[1],0]],[coords[e[0],1],coords[e[1],1]],color='gray')
            plt.plot([coords[e[1],0],coords[e[2],0]],[coords[e[1],1],coords[e[2],1]],color='gray')
            plt.plot([coords[e[2],0],coords[e[3],0]],[coords[e[2],1],coords[e[3],1]],color='gray')
            plt.plot([coords[e[3],0],coords[e[0],0]],[coords[e[3],1],coords[e[0],1]],color='gray')
            
            if annotate:
                # Annotate element index.
                dx = abs(coords[e[0],0] - coords[e[1],0])
                x_mid = (coords[e[0],0] + coords[e[1],0])/2.
                dy = abs(coords[e[2],1] - coords[e[1],1])
                y_mid = (coords[e[2],1] + coords[e[1],1])/2.
                if self.elem_type == 'Q2':
                    offset_x =0.
                    offset_y = -0.25*dy
                else: 
                    offset_x = 0.
                    offset_y = 0.
                plt.annotate(f'{i_elem}', (x_mid+offset_x, y_mid+offset_y), bbox = dict(boxstyle="round", fc="0.8"))

    def plot(self):
        if self.elem_type in ['Q1','Q2']:
            coords = self.coords
            elements = self.elements
            boundary_nodes = self.boundary_nodes
            fig = plt.figure(figsize=(8, 8))
            self.draw_elements(annotate=True)

            for i in range(len(coords)):
                plt.scatter(coords[i,0],coords[i,1], color='gray')
                plt.annotate(f'{i}', (coords[i,0],coords[i,1]))

            boundary_colors = plt.cm.tab20b(np.linspace(0, 1, len(boundary_nodes.keys())))
            for i_boundary, (boundary, nodes) in enumerate(boundary_nodes.items()):
                x = [coords[node, 0] for node in nodes]
                y = [coords[node, 1] for node in nodes]
                plt.scatter(x, y, color=boundary_colors[i_boundary], label=boundary)
            plt.legend()
            plt.show()
        else:
            raise Exception(f"Plotting the mesh not implemented for element type '{self.elem_type}'")

    def plot_element_quantity(self, quantity, quantity_min, quantity_max, cmap='jet'):
        ax = plt.axes()
        cm = plt.get_cmap(cmap)
        norm = colors.Normalize(vmin=quantity_min, vmax=quantity_max)
        scalar_map = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        scalar_map.set_array([])
        for i in range(self.n_elem):
            c_ind = (quantity[i]-quantity_min)/(quantity_max-quantity_min)
            r = patches.Rectangle(xy=self.coords[self.elements[i,0]], 
                                width=self.elem_width, height=self.elem_height,
                                fc=colors.rgb2hex(cm(c_ind)),
                                ec='w')
            ax.add_patch(r)
            plt.axis('scaled')
            ax.set_aspect('equal')

        cbar = plt.colorbar(scalar_map)
        #cbar.set_label('Quantity')
        plt.show()

    def plot_vector_field(self, vector_field):
        centr = np.zeros((2,self.n_elem))
        for i in range(self.n_elem):
            centr[0,i]=np.mean(self.coords[self.elements[i,:], 0])
            centr[1,i]=np.mean(self.coords[self.elements[i,:], 1])
        plt.figure(figsize=(6,6))
        self.draw_elements()

        for i in range(self.n_elem):
            plt.quiver(centr[0,i], centr[1,i], vector_field[0][i], vector_field[1][i],angles='xy', scale_units='xy', scale=7, width=0.005,color='dodgerblue')
        plt.show()

class MeshLChannel(Mesh):
    def __init__(self, elem_type, n_elem_for_width):
        self.elem_type = elem_type
        self.n_elem_for_width = n_elem_for_width
        self.elem_width = self.elem_height = 1./n_elem_for_width
        self.area = 3*n_elem_for_width**2
        if elem_type == 'Q1':
            self.coords, self.elements, self.boundary_nodes, self.neighbor_elements = self.create_mesh_Q1(n_elem_for_width)
        elif elem_type == 'Q2':
            self.coords, self.elements, self.boundary_nodes, self.neighbor_elements = self.create_mesh_Q2(n_elem_for_width)
        else:
            raise Exception(f"Unknown element type for L-channel mesh.")
        super().__init__()

    def create_nodes(self, n_nodes_for_width):

        dx = 1./(n_nodes_for_width-1)
        dy = dx
        # Nodes for each of the three blocks minus interface nodes.
        n_nodes = 3*(n_nodes_for_width*n_nodes_for_width) - 2*n_nodes_for_width
        coords = np.zeros((n_nodes,2))
        x = 0

        boundary_nodes = {'inlet': [], 'outlet': [], 'wall': []}
        # Create nodes for lower left (LL) block without interface nodes.
        for i in range(n_nodes_for_width-1):
            y = 0
            for j in range(n_nodes_for_width):
                i_node = i*n_nodes_for_width+j
                coords[i_node,:] = np.array([x,y])
                if i == 0: # LL left = outlet
                    boundary_nodes['outlet'].append(i_node)
                elif ((j == 0) or # LL bottom = wall
                    (j == n_nodes_for_width-1)): # LL top = wall
                    boundary_nodes['wall'].append(i_node)
                y = y+dy
            x = x+dx
        # Store max. index of nodes already created as offset.
        i_offset =  (n_nodes_for_width-1)*n_nodes_for_width    
        # Create node for lower and upper right (LR and UR) blocks.
        for i in range(n_nodes_for_width):
            y = 0
            for j in range(2*n_nodes_for_width-1):
                i_node = i_offset + i*(2*n_nodes_for_width-1)+j
                coords[i_node,:] = np.array([x,y])
                if ((i == 0 and j >= n_nodes_for_width-1) or # UR left = wall
                    (i == n_nodes_for_width-1) or # right = wall
                    (j == 0)): # bottom = wall
                    boundary_nodes['wall'].append(i_node)
                elif (j == 2*n_nodes_for_width-2): # top = inlet
                    boundary_nodes['inlet'].append(i_node)
                y = y+dy
            x = x+dx

        return coords, boundary_nodes

    def create_neighbor_elements(self, n_elem_for_width):
        neighbor_elements = []
        for i in range(n_elem_for_width*n_elem_for_width):
            t_adj = []
            if i%n_elem_for_width!=0:
                t_u = i-1
                t_adj.append(t_u)
            if i>=n_elem_for_width:
                t_l = i-n_elem_for_width
                t_adj.append(t_l)
            if i%n_elem_for_width!=n_elem_for_width-1:
                t_o = i+1
                t_adj.append(t_o)
            if (n_elem_for_width*n_elem_for_width)-i>n_elem_for_width:
                t_r = i+n_elem_for_width
                t_adj.append(t_r)
            neighbor_elements.append(t_adj)
        for i in range(n_elem_for_width*n_elem_for_width*2):
            j = i+n_elem_for_width**2
            t_adj = []
            if i%(2*n_elem_for_width)!=0:
                t_u = j-1
                t_adj.append(t_u)
            if i<n_elem_for_width:
                t_l = j-n_elem_for_width
                t_adj.append(t_l)
            elif i>=2*n_elem_for_width:
                t_l = j-2*n_elem_for_width
                t_adj.append(t_l)
            if i%(2*n_elem_for_width)!=2*n_elem_for_width-1:
                t_o = j+1
                t_adj.append(t_o)
            if i<(n_elem_for_width-1)*n_elem_for_width*2:
                t_r = j+2*n_elem_for_width
                t_adj.append(t_r)
            neighbor_elements.append(t_adj)
        return neighbor_elements

    def create_mesh_Q1(self, n_elem_for_width):
        '''  
                  o----o----o
                  |         |
                  o    UR   o
                  |         |
        o----o----o ........o
        |         .         |
        o    LL   .    LR   o
        |         .         |
        o----o----o----o----o
        '''

        n_nodes_for_width = n_elem_for_width+1
        coords, boundary_nodes = self.create_nodes(n_nodes_for_width)

        # Create elements, i.e., the connectivity.
        # Elements for all three blocks.
        n_elem = 3*n_elem_for_width*n_elem_for_width
        elements = np.zeros((n_elem,4), dtype=int)
        i_elem = 0

        # Create elements for lower left (LL) block.
        for i in range((n_elem_for_width+1)*n_elem_for_width):
            if i%(n_elem_for_width+1)!=n_elem_for_width:
                # elements[i_elem,:] = i, int(i+n_elem_for_width+1), int(i+n_elem_for_width+2), int(i+1)
                elements[i_elem,:] = [i, 
                                    i+n_elem_for_width+1, 
                                    i+n_elem_for_width+2, 
                                    i+1]
                i_elem = i_elem+1
        
        i_offset =  (n_nodes_for_width-1)*n_nodes_for_width   
        # Create elements for lower and upper right (LR and UR) blocks.
        for i in range((2*n_elem_for_width+1)*n_elem_for_width):
            j = i_offset + i
            if i%(2*n_elem_for_width+1)!=2*n_elem_for_width:
                # elements[i_elem,:] = int(j),int(j+2*n_elem_for_width+1),int(j+2*n_elem_for_width+2),int(j+1)
                elements[i_elem,:] = [j,
                                    j+2*n_elem_for_width+1,
                                    j+2*n_elem_for_width+2,
                                    j+1]
                i_elem = i_elem+1

        neighbor_elements = self.create_neighbor_elements(n_elem_for_width)

        
        return coords, elements, boundary_nodes, neighbor_elements

    def create_mesh_Q2(self, n_elem_for_width):

        '''  
                            o----o----o----o----o
                            |         |         |
                            o    o    o    o    o  
                            |         |         |
                            o----o----o----o----o
                            |                   |
                            :        UR         :
                            |                   |
        o----o----o----o----o....o....o....o....o
        |                   |                   |
        :         LL        :        LR         :
        |                   |                   |
        o----o----o----o----o----o----o----o----o
        |         |         :         |         |
        o    o    o    o    o    o    o    o    o
        |         |         :         |         |
        o----o----o----o----o----o----o----o----o
        '''

        n_nodes_for_width = 2*n_elem_for_width+1
        coords, boundary_nodes = self.create_nodes(n_nodes_for_width)
        # Create elements, i.e., the connectivity.
        # Elements for all three blocks.
        n_elem = 3*n_elem_for_width*n_elem_for_width
        elements = np.zeros((n_elem,9), dtype=int)
        i_elem = 0

        # Create elements for lower left (LL) block.
        for j in range(2*n_elem_for_width):
            if j%2==0:
                for k in range(2*n_elem_for_width+1):
                    if k%2==0 and k%(2*n_elem_for_width+1)!=2*n_elem_for_width:
                        i = j*(2*n_elem_for_width+1)+k
                        elements[i_elem,:] = [i,
                                        i+2*2*n_elem_for_width+2,
                                        i+2*2*n_elem_for_width+4,
                                        i+2,
                                        i+2*n_elem_for_width+1,
                                        i+2*2*n_elem_for_width+3,
                                        i+2*n_elem_for_width+3,
                                        i+1,
                                        i+2*n_elem_for_width+2]
                        i_elem = i_elem+1

        i_offset =  (n_nodes_for_width-1)*n_nodes_for_width   
        # Create elements for lower and upper right (LR and UR) blocks.
        for k in range(2*n_elem_for_width):
            if k%2==0:
                for l in range(2*2*n_elem_for_width+1):
                    if l%2==0 and l%(2*2*n_elem_for_width+1)!=2*2*n_elem_for_width:
                        i = k*(2*2*n_elem_for_width+1)+l
                        j = i+(2*n_elem_for_width+1)*2*n_elem_for_width
                        elements[i_elem,:] = [int(j),
                                        j+4*2*n_elem_for_width+2,
                                        j+4*2*n_elem_for_width+4,
                                        j+2,
                                        j+2*2*n_elem_for_width+1,
                                        j+4*2*n_elem_for_width+3,
                                        j+2*2*n_elem_for_width+3,
                                        j+1,
                                        j+2*2*n_elem_for_width+2]
                        i_elem = i_elem+1

        neighbor_elements = self.create_neighbor_elements(n_elem_for_width)

        return coords, elements, boundary_nodes, neighbor_elements

class MeshRectangle(Mesh):
    def __init__(self, elem_type, n_elem_for_width, n_elem_for_height, width=1.0, height=1.0):
        self.elem_type = elem_type
        self.n_elem_for_width = n_elem_for_width
        self.n_elem_for_height = n_elem_for_height
        self.width = width
        self.height = height
        self.elem_width = width/n_elem_for_width
        self.elem_height = height/n_elem_for_height
        self.area = width*height
        if elem_type == 'Q1':
            self.coords, self.elements, self.boundary_nodes, self.neighbor_elements = self.create_mesh_Q1(n_elem_for_width, n_elem_for_height, width, height)
        elif elem_type == 'Q2':
            self.coords, self.elements, self.boundary_nodes, self.neighbor_elements = self.create_mesh_Q2(n_elem_for_width, n_elem_for_height, width, height)
        else:
            raise Exception(f"Unknown element type for L-channel mesh.")
        super().__init__()

    def create_nodes(self, n_nodes_for_width, n_nodes_for_height, width, height):

        dx = width/(n_nodes_for_width-1)
        dy = height/(n_nodes_for_height-1)

        n_nodes = n_nodes_for_width*n_nodes_for_height
        coords = np.zeros((n_nodes,2))
        x = 0
        for i in range(n_nodes_for_width):
            y = 0
            for j in range(n_nodes_for_height):
                i_node = i*(n_nodes_for_height)+j
                coords[i_node,:] = np.array([x,y])
                y = y+dy
            x = x+dx

        boundary_nodes = self.create_boundary_nodes(n_nodes_for_width, n_nodes_for_height, coords)

        return coords, boundary_nodes

    def create_boundary_nodes(self, n_nodes_for_width, n_nodes_for_height, coords):

        boundary_nodes = {'left': [], 'bottom': [], 'right': [], 'top': []}

        for i in range(n_nodes_for_width):
            for j in range(n_nodes_for_height):
                i_node = i*(n_nodes_for_height)+j
                if i == 0:
                    boundary_nodes['left'].append(i_node)
                elif i == n_nodes_for_width-1:
                    boundary_nodes['right'].append(i_node)
                elif j == 0:
                    boundary_nodes['bottom'].append(i_node)
                elif j == n_nodes_for_height-1:
                     boundary_nodes['top'].append(i_node)

        return boundary_nodes

    def create_neighbor_elements(self, n_elem_for_width, n_elem_for_height):
        neighbor_elements = []
        for i in range(n_elem_for_width*n_elem_for_height):
            t_adj = []
            if i%n_elem_for_height!=0:
                t_u = i-1
                t_adj.append(t_u)
            if i>=n_elem_for_height:
                t_l = i-n_elem_for_height
                t_adj.append(t_l)
            if i%n_elem_for_height!=n_elem_for_height-1:
                t_o = i+1
                t_adj.append(t_o)
            if (n_elem_for_width*n_elem_for_height)-i>n_elem_for_width:
                t_r = i+n_elem_for_height
                t_adj.append(t_r)
            neighbor_elements.append(t_adj)

        return neighbor_elements

    def create_mesh_Q1(self, n_elem_for_width, n_elem_for_height, width, height):
        '''
            o----o----o----o
            |    |    |    |
            o ---o----o----o
            |    |    |    |
            o----o----o----o
        '''

        n_nodes_for_width  = n_elem_for_width+1
        n_nodes_for_height = n_elem_for_height+1
        coords, boundary_nodes = self.create_nodes(n_nodes_for_width, n_nodes_for_height, width, height)

        # Create elements, i.e., the connectivity.
        n_elem = n_elem_for_width*n_elem_for_height
        elements = np.zeros((n_elem,4), dtype=int)
        i_elem = 0

        # Create elements.
        for i in range(n_elem_for_width*(n_elem_for_height+1)):
            if i%(n_elem_for_height+1)!=n_elem_for_height:
                elements[i_elem,:] = [i,
                                    i+n_elem_for_height+1,
                                    i+n_elem_for_height+2,
                                    i+1]
                i_elem = i_elem+1

        neighbor_elements = self.create_neighbor_elements(n_elem_for_width, n_elem_for_height)

        return coords, elements, boundary_nodes, neighbor_elements

    def create_mesh_Q2(self, n_elem_for_width, n_elem_for_height, width, height):
        '''
            o----o----o----o----o----o----o
            |         |         |         |
            o    o    o    o    o    o    o
            |         |         |         |
            o----o----o----o----o----o----o
            |         |         |         |
            o    o    o    o    o    o    o
            |         |         |         |
            o----o----o----o----o----o----o
        '''
        n_nodes_for_width  = 2*n_elem_for_width+1
        n_nodes_for_height = 2*n_elem_for_height+1
        coords, boundary_nodes = self.create_nodes(n_nodes_for_width, n_nodes_for_height, width, height)

        # Create elements, i.e., the connectivity.
        n_elem = n_elem_for_width*n_elem_for_height
        elements = np.zeros((n_elem,9), dtype=int)
        i_elem = 0

        # Create elements.
        for j in range(2*n_elem_for_width):
            if j%2==0:
                for k in range(2*n_elem_for_height+1):
                    if k%2==0 and k%(2*n_elem_for_height+1)!=2*n_elem_for_height:
                        i = j*(2*n_elem_for_height+1)+k
                        elements[i_elem,:] = [i,
                                        i+2*2*n_elem_for_height+2,
                                        i+2*2*n_elem_for_height+4,
                                        i+2,
                                        i+2*n_elem_for_height+1,
                                        i+2*2*n_elem_for_height+3,
                                        i+2*n_elem_for_height+3,
                                        i+1,
                                        i+2*n_elem_for_height+2]
                        i_elem = i_elem+1

        neighbor_elements = self.create_neighbor_elements(n_elem_for_width, n_elem_for_height)

        return coords, elements, boundary_nodes, neighbor_elements

class MeshDiffuser(MeshRectangle):

    def __init__(self, elem_type, n_elem_for_width, n_elem_for_height, width=1, height=1, outlet_height=1./3.):

        self.outlet_height = outlet_height

        if (height - 3*outlet_height) < 0.0:
            raise Exception('Total height too small for outlet height!')

        super().__init__(elem_type, n_elem_for_width, n_elem_for_height, width, height)

    def create_boundary_nodes(self, n_nodes_for_width, n_nodes_for_height, coords):

        boundary_nodes = {'inlet': [],
                          'outlet': [],
                          'wall': []}

        for i in range(n_nodes_for_width):
            for j in range(n_nodes_for_height):
                i_node = i*(n_nodes_for_height)+j
                if i == 0:
                    boundary_nodes['inlet'].append(i_node)
                elif i == n_nodes_for_width-1:
                    y = coords[i_node, 1]
                    if y >= self.outlet_height and y <= 2*self.outlet_height:
                        boundary_nodes['outlet'].append(i_node)
                    else:
                        boundary_nodes['wall'].append(i_node)
                elif j == 0 or j == n_nodes_for_height-1:
                    boundary_nodes['wall'].append(i_node)

        return boundary_nodes

class MeshDoublePipe(MeshRectangle):

    def __init__(self, elem_type, n_elem_for_width, n_elem_for_height, width=1., height=1., inlet_height=1./6., outlet_height=1./6.):
        self.inlet_height = inlet_height
        self.outlet_height = outlet_height

        if (height - 4*inlet_height) < 0.0:
            raise Exception('Total height too small for inlet height!')
        elif (height - 4*outlet_height) < 0.0:
            raise Exception('Total height too small for outlet height!')

        super().__init__(elem_type, n_elem_for_width, n_elem_for_height, width, height)

    def create_boundary_nodes(self, n_nodes_for_width, n_nodes_for_height, coords):

        boundary_nodes = {'inlet_lower': [],
                          'inlet_upper': [],
                          'outlet_lower': [],
                          'outlet_upper': [],
                          'wall': []}

        for i in range(n_nodes_for_width):
            for j in range(n_nodes_for_height):
                i_node = i*(n_nodes_for_height)+j
                if i == 0:
                    y = coords[i_node, 1]
                    if y >= self.inlet_height and y <= 2*self.inlet_height:
                        boundary_nodes['inlet_lower'].append(i_node)
                    elif y >= self.height - 2*self.inlet_height and y <= self.height-self.inlet_height:
                        boundary_nodes['inlet_upper'].append(i_node)
                    else:
                        boundary_nodes['wall'].append(i_node)
                elif i == n_nodes_for_width-1:
                    y = coords[i_node, 1]
                    if y >= self.inlet_height and y <= 2*self.inlet_height:
                        boundary_nodes['outlet_lower'].append(i_node)
                    elif y >= self.height - 2*self.inlet_height and y <= self.height-self.inlet_height:
                        boundary_nodes['outlet_upper'].append(i_node)
                    else:
                        boundary_nodes['wall'].append(i_node)
                elif j == 0 or j == n_nodes_for_height-1:
                    boundary_nodes['wall'].append(i_node)

        return boundary_nodes
