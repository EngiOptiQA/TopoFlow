import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix, hstack, vstack, linalg
import numpy.matlib
import pandas as pd
import matplotlib.patches as patches
import matplotlib.colors as colors

"""
Function for mesh
"""
def SquareMeshQ1(nelm):
    dx = 1/nelm
    dy = dx
    nnode = (nelm+1)*(3*nelm+1)
    Node = np.zeros((nnode,2))
    x = 0
    for i in range(nelm):
        y = 0
        for j in range(nelm+1):
            ind = i*(nelm+1)+j
            Node[ind,:] = np.array([x,y])
            y = y+dy
        x = x+dx
    for i in range(nelm+1):
        y = 0
        for j in range(2*nelm+1):
            ind = nelm*(nelm+1)+i*(2*nelm+1)+j
            Node[ind,:] = np.array([x,y])
            y = y+dy
        x = x+dx
    elmN = nelm*nelm*3
    conec = np.zeros((elmN,4))
    ind = 0
    for i in range((nelm+1)*nelm):
        if i%(nelm+1)!=nelm:
            conec[ind,:] = i,int(i+nelm+1), int(i+nelm+2), int(i+1)
            ind = ind+1
    for i in range((2*nelm+1)*nelm):
        j = i+(nelm+1)*nelm
        if i%(2*nelm+1)!=2*nelm:
            conec[ind,:] = int(j),int(j+2*nelm+1),int(j+2*nelm+2),int(j+1)
            ind = ind+1
    Element = conec.astype(int)
    NodeBC = BoundaryCndQ1(nelm,Node)
    return Node, Element, NodeBC

def BoundaryCndQ1(nelm,Node):
    BC=[]
    for i in range(nelm):
        BC.append([(i+1)*(nelm+1),0,0])
        BC.append([(i+1)*(nelm+1),1,0])
        BC.append([(i+1)*(nelm+1)+nelm,0,0])
        BC.append([(i+1)*(nelm+1)+nelm,1,0])
        BC.append([(nelm+1)*nelm+(i+1)*(2*nelm+1),0,0])
        BC.append([(nelm+1)*nelm+(i+1)*(2*nelm+1),1,0])
    for i in range(nelm-1):
        BC.append([(nelm+1)*(nelm+1)+i,0,0])
        BC.append([(nelm+1)*(nelm+1)+i,1,0])
    for i in range(nelm*2-1):
        BC.append([(nelm+1)*nelm+(2*nelm+1)*nelm+1+i,0,0])
        BC.append([(nelm+1)*nelm+(2*nelm+1)*nelm+1+i,1,0])
    for i in range(nelm+1):
        v_x = -(1-4*(1/2-Node[i,1])**2)
        BC.append([i,0,v_x])
        BC.append([i,1,0])
        v_y = -(1-4*(3/2-Node[(nelm+1)*nelm+(2*nelm+1)*(i+1)-1,0])**2)
        BC.append([(nelm+1)*nelm+(2*nelm+1)*(i+1)-1,0,0])
        BC.append([(nelm+1)*nelm+(2*nelm+1)*(i+1)-1,1,v_y])
    BC.sort()
    NodeBC = np.array(BC)
    return NodeBC

def SquareMeshQ2(nelm_1):
    nelm = nelm_1*2
    dx = 1/nelm
    dy = dx
    nnode = (nelm+1)*(3*nelm+1)
    Node = np.zeros((nnode,2))
    x = 0
    for i in range(nelm):
        y = 0
        for j in range(nelm+1):
            ind = i*(nelm+1)+j
            Node[ind,:] = np.array([x,y])
            y = y+dy
        x = x+dx
    for i in range(nelm+1):
        y = 0
        for j in range(2*nelm+1):
            ind = nelm*(nelm+1)+i*(2*nelm+1)+j
            Node[ind,:] = np.array([x,y])
            y = y+dy
        x = x+dx
    elmN = nelm_1*nelm_1*3
    conec = np.zeros((elmN,9))
    ind = 0
    for j in range(nelm):
        if j%2==0:
            for k in range(nelm+1):
                if k%2==0 and k%(nelm+1)!=nelm:
                    i = j*(nelm+1)+k
                    conec[ind,:] = i,int(i+2*nelm+2),int(i+2*nelm+4),int(i+2),int(i+nelm+1),int(i+2*nelm+3),int(i+nelm+3),int(i+1),int(i+nelm+2)
                    ind = ind+1
    for k in range(nelm):
        if k%2==0:
            for l in range(2*nelm+1):
                if l%2==0 and l%(2*nelm+1)!=2*nelm:
                    i = k*(2*nelm+1)+l
                    j = i+(nelm+1)*nelm
                    conec[ind,:] = int(j),int(j+4*nelm+2),int(j+4*nelm+4),int(j+2),int(j+2*nelm+1),int(j+4*nelm+3),int(j+2*nelm+3),int(j+1),int(j+2*nelm+2)
                    ind = ind+1
    Element = conec.astype(int)
    NodeBC = BoundaryCndQ2(nelm_1,Node)
    return Node, Element, NodeBC

def BoundaryCndQ2(nelm_1,Node):
    nelm=nelm_1*2
    BC=[]
    for i in range(nelm):
        BC.append([(i+1)*(nelm+1),0,0])
        BC.append([(i+1)*(nelm+1),1,0])
        BC.append([(i+1)*(nelm+1)+nelm,0,0])
        BC.append([(i+1)*(nelm+1)+nelm,1,0])
        BC.append([(nelm+1)*nelm+(i+1)*(2*nelm+1),0,0])
        BC.append([(nelm+1)*nelm+(i+1)*(2*nelm+1),1,0])
    for i in range(nelm-1):
        BC.append([(nelm+1)*(nelm+1)+i,0,0])
        BC.append([(nelm+1)*(nelm+1)+i,1,0])
    for i in range(nelm*2-1):
        BC.append([(nelm+1)*nelm+(2*nelm+1)*nelm+1+i,0,0])
        BC.append([(nelm+1)*nelm+(2*nelm+1)*nelm+1+i,1,0])
    for i in range(nelm+1):
        v_x = -(1-4*(1/2-Node[i,1])**2)
        BC.append([i,0,v_x])
        BC.append([i,1,0])
        v_y = -(1-4*(3/2-Node[(nelm+1)*nelm+(2*nelm+1)*(i+1)-1,0])**2)
        BC.append([(nelm+1)*nelm+(2*nelm+1)*(i+1)-1,0,0])
        BC.append([(nelm+1)*nelm+(2*nelm+1)*(i+1)-1,1,v_y])
    BC.sort()
    NodeBC = np.array(BC)
    return NodeBC


def RecSquareMeshQ1(nelm,left,right,wid,spa):
    dx = 1/nelm
    dy = dx
    nnode = (nelm+1)*(nelm+1)
    Node = np.zeros((nnode,2))
    x = 0
    for i in range(nelm+1):
        y = 0
        for j in range(nelm+1):
            ind = i*(nelm+1)+j
            Node[ind,:] = np.array([x,y])
            y = y+dy
        x = x+dx
    elmN = nelm*nelm
    conec = np.zeros((elmN,4))
    ind = 0
    for i in range((nelm+1)*nelm):
        if i%(nelm+1)!=nelm:
            conec[ind,:] = i,int(i+nelm+1), int(i+nelm+2), int(i+1)
            ind = ind+1
    Element = conec.astype(int)
    NodeBC = RecBoundaryCndQ1(nelm,Node,left,right,wid,spa)
    return Node, Element, NodeBC
def RecBoundaryCndQ1(nelm,Node,left,right,wid,spa):
    BC=[]
    for i in range(nelm-1):
        BC.append([(nelm+1)*(i+1),0,0])
        BC.append([(nelm+1)*(i+1),1,0])
        BC.append([(nelm+1)*(i+1)+nelm,0,0])
        BC.append([(nelm+1)*(i+1)+nelm,1,0])
    for i in range(nelm+1):
        y_in = Node[i,1]
        if y_in>=left-wid and y_in<=left:
            v_in = 1-((2/wid)**2)*((y_in-(left-wid/2))**2)
        elif y_in>=left-spa-wid*2 and y_in<=left-spa-wid:
            v_in = 1-((2/wid)**2)*((y_in-(left-spa-3*wid/2))**2)
        else:
            v_in=0
        BC.append([i,0,v_in])
        BC.append([i,1,0])
        y_out = Node[(nelm+1)*nelm+i,1]
        if y_out>=right-wid and y_out<=right:
            v_out = 1-((2/wid)**2)*((y_out-(right-wid/2))**2)
        elif y_out>=right-spa-wid*2 and y_out<=right-spa-wid:
            v_out = 1-((2/wid)**2)*((y_out-(right-spa-3*wid/2))**2)
        else:
            v_out=0
        BC.append([(nelm+1)*nelm+i,0,v_out])
        BC.append([(nelm+1)*nelm+i,1,0])
    BC.sort()
    NodeBC = np.array(BC)
    return NodeBC

def RecSquareMeshQ2(nelm_1,left,right,wid,spa):
    nelm=nelm_1*2
    dx = 1/nelm
    dy = dx
    nnode = (nelm+1)*(nelm+1)
    Node = np.zeros((nnode,2))
    x = 0
    for i in range(nelm+1):
        y = 0
        for j in range(nelm+1):
            ind = i*(nelm+1)+j
            Node[ind,:] = np.array([x,y])
            y = y+dy
        x = x+dx
    elmN = nelm_1*nelm_1
    conec = np.zeros((elmN,9))
    ind = 0
    for j in range(nelm):
        if j%2==0:
            for k in range(nelm+1):
                if k%2==0 and k%(nelm+1)!=nelm:
                    i = j*(nelm+1)+k
                    conec[ind,:] = i,int(i+2*nelm+2),int(i+2*nelm+4),int(i+2),int(i+nelm+1),int(i+2*nelm+3),int(i+nelm+3),int(i+1),int(i+nelm+2)
                    ind = ind+1
    Element = conec.astype(int)
    NodeBC = RecBoundaryCndQ1(nelm,Node,left,right,wid,spa)
    return Node, Element, NodeBC
def RecBoundaryCndQ2(nelm_1,Node,left,right,wid,spa):
    nelm=nelm_1*2
    BC=[]
    for i in range(nelm-1):
        BC.append([(nelm+1)*(i+1),0,0])
        BC.append([(nelm+1)*(i+1),1,0])
        BC.append([(nelm+1)*(i+1)+nelm,0,0])
        BC.append([(nelm+1)*(i+1)+nelm,1,0])
    for i in range(nelm+1):
        y_in = Node[i,1]
        if y_in>=left-wid and y_in<=left:
            v_in = 1-((2/wid)**2)*((y_in-(left-wid/2))**2)
        elif y_in>=left-spa-wid*2 and y_in<=left-spa-wid:
            v_in = 1-((2/wid)**2)*((y_in-(left-spa-3*wid/2))**2)
        else:
            v_in=0
        BC.append([i,0,v_in])
        BC.append([i,1,0])
        y_out = Node[(nelm+1)*nelm+i,1]
        if y_out>=right-wid and y_out<=right:
            v_out = 1-((2/wid)**2)*((y_out-(right-wid/2))**2)
        elif y_out>=right-spa-wid*2 and y_out<=right-spa-wid:
            v_out = 1-((2/wid)**2)*((y_out-(right-spa-3*wid/2))**2)
        else:
            v_out=0
        BC.append([(nelm+1)*nelm+i,0,v_out])
        BC.append([(nelm+1)*nelm+i,1,0])
    BC.sort()
    NodeBC = np.array(BC)
    return NodeBC


"""
Main Function
"""
def Q2Q1FEM(Node,Element,NodeBC,Node_P,Element_P,NodeBC_P,zIni):
    fem_NNode = Node.shape[0]    # Number of nodes
    fem_NElem = Element.shape[0] # Number of elements
    fem_Node = Node           # [NNode x 2] array of nodes
    fem_Element = Element     # [NElement x Var] cell array of elements
    fem_NodeBC = NodeBC       # Array of velocity boundary conditions

    fem_NNode_P = Node_P.shape[0]    # Number of nodes
    fem_NElem_P = Element_P.shape[0] # Number of elements
    fem_Node_P = Node_P           # [NNode x 2] array of nodes
    fem_Element_P = Element_P     # [NElement x Var] cell array of elements
    fem_NodeBC_P = NodeBC_P       # Array of velocity boundary conditions

    fem_mu0 = 1               # Dynamic viscosity
    fem_Reg = 0               # Tag for regular meshes
    R = -1
    Volfrac = 1
    P_ori = np.identity(fem_NElem)
    P_ori_P = np.identity(fem_NElem_P)
    P = csr_matrix(P_ori)
    P_P = csr_matrix(P_ori_P)
    
    opt_zMin=0.0    # Lower bound for design variables
    opt_zMax=1.0    # Upper bound for design variables
    opt_zIni = zIni # Initial design variables
    opt_zIni_P = zIni
    opt_P = P       # Matrix that maps design to element vars.
    opt_P_P = P_P
    opt_VolFrac = Volfrac  # Specified volume fraction cosntraint
    opt_Tol = 0.01   # Convergence tolerance on design vars.
    opt_MaxIter = 150  # Max. number of optimization iterations
    opt_OCMove = 0.2  # Allowable move step in OC update scheme
    opt_OCEta = 0.5  # Exponent used in OC update scheme
    
    Tol=opt_Tol*(opt_zMax-opt_zMin)
    Change=2*Tol
    z=opt_zIni
    z_P = opt_zIni_P
    P=opt_P
    P_P = opt_P_P
    
    E,dEdy,V,dVdy = MatIntFnc(P*z,np.array([fem_mu0,0.01]))
    E_P,dEdy_P,V_P,dVdy_P = MatIntFnc(P_P*z_P,np.array([fem_mu0,0.01]))
    g,dgdE,dgdV,fem_ElemArea = ConstraintFncQ2(fem_NElem,fem_Node,fem_Element,E,V,Volfrac)
    g_P,dgdE_P,dgdV_P,fem_ElemArea_P = ConstraintFncQ1(fem_NElem_P,fem_Node_P,fem_Element_P,E_P,V_P,Volfrac)
    
    fem_ElemNDofA = (2*fem_Element.shape[1])*np.ones(fem_Element.shape[0])
    fem_ElemNDofA_P = (fem_Element_P.shape[1])*np.ones(fem_Element_P.shape[0])
    fem_iA = np.zeros(int(sum(fem_ElemNDofA**2)))
    fem_jA=np.zeros(int(sum(fem_ElemNDofA**2)))
    fem_kAmu=np.zeros(int(sum(fem_ElemNDofA**2))) 
    fem_kAalpha=np.zeros(int(sum(fem_ElemNDofA**2)))
    fem_e=np.zeros(int(sum(fem_ElemNDofA**2)))

    fem_iB=np.zeros(int((sum(fem_ElemNDofA_P))*(sum(fem_ElemNDofA))))
    fem_jB=np.zeros(int((sum(fem_ElemNDofA_P))*(sum(fem_ElemNDofA))))
    fem_kB=np.zeros(int((sum(fem_ElemNDofA_P))*(sum(fem_ElemNDofA))))

    fem_NDof = (2*fem_NNode+fem_NNode_P)+1
    indexA = 0
    indexB = 0

    fem_ShapeFnc_W,fem_ShapeFnc_X = quad_order_2_gauss()
    fem_ShapeFnc_W_P,fem_ShapeFnc_X_P = quad_order_1_gauss()
    fem_ShapeFnc_N = ShapefuncQ2(fem_ShapeFnc_X)
    fem_ShapeFnc_N_P = ShapefuncQ1(fem_ShapeFnc_X)
    fem_ShapeFnc_dNdxi = Shapefunc_gradientsQ2(fem_ShapeFnc_X)
    fem_ShapeFnc_dNdxi_P = Shapefunc_gradientsQ1(fem_ShapeFnc_X)

    np.set_printoptions(threshold=np.inf)

    for el in range(fem_NElem):  
        eNode = fem_Element[el,:]
        eNode_P = fem_Element_P[el,:]
        eNDof = int(fem_ElemNDofA[el])
        eNDof_P = int(fem_ElemNDofA_P[el])
        if el==0:
            Amu_e,Aalpha_e=LocalKQ2(fem_mu0,fem_ShapeFnc_W,fem_ShapeFnc_dNdxi,fem_ShapeFnc_N,fem_Node,eNode)
            Be = LocalKQ1(fem_mu0,fem_ShapeFnc_W,fem_ShapeFnc_dNdxi,fem_ShapeFnc_N,fem_Node,eNode,fem_ShapeFnc_dNdxi_P,fem_ShapeFnc_N_P,fem_Node_P,eNode_P)
        eDofA = np.concatenate([(2*eNode).reshape(9,1),(2*eNode+1).reshape(9,1)],axis=1).reshape(eNDof,1)
        eDofA_P = eNode_P.reshape(eNDof_P,1)
        I=np.matlib.repmat(eDofA ,1,eNDof)
        J=I.T
        I_P=np.matlib.repmat(eDofA ,1,eNDof_P)
        J_P=np.matlib.repmat(eDofA_P ,1,eNDof)
        fem_iA[indexA:indexA+eNDof**2] = I.T.flatten()
        fem_jA[indexA:indexA+eNDof**2] = J.T.flatten()
        fem_kAmu[indexA:indexA+eNDof**2] = Amu_e.flatten()
        fem_kAalpha[indexA:indexA+eNDof**2] = Aalpha_e.flatten()
        fem_e[indexA:indexA+eNDof**2] = el
        fem_iB[indexB:indexB+eNDof*eNDof_P] = I_P.T.flatten()
        fem_jB[indexB:indexB+eNDof*eNDof_P] = J_P.T.flatten()
        fem_kB[indexB:indexB+eNDof*eNDof_P] = Be.T.flatten()
        indexA = indexA + eNDof**2
        indexB = indexB + eNDof_P*eNDof
    fem_FixedDofs = np.zeros((1,fem_NodeBC.shape[0]))
    fem_G = np.zeros((fem_NodeBC.shape[0],1))
    for i in range(fem_NodeBC.shape[0]):
        fem_FixedDofs[0,i] = 2*(fem_NodeBC[i,0]) + fem_NodeBC[i,1]
        fem_G[i] = fem_NodeBC[i,2]
    fem_FreeDofs = np.setdiff1d(np.arange(fem_NDof),fem_FixedDofs)
    A = csr_matrix((fem_kAmu+E[fem_e.astype(int)]*fem_kAalpha,(fem_iA.astype(int),fem_jA.astype(int))))
    B = csr_matrix((fem_kB,(fem_iB.astype(int),fem_jB.astype(int))))
    Z=np.zeros((2*fem_NNode,1))
    O=csr_matrix((fem_NNode_P,fem_NNode_P),dtype='int')
    K_1 = hstack([A,B,Z])
    K_2 = hstack(([B.T,O,csr_matrix(fem_ElemArea[0]*np.ones((fem_NNode_P,1)))]))
    K_3 = hstack([Z.T,csr_matrix(fem_ElemArea[0]*np.ones(fem_NNode_P)),0])
    K = vstack([K_1,K_2,K_3])
    K = (K+K.transpose())/2
    S = np.zeros((fem_NDof,1)) 
    S[fem_FixedDofs.astype(int),:]=fem_G

    S[fem_FreeDofs,:] = linalg.spsolve(K[fem_FreeDofs,:][:,fem_FreeDofs], \
                                       -K[fem_FreeDofs.flatten(),:][:,fem_FixedDofs.flatten()]*S[fem_FixedDofs.astype(int),:].flatten())[:,np.newaxis]
    U = S[:2*fem_NNode]
    p = S[2*fem_NNode:]
    F = A*U
    f = 1/2*np.dot(F.flatten(),U.flatten())
    
    u_plot = np.zeros(int(U.shape[0]/2))
    v_plot = np.zeros(int(U.shape[0]/2))
    for i in range(u_plot.shape[0]):
        u_plot[i] = U[2*i]
        v_plot[i] = U[2*i+1]
    u_plot_elem = np.zeros(fem_NElem)
    v_plot_elem = np.zeros(fem_NElem)
    for i in range(fem_NElem):
        u_plot_elem[i] = np.mean(u_plot[Element[i,:]])
        v_plot_elem[i] = np.mean(v_plot[Element[i,:]])
    p_plot_elem = np.zeros(fem_NElem)
    for i in range(fem_NElem):
        p_plot_elem[i]=np.mean(p[Element_P[i,:]])
    
    return p_plot_elem,f,u_plot_elem,v_plot_elem


"""
Function for FEM
"""
def quad_order_1_gauss():
    s = np.sqrt(1/3)
    x = np.array([[-s, -s], [s, -s], [s, s], [-s, s]]) # Coordinates of quadrature points
    w = np.array([1., 1., 1., 1.])  # Weights of quadrature points
    return w,x  # Return the coordinates and weights of the quadrature rule

def quad_order_2_gauss():
    s = np.sqrt(3/5)
    x = np.array([[-s, -s], [0, -s], [s, -s],
               [-s, 0], [0, 0], [s, 0],
               [-s, s], [0, s], [s, s]])  # Coordinates of quadrature points
    w = np.array([25., 40., 25., 40., 64., 40., 25., 40., 25.]) / 81.0  # Weights of quadrature points
    return w,x  # Return the coordinates and weights of the quadrature rule

def ShapefuncQ1(X):
    N_array = np.zeros((4,X.shape[0]))
    for i in range(X.shape[0]):
        ref_coordinates = X[i,:]
        xi = ref_coordinates[0]   
        eta = ref_coordinates[1]
        # Calculate shape function values for nodes 1 to 4
        N = [(1 - xi) * (1 - eta) * 0.25,  # Element node 1
             (1 + xi) * (1 - eta) * 0.25,  # Element node 2
             (1 + xi) * (1 + eta) * 0.25,  # Element node 3
             (1 - xi) * (1 + eta) * 0.25]  # Element node 4
        N_array[:,i] = N
    return N_array

def ShapefuncQ2(X):
    N_array = np.zeros((9,X.shape[0]))
    for i in range(X.shape[0]):
        ref_coordinates = X[i,:]
        xi = ref_coordinates[0]
        eta = ref_coordinates[1]
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
        N_array[:,i] = N
    return N_array

def Shapefunc_gradientsQ1(X):
    dN_array = np.zeros((4,2,X.shape[0]))
    for i in range(X.shape[0]):
        ref_coordinates = X[i,:]
        xi = ref_coordinates[0]
        eta = ref_coordinates[1]
        # Derivatives of shape function with respect to the reference coordinates
        dN = [[-1 * (1 - eta) * 0.25, -1 * (1 - xi) * 0.25],  # Element node 1
              [(1 - eta) * 0.25, -1 * (1 + xi) * 0.25],       # Element node 2
              [(1 + eta) * 0.25, (1 + xi) * 0.25],            # Element node 3
              [-1 * (1 + eta) * 0.25, (1 - xi) * 0.25]]       # Element node 4
        dN_array[:,:,i] = np.asarray(dN)
    
    return dN_array  # Return gradient of shape function as array

def Shapefunc_gradientsQ2(X):
    dN_array = np.zeros((9,2,X.shape[0]))
    for i in range(X.shape[0]):
        ref_coordinates = X[i,:]
        xi = ref_coordinates[0]
        eta = ref_coordinates[1]
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
        dN_array[:,:,i] = np.asarray(dN)
    return dN_array  # Return gradient of shape function as array

def LocalKQ1(fem_mu0,fem_ShapeFnc_W,fem_ShapeFnc_dNdxi,fem_ShapeFnc_N,fem_Node,eNode,fem_ShapeFnc_dNdxi_P,fem_ShapeFnc_N_P,fem_Node_P,eNode_P):
    nn=eNode.shape[0]
    nn_P = eNode_P.shape[0]
    W = fem_ShapeFnc_W
    Be=np.zeros((2*nn,nn_P))
    for q in range(W.shape[0]):   #quadrature loop
        Ge = np.zeros((2*nn,nn_P))
        dNdxi = fem_ShapeFnc_dNdxi[:,:,q]
        J0 = np.dot(fem_Node[eNode,:].T,dNdxi)
        dNdx = np.dot(dNdxi,np.linalg.inv(J0))
        dNudx = dNdx.flatten()
        dNdx_1 = dNdxi[:,0]
        dNdx_2 = dNdxi[:,1]
        gradient_operator_1 = np.zeros((nn, nn_P))
        gradient_operator_2 = np.zeros((nn, nn_P))
        integrator = W[q]*np.linalg.det(J0)
        for j in range(nn):
            grad_Ni = dNdx_1[j]
            grad_Nj = dNdx_2[j]
            for i in range(nn_P):
                N_P = fem_ShapeFnc_N_P[i,q]
                gradient_operator_1[j,i] += N_P * grad_Ni
                gradient_operator_2[j,i] += N_P * grad_Nj
        Ge[[0,2,4,6,8,10,12,14,16],:nn_P] -= gradient_operator_1
        Ge[[1,3,5,7,9,11,13,15,17],:nn_P] -= gradient_operator_2
        Be[:,:] += integrator*Ge
    return Be

def LocalKQ2(fem_mu0,fem_ShapeFnc_W,fem_ShapeFnc_dNdxi,fem_ShapeFnc_N,fem_Node,eNode):
    Cmu = fem_mu0*np.array([[2,0,0],[0,2,0],[0,0,1]])
    nn = eNode.shape[0]
    Amu_e = np.zeros((2*nn,2*nn))
    Aalpha_e = np.zeros((2*nn,2*nn))
    W = fem_ShapeFnc_W
    for q in range(W.shape[0]):   #quadrature loop
        dNdxi = fem_ShapeFnc_dNdxi[:,:,q]
        J0 = np.dot(fem_Node[eNode,:].T,dNdxi)
        dNdx = np.dot(dNdxi,np.linalg.inv(J0))
        B = np.zeros((3,2*nn))
        B[0,[0,2,4,6,8,10,12,14,16]] = dNdxi[:,0] 
        B[1,[1,3,5,7,9,11,13,15,17]] = dNdxi[:,1] 
        B[2,[0,2,4,6,8,10,12,14,16]] = dNdxi[:,1] 
        B[2,[1,3,5,7,9,11,13,15,17]] = dNdxi[:,0]
        integrator = W[q]*np.linalg.det(J0)
        Amu_e = Amu_e +(np.dot(np.dot(B.T,Cmu),B)*W[q]*np.linalg.det(J0))
        N = fem_ShapeFnc_N[:,q]
        Nu = np.zeros((2,2*nn))
        Nu[0,[0,2,4,6,8,10,12,14,16]] = N.reshape(9)
        Nu[1,[1,3,5,7,9,11,13,15,17]] = N.reshape(9)
        Aalpha_e = Aalpha_e + np.dot(Nu.T,Nu)*integrator
    return Amu_e,Aalpha_e

def ConstraintFncQ1(fem_NElem,fem_Node,fem_Element,E,V,Volfrac):
    fem_ElemArea = np.zeros(fem_NElem)
    for el in range(fem_NElem):
        vx = fem_Node[fem_Element[el,:],0]
        vy = fem_Node[fem_Element[el,:],1]
        vx_s = vx[[1,2,3,0]]                        #🫠
        vy_s = vy[[1,2,3,0]]
        fem_ElemArea[el] = 0.5*sum(vx*vy_s-vy*vx_s)
    g = sum(fem_ElemArea*V)/sum(fem_ElemArea)-Volfrac
    dgdE = np.zeros(E.shape)
    dgdV = fem_ElemArea/sum(fem_ElemArea)
    return g,dgdE,dgdV,fem_ElemArea

def ConstraintFncQ2(fem_NElem,fem_Node,fem_Element,E,V,Volfrac):
    fem_ElemArea = np.zeros(fem_NElem)
    for el in range(fem_NElem):
        vx = fem_Node[fem_Element[el,:],0][:4]
        vy = fem_Node[fem_Element[el,:],1][:4]
        vx_s = vx[[1,2,3,0]]                        #🫠
        vy_s = vy[[1,2,3,0]]
        fem_ElemArea[el] = 0.5*sum(vx*vy_s-vy*vx_s)
    g = sum(fem_ElemArea*V)/sum(fem_ElemArea)-Volfrac
    dgdE = np.zeros(E.shape)
    dgdV = fem_ElemArea/sum(fem_ElemArea)
    return g,dgdE,dgdV,fem_ElemArea

def MatIntFnc(y,param):
    mu0 = param[0]
    q = param[1]
    epsilon = 4*(10**-3)
    E = (mu0/epsilon)*q*(1-y)/(y+q)
    dEdy = -(mu0/epsilon)*(1+q)*q/((y+q)**2)
    V = y
    dVdy = np.ones(y.shape[0])
    return E,dEdy,V,dVdy


"""
Function for Plot
"""
def Plot_initQ2_check(Node,Element,NodeBC):
    Node_T=Node.T
    for temp_t in Element:
        plt.plot([Node_T[0,temp_t[0]],Node_T[0,temp_t[1]]],[Node_T[1,temp_t[0]],Node_T[1,temp_t[1]]],color='magenta')
        plt.plot([Node_T[0,temp_t[1]],Node_T[0,temp_t[2]]],[Node_T[1,temp_t[1]],Node_T[1,temp_t[2]]],color='magenta')
        plt.plot([Node_T[0,temp_t[2]],Node_T[0,temp_t[3]]],[Node_T[1,temp_t[2]],Node_T[1,temp_t[3]]],color='magenta')
        plt.plot([Node_T[0,temp_t[3]],Node_T[0,temp_t[4]]],[Node_T[1,temp_t[3]],Node_T[1,temp_t[4]]],color='magenta')
        plt.plot([Node_T[0,temp_t[4]],Node_T[0,temp_t[5]]],[Node_T[1,temp_t[4]],Node_T[1,temp_t[5]]],color='magenta')
        plt.plot([Node_T[0,temp_t[5]],Node_T[0,temp_t[6]]],[Node_T[1,temp_t[5]],Node_T[1,temp_t[6]]],color='magenta')
        plt.plot([Node_T[0,temp_t[6]],Node_T[0,temp_t[7]]],[Node_T[1,temp_t[6]],Node_T[1,temp_t[7]]],color='magenta')
        plt.plot([Node_T[0,temp_t[7]],Node_T[0,temp_t[0]]],[Node_T[1,temp_t[7]],Node_T[1,temp_t[0]]],color='magenta')
    for i in range(len(Node)):
        plt.scatter(Node[i,0],Node[i,1],color = 'magenta')
    for i in range(len(NodeBC)):
        plt.scatter(Node[int(NodeBC[i,0]),0],Node[int(NodeBC[i,0]),1],color = 'dodgerblue')
    plt.show()
    
def Plot_initQ2(Node,Element,NodeBC):
    Node_T=Node.T
    for temp_t in Element:
        plt.plot([Node_T[0,temp_t[0]],Node_T[0,temp_t[1]]],[Node_T[1,temp_t[0]],Node_T[1,temp_t[1]]],color='magenta')
        plt.plot([Node_T[0,temp_t[1]],Node_T[0,temp_t[2]]],[Node_T[1,temp_t[1]],Node_T[1,temp_t[2]]],color='magenta')
        plt.plot([Node_T[0,temp_t[2]],Node_T[0,temp_t[3]]],[Node_T[1,temp_t[2]],Node_T[1,temp_t[3]]],color='magenta')
        plt.plot([Node_T[0,temp_t[3]],Node_T[0,temp_t[0]]],[Node_T[1,temp_t[3]],Node_T[1,temp_t[0]]],color='magenta')
    for i in range(len(Node)):
        plt.scatter(Node[i,0],Node[i,1],color = 'magenta')
    for i in range(len(NodeBC)):
        plt.scatter(Node[int(NodeBC[i,0]),0],Node[int(NodeBC[i,0]),1],color = 'dodgerblue')
    plt.show()
    
def Plot_initQ1(Node,Element,NodeBC):
    Node_T=Node.T
    for temp_t in Element:
        plt.plot([Node_T[0,temp_t[0]],Node_T[0,temp_t[1]]],[Node_T[1,temp_t[0]],Node_T[1,temp_t[1]]],color='magenta')
        plt.plot([Node_T[0,temp_t[1]],Node_T[0,temp_t[2]]],[Node_T[1,temp_t[1]],Node_T[1,temp_t[2]]],color='magenta')
        plt.plot([Node_T[0,temp_t[2]],Node_T[0,temp_t[3]]],[Node_T[1,temp_t[2]],Node_T[1,temp_t[3]]],color='magenta')
        plt.plot([Node_T[0,temp_t[3]],Node_T[0,temp_t[0]]],[Node_T[1,temp_t[3]],Node_T[1,temp_t[0]]],color='magenta')
    for i in range(len(Node)):
        plt.scatter(Node[i,0],Node[i,1],color = 'magenta')
    for i in range(len(NodeBC)):
        plt.scatter(Node[int(NodeBC[i,0]),0],Node[int(NodeBC[i,0]),1],color = 'dodgerblue')
    plt.show()
    
def Plot_quiver(Node,Element,u_plot_elem,v_plot_elem):
    fem_NElem = Element.shape[0]
    Node_T=Node.T
    centr = np.zeros((2,fem_NElem))
    for i in range(fem_NElem):
        centr[0,i]=np.mean(Node_T[0,Element[i,:]])
        centr[1,i]=np.mean(Node_T[1,Element[i,:]])
    plt.figure(figsize=(6,6))
    for temp_t in Element:
        plt.plot([Node_T[0,temp_t[0]],Node_T[0,temp_t[1]]],[Node_T[1,temp_t[0]],Node_T[1,temp_t[1]]],color='magenta')
        plt.plot([Node_T[0,temp_t[1]],Node_T[0,temp_t[2]]],[Node_T[1,temp_t[1]],Node_T[1,temp_t[2]]],color='magenta')
        plt.plot([Node_T[0,temp_t[2]],Node_T[0,temp_t[3]]],[Node_T[1,temp_t[2]],Node_T[1,temp_t[3]]],color='magenta')
        plt.plot([Node_T[0,temp_t[3]],Node_T[0,temp_t[0]]],[Node_T[1,temp_t[3]],Node_T[1,temp_t[0]]],color='magenta')

    for i in range(fem_NElem):
        plt.quiver(centr[0,i], centr[1,i], u_plot_elem[i], v_plot_elem[i],angles='xy', scale_units='xy', scale=7, width=0.005,color='dodgerblue')
    plt.show()
    
def Plot_patch(nelm,fem_Node,fem_Element,p_plot,max_p,min_p,cmap):
    fem_NElem = fem_Element.shape[0]
    ax = plt.axes()
    cm = plt.get_cmap(cmap)
    for i in range(fem_NElem):
        c_ind = (p_plot[i]-min_p)/(max_p-min_p)
        r = patches.Rectangle(xy=fem_Node[int(fem_Element[i,0])], width=1/nelm, height=1/nelm, fc=colors.rgb2hex(cm(c_ind)),ec='w')
        ax.add_patch(r)
        plt.axis('scaled')
        ax.set_aspect('equal')
    plt.show()
