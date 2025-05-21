from amplify.client import FixstarsClient
from matplotlib import pyplot as plt
import numpy as np
import os
import tikzplotlib

from flow_solver.finite_element_model import FEM
from flow_solver.mesh_generator import MeshDiffuser
from optimizer import Annealing, AnnealingSolver
from problems.topo_opt_level_set_fluid_flow import TopologyOptimizationProblem

output_path = './'

# Create Meshes for Diffuser Problem
n_elem_for_width = 32
n_elem_for_height = 32

mesh_v = MeshDiffuser('Q2', n_elem_for_width, n_elem_for_height)
mesh_p = MeshDiffuser('Q1', n_elem_for_width, n_elem_for_height)

## Boundary Conditions.
# Velocity profile at inlet.
def inlet_velocity(mesh, i_node):
    x = mesh.coords[i_node][0]; y = mesh.coords[i_node][1]
    v_x = 4*(1.-y)*y; v_y = 0.0
    return np.array([v_x, v_y])

# Compute boundary velocities.
boundary_velocity = {key: [] for key in mesh_v.boundary_nodes.keys()}
for boundary in mesh_v.boundary_nodes.keys():
    for i_node in mesh_v.boundary_nodes[boundary]:
        if boundary == 'inlet':
            boundary_velocity[boundary].append(inlet_velocity(mesh_v, i_node))
        elif boundary == 'wall':
            boundary_velocity[boundary].append(np.zeros(2))

# Set values for Dirichlet boundary conditions.
# Velocity.
BC = []
for boundary in mesh_v.boundary_nodes.keys():
    for i_node, node in enumerate(mesh_v.boundary_nodes[boundary]):
        if boundary == 'inlet' or boundary == 'wall' :
            BC.append([node, 0, boundary_velocity[boundary][i_node][0]])
            BC.append([node, 1, boundary_velocity[boundary][i_node][1]])
        elif boundary == 'outlet':
            # Parallel outflow, i.e., no vertical velocity.
            BC.append([node, 1, 0.0])
boundary_values_v = np.array(BC)

# Material properties.
viscosity = 1.0
density = 1.0
alpha_solid = 12.5
epsilon = viscosity/alpha_solid

## Finite Element Model.
fem = FEM(mesh_v, mesh_p, viscosity, density, epsilon=epsilon)
fem.set_boudary_values_v(boundary_values_v)

## Optimization.
# Volume fraction.
volume_fraction_max = 0.5
# Initial design, i.e., no material .
design_initial = np.ones(mesh_v.n_elem) 

# Annealing Solver.
client = FixstarsClient()
client.parameters.timeout = 1000
client.parameters.outputs.duplicate = True

file_token_fixstars = './token_Fixstars.txt'
file_proxy_settings = './proxy.txt'
if os.path.exists(file_token_fixstars):
    client.token = open(file_token_fixstars,"r").read().replace('\n', '')
if os.path.exists(file_proxy_settings):
    client.proxy = open(file_proxy_settings,"r").read().replace('\n', '')
    
annealing_solver = AnnealingSolver(client)

# Topology Optimization Problem
n_qubits_per_variable = 9
hyperparameters = {
        'energy_dissipation': 1.,
        'regularization': 0.,
        'volume_constraint': 0.2,
        'char_func': 0.
}
topo_opt_problem = TopologyOptimizationProblem(fem.ne, n_qubits_per_variable, hyperparameters, volume_fraction_max)
topo_opt_problem.generate_discretizaton()

# Annealing-Based Optimizer (Two-Step Optimization)
annealing_optimizer = Annealing(fem)
max_opt_steps = 10

# Hyperparameter Study.
lambda_vol_initial = 0.2
lambda_vol_delta = 0.5
n_steps_vol = 5
lambda_vol_values = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
n_steps_vol = len(lambda_vol_values)

objective_functions_values = []
volume_fraction_values = []
inconsistency_values = []
char_funcs = []
level_sets = []
for lambda_reg_i in range(n_steps_vol):
    hyperparameters['volume_constraint'] = lambda_vol_values[lambda_reg_i]
    # lambda_vol_values.append(hyperparameters['volume_constraint'])
    print(f' LAMBDA_VOL = {lambda_vol_values[lambda_reg_i]} '.center(80, '#'))

    output_path_current = os.path.join(output_path, str(lambda_vol_values[lambda_reg_i]))
    os.makedirs(output_path_current, exist_ok=True)

    topo_opt_problem.set_hyperparameters(hyperparameters)
    annealing_optimizer.optimize(annealing_solver, 
                                topo_opt_problem,
                                design_initial, 
                                max_opt_steps, tol=1e-2,
                                plot_mode='final',
                                output_path=output_path_current,
                                tikz = True)
    objective_functions_values.append(annealing_optimizer.objective_function)
    volume_fraction_values.append(annealing_optimizer.volume_fraction)
    inconsistency_values.append(annealing_optimizer.n_inconsistencies)
    level_set, level_set_scaled, char_func = topo_opt_problem.get_functions_from_binary_solutions(annealing_optimizer.binary_solutions_optimum)
    char_funcs.append(char_func)
    level_sets.append(level_set)

    # Create figures.
    textwidth_in_inches = 500.484/72.27
    aspect_ratio = 0.5

    # Objective function.
    fig, ax = plt.subplots(figsize=(textwidth_in_inches, textwidth_in_inches * aspect_ratio))
    ax.set_title('Comparison Objective Function')
    ax.plot(np.arange(1, len(annealing_optimizer.objective_function_list)+1),annealing_optimizer.objective_function_list,label="Annealing", marker='s', color='k',zorder=2)
    ax.axhline(y=annealing_optimizer.objective_function, color='k', label='Annealing (final)', zorder=2)


    # natural_numbers = np.arange(0, len(annealing_optimizer.objective_function_list),2)
    # plt.xticks(natural_numbers)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel(r'Objective Function $J$')
    ax.legend(loc='best', fontsize='medium')

    plt.savefig(os.path.join(output_path_current, 'objective.png'))
    tikzplotlib.save(os.path.join(output_path_current, 'objective.tex'))
    plt.close(fig)

    # Volume fraction.
    dV = 1./mesh_p.n_elem
    fig, ax = plt.subplots(figsize=(textwidth_in_inches, textwidth_in_inches * aspect_ratio))
    ax.set_title('Comparison Volume Fraction')
    ax.plot(np.arange(1, len(annealing_optimizer.volume_fraction_list)+1), annealing_optimizer.volume_fraction_list,label='Annealing',marker='s',color='k',zorder=4)
    
   
    ax.axhline(y=volume_fraction_max, label=r'$V_{\mathrm{max}}$', color='k', linestyle='dotted')
    x_limits = fig.gca().get_xlim()
    x_fill = np.linspace(x_limits[0], x_limits[1], 100)
    ax.axhline(y=volume_fraction_max+1*dV, color='lightgray')
    ax.axhline(y=volume_fraction_max-1*dV, color='lightgray')
    ax.fill_between(x_fill, volume_fraction_max-1*dV, volume_fraction_max+1*dV, color='lightgray', alpha=0.2, label='Binary Resolution', zorder=1)
    ax.set_xlim(x_limits)
    ax.set_ylim([0.49,0.52])
    natural_numbers = np.arange(0, len(annealing_optimizer.objective_function_list),2)
    plt.xticks(natural_numbers)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel(r'Volume Fraction $V_f/|\Omega|$')
    ax.legend(loc='best', fontsize='medium')

    plt.savefig(os.path.join(output_path_current, 'volume_fraction.png'))
    tikzplotlib.save(os.path.join(output_path_current, 'volume_fraction.tex'))
    plt.close(fig)


fig, ax = plt.subplots(figsize=(textwidth_in_inches, textwidth_in_inches * aspect_ratio))
ax.set_title('Comparison Objective Function')
ax.plot(lambda_vol_values, objective_functions_values)
ax.set_xlabel(r'$\lambda_{\mathrm{vol}}$')
ax.set_ylabel(r'Objective Function $J$')
plt.savefig(os.path.join(output_path, 'objective_over_lambda_vol.png'))
tikzplotlib.save(os.path.join(output_path, 'objective_over_lambda_vol.tex'))
plt.close(fig)

fig, ax = plt.subplots(figsize=(textwidth_in_inches, textwidth_in_inches * aspect_ratio))
ax.set_title('Comparison Volume Fraction Function')
ax.plot(lambda_vol_values, volume_fraction_values)
ax.set_xlabel(r'$\lambda_{\mathrm{vol}}$')
ax.set_ylabel(r'Volume Fraction $V_f/|\Omega|$')
plt.savefig(os.path.join(output_path, 'volume_fraction_over_lambda_vol.png'))
tikzplotlib.save(os.path.join(output_path, 'volume_fraction_over_lambda_vol.tex'))
plt.close(fig)

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(textwidth_in_inches, textwidth_in_inches * aspect_ratio))

# Plot 1: Objective Function
ax1.set_title('Comparison of Objective Function and Volume Fraction')
ax1.set_xlabel(r'$\lambda_{\mathrm{vol}}$')
ax1.set_ylabel(r'Objective Function $J$')
line1, = ax1.plot(lambda_vol_values, objective_functions_values, color='black', linestyle='-', marker='x', label='Objective Function $J$')
ax1.tick_params(axis='y', colors='black')

# Create twin axis for second y-axis
ax2 = ax1.twinx()
ax2.set_ylabel(r'Volume Fraction $V_f/|\Omega|$')
line2, = ax2.plot(lambda_vol_values, volume_fraction_values, color='gray', linestyle='--', marker='o', label=r'Volume Fraction $V_f/|\Omega|$')
ax2.tick_params(axis='y', colors='black')

# Optional: add combined legend
lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='center right')

# Save the combined figure
plt.tight_layout()
combined_filename_base = os.path.join(output_path, 'combined_objective_volume_fraction_over_lambda_vol')
plt.savefig(combined_filename_base + '.png', dpi=300)
tikzplotlib.save(combined_filename_base + '.tex')
plt.close(fig)