from amplify.client import FixstarsClient
import csv
from matplotlib import pyplot as plt
import numpy as np
import os
import tikzplotlib

from flow_solver.finite_element_model import FEM
from flow_solver.mesh_generator import MeshDoublePipe
from optimizer import Annealing, AnnealingSolver, PolyTop
from problems.topo_opt_level_set_fluid_flow import TopologyOptimizationProblem

output_path = './'

# Create Meshes for double pipe problem.
n_elem_for_width = 48
n_elem_for_height = 32
mesh_v = MeshDoublePipe('Q2', n_elem_for_width, n_elem_for_height, width=1.5, height=1.0, inlet_height=1./6., outlet_height=1./6.)
mesh_p = MeshDoublePipe('Q1', n_elem_for_width, n_elem_for_height, width=1.5, height=1.0, inlet_height=1./6., outlet_height=1./6.)

## Boundary Conditions.
# Velocity profile at inlets.
def upper_inlet_velocity(mesh, i_node):
    x = mesh.coords[i_node][0]; y = mesh.coords[i_node][1]
    v_x = 1-144*((y-3/4)**2); v_y = 0.0
    return np.array([v_x, v_y])

def lower_inlet_velocity(mesh, i_node):
    x = mesh.coords[i_node][0]; y = mesh.coords[i_node][1]
    v_x = 1-144*((y-1/4)**2); v_y = 0.0
    return np.array([v_x,v_y])

# Compute boundary velocities.
boundary_velocity = {key: [] for key in mesh_v.boundary_nodes.keys()}
for boundary in mesh_v.boundary_nodes.keys():
    for i_node in mesh_v.boundary_nodes[boundary]:
        if boundary == 'inlet_upper':
            boundary_velocity[boundary].append(upper_inlet_velocity(mesh_v, i_node))
        elif boundary == 'inlet_lower':
            boundary_velocity[boundary].append(lower_inlet_velocity(mesh_v, i_node))
        elif boundary == 'wall':
            boundary_velocity[boundary].append(np.zeros(2))
        if boundary == 'outlet_upper':
            boundary_velocity[boundary].append(upper_inlet_velocity(mesh_v, i_node))
        elif boundary == 'outlet_lower':
            boundary_velocity[boundary].append(lower_inlet_velocity(mesh_v, i_node))

# Set values for Dirichlet boundary conditions.
# Velocity.
BC = []
for boundary in mesh_v.boundary_nodes.keys():
    for i_node, node in enumerate(mesh_v.boundary_nodes[boundary]):
        if (boundary == 'inlet_upper' or 
            boundary == 'inlet_lower' or 
            boundary == 'wall' or 
            boundary == 'outlet_upper' or 
            boundary == 'outlet_lower'):
            BC.append([node, 0, boundary_velocity[boundary][i_node][0]])
            BC.append([node, 1, boundary_velocity[boundary][i_node][1]])
boundary_values_v = np.array(BC)

# Material properties.
viscosity = 1.0
density = 1.0
alpha_solid = 12.5
epsilon = viscosity/alpha_solid

## Finite Element Model.
fem = FEM(mesh_v, mesh_p, viscosity, density, epsilon)
fem.set_boudary_values_v(boundary_values_v)

## Optimization.
# Volume fraction..
volume_fraction_max = 1./3.
design_tolerance = 0.01

## Classical.
density_initial = np.ones(mesh_v.n_elem) # Initial design, i.e., no material.
density_min = 0.0; density_max = 1.0 # Lower and upper bound for design variables.

max_iterations = 150 # Max. number of optimization steps.

opt_OCMove = 0.2  # Allowable move step in OC update scheme
opt_OCEta = 0.5  # Exponent used in OC update scheme

q_values =  10. ** -np.arange(4, -1, -1)

poly_top = PolyTop(fem)

density = poly_top.optimize(density_initial, density_min, density_max, volume_fraction_max,
                design_tolerance, max_iterations, q_values,
                opt_OCMove, opt_OCEta)

binary_array = np.where(density >= 0.95, 1, 0)
E = fem.viscosity/fem.epsilon*(1-binary_array)
_, u, v, _, _, f = fem.solve(E)
print(f'Final objective function (binary design): {f}, volume fraction {sum(binary_array)/mesh_p.area}')

## Annealing.
client = FixstarsClient()
client.parameters.timeout = 10000
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
        'volume_constraint': 0.05,
        'char_func': 0.
}

topo_opt_problem = TopologyOptimizationProblem(fem.ne, n_qubits_per_variable, hyperparameters, volume_fraction_max)
topo_opt_problem.generate_discretizaton()

# Initial design, i.e., no material.
level_set_scaled_initial = np.ones(mesh_v.n_elem) 

# Annealing-Based Optimizer (Two-Step Optimization)
max_opt_steps = 15
annealing_optimizer = Annealing(fem)

char_func = annealing_optimizer.optimize(annealing_solver,
                            topo_opt_problem, 
                            level_set_scaled_initial,
                            max_opt_steps,
                            design_tolerance)

## Comparison
# Number of optimization steps
n_C = len(poly_top.objective_function_list)
n_A = len(annealing_optimizer.objective_function_list)
n_rel_diff = (n_A-n_C)/n_C*100
print('Number of optimization steps:')
print('\tClassical:', n_C)
print('\tAnnealing:', n_A)
print('\tRel. Diff. (%):', n_rel_diff)
# Objective function
obj_rel_diff = (annealing_optimizer.objective_function-f)/f*100
print('Objective function:')
print('\tClassical: '+'{:.3e}'.format(f))
print('\tAnnealing: '+'{:.3e}'.format(annealing_optimizer.objective_function))
print('\tRel. Diff (%).:', obj_rel_diff)

# Save as csv file.
filename = os.path.join(output_path, 'comparison.csv')
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Quantity", "Classical", "Annealing", "Relative Difference"])
    writer.writerow(["Number of Opt. Steps", n_C, n_A, n_rel_diff])
    writer.writerow(["Objective Fct.", f, annealing_optimizer.objective_function, obj_rel_diff])


textwidth_in_inches = 500.484/72.27
aspect_ratio = 1.
# History objective function.
# Save as csv file.
filename = os.path.join(output_path, 'comparison_objective.csv')
max_steps = max(len(poly_top.objective_function_list), len(annealing_optimizer.objective_function_list))
rows = [["Opt. Step", "Classical", "Annealing"]] 
for step in range(max_steps):
    value_1 = poly_top.objective_function_list[step] if step < len(poly_top.objective_function_list) else ""
    value_2 = annealing_optimizer.objective_function_list[step] if step < len(annealing_optimizer.objective_function_list) else ""
    rows.append([step + 1, value_1, value_2]) 
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(rows)

# Create figures.
fig, ax = plt.subplots(figsize=(textwidth_in_inches, textwidth_in_inches * aspect_ratio))
ax.set_title('Comparison Objective Function')
ax.plot(np.arange(1, len(annealing_optimizer.objective_function_list)+1),annealing_optimizer.objective_function_list,label="Annealing", marker='s', color='k',zorder=2)
ax.axhline(y=annealing_optimizer.objective_function, color='k', label='Annealing (final)', zorder=2)
ax.plot(np.arange(1, len(poly_top.objective_function_list)+1),poly_top.objective_function_list,label="Classical",marker='x',color='gray', linestyle='dashed',zorder=1)
ax.axhline(y=f, color='gray', label='Classical (final, filtered)', linestyle='dashdot',zorder=1)

natural_numbers = np.arange(0, max(len(annealing_optimizer.objective_function_list),len(poly_top.objective_function_list)),2)
plt.xticks(natural_numbers)
ax.set_xlabel('Optimization Step')
ax.set_ylabel(r'Objective Function $J$')
ax.legend(loc='best', fontsize='medium')

plt.savefig(os.path.join(output_path, 'comparison_objective.png'))
tikzplotlib.save(os.path.join(output_path, 'comparison_objective.tex'))

# History volume fraction.
# Save as csv file.
filename = os.path.join(output_path, 'comparison_volume_fraction.csv')
max_steps = max(len(poly_top.volume_fraction_list), len(annealing_optimizer.volume_fraction_list))
rows = [["Opt. Step", "Classical", "Annealing"]] 
for step in range(max_steps):
    value_1 = poly_top.volume_fraction_list[step] if step < len(poly_top.volume_fraction_list) else ""
    value_2 = annealing_optimizer.volume_fraction_list[step] if step < len(annealing_optimizer.volume_fraction_list) else ""
    rows.append([step + 1, value_1, value_2]) 
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(rows)

# Create figures.
dV = 1./mesh_p.n_elem
fig, ax = plt.subplots(figsize=(textwidth_in_inches, textwidth_in_inches * aspect_ratio))
ax.set_title('Comparison Volume Fraction')
ax.plot(np.arange(1, len(annealing_optimizer.volume_fraction_list)+1), annealing_optimizer.volume_fraction_list,label='Annealing',marker='s',color='k',zorder=4)
ax.plot(np.arange(1, len(poly_top.volume_fraction_list)+1), poly_top.volume_fraction_list,label='Classical',marker='x',color='gray', linestyle='dashed',zorder=3)

ax.axhline(y=sum(binary_array)/mesh_p.area, label='Classical (filtered)', color='gray', linestyle='dashdot', zorder=3)
ax.axhline(y=volume_fraction_max, label=r'$V_{\mathrm{max}}$', color='k', linestyle='dotted')
x_limits = fig.gca().get_xlim()
x_fill = np.linspace(x_limits[0], x_limits[1], 100)
ax.axhline(y=volume_fraction_max+1*dV, color='lightgray')
ax.axhline(y=volume_fraction_max-1*dV, color='lightgray')
ax.fill_between(x_fill, volume_fraction_max-1*dV, volume_fraction_max+1*dV, color='lightgray', alpha=0.2, label='Binary Resolution', zorder=1)
ax.set_xlim(x_limits)
ax.set_ylim([0.325,0.345])
natural_numbers = np.arange(0, max(len(annealing_optimizer.objective_function_list),len(poly_top.objective_function_list)),2)
plt.xticks(natural_numbers)
ax.set_xlabel('Optimization Step')
ax.set_ylabel(r'Volume Fraction $V_f/|\Omega|$')
ax.legend(loc='best', fontsize='medium')

plt.savefig(os.path.join(output_path, 'comparison_volume_fraction.png'))
tikzplotlib.save(os.path.join(output_path, 'comparison_volume_fraction.tex'))

# Final designs.
file_name = None
tikz = True
title = 'Classical'
file_name = os.path.join(output_path, 'final_design_classical')
mesh_v.plot_element_quantity(binary_array, cmap='gray', cbar=False, title=title, file_name=file_name, tikz=tikz)
file_name = os.path.join(output_path, 'final_design_annealing')
title = 'Annealing'
mesh_v.plot_element_quantity(char_func, cmap='gray', cbar=False, title=title, file_name=file_name, tikz=tikz)