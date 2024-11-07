# TopoFlow
This subproject of _EngiOptiQA_ provides a framework to perform topology optimization for fluid flow problems with design updates based on an Ising machine formulation. The latter allows the use of optimization techniques such as GPU-based annealing or (hybrid) quantum annealing (QA).

## Overview
### Code Structure
This project is based on three modules:
   1. A flow solver based on the Finite Element Method ([flow_solver](flow_solver) folder)
   2. An optimization method with design updates either based on gradient information or the Ising machine formulation ([optimizer](optimizer) folder)
   3. The Ising machine formulation as quadratic unconstrained binary optimization (QUBO) problem ([problems](problems) folder)

In addition, _Jupyter_ notebooks for the individual test cases can be found in the [notebooks](notebooks) folder. 
### Obsolete Code
Initially, this repository contained the code Yudai developed in the beginning (October 2023) as a reference ([original](original) folder). The code was extended by Shiori and can be found in the [Lshape16](Lshape16) folder. 
