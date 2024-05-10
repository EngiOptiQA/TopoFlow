# TopoFlow
This subproject of _EngiOptiQA_ provides a framework to perform topology optimization for fluid flow problems with annealing methods such as GPU-based annealing or hybrid quantum annealing (QA).

## Overview
### Code Structure
This project is based on three modules:
   1. A flow solver based on the Finite Element Method ([flow_solver](flow_solver) folder)
   2. An optimization method either based on gradient information or an update found by annealing ([optimizer](optimizer) folder)
   3. The problem formulation as quadratic unconstrained binary optimization (QUBO) problem ([problems](problems) folder)

In addition, _Jupyter_ notebooks for the individual test cases can be found in the [notebooks](notebooks) folder. 
### Obsolete Code
Initially, this repository contained the code Yudai developed in the beginning (October 2023) as a reference ([original](original) folder). The code was extended by Shiori and can be found in the [Lshape16](Lshape16) folder. 
