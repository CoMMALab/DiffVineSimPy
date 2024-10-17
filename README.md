# Differentiable Vine Simulator

This is a differentiable forward-dynamics solver for an extendable, soft vine robot. Due to the high DoF of such a robot and complex propagation of forces through the body, we find that naive force-based methods are not sufficent for a stable simulation. Instead, we use a QP solver to find the global optimum next state based on a set of explicit constraints and energy-minimization formulas. This gurantees that the simulated robot will never be in impossible states like penertrating an obstacle, stretch, break apart, or exhibit jitters common to force-based sims.

This work is based on https://github.com/charm-lab/Vine_Simulator. 

## Install
Make sure you have torch and qpth in your python environment.

## Run

```bash
python -m sim.main
```