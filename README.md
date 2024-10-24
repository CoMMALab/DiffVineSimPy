# Differentiable Vine Simulator

This is a differentiable forward-dynamics solver for an extendable, soft vine robot. Due to the high DoF of such a robot and complex propagation of forces through the body, we find that naive force-based methods are not sufficent for a stable simulation. Instead, we use a QP solver to find the global optimum next state based on a set of explicit constraints and energy-minimization formulas. This gurantees that the simulated robot will never be in impossible states like penertrating an obstacle, stretch, break apart, or exhibit jitters common to force-based sims.

This work is based on https://github.com/charm-lab/Vine_Simulator. 

Also uses [sqrtm](https://github.com/steveli/pytorch-sqrtm)

## Install
Make sure you have torch and qpth in your python environment.

## Run

```bash
python -m sim.main
```
## TODO

Change to use extension (think about it. friction? bending? batchable?)
   - Consider a small crack. There should be real strong friction here, vine will not slide backwards
        So using the constant model will not work
   - Using extendo model means we can change the resolution by increasing/decreasing resolution
   - Using constant model means easy batching and gradients
   
So we need to batch no matter what. The state vector MUST be constant size.
We want to compute d x_{t+1} / d x_t  and  d x_{t+1} / d params
   - Some d values will be invalid. small -> large means some jacobian will be zero
       That's fine as long as the graph is properly built
   - During forward sim, need to track the uninited ones


lqp_py

arbitrary start angle and position

parameter fitting on sim
- how to deal with hidden velocity
- choose nonlinear optimizer

## Tuning considerations

iters,
tol_rel


in cone_program.py: mode: (optional) Which mode to compute derivative with, options are
          ["dense", "lsqr", "lsmr"].