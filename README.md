# Differentiable Vine Simulator

This is a differentiable forward-dynamics solver for an extendable, soft vine robot. Due to the high DoF of such a robot and complex propagation of forces through the body, we find that naive force-based methods are not sufficent for a stable simulation. Instead, we use a QP solver to find the global optimum next state based on a set of explicit constraints and energy-minimization formulas. This gurantees that the simulated robot will never be in impossible states like penertrating an obstacle, stretch, break apart, or exhibit jitters common to force-based sims.

This work is based on https://github.com/charm-lab/Vine_Simulator. 

Also uses [sqrtm](https://github.com/steveli/pytorch-sqrtm)

## Install
Make sure you have torch and cvxpylayers in your python environment.

## Run

To see the simulation rollout with some default params and scene, run this. It's mainly for debugging changes in the sim itself
```bash
python -m sim.main
```

To do fitting, run: (It uses the test rollouts in `sim_results`, but the full dataset of 500 (1.3G) rollouts need to be uploaded somewhere)
```bash
python -m sim.fitting
```

During fitting you can run `tensorboard --logdir=runs` to see tensorboard, but I recommend using the vscode integration, the button above `import tensorboard` opens it as a tab in vscode

## Notes

For the fitting, the starting legnth is different so that makes it have a constant offset throughout. Actually I have no idea why.

Also the magnitude of gradients (and by extension param values) matters a lot

To try: different start guesses

Different LR/clipping params

nonlinear buckling

torch.optim.LBFGS?

Witht he hidden velocity variable: Not as convex as we thought!