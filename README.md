# Differentiable Vine Simulator

A differentiable forward-dynamics simulator for an extendable, soft vine robot. This code release accompanies our RoboSoft paper _"Physics-Grounded Differentiable Simulation for Soft Growing Robots"_. In this repo, we also provide the complete code for fitting our vine model to real vine trials, as well as the code for extracting vine positions from video data.

## Install
Make sure you have torch in your python environment.

Install the dependencies in `requirements.txt`

Alternatively, there is a conda `environment.yml` but it is unlikely to work due to specific cuda driver versions.

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

## File-by-file

`sim/main.py`: Simulates and displays a simple rollout with hardcoded obstacles. This is good for if you're getting started or making tweaks to the physics themselves

`paper_vis.py` Script to generate the timing benchmark figure.

`simulated_data`: Contains code and data relating to generating an using simulated data from the other [vine simulator](https://github.com/charm-lab/Vine_Simulator). The file `gen_rects.py` creates a dataset of random rects, which can be fed into the other sim, and `sim_results` renders the sim rollouts. Some sample rollouts have been provided in `sim_output`

`videoparser` Code for converting videos of real trials into frame-by-frame vine positions (as a sequence of points) as well as obstacle positions (as a set of line segments). _Check the `videoparser/README.md` file in there for more details_
- `data` is a directory of trial videos as well as extracted vine segmentations. `sim_out` are the vine positions from each simulated rollout
- `framer.py` converts videos into frames, performs the right homography tranformation to align the testbed surface onto the corners of the frame, optical flow, segmentation, and centerline extraction for the vine.
- `classifier` is an early version of framer where we tried other segmentation methods, like k-means color thresholding, which didn't work because color is too variable to be the only discriminating feature.
- `processor.py` Manual parts of labelling the obstacles and workspace boundaries.
- `betterprocessor.py` Takes simulated vine rollouts and overlays them onto the real video, to generate certain figures in the paper. You can see these results as the png images in this directory.

`sim` The simulator and fitting code itself. There are a bunch of variants which we used for the trials in the paper. However, they are based on the structure in `fitting_real.py`
- `vine.py` The core simulator code. Defines vine parameters and state. Defines `evolve()` function, which takes in a state (position and velocity), then solves the QP to generate the next state.
- `solver.py` Called from `vine.py` and performs the actual QP solving. There are a bunch of QP solvers here, batched, unbatch, gradients, no gradients. 
- `render.py` Vine rendering code. Also has a sns variant.
- `fitting_*` These files do fitting on real data (stuff from video parser), data from the other simulator, and a [small dataset](https://github.com/charm-lab/Vine_Simulator) we found in the repo.
- `read_yitian` Small converter from videoparser outputs to a format usable by `vine.py`.
- `sqrtm` sqrtm implementation from [this](https://github.com/steveli/pytorch-sqrtm)
- `test_*` Various tests.

`models` Trained MLPs for out bending model. All of them work pretty much the same, but model_360_good is a bit better

`goodruns` Tensorboard logs of good fitting sessions. For referencing what the loss curves should look like.

