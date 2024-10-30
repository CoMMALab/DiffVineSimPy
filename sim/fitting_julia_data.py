import math
from matplotlib import pyplot as plt
import numpy as np
import torch
import h5py
from sim.render import draw_batched, vis_init
from sim.vine import VineParams

torch.set_printoptions(profile = 'full', linewidth = 900, precision = 2)

if __name__ == '__main__':

    # Open the HDF5 file
    with h5py.File('fitting_problem_data.h5', 'r') as f:
        # Load each dataset directly into NumPy arrays and convert to PyTorch tensors
        A = torch.from_numpy(np.array(f['A']))
        U = torch.from_numpy(np.array(f['U']))
        n_opt = torch.from_numpy(np.array(f['n_opt']))
        z0 = torch.from_numpy(np.array(f['z0']))

        # Load and stack x_vid and y_vid
        x_vid = torch.from_numpy(
            np.stack([np.array(f['x_vid'][f'x_vid_{i+1}']) for i in range(len(f['x_vid']))])
            )
        y_vid = torch.from_numpy(
            np.stack([np.array(f['y_vid'][f'y_vid_{i+1}']) for i in range(len(f['y_vid']))])
            )

    # Define problem dimensions
    T = U.shape[0]     # Time steps
    nq = 30            # Size of configuration vector
    print('Time steps:', T)
    print(f'A: {A.shape}')

    # Calculate indices for extracting q, v, r, s, stiffness, and damping
    q_idx = [(t - 1) * (2 * nq) + torch.arange(nq) for t in range(1, T + 1)]
    v_idx = [(t - 1) * (2 * nq) + nq + torch.arange(nq) for t in range(1, T + 1)]
    r_idx = [2 * nq * T + (t - 1) * nq + torch.arange(nq) for t in range(1, T)]
    s_idx = [2 * nq * T + nq * (T - 1) + (t - 1) * nq + torch.arange(nq) for t in range(1, T)]

    # Extract q, v, r, s from z0
    q = torch.stack([z0[idx] for idx in q_idx])
    v = torch.stack([z0[idx] for idx in v_idx])
    # r = torch.stack([z0[idx] for idx in r_idx])
    # s = torch.stack([z0[idx] for idx in s_idx])

    q = q.view(T, -1)
    v = v.view(T, -1)

    # Extract stiffness and damping from the end of z0
    stiffness = 1000 * z0[-2]
    damping = z0[-1]

    # Print extracted components to verify
    print("q shape:", q.shape)
    print("v shape:", v.shape)
    print("stiffness:", stiffness)
    print("damping:", damping)

    # init heading = 1.31
    # diam = 24.0
    # d = 16.845
    # stiffness 50000.0
    # damping 10
    # M 0.001
    # I 200
    # bodies 10

    max_bodies = 10
    init_bodies = 2
    batch_size = 1

    x = 248.8899876704202
    y = 356.2826607726847
    width = 30.0
    height = 502.2832775276208

    obstacles = [[x - width / 2, y - height / 2, x + width / 2, y + height]]

    params = VineParams(
        max_bodies = 10,
        obstacles = obstacles,
        grow_rate = 8 * 10 / 1000,
        )

    params.m = torch.tensor([0.001], dtype = torch.float32, requires_grad = False)
    params.I = torch.tensor([200], dtype = torch.float32, requires_grad = False)
    params.half_len = 16.845
    params.radius = 24.0 / 2
    params.stiffness = torch.tensor([50000.0], dtype = torch.float32, requires_grad = True)
    params.damping = torch.tensor([10], dtype = torch.float32, requires_grad = True)

    # Init
    init_headings = torch.full((batch_size, 1), fill_value = 1.31)
    init_headings += torch.randn_like(init_headings) * math.radians(0)
    init_x = torch.full((batch_size, 1), 0.0)
    init_y = torch.full((batch_size, 1), 0.0)

    # Render the loaded data
    vis_init()

    plt.pause(0.001)
    for frame in range(T):
        print(f'Frame {frame}')
        draw_batched(params, q[None, frame], [10], lims=False)
        plt.pause(0.001)
