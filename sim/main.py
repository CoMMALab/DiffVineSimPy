from sim.solver import *
from .vine import *
from .render import *
from typing import Callable

torch.set_printoptions(profile = 'full', linewidth = 900, precision = 2)
import time


def solve(
        params: VineParams, dstate, forces, growth, sdf_now, deviation_now, L, J, growth_wrt_state, growth_wrt_dstate
    ):
    # Convert the values to a shape that qpth can understand
    N = params.max_bodies * 3
    dt = params.dt
    M = create_M(params.m, params.I, params.max_bodies)

    # Compute c
    p = forces * dt - torch.matmul(dstate, M)

    # Expand Q to [batch_size, N, N]
    Q = M
    # Q = params.M.unsqueeze(0).expand(batch_size, -1, -1)

    # Inequality constraints
    G = -L * dt
    h = sdf_now

    # Equality constraints
    # Compute growth constraint components
    g_con = (
        growth.squeeze(1) - params.grow_rate -
        torch.bmm(growth_wrt_dstate, dstate.unsqueeze(2)).squeeze(2).squeeze(1)
        )
    g_coeff = (growth_wrt_state * dt + growth_wrt_dstate)

    # Prepare equality constraints
    A = torch.cat([J * dt, g_coeff], dim = 1)
    b = torch.cat([-deviation_now, -g_con.unsqueeze(1)], dim = 1) # [batch_size, N]

    init_layers(N, Q.shape, p.shape[1:], G.shape[1:], h.shape[1:], A.shape[1:], b.shape[1:])
    next_dstate_solution = solve_layers(Q, p, G, h, A, b)

    return next_dstate_solution


if __name__ == '__main__':

    ipm = 39.3701 / 1000    # inches per mm
    b1 = [5.5 / ipm, -5 / ipm, 4 / ipm, 7 / ipm]
    b2 = [4 / ipm, -17 / ipm, 7 / ipm, 5 / ipm]
    b3 = [15.5 / ipm, -17 / ipm, 8 / ipm, 11 / ipm]
    b4 = [13.5 / ipm, -30 / ipm, 4 / ipm, 5 / ipm]
    b5 = [20.5 / ipm, -30 / ipm, 4 / ipm, 5 / ipm]
    b6 = [13.5 / ipm, -30 / ipm, 10 / ipm, 1 / ipm]

    obstacles = [b1, b2, b3, b4, b5, b6]

    for i in range(len(obstacles)):
        obstacles[i][0] -= 20
        obstacles[i][1] -= 0

        obstacles[i][2] = obstacles[i][0] + obstacles[i][2]
        obstacles[i][3] = obstacles[i][1] + obstacles[i][3]

    max_bodies = 40
    init_bodies = 2
    batch_size = 1

    init_headings = torch.full((batch_size, 1), math.radians(-20))
    init_headings += torch.randn_like(init_headings) * math.radians(0)
    init_x = torch.full((batch_size, 1), 0.0)
    init_y = torch.full((batch_size, 1), 0.0)

    params = VineParams(
        max_bodies,
        obstacles = obstacles,
        grow_rate = 150,
        )

    state, dstate = create_state_batched(batch_size, max_bodies)
    bodies = torch.full((batch_size, 1), fill_value = init_bodies)

    init_state_batched(params, state, bodies, init_headings)

    vis_init()
    draw_batched(params, state, bodies)
    plt.pause(0.001)

    forward_batched: Callable = torch.func.vmap(partial(forward, params))

    # Measure time per frame
    total_time = 0
    total_frames = 0

    # Init optimization targets
    params.m = torch.tensor([params.m], dtype = torch.float32, requires_grad = True)
    params.I = torch.tensor([params.I], dtype = torch.float32, requires_grad = True)

    for frame in range(1000):
        start = time.time()

        forces, growth, sdf_now, deviation_now, L, J, growth_wrt_state, growth_wrt_dstate \
              = forward_batched(init_headings, init_x, init_y, state, dstate, bodies, )

        next_dstate_solution = solve(
            params, dstate, forces, growth, sdf_now, deviation_now, L, J, growth_wrt_state, growth_wrt_dstate
            )

        # Update state and dstate
        state += next_dstate_solution.detach() * params.dt
        dstate = next_dstate_solution.detach()

        if frame > 5:
            total_time += time.time() - start
            total_frames += 1
            print('Time per frame: ', total_time / total_frames)

        if frame % 2 == 0:
            draw_batched(params, state, bodies)
            plt.pause(0.001)

        if torch.any(bodies >= params.max_bodies):
            raise Exception('At least one instance has reached the max body count.')

        print('===========step end============\n\n')

    plt.show()
