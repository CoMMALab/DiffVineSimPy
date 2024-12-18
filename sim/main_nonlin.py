from sim.solver import *
from .vine import *
from .render import *

torch.set_printoptions(profile = 'full', linewidth = 900, precision = 2)
import time

if __name__ == '__main__':

    ipm = 39.3701 / 600    # inches per mm
    b1 = [5.5 / ipm, -5 / ipm, 4 / ipm, 7 / ipm]
    b2 = [4 / ipm, -17 / ipm, 7 / ipm, 5 / ipm]
    b3 = [13.5 / ipm, -17 / ipm, 8 / ipm, 11 / ipm]
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
    init_bodies = 4
    batch_size = 8
    params = VineParams(max_bodies, init_heading_deg = -45, obstacles = obstacles, grow_rate = 100)

    state, dstate = create_state_batched(batch_size, max_bodies)
    bodies = torch.full((batch_size, 1), fill_value = init_bodies)

    for i in range(state.shape[0]):
        init_state(params, state[i], dstate[i], bodies[i], noise_theta_sigma = 0.4, heading_delta = 0)

    vis_init()
    draw_batched(params, state, dstate, bodies)
    plt.pause(0.001)
    # plt.pause(8)
    # plt.show()

    forward_batched = torch.func.vmap(partial(forward_batched_part, params))

    next_dstate_solution = torch.zeros((
        batch_size,
        params.max_bodies * 3,
        ))

    # Measure time per frame
    total_time = 0
    total_frames = 0

    import theseus as th

    for frame in range(1000):
        start = time.time()

        state = th.Variable(state)
        dstate = th.Variable(dstate)
        bodies = th.Variable(bodies)

        forces, growth, sdf_now, deviation_now, L, J, growth_wrt_state, growth_wrt_dstate \
              = forward_batched(state, dstate, bodies)

        # Convert the values to a shape that qpth can understand
        N = params.max_bodies * 3
        dt = params.dt

        # Compute c
        p = forces * dt - torch.matmul(dstate, params.M)

        # Expand Q to [batch_size, N, N]
        Q = params.M.unsqueeze(0).expand(batch_size, -1, -1)

        # Inequality constraints
        G = -L * dt
        h = sdf_now

        # Equality constraints
        # Compute growth constraint components
        g_con = (
            growth.squeeze(1) - params.grow_rate -
            torch.bmm(growth_wrt_dstate, dstate.unsqueeze(2)).squeeze(2).squeeze(1)
            )
        g_coeff = (growth_wrt_state * dt + growth_wrt_dstate) # [batch_size, 1, 20N]

        # Prepare equality constraints
        A = torch.cat([J * dt, g_coeff], dim = 1)                     # [batch_size, N, N + 9]
        b = torch.cat([-deviation_now, -g_con.unsqueeze(1)], dim = 1) # [batch_size, N]

        # Solve the batched QP problem
        '''
             \hat z = argmin_z 1/2 z^T Q z + p^T z
                        subject to Gz <= h
                                    Az  = b
        '''
        # next_dstate_solution = solve_cvxpy(Q, p, G, h, A, b, solver=cp.SCS, verbose=False, requires_grad=True)
        # next_dstate_solution = solve_qpth(Q, p, G, h, A, b)
        init_layers(
            params.M,
            max_bodies * 3,
            Q.shape[1:],
            p.shape[1:],
            G.shape[1:],
            h.shape[1:],
            A.shape[1:],
            b.shape[1:]
            )
        next_dstate_solution = solve_layers(p, G, h, A, b)

        # Update state and dstate
        state += next_dstate_solution * dt
        dstate = next_dstate_solution

        if frame > 5:
            total_time += time.time() - start
            total_frames += 1
            print('Time per frame: ', total_time / total_frames)

        if frame % 5 == 0:
            draw_batched(params, state, dstate, bodies)
            plt.pause(0.001)

        if torch.any(bodies >= params.max_bodies):
            raise Exception('At least one instance has reached the max body count.')

        print('===========step end============\n\n')

    plt.show()
