from sim.solver import solve_cvxpy, solve_qpth
from .vine import *
from .render import *
import osqpth
import lovely_tensors as lt
# lt.monkey_patch()
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

    max_bodies = 7
    init_bodies = 4
    batch_size = 6
    params = VineParams(max_bodies, init_heading_deg = -45, obstacles = obstacles, grow_rate = 250)

    state, dstate = create_state_batched(batch_size, max_bodies)
    bodies = torch.full((batch_size, 1), fill_value = init_bodies)

    for i in range(state.shape[0]):
        init_state(params, state[i], dstate[i], bodies[i], noise_theta_sigma = 0.4, heading_delta = 0)

    vis_init()
    draw(params, state, dstate, bodies)
    plt.pause(0.001)
    # plt.pause(8)
    # plt.show()

    # torch.compile() does absolutely nothing
    forward_batched = torch.func.vmap(partial(forward, params))

    next_dstate_solution = torch.zeros((
        batch_size,
        params.max_bodies * 3,
        ))

    # Measure time per frame
    total_time = 0
    total_frames = 0

    for frame in range(1000):
        start = time.time()

        forces, growth, sdf_now, deviation_now, L, J, growth_wrt_state, growth_wrt_dstate \
              = forward_batched(state, dstate, bodies)

        # Convert the values to a shape that qpth can understand
        N = params.max_bodies * 3
        dt = params.dt

        # Compute c
        p = forces * dt - torch.matmul(dstate, params.M)

        # Expand Q to [batch_size, N, N]
        Q = params.M   # .unsqueeze(0).expand(batch_size, -1, -1)

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

        print('J', J.shape)
        print('g_coeff', g_coeff.shape)
        print('A', A.shape)

        # A += 1e-4 * torch.eye(A.shape[2]).unsqueeze(0).expand_as(A)

        # det_G = torch.linalg.det(G[0])
        # if torch.any(det_G.abs() < 1e-10):
        #     print("G is singular or ill-conditioned.")

        try:
            m = torch.linalg.lu_factor(Q)
        except:
            print('Q is not positive definite')
        else:
            print('Q is positive definite')

        for i in range(batch_size):
            rank_A = torch.linalg.matrix_rank(A[i])
            if rank_A < A[i].shape[1]:
                print(f"Batch {i}: Matrix A is rank deficient. Rank: {rank_A}, Expected: {A[i].shape[1]}")

        # Solve the batched QP problem
        '''
        \hat z =   argmin_z 1/2 z^T Q z + p^T z
                        subject to Gz <= h
                                    Az  = b
        '''
        # next_dstate_solution = solve_qpth(Q, p, G, h, A, b)

        # Define epsilon for approximating equality constraints
        epsilon = 1e-3
        inf = torch.tensor(float('inf'), device = h.device)

        # Combine inequality and equality constraints
        A_osqp = torch.cat([G, A], dim = 1)    # [batch_size, m_total, N]
        l_osqp = torch.cat([-inf * torch.ones_like(h), b - epsilon], dim = 1)
        u_osqp = torch.cat([h, b + epsilon], dim = 1)

        # Prepare P_val, q_val, A_val
        P_val = Q.flatten().unsqueeze(0).repeat(batch_size, 1)
        q_val = p  # [batch_size, N]
        A_val = A_osqp.reshape(batch_size, -1)
        l_val = l_osqp
        u_val = u_osqp

        # Get indices and shapes for P and A
        P_shape = Q.shape
        A_shape = A_osqp[0].shape  # Assuming same shape across batch

        # Prepare P_idx and A_idx (since we are using dense matrices, indices can be None)
        P_idx = None
        A_idx = None

        # Initialize OSQP solver
        solver = OSQP(
            P_idx,
            P_shape,
            A_idx,
            A_shape,
            eps_rel = 1e-5,
            eps_abs = 1e-5,
            verbose = False,
            max_iter = 10000,
            diff_mode = DiffModes.ACTIVE
            )

        # Solve the QP problem
        next_dstate_solution = solver(P_val, q_val, A_val, l_val, u_val)

        # Update state and dstate
        state += next_dstate_solution * dt
        dstate = next_dstate_solution

        if frame > 5:
            total_time += time.time() - start
            total_frames += 1
            # print('Time per frame: ', total_time / total_frames)

        if frame % 2 == 0:
            draw(params, state, dstate, bodies)
            plt.pause(0.001)

        if torch.any(bodies >= params.max_bodies):
            raise Exception('At least one instance has reach the max bodies.')

        print('===========step end============\n\n')

    plt.show()
