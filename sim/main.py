from sim.solver import solve_cvxpy, solve_qpth
from .vine import *
from .render import *

import lovely_tensors as lt
# lt.monkey_patch()

if __name__ == '__main__':
    
    ipm = 39.3701/600 # inches per mm
    b1 = [5.5/ipm,-5/ipm,4/ipm,7/ipm]
    b2 = [4/ipm,-17/ipm,7/ipm,5/ipm]
    b3 = [13.5/ipm,-17/ipm,8/ipm,11/ipm]
    b4 = [13.5/ipm,-30/ipm,4/ipm,5/ipm]
    b5 = [20.5/ipm,-30/ipm,4/ipm,5/ipm]
    b6 = [13.5/ipm,-30/ipm,10/ipm,1/ipm]
    
    obstacles = [
            b1, b2, b3, b4, b5, b6
        ]
    
    for i in range(len(obstacles)):
        obstacles[i][0] -= 20
        obstacles[i][1] -= 0
        
        obstacles[i][2] = obstacles[i][0] + obstacles[i][2]
        obstacles[i][3] = obstacles[i][1] + obstacles[i][3]
    
    max_bodies = 60
    init_bodies= 4
    batch_size = 6
    params = VineParams(max_bodies, init_heading_deg=-45, obstacles=obstacles, grow_rate=250) 
    
    state, dstate = create_state_batched(batch_size, max_bodies)
    bodies = torch.full((batch_size,1), fill_value=init_bodies)
    
    for i in range(state.shape[0]):
        init_state(params, state[i], dstate[i], bodies[i], noise_theta_sigma=0.4, heading_delta=0)
    
    vis_init()
    draw(params, state, dstate, bodies)
    plt.pause(0.001)
    # plt.pause(8)
    # plt.show()
    
    # FIXME torch.compile(
    forward_batched =  torch.func.vmap(partial(forward, params))
    
    next_dstate_solution = torch.zeros((batch_size, params.max_bodies * 3,))
    
    for frame in range(1000):
        forces, growth, sdf_now, deviation_now, L, J, growth_wrt_state, growth_wrt_dstate \
              = forward_batched(state, dstate, bodies)
        
        # Solving step

        N = params.max_bodies * 3
        batch_size = state.shape[0]
        dt = params.dt

        # Compute c
        p = forces * dt - torch.matmul(dstate, params.M)

        # Expand Q to [batch_size, N, N]
        eps_eye = + 1e-4 * torch.eye(N)
        Q = params.M

        # Inequality constraints
        G = -L * dt
        h = sdf_now

        # Equality constraints
        # Compute growth constraint components
        g_con = (growth.squeeze(1) - params.grow_rate - 
                torch.bmm(growth_wrt_dstate, dstate.unsqueeze(2)).squeeze(2).squeeze(1))
        g_coeff = (growth_wrt_state * dt + growth_wrt_dstate).squeeze(1)

        # Prepare equality constraints
        A = torch.cat([J * dt, g_coeff.unsqueeze(1)], dim=1)
        b = torch.cat([-deviation_now, -g_con.unsqueeze(1)], dim=1)

        # Solve the batched QP problem
        next_dstate_solution = solve_qpth(Q, p, G, h, A, b)
        
        # Update state and dstate
        state += next_dstate_solution * dt
        dstate = next_dstate_solution
        
        draw(params, state, dstate, bodies)
        
        if torch.any(bodies >= params.max_bodies):
            raise Exception('At least one instance has reach the max bodies.')
        
        plt.pause(0.0001)
        # print('===========step end============\n\n')
        
    plt.show()