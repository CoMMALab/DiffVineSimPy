from sim.solver import *
from .vine import *
from .render import *
from typing import Callable
from .fitting_real import read_yitian

torch.set_printoptions(profile = 'full', linewidth = 900, precision = 2)
# torch.autograd.set_detect_anomaly(True)

import time

if __name__ == '__main__':

    ipm = 39.3701 / 1000   # inches per mm
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

    max_bodies = 70
    init_bodies = 2
    batch_size = 1
    
    # Control the initial heading of each vine in the batch
    init_headings = torch.full((batch_size, 1), math.radians(-20))
    
    # Add some noise to the initial headings
    init_headings += torch.randn_like(init_headings) * math.radians(10)
    
    init_x = torch.full((batch_size, 1), 0.0)
    init_y = torch.full((batch_size, 1), 0.0)

    params = VineParams(
        max_bodies = max_bodies,
        obstacles = [[0, 0, 0, 0]],
        grow_rate = -1,
        stiffness_mode = 'linear', # 'linear' or 'mlp' or 'real
        stiffness_val = torch.tensor([30_000.0 / 100_000.0], dtype = torch.float32)
        )

    # Initial guess values
    params.half_len = 5
    params.radius = 7
    params.m = torch.tensor([0.0313], dtype = torch.float32)
    params.I = torch.tensor([0.1691], dtype = torch.float32)
    params.stiffness = torch.tensor([30_000.0 / 100_000.0], dtype = torch.float32)
    params.damping = torch.tensor(.18, dtype = torch.float32) / 100
    params.grow_rate = torch.tensor(0.1647, dtype = torch.float32)
    params.sicheng = torch.tensor(1, dtype=torch.float32)
    
    # Load MLP from weights
    if params.stiffness_mode == 'mlp':
        print('Loading MLP weights from models/model_360_good.pt')
        params.stiffness_func.load_state_dict(torch.load('models/model_360_good.pt', weights_only=True))
        
    assert params.stiffness_val.dtype == torch.float32
    assert params.m.dtype == torch.float32
    assert params.grow_rate.dtype == torch.float32
    assert params.I.dtype == torch.float32
    
    # Create empty state arrays with the right shape
    state, dstate = create_state_batched(batch_size, max_bodies)
    bodies = torch.full((batch_size, 1), fill_value = init_bodies)
    
    # Fill the state arrays using init_headings
    init_state_batched(params, state, bodies, init_headings)

    vis_init()
    draw_batched(params, state, bodies)
    plt.pause(0.001)

    forward_batched: Callable = torch.func.vmap(partial(forward_batched_part, params))

    # Measure time per frame
    total_time = 0
    total_frames = 0
    truth_states, truth_bodies, scene = read_yitian(3)

    # Convert the obstacles to segments in a form we can use later
    params.obstacles = None
    params.segments = torch.concat((scene[:, 0, :], scene[:, 1, :]), axis = 1)
    for frame in range(1000):
        start = time.time()

        bodies, forces, growth, sdf_now, deviation_now, L, J, growth_wrt_state, growth_wrt_dstate \
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

        if frame % 1 == 0:
            draw_batched(params, state, bodies, c='blue')
            plt.gcf().set_size_inches(10, 10)
            plt.pause(0.001)

        if torch.any(bodies >= params.max_bodies):
            raise Exception('At least one instance has reached the max body count.')

        print('===========step end============\n\n')

    plt.show()
