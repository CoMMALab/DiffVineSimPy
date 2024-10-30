from sim.solver import *
from .vine import *
from .sns_render import *
from typing import Callable

torch.set_printoptions(profile = 'full', linewidth = 900, precision = 2)
import time

if __name__ == '__main__':
    draw = False
    ipm = 39.3701 / 1000   # inches per mm
    b1 = [2.5 / ipm, -5.5 / ipm, 3 / ipm, 3 / ipm]
    b2 = [ 1.5/ ipm, -11 / ipm, 3 / ipm, 3 / ipm]
    b3 = [8 / ipm, -4.5 / ipm, 3 / ipm, 3 / ipm]
    b4 = [6 / ipm, -13.5 / ipm, 2 / ipm, 2 / ipm]
    b5 = [11.5 / ipm, -8 / ipm, 2 / ipm, 2 / ipm]
    b6 = [8 / ipm, -9.5 / ipm, 1.5 / ipm, 1.5 / ipm]
    b7 = [600, 0, 5, 5]
    b8 = [600, 0, 5, 5]
    #obstacles = [b1, b2, b3, b4, b5, b6]
    obstacles = [b7, b8]

    for i in range(len(obstacles)):
        obstacles[i][0] -= 0
        obstacles[i][1] -= 0

        obstacles[i][2] = obstacles[i][0] + obstacles[i][2]
        obstacles[i][3] = obstacles[i][1] + obstacles[i][3]

    max_bodies = 40
    init_bodies = 2
    batch_sizes = [1, 1, 1, 1]
    master = []
    for batch_size in batch_sizes:
        data = []
        total_divs = []
        for i in range(2):
            # Control the initial heading of each vine in the batch
            init_headings = torch.full((batch_size, 1), math.radians(-45))
            
            # Add some noise to the initial headings
            init_headings += torch.randn_like(init_headings) * math.radians(10)
            
            
            
            init_x = torch.full((batch_size, 1), 0.0)
            init_y = torch.full((batch_size, 1), 0.0)

            params = VineParams(
                max_bodies,
                obstacles = obstacles,
                grow_rate = 150,
                stiffness_mode='linear',
                stiffness_val = torch.tensor([30_000.0 / 1_000_000.0], dtype = torch.float32) 
                )
            
            # Create empty state arrays with the right shape
            state, dstate = create_state_batched(batch_size, max_bodies)
            bodies = torch.full((batch_size, 1), fill_value = init_bodies)
            
            # Fill the state arrays using init_headings
            init_state_batched(params, state, bodies, init_headings)
            if draw:
                vis_init()
                draw_batched(params, state, bodies)
                plt.pause(0.001)

            forward_batched: Callable = torch.func.vmap(partial(forward_batched_part, params))

            # Measure time per frame
            total_time = 0
            total_frames = 0
            body_count = []
            times = []
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
                    tick = time.time() - start
                    total_time += tick
                    total_frames += 1
                    print('Time per frame: ', total_time / total_frames)
                    body_count.append(bodies.item())
                    times.append(tick)

                if frame % 2 == 0:
                    if draw:
                        draw_batched(params, state, bodies, c='green')
                        plt.pause(0.001)

                if torch.any(bodies >= params.max_bodies):
                    #raise Exception('At least one instance has reached the max body count.')
                    break

                print('===========step end============\n\n')

            counting = False
            parts = [5, 15, 25, 35]
            total = 0
            div = 0
            #print(body_count, times)
            output = []
            divs = []
            for i, num in enumerate(body_count):
                if any(abs(num-x) <= 2 for x in parts):
                    #print(num, counting)
                    if counting:
                        total += times[i]
                        div += 1
                    else:
                        counting = True
                        if div > 0:
                            output.append(total)
                            divs.append(div)
                            total = 0
                            div = 0
                        total += times[i]
                        div += 1
                else:
                    counting = False
            output.append(total)
            divs.append(div)
            data.append(output)
            total_divs.append(divs)
        data = np.sum(np.array(data), axis=0)
        total_divs = np.sum(np.array(total_divs), axis=0)
        data /= total_divs
        master.append(data)
    print(master)
        
