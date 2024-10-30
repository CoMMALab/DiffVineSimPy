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
    batch_sizes = [1]
    master_mean = []
    master_var = []
    for batch_size in batch_sizes:
        parts = [5, 15, 25, 35]
        output = [[],[],[],[]]
        for i in range(10):
            # Control the initial heading of each vine in the batch
            init_headings = torch.full((batch_size, 1), math.radians(-45))
            
            # Add some noise to the initial headings
            init_headings += torch.randn_like(init_headings) * math.radians(10)
            
            
            
            init_x = torch.full((batch_size, 1), 0.0)
            init_y = torch.full((batch_size, 1), 0.0)

            params = VineParams(
                max_bodies,
                obstacles = obstacles,
                grow_rate = 150 / 1000,
                stiffness_mode='linear',
                stiffness_val = torch.tensor([30_000.0 / 100_000.0]) 
                )
            
            assert params.stiffness_val.dtype == torch.float32
            assert params.m.dtype == torch.float32
            assert params.grow_rate.dtype == torch.float32
            assert params.I.dtype == torch.float32
            
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
            try:
                for frame in range(1000):
                    start = time.time()
                    # Calculate dynamics
                    bodies, forces, growth, sdf_now, deviation_now, L, J, growth_wrt_state, growth_wrt_dstate = \
                        forward_batched(init_headings, init_x, init_y, state, dstate, bodies)

                    # Call solve and handle potential exceptions
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
            except KeyboardInterrupt:
                    # Re-raise the KeyboardInterrupt exception to allow Ctrl+C to stop the program
                    print("Keyboard interrupt detected. Exiting loop.")
                    raise
            except Exception as e:
                print(f"Error in solve(): {e}. Restarting loop iteration.")
                continue  # Restart loop iteration on error
            for i, num in enumerate(body_count):
                for j, x in enumerate(parts):
                    if abs(num-x) <= 2:
                        output[j].append(times[i])
        means = []
        vars = []
        for n_bod in output:
            n_bod = np.array(n_bod)
            mean = np.mean(n_bod, axis=0)
            var = np.var(n_bod, axis=0)
            means.append(mean)
            vars.append(var)
        master_mean.append(means)
        master_var.append(vars)
    print(master_mean, master_var)
    np.save('means.npy', np.array(master_mean))
    np.save('vars.npy', np.array(master_var))
        
