from functools import partial
import math
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from sim.main import solve
from sim.render import draw_batched, vis_init
from sim.vine import VineParams, create_state_batched, forward, generate_segments_from_rectangles, init_state_batched
from sim_results import load_vine_robot_csv
import tqdm
torch.set_printoptions(profile = 'full', linewidth = 900, precision = 2)

ipm = 39.3701 / 1000    # inches per mm

if __name__ == '__main__':

    # Directory containing the CSV files
    directory = './sim_output'     # Replace with your actual directory
    
    states = []
    scenes = []
    
    num_to_read = 10
    
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            print(f"Reading file: {filename}")

            # Load vine robot data from CSV
            # RECTS ARE xywh HERE
            data, rects = load_vine_robot_csv(filepath)
            
            states.append(data)
            scenes.append(rects)
            
            num_to_read -= 1
            if num_to_read == 0:
                break

    # init heading = 1.31
    # diam = 24.0
    # d = 16.845
    # stiffness 50000.0
    # damping 10
    # M 0.001
    # I 200
    # bodies 10

    max_bodies = 25 * 2
    init_bodies = 2
    batch_size = 1

    # x = 248.8899876704202
    # y = 356.2826607726847
    # width = 30.0
    # height = 502.2832775276208

    # obstacles = [[x - width / 2, y - height / 2, x + width / 2, y + height]]

    params = VineParams(
        max_bodies = max_bodies,
        obstacles = [[0, 0, 0, 0]],
        grow_rate = 8 / ipm / 1000 * 1.5,
        )

    params.m = torch.tensor([0.002], dtype = torch.float32, requires_grad = False)
    params.I = torch.tensor([5], dtype = torch.float32, requires_grad = False)
    params.half_len = 3.0 / ipm / 1000 / 2
    params.radius = 15.0 / 2 / ipm / 1000 * 0.1
    params.stiffness = torch.tensor([30_000.0], dtype = torch.float32, requires_grad = True)
    params.damping = torch.tensor([10.0], dtype = torch.float32, requires_grad = True)

    

    # Render the loaded data
    vis_init()

    for state, scene in zip(states, scenes):
        # Extract the scene
        for i in range(len(scene)):
            # Convert xywh to xyxy
            scene[i] = list(scene[i])
            scene[i][2] = scene[i][0] + scene[i][2]
            scene[i][3] = scene[i][1] + scene[i][3]
        
        params.obstacles = torch.tensor(scene)
        params.segments = generate_segments_from_rectangles(params.obstacles)
        
        # Index of 0,1,2, 6,7,8, 12,13,14, ...
        idx = [i + j for i in range(0, state.shape[1], 6) for j in range(3)]
        
        truth_states = torch.from_numpy(state)
        truth_states = truth_states[:, idx]
        T = state.shape[0]
        nbodies = 25
                
        # Now do sim rollout
        forward_batched = torch.func.vmap(partial(forward, params))
        
        # Init
        init_headings = torch.full((batch_size, 1), fill_value = -52 * math.pi / 180)
        init_x = torch.full((batch_size, 1), 0.0)
        init_y = torch.full((batch_size, 1), 0.0)
        
        state, dstate = create_state_batched(batch_size, max_bodies)
        bodies = torch.full((batch_size, 1), fill_value = init_bodies)
        init_state_batched(params, state, bodies, init_headings)
        
        # Result
        sim_states = []
        bodies_over_time = []
        
        draw_batched(params, state, bodies, lims=False)
        plt.pause(0.001)
        
        for frame in tqdm.tqdm(range(T)):
            
            forces, growth, sdf_now, deviation_now, L, J, growth_wrt_state, growth_wrt_dstate \
                = forward_batched(init_headings, init_x, init_y, state, dstate, bodies, )

            next_dstate_solution = solve(
                params, dstate, forces, growth, sdf_now, deviation_now, L, J, growth_wrt_state, growth_wrt_dstate
                )

            # Update state and dstate
            new_state = state + next_dstate_solution * params.dt
            dstate = next_dstate_solution.detach()
            
            loss = truth_states[None, frame, :] - new_state
            
            state = new_state.detach()
            
            draw_batched(params, 
                        torch.cat([state, truth_states[None, frame, :]]),
                        (bodies[0][0], nbodies), lims=False)
                        
            plt.xlim([-0.1, 1])
            plt.ylim([-1, 0.1])
            plt.pause(0.001)
            # plt.show()
            
            sim_states.append(state[0])
            bodies_over_time.append(bodies[0])
                    
        sim_states = torch.stack(sim_states, dim = 0)
        
        # for frame in range(T):       
        #     if frame % 40 != 0:     
        #         continue
            
        #     draw_batched(params, truth_states[None, frame, :], [nbodies], lims=False)
            
        #     draw_batched(params, sim_states[None, frame, :], bodies_over_time[frame], lims=False)
            
        #     plt.pause(0.001)
