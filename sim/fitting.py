from functools import partial
import math
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from sim.main import solve
from sim.render import draw_batched, vis_init
from sim.vine import StateTensor, VineParams, create_state_batched, forward, generate_segments_from_rectangles, init_state_batched, zero_out, zero_out_custom
from sim_results import load_vine_robot_csv
import tqdm
torch.set_printoptions(profile = 'full', linewidth = 900, precision = 2)

ipm = 39.3701 / 1000    # inches per mm

def distance(state, bodies, true_state):
    '''
    Given batched BxS (state) and BxS' (true_state),
    compute the shortest Euclidean distance between each point on `state` to `true_state`,
    using only the x, y components in the state (ignoring bodies and other parameters).
    
    Parameters:
    - state: BxS tensor, where each row represents the states for each batch item as [x, y, theta, x2, y2, theta2, ...]
    - true_state: BxS' tensor (fixed-size truth states).
    
    Returns:
    - distances: Bx1 tensor containing the shortest distances.
    '''
    
    # Extract x and y components from state and true_state
    state_x = StateTensor(state).x  # Extracts x values
    state_y =  StateTensor(state).y  # Extracts y values
    true_state_x = true_state[:, ::3]  # Extracts x values from true_state
    true_state_y = true_state[:, 1::3]  # Extracts y values from true_state
    
    # Stack the x and y components along the last dimension to get a 2D point (x, y)
    state_xy = torch.stack((state_x, state_y), dim=-1)  # BxNx2 tensor
    true_state_xy = torch.stack((true_state_x, true_state_y), dim=-1)  # BxMx2 tensor

    # Compute pairwise Euclidean distances between points in `state_xy` and `true_state_xy`
    diff = state_xy[:, :, None, :] - true_state_xy[:, None, :, :]
    distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))  # Euclidean distance
    
    # Find minimum distance for each point in the batch
    min_distances, _ = distances.min(dim=2)
        
    # Zero out
    zero_out_custom(min_distances, bodies)
    
    return min_distances

# def convert_truths(truth_states):
#     # timesteps = truth_states.shape[0]
#     # num_bodies = truth_states.shape[1] // 3
#     # truth_states = truth_states.view(-1, num_bodies, 3)
    
#     # # Take the mean of each body pair
#     # truth_states_mean = (truth_states[:, 0::2, :] + truth_states[:, 1::2, :]) / 2
#     # truth_states_mean = truth_states_mean.view(timesteps, -1)
    
#     # assert truth_states_mean.shape[1] == num_bodies * 3 // 2
    
#     # return truth_states_mean.type(torch.float32)

def convert_to_valid_state(start_x, start_y, state, half_len, max_bodies):
    
    '''
    Given a batch of N points BxN, where N is the sequence (x, y, t, x2, y2, t2, ...),
    convert this to a sequence of points that are exactly d units apart,
    as close as possible to the original sequence.

    Parameters:
    - state: Tensor of shape [B, N], where B is batch size and N is the sequence length.
    - d: Desired distance between points.

    Returns:
    - points: Tensor of shape [B, max_bodies, 3], where M is the number of points that are d units apart.
    - num_bodies_tensor: Tensor of shape [B, 1], the number of bodies computed for each batch item.
    '''
    d = half_len * 2
    B, N = state.shape
    links = 50 # Trust me
    
    points_tensor = torch.zeros((B, max_bodies, 3), dtype=torch.float32)
    bodies = torch.zeros((B, 1), dtype=torch.int64)
    
    points_tensor[:, 0, 0] = start_x.flatten()
    points_tensor[:, 0, 1] = start_y.flatten()

    for b in range(B):
        # plot the truth state
        # plt.figure(2)
        # # Make the color a rainbow
        # cmap = plt.get_cmap('rainbow')
        # plt.cla()
        # plt.scatter(state[b, 0:links*3:3], state[b, 1:links*3:3], c=range(len(state[b, 0:links*3:3])), cmap=cmap)
        # plt.pause(0.001)
        
        # Initialize the starting point
        current_x = start_x[b].item()
        current_y = start_y[b].item()
        current_point = torch.tensor([current_x, current_y])

        total_bodies = 0  # Start with the first point
        i = 0  # Index for iterating through input points

        # Loop until all input points are processed
        while i < links:
            next_x = state[b, i*3].item()
            next_y = state[b, i*3 + 1].item()
            next_point = torch.tensor([next_x, next_y])
            
            # Calculate the vector from the current point to the next point
            delta = next_point - current_point
            segment_length = torch.norm(delta)

            # Skip if the points are too close to avoid division by zero
            if segment_length < 1e-6:
                i += 1
                continue

            # Check if the segment length is greater than or equal to the desired distance
            if segment_length >= d:
                alpha = d / segment_length.item()
                new_point = current_point + alpha * delta
                
                # We cannot add any more
                if total_bodies == max_bodies:
                    break
                
                # Add the segment that is the CENTER of the new and old point
                points_tensor[b, total_bodies, 0:2] = (new_point + current_point) / 2
                points_tensor[b, total_bodies, 2] = math.atan2(delta[1], delta[0])
                
                # Update the current point
                current_point = new_point
                # Increment the total bodies count
                total_bodies += 1
                # Do not increment i to check if more points fit on the same segment
            else:
                # Check next point
                i += 1
        
        # Set points tensor to the center of each body
        
        
        # Check, are there any points we did not add, and have not hit max_size?
        # print('At end', i, num_triplets, total_bodies, max_bodies)
        if i <= links and total_bodies < max_bodies:
            if i == links: i -= 1
            
            # Then add the next one
            next_x = state[b, i*3].item()
            next_y = state[b, i*3 + 1].item()
            points_tensor[b, total_bodies, 0] = next_x
            points_tensor[b, total_bodies, 1] = next_y
            points_tensor[b, total_bodies, 2] = math.atan2(next_y - points_tensor[b, total_bodies-1, 1], 
                                                            next_x - points_tensor[b, total_bodies-1, 0])
            
            total_bodies += 1

        bodies[b] = total_bodies
    
    return points_tensor.view(B, -1), bodies
    
def train(params: VineParams, truth_states):
    '''
    Trains update 
    Args:
        params (VineParams): 
        truth_states (torch.Tensor) [T, 3N]: 
    '''
    
    # train_batch_size = truth_states.shape[0] - 1
    print('Total GT frames', truth_states.shape[0])
    train_batch_size = 200 # FIXME
        
    # Construct initial state
    init_headings = torch.full((train_batch_size, 1), fill_value = -52 * math.pi / 180)
    init_x = torch.full((train_batch_size, 1), 0.0)
    init_y = torch.full((train_batch_size, 1), 0.0)
    
    _, pred_dstate = create_state_batched(train_batch_size, params.max_bodies)
    # bodies = torch.full((model_input_batch, 1), fill_value = init_bodies, dtype=torch.long)
    # init_state_batched(params, pred_state, bodies, init_headings)
    
    # draw_batched(params, state, bodies, lims=False)
    # plt.pause(0.001)
    
    forward_batched = torch.func.vmap(partial(forward, params))
    
    def evolve(init_headings, init_x, init_y, state, dstate, bodies):
        
        # Assert all types are float32
        assert state.dtype == torch.float32
        assert dstate.dtype == torch.float32
        
        bodies, forces, growth, sdf_now, deviation_now, L, J, growth_wrt_state, growth_wrt_dstate \
            = forward_batched(init_headings, init_x, init_y, state, dstate, bodies)
        
        next_dstate_solution = solve(
            params, dstate, forces, growth, sdf_now, deviation_now, L, J, growth_wrt_state, growth_wrt_dstate
            )

        # Update state and dstate
        new_state = state + next_dstate_solution * params.dt
        new_dstate = next_dstate_solution
        
        return new_state, new_dstate, bodies
    
    # Set up training
    params.m.requires_grad_()
    params.I.requires_grad_()
    params.stiffness.requires_grad_()
    params.damping.requires_grad_()
    params.grow_rate.requires_grad_()
    
    optimizer = torch.optim.SGD([params.m, params.I, params.stiffness, params.damping, params.grow_rate], 
                lr = 0.0001)
    
    # Get truth state except for the last frame
    init_states, init_bodies = convert_to_valid_state(init_x, init_y, truth_states[:train_batch_size], params.half_len, params.max_bodies)
    
    # Evolve and visualize the state at truth_states[0]
    # state_now = init_states[0:1]
    # dstate_now = torch.zeros_like(state_now)
    # bodies_now = init_bodies[0:1]
    # for frame in range(0, init_states.shape[0], 13):
    #     # Evolve
    #     state_now, dstate_now, bodies_now = evolve(init_headings[0:1], init_x[0:1], init_y[0:1], state_now, dstate_now, bodies_now)
        
    #     # Loss
    #     distances = distance(state_now, bodies_now, truth_states[frame:frame+1])
    #     print('distances', distances)
    #     loss = distances.mean()
    #     loss.backward()
    #     # optimizer.step()
    #     # state_now.mean().backward()
    #     print(state_now.grad, bodies_now.grad)
    #     print('new M', params.m, params.m.grad)
        
    #     # Draw
    #     state_now = state_now.detach()
    #     bodies_now = bodies_now.detach()
    #     draw_batched(params, init_states[frame:frame+1], init_bodies[frame:frame+1], lims=False, clear=True, obstacles=True)
    #     draw_batched(params, state_now, bodies_now, lims=False, clear=False, obstacles=False)
    #     plt.pause(0.001)
        
    #     # bodies[frame:frame+1]
        
    
    # Train loop
    for iter in tqdm.tqdm(range(100)):
        # On each loop, we feed the truth state into forward, and get the 
        # predicted next state. Then compute loss and backprop
        # Also, for the hidden velocty value, we start from an initial guess of 0,
        # and feed the estimate_dstate as the input to the next batch 
        
        pred_state, pred_dstate, pred_bodies = evolve(init_headings, init_x, init_y, init_states, pred_dstate, init_bodies)
        
        # Set the predicted velocity as the input to the next tiemstep
        pred_dstate[:, 1:] = pred_dstate[:, :-1]
        pred_dstate[:, 0] = 0 # Begin with 0 velocity
        
        # Compute loss
        distances = distance(pred_state, pred_bodies, init_states)
        loss = distances.mean()
        loss.backward()
        
        print('M value', params.m, params.m.grad)
        
        optimizer.step()
        
        iter_to_view = iter % train_batch_size
        draw_batched(params, init_states[None, iter_to_view], init_bodies[None, iter_to_view], lims=False, clear=True, obstacles=True)
        
        # draw_batched(params, pred_state[None, iter_to_view].detach(), pred_bodies[None, iter_to_view].detach(), 
        #             lims=False, clear=True, obstacles=True)
                    
                    
        plt.xlim([-0.1, 1])
        plt.ylim([-1, 0.1])
        plt.pause(0.001)
        # plt.show()
        
        
        # sim_states.append(state[0])
        # bodies_over_time.append(bodies[0])
                
    # sim_states = torch.stack(sim_states, dim = 0)
        
    
if __name__ == '__main__':

    # Directory containing the CSV files
    directory = './sim_output'     # Replace with your actual directory
    
    states = []
    scenes = []
    
    num_to_read = 4
    
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

    # x = 248.8899876704202
    # y = 356.2826607726847
    # width = 30.0
    # height = 502.2832775276208

    # obstacles = [[x - width / 2, y - height / 2, x + width / 2, y + height]]

    params = VineParams(
        max_bodies = max_bodies,
        obstacles = [[0, 0, 0, 0]],
        grow_rate = -1,
        )

    params.half_len = 3.0 / ipm / 1000 / 2
    params.radius = 15.0 / 2 / ipm / 1000 * 0.1
    params.m = torch.tensor([0.002], dtype = torch.float32)
    params.I = torch.tensor([5], dtype = torch.float32)
    params.stiffness = torch.tensor([30_000.0], dtype = torch.float32)
    params.damping = torch.tensor([10.0], dtype = torch.float32)
    params.grow_rate = torch.tensor([100.0 / ipm / 1000], dtype = torch.float32)

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
        # idx = [i + j for i in range(0, state.shape[1], 6) for j in range(3)]
        
        truth_states = torch.from_numpy(state)
        T = state.shape[0]
        truth_bodies = 25 * 2
        
        # Train, using the given params as config and (for optimzied vars)
        # initial guess. Then truth states should be a TxS trajectory over T timesteps
        # and fixed-size states of size S. Make sure dt matches
        train(params, truth_states)
                
        # for frame in range(T):       
        #     if frame % 40 != 0:     
        #         continue
            
        #     draw_batched(params, truth_states[None, frame, :], [nbodies], lims=False)
            
        #     draw_batched(params, sim_states[None, frame, :], bodies_over_time[frame], lims=False)
            
        #     plt.pause(0.001)
