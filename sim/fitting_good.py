import os
import math
import torch

from sim.vine import StateTensor, VineParams, create_state_batched, forward_batched_part, forward, generate_segments_from_rectangles, init_state_batched, zero_out, zero_out_custom
from sim_results import load_vine_robot_csv
from sim.solver import sqrtm_module

from matplotlib import pyplot as plt
from sim.render import draw_batched, vis_init
from torch.utils.tensorboard import SummaryWriter

torch.set_printoptions(profile = 'full', linewidth = 900, precision = 2)
# torch.autograd.set_detect_anomaly(True)

ipm = 39.3701 / 1000   # inches per mm

class MutableInt:
    def __init__(self, value):
        self.value = value
        
def distance(state, bodies, true_state, true_bodies):
    '''
    Given batched BxS (state) and BxS' (true_state),
    compute the shortest Euclidean distance from each point on `state` to `true_state`,
    using only the x, y components in the state (ignoring theta).

    Parameters:
    - state: B x S tensor, where each row represents the states for each batch item as [x, y, theta, x2, y2, theta2, ...]
    - bodies: Bx1 tensor, number of valid bodies in each state batch item
    - true_state: B x S' tensor (fixed-size truth states).
    - true_bodies: Bx1 tensor, number of valid bodies in each true_state batch item

    Returns:
    - min_distances: B x 1 tensor containing the mean distances from each state.
    '''

    # Constants for large finite values
    LARGE_POSITIVE = 1e9
    LARGE_NEGATIVE = -1e9

    # Extract x and y components from state and true_state
    state_x = StateTensor(state).x             # B x N
    state_y = StateTensor(state).y             # B x N
    true_state_x = StateTensor(true_state).x   # B x M
    true_state_y = StateTensor(true_state).y   # B x M

    # Get batch size and maximum number of bodies
    B, N = state_x.shape
    _, M = true_state_x.shape

    # Create masks for valid bodies
    idx_state = torch.arange(N).unsqueeze(0)       # 1 x N
    idx_true_state = torch.arange(M).unsqueeze(0)  # 1 x M

    state_mask = idx_state < bodies                # B x N
    true_state_mask = idx_true_state < true_bodies # B x M

    # Assign large values to invalid bodies in state
    state_x = state_x.masked_fill(~state_mask, LARGE_POSITIVE)
    state_y = state_y.masked_fill(~state_mask, LARGE_POSITIVE)

    # Assign large values to invalid bodies in true_state
    true_state_x = true_state_x.masked_fill(~true_state_mask, LARGE_NEGATIVE)
    true_state_y = true_state_y.masked_fill(~true_state_mask, LARGE_NEGATIVE)

    # Stack the x and y components
    state_xy = torch.stack((state_x, state_y), dim = -1)                # B x N x 2
    true_state_xy = torch.stack((true_state_x, true_state_y), dim = -1) # B x M x 2

    # Compute pairwise Euclidean distances
    diff = state_xy[:, :, None, :] - true_state_xy[:, None, :, :] # B x N x M x 2
    distances = torch.linalg.vector_norm(diff, dim = -1)          # B x N x M

    # Find minimum distance for each point in state (over true_state points)
    min_distances, _ = distances.min(dim = 2)  # B x N

    # Calculate average distance per batch, ignoring LARGE_POSITIVE entries
    avg_distance = min_distances[min_distances < LARGE_POSITIVE / 16].mean()

    return avg_distance    # [1] tensor


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
    links = 50     # Trust me

    points_tensor = torch.zeros((B, max_bodies, 3), dtype = torch.float32)
    bodies = torch.zeros((B, 1), dtype = torch.int64)

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

        total_bodies = 0   # Start with the first point
        i = 0              # Index for iterating through input points

        # Loop until all input points are processed
        while i < links:
            next_x = state[b, i * 3].item()
            next_y = state[b, i * 3 + 1].item()
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
            last_x = points_tensor[b, total_bodies - 1, 0]
            last_y = points_tensor[b, total_bodies - 1, 1]
            next_x = state[b, i * 3].item()
            next_y = state[b, i * 3 + 1].item()
            points_tensor[b, total_bodies, 0] = next_x
            points_tensor[b, total_bodies, 1] = next_y
            points_tensor[b, total_bodies, 2] = math.atan2(
                next_y - last_y,
                next_x - last_x
                )

            total_bodies += 1

        bodies[b] = total_bodies

    return points_tensor.view(B, -1), bodies


def train(params: VineParams, truth_states, optimizer, writer, mutable_iter):
    '''
    Trains one batch with shared params.obstacles 
    Args:
        params (VineParams): 
        truth_states (torch.Tensor) [T, 3N]: 
    '''

    # train_batch_size = truth_states.shape[0] - 1
    train_batch_size = 50     # FIXME I made the batch smaller for testing

    # Construct initial state
    init_headings = torch.full((train_batch_size, 1), fill_value = -52 * math.pi / 180)
    init_x = torch.full((train_batch_size, 1), 0.0)
    init_y = torch.full((train_batch_size, 1), 0.0)
    
    # Create empty pred_dstate so we can save it across iterations
    _, pred_dstate = create_state_batched(train_batch_size, params.max_bodies)
    
    print('Converting raw data')
    
    # Convert the ground truth data (arbitrarily spaced points) to a sequence of valid vine states
    true_states, true_nbodies = convert_to_valid_state(init_x, init_y, truth_states[:train_batch_size], params.half_len, params.max_bodies)

    print('Begin loop')
    
    # Train loop
    for _ in range(10000000):
        iter = mutable_iter.value
        # On each loop, we feed the truth state into forward, and get the
        # predicted next state. Then compute loss and backprop
        # Also, for the hidden velocty value, we start from an initial guess of 0,
        # feed this into forward(), and then use the predictions as the 'ground truth' for the
        # next iteration. This is really crude and I'm not sure it works but will do for now

        optimizer.zero_grad()
        # sqrtm_module.

        # HACK: Detach the prior iteration so gradients won't depend on non-existant parts of the computational graph
        pred_dstate = pred_dstate.detach().clone()
        true_states = true_states.detach().clone()
        true_nbodies = true_nbodies.detach().clone()
        
        pred_state, pred_dstate, pred_bodies = forward(params, init_headings, init_x, init_y,
                true_states, pred_dstate, true_nbodies)

        # Set the predicted velocity as the input to the next timestep
        pred_dstate[:, 1:] = pred_dstate[:, :-1]
        pred_dstate[:, 0] = 0  # Time=0 has 0 velocity by definition

        # Compute loss (note prediction for idx should be compared to truth at idx+1)
        distances = distance(pred_state[:-1], pred_bodies[:-1], true_states[1:], true_nbodies[1:])

        loss = distances

        loss.backward()
        
        # Custom coefficients for grads
        # Determined with sweat and tears
        params.m.grad *= 0.01
        params.I.grad *= 2e6
        params.damping.grad *= 1e6
        params.stiffness.grad *= 1
        params.grow_rate.grad *= 3e4
        
        # Clip gradients, FIXME sometimes the grads explode for no reason
        # Set a bit conservative, so worst case it takes a bit longer but won't explode
        torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm = 1e-2, norm_type = 2)

        # Debug see if grads are reasonable
        print(f"{'Parameter':<15}{'Value':<20}{'Gradient':<20}")
        print("-" * 55)

        print(f"{'M':<15}{params.m.item():<20.11f}{params.m.grad.item():<20.11f}")
        print(f"{'I':<15}{params.I.item():<20.11f}{params.I.grad.item():<20.11f}")
        print(f"{'Stiffness':<15}{params.stiffness.item():<20.11f}{params.stiffness.grad.item():<20.11f}")
        print(f"{'Damping':<15}{params.damping.item():<20.11f}{params.damping.grad.item():<20.11f}")

        # FIXME This has no grads, I think i know why but wontfix for now
        print(f"{'Grow Rate':<15}{params.grow_rate.item():<20.11f}{params.grow_rate.grad.item():<20.11f}")

        # Log loss
        writer.add_scalar('Loss/train', loss.item(), iter)

        # Log parameters
        writer.add_scalar('Parameters/M', params.m.item(), iter)
        writer.add_scalar('Parameters/I', params.I.item(), iter)
        writer.add_scalar('Parameters/Stiffness', params.stiffness.item(), iter)
        writer.add_scalar('Parameters/Damping', params.damping.item(), iter)
        writer.add_scalar('Parameters/Grow_Rate', params.grow_rate.item(), iter)

        # Log gradients
        writer.add_scalar('Gradients/M_grad', params.m.grad.item(), iter)
        writer.add_scalar('Gradients/I_grad', params.I.grad.item(), iter)
        writer.add_scalar('Gradients/Stiffness_grad', params.stiffness.grad.item(), iter)
        writer.add_scalar('Gradients/Damping_grad', params.damping.grad.item(), iter)
        writer.add_scalar('Gradients/Grow_Rate_grad', params.grow_rate.grad.item(), iter)

        optimizer.step()

        # Every step, we'll visualize a different batch item
        idx_to_view = (iter * 9) % (train_batch_size - 1)

        # Visualize the ground truth at idx+1
        draw_batched(
            params,
            true_states[None, idx_to_view + 1],
            true_nbodies[None, idx_to_view + 1],
            clear = True,
            obstacles = True,
            c = 'g'
            )

        # Visualize the predicted state _from_ idx, essentially idx + 1
        draw_batched(
            params,
            pred_state[None, idx_to_view].detach(),
            pred_bodies[None, idx_to_view].detach(),
            clear = False,
            obstacles = False,
            c = 'b'
            )

        plt.xlim([-0.1, 1])
        plt.ylim([-1, 0.1])
        plt.title(f'Comparing truth (green) and pred (blue) for t = {idx_to_view}')
        plt.pause(0.001)

        print(f'End of iter {iter}, loss {loss.item()}\n')
        
        mutable_iter.value += 1

    # Evolve and visualize the state starting from truth_states[0]
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


if __name__ == '__main__':
    # Good runs
    # Oct29_00-16-51_Seele
    # Oct29_01-54-14_Seele
    
    # Directory containing the CSV files
    directory = './sim_output'     # Replace with your actual directory

    filenames = []

    # Get files sorted in number order
    files = os.listdir(directory)
    files_sorted = sorted(files, key = lambda x: int(x.split('_')[1].split('.')[0]))

    # Load all the data into `states`` and `scenes``
    for filename in files_sorted:
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            filenames.append(filename)

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

    # Initial guess values
    # params.half_len = 3.0 / ipm / 1000 / 2
    # params.radius = 15.0 / 2 / ipm / 1000 * 0.05
    # params.m = torch.tensor([0.002], dtype = torch.float32)
    # params.I = torch.tensor([5], dtype = torch.float32)
    # params.stiffness = torch.tensor([30_000.0 / 1_000_000.0], dtype = torch.float32)
    # params.damping = torch.tensor([10.0], dtype = torch.float32)
    # params.grow_rate = torch.tensor([100.0 / ipm / 1000], dtype = torch.float32)
    
    params.half_len = 3.0 / ipm / 1000 / 2
    params.radius = 15.0 / 2 / ipm / 1000 * 0.05
    params.m = 2 * torch.tensor([0.002], dtype = torch.float32)
    params.I = 2 * torch.tensor([5], dtype = torch.float32)
    params.stiffness = 0.5 * torch.tensor([30_000.0 / 1_000_000.0], dtype = torch.float32)
    params.damping = 2 * torch.tensor([10.0], dtype = torch.float32)
    
    # Note to tuners setting this low cheats the loss function 
    params.grow_rate = torch.tensor([100.0 / ipm / 1000], dtype = torch.float32)
    
    # Declare optimization variables
    params.m.requires_grad_()
    params.I.requires_grad_()
    params.stiffness.requires_grad_()
    params.damping.requires_grad_()
    params.grow_rate.requires_grad_()
    
    # Declare a 2-layer MLP for stiffness
    # Takes 1 scalat input and outputs 1 scalar output
    params.stiffness_func = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.Tanh(),
        torch.nn.Linear(10, 1)
        )
        
    # Initialize the weights with xavier normal
    for layer in params.stiffness_func:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)    
        
    # Set up optimizer
    optimizer_params = [params.m, params.I, params.stiffness, params.damping, params.grow_rate]
    
    # FIXME Not sure if adam is working for or against us, but everything is tuned with this set up
    # so we'll stick with it for now
    optimizer = torch.optim.AdamW(optimizer_params, lr = 1e-3, betas = (0.8, 0.95), weight_decay = 0)
    # optimizer = torch.optim.LBFGS(optimizer_params, lr = 1e-4, max_iter = 20, history_size = 10, line_search_fn = 'strong_wolfe')
    # optimizer = torch.optim.SGD(optimizer_params, lr = 1e-4, weight_decay = 0)
    
    # Set up tensorboard
    writer = SummaryWriter()
    
    vis_init()
    
    # Step count
    mutable_iter = MutableInt(0)

    # Run training separately for each scene
    # train() only works if all the data share obstacles
    # TODO In the future we may rewrite the code to support different obstacles within a batch
    for filename in filenames:
        print(f"Loading file: {filename}")

        # Load vine robot data from CSV
        # RECTS ARE xywh HERE
        state, scene = load_vine_robot_csv(filepath)
           
        # Extract the scene
        for i in range(len(scene)):
            # Convert xywh to xyxy
            scene[i] = list(scene[i])
            scene[i][2] = scene[i][0] + scene[i][2]
            scene[i][3] = scene[i][1] + scene[i][3]
                    
        # Convert the obstacles to segments in a form we can use later
        params.obstacles = torch.tensor(scene)
        params.segments = generate_segments_from_rectangles(params.obstacles)

        truth_states = torch.from_numpy(state)
        
        # Train, using the given params as config and (for optimzied vars)
        # initial guess. Then truth states should be a TxS trajectory over T timesteps
        # and fixed-size states of size S. Make sure dt matches
        train(params, truth_states, optimizer, writer, mutable_iter)
        
    # Close tensorboard writer
    writer.close()
