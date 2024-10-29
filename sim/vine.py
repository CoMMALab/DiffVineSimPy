import math
import torch
import functorch
from torch.autograd.functional import jacobian
import cvxpy as cp
from functools import partial

from sim.solver import init_layers, solve_layers


def finite_changes(h, init_val):
    h_rel = torch.empty_like(h)
    h_rel[0] = h[0] - init_val
    h_rel[1:] = h[1:] - h[:-1]

    return h_rel


def generate_segments_from_rectangles(obstacles):
    '''
    obstacles: tensor of shape (num_rectangles, 4), each row is [x1, y1, x2, y2]
    Returns:
        segments: tensor of shape (num_rectangles * 4, 4), each row is [x_start, y_start, x_end, y_end]
    '''
    x1 = obstacles[:, 0]
    y1 = obstacles[:, 1]
    x2 = obstacles[:, 2]
    y2 = obstacles[:, 3]

    # Side 1: bottom edge
    starts1 = torch.stack([x1, y1], dim = 1)
    ends1 = torch.stack([x2, y1], dim = 1)

    # Side 2: right edge
    starts2 = torch.stack([x2, y1], dim = 1)
    ends2 = torch.stack([x2, y2], dim = 1)

    # Side 3: top edge
    starts3 = torch.stack([x2, y2], dim = 1)
    ends3 = torch.stack([x1, y2], dim = 1)

    # Side 4: left edge
    starts4 = torch.stack([x1, y2], dim = 1)
    ends4 = torch.stack([x1, y1], dim = 1)

    # Stack all starts and ends
    starts = torch.cat([starts1, starts2, starts3, starts4], dim = 0) # shape (num_rectangles * 4, 2)
    ends = torch.cat([ends1, ends2, ends3, ends4], dim = 0)           # shape (num_rectangles * 4, 2)

    # Combine starts and ends into segments
    segments = torch.cat([starts, ends], dim = 1)  # shape (num_rectangles * 4, 4)

    return segments


def dist2segments(points, segments):
    '''
    points: tensor of shape (num_points, 2)
    segments: tensor of shape (num_segments, 4)
    Returns:
        min_distances: tensor of shape (num_points,)
        closest_points: tensor of shape (num_points, 2)
        min_indices: tensor of shape (num_points,)  # indices of the closest segments
    '''
    # Extract starts and ends
    starts = segments[:, :2]               # (num_segments, 2)
    ends = segments[:, 2:]                 # (num_segments, 2)
    AB = ends - starts                     # (num_segments, 2)
    AB_dot = torch.sum(AB * AB, dim = 1)   # (num_segments,)

    # Expand dimensions for broadcasting
    points_expanded = points[:, None, :]   # (num_points, 1, 2)
    starts_expanded = starts[None, :, :]   # (1, num_segments, 2)
    AB_expanded = AB[None, :, :]           # (1, num_segments, 2)
    AB_dot_expanded = AB_dot[None, :]      # (1, num_segments)

    # Compute AP
    AP = points_expanded - starts_expanded     # (num_points, num_segments, 2)

    # Compute numerator: AP ⋅ AB
    numerator = torch.sum(AP * AB_expanded, dim = 2) # (num_points, num_segments)

    # Compute t
    t = numerator / AB_dot_expanded    # (num_points, num_segments)
    t_clamped = torch.clamp(t, 0.0, 1.0)

    # Compute closest points
    C = starts_expanded + t_clamped[..., None] * AB_expanded # (num_points, num_segments, 2)

    # Compute distance vectors
    distance_vectors = C - points_expanded            # (num_points, num_segments, 2)
    distances = torch.norm(distance_vectors, dim = 2) # (num_points, num_segments)

    # Find minimum distances and indices
    min_distances, min_indices = torch.min(distances, dim = 1)        # (num_points,)
    closest_points = C[torch.arange(points.shape[0]), min_indices, :] # (num_points, 2)

    return min_distances, closest_points


def isinside(points, obstacles):
    '''
    Determines if each point is inside any of the given rectangular obstacles.
    Args:
        points (torch.Tensor): A tensor of shape (N, 2) representing N points with (x, y) coordinates.
        obstacles (torch.Tensor): A tensor of shape (M, 4) representing M rectangular obstacles with 
                                  (x_min, y_min, x_max, y_max) coordinates.
    Returns:
        torch.Tensor: A boolean tensor of shape (N,) where each element is True if the corresponding 
                      point is inside any of the obstacles, and False otherwise.    
    '''
    x_min = obstacles[:, 0].unsqueeze(0)   # (1, M)
    y_min = obstacles[:, 1].unsqueeze(0)   # (1, M)
    x_max = obstacles[:, 2].unsqueeze(0)   # (1, M)
    y_max = obstacles[:, 3].unsqueeze(0)   # (1, M)

    point_x = points[:, 0].unsqueeze(1)    # (N, 1)
    point_y = points[:, 1].unsqueeze(1)    # (N, 1)

    # Check if points are within the bounds of any obstacle (batched comparison)
    inside = (point_x <= x_max) & (point_x >= x_min) & (point_y <= y_max) & (point_y >= y_min) # (N, M)

    return inside.any(dim = 1)     # (N,), True if inside any obstacle


class StateTensor:
    '''
    Helpful wrapper around a position or velocity state vector, providing
    convenience functions for getting x, y, theta data
    '''

    def __init__(self, tensor):
        if isinstance(tensor, StateTensor):
            raise ValueError("You passed a StateTensor to construct a StateTensor")

        self.tensor = tensor
        if tensor.shape[-1] % 3 != 0:
            raise ValueError(f"Tensor size {tensor.shape} must be 3N")

    # Note the following are views
    @property
    def x(self):
        return self.tensor[..., 0::3]

    @property
    def y(self):
        return self.tensor[..., 1::3]

    @property
    def theta(self):
        return self.tensor[..., 2::3]


class VineParams:
    '''
    Time indepedent parameters.
    TODO make tensors and differentiable
    '''

    def __init__(self, max_bodies, obstacles = [], grow_rate = 10.0):
        self.max_bodies = max_bodies
        self.dt = 1 / 90               # 1/90  # Time step
        self.radius = 25.0 / 2         # Uncompressed collision radius of each body
        self.grow_rate = grow_rate     # Length grown per unit time

        # Robot parameters
        self.m = 0.01  # 0.002  # Mass of each body
        self.I = 10    # Moment of inertia of each body
        self.half_len = 9

        # Stiffness and damping coefficients
        self.stiffness = 15_000.0 / 1_000_1000  # 30_000.0  # Stiffness coefficient (too large is instable!)
        self.damping = 50.0        # Damping coefficient (too large is instable!)
        self.vel_damping = 0.1     # Damping coefficient for velocity (too large is instable!)

        # Environment obstacles (rects only for now)
        self.obstacles = obstacles

        # Make sure x < x2 y < y2 for rectangles
        for i in range(len(self.obstacles)):
            # Swap x
            if self.obstacles[i][0] > self.obstacles[i][2]:
                self.obstacles[i][0], self.obstacles[i][2] = self.obstacles[i][2], self.obstacles[i][0]
            # Swap y
            if self.obstacles[i][1] > self.obstacles[i][3]:
                self.obstacles[i][1], self.obstacles[i][3] = self.obstacles[i][3], self.obstacles[i][1]

        # Make a tensor containing a flattened list of segments of shape  4NxN
        self.obstacles = torch.tensor(self.obstacles)
        self.segments = generate_segments_from_rectangles(self.obstacles)


def create_M(m, I, max_bodies):
    # Update mass matrix M (block diagonal)
    diagonal_elements = torch.cat([m, m, I]).repeat(max_bodies)
    return torch.diag(diagonal_elements)   # Shape: (nq, nq))


def create_state_batched(batch_size, max_bodies):
    state = torch.zeros(batch_size, max_bodies * 3)
    dstate = torch.zeros(batch_size, max_bodies * 3)
    return state, dstate


def init_state_batched(params: VineParams, state, bodies, init_headings) -> StateTensor:
    '''
    Create vine state vectors from params
    '''
    batch_size = state.shape[0]
    state = StateTensor(state)

    # Init first body position
    for i in range(batch_size):
        state.theta[i, :] = init_headings[i]

    state.x[:, 0] = params.half_len * torch.cos(state.theta[:, 0])
    state.y[:, 0] = params.half_len * torch.sin(state.theta[:, 0])

    # Init all body positions
    for batch_i in range(batch_size):
        for i in range(1, bodies[i]):
            lastx = state.x[batch_i, i - 1]
            lasty = state.y[batch_i, i - 1]
            lasttheta = state.theta[batch_i, i - 1]
            thistheta = state.theta[batch_i, i]

            state.x[
                batch_i,
                i] = lastx + params.half_len * torch.cos(lasttheta) + params.half_len * torch.cos(thistheta)
            state.y[
                batch_i,
                i] = lasty + params.half_len * torch.sin(lasttheta) + params.half_len * torch.sin(thistheta)


def zero_out(state, bodies):
    state = StateTensor(state)

    idx = torch.arange(state.x.shape[-1])
    mask = idx >= bodies   # Mask of uninited values

    state.x[:] = torch.where(mask, 0, state.x)
    state.y[:] = torch.where(mask, 0, state.y)
    state.theta[:] = torch.where(mask, 0, state.theta)


def zero_out_custom(state, bodies):
    idx = torch.arange(state.shape[-1])
    mask = idx >= bodies   # Mask of uninited values
    state[:] = torch.where(mask, 0, state)


def sdf(params: VineParams, state, bodies):
    """
    Computes the signed distance function (SDF) for given points and obstacles.
    Args:
        params:
        state: vine state N
        bodies: num bodies per batch item
    Returns:
        tuple: 
            - min_dist (torch.Tensor): N distance from body collider to nearest obstacle
            - contact_forces (torch.Tensor): Nx2 tensor of forces accounting for squishiness of the vine
    """

    state = StateTensor(state)

    points = torch.stack((state.x, state.y), dim = -1)
    min_dist, min_contactpts = dist2segments(points, params.segments)

    inside_points = isinside(points, params.obstacles)

    min_dist = torch.where(inside_points, -min_dist, min_dist)

    # Make sdf_now negative when uncompressible body penetrates
    min_dist -= params.radius

    zero_out_custom(min_dist, bodies)

    return min_dist, min_contactpts


def joint_deviation(params: VineParams, init_x, init_y, state: torch.Tensor, bodies):

    # Vector of deviation per-joint, [x y x2 y2 x3 y3],
    # where each coordinate pair is the deviation with the last body
    length = state.shape[-1] // 3
    constraints = state.new_zeros(length * 2)

    state = StateTensor(state)

    x = state.x
    y = state.y
    theta = state.theta

    constraints[0] = (x[0] - init_x) - params.half_len * torch.cos(theta[0])
    constraints[1] = (y[0] - init_y) - params.half_len * torch.sin(theta[0])

    constraints[2::2] = (x[1:] - x[:-1]) - params.half_len * torch.cos(theta[1:]) \
                                         - params.half_len * torch.cos(theta[:-1])

    constraints[3::2] = (y[1:] - y[:-1]) - params.half_len * torch.sin(theta[1:]) \
                                         - params.half_len * torch.sin(theta[:-1])

    # Last body is special. It has a sliding AND rotation joint with the second-last body
    endx = x[bodies - 2] + params.half_len * torch.cos(theta[bodies - 2])
    endy = y[bodies - 2] + params.half_len * torch.sin(theta[bodies - 2])

    angle_diff = torch.atan2(y[bodies - 1] - endy, x[bodies - 1] - endx) - theta[bodies - 1]

    # So if we have 4 bodies, we have constraints: [x y x y x y dtheta]
    #                                               0 1 2 3 4 5 6
    # So the last constraint index is 6 = (bodies-1)*2
    constraints[(bodies - 1) * 2] = angle_diff

    zero_out_custom(constraints, bodies * 2 - 1)

    return constraints


def bending_energy(params: VineParams, theta_rel, dtheta_rel, bodies):
    # Compute the response (like potential energy of bending)
    # Can think of the system as always wanting to get rid of potential
    # Generally, \tau = - stiffness * benderino - damping * d_benderino
    
    # FIXME Stiffness gets a constant so the grads are balanced with the rest of the parameters
    # bend = -1 * 1_000_000 * params.stiffness.abs() * theta_rel - params.damping.abs() * dtheta_rel
    
    # buckling bending
    # FIXME Comment out to switch bending modes
    
    print('input theta_rel', theta_rel.shape, theta_rel.unsqueeze(-1).shape)
    print('output', params.stiffness_func(theta_rel.unsqueeze(-1)).shape)
    bend = -1 * 1_000_000 * params.stiffness_func(theta_rel.unsqueeze(-1)).squeeze() - params.damping.abs() * dtheta_rel
    
    
    # bend = -1 * theta_rel.sign() * params.stiffness * (0.5 - (1.5 * theta_rel.abs() - 0.7)**2) - params.damping * dtheta_rel
    # bend = -1 * theta_rel.sign() * 1 * params.stiffness * torch.log(theta_rel.abs()*2 + 1) - params.damping * dtheta_rel

    zero_out_custom(bend, bodies)

    return bend
    # return -self.stiffness * torch.where(torch.abs(theta) < 0.3, theta * 2, theta * 0.5) - self.damping * dtheta
    # return torch.sign(theta) * -(torch.sin((torch.abs(theta) + 0.1) / 0.3) + 1.4)


def extend(params: VineParams, state, dstate, bodies):
    state = StateTensor(state)
    dstate = StateTensor(dstate)

    new_i = bodies
    last_i = bodies - 1
    penult_i = bodies - 2

    # Compute position of second last seg
    endingx = state.x[penult_i] + params.half_len * torch.cos(state.theta[penult_i])
    endingy = state.y[penult_i] + params.half_len * torch.sin(state.theta[penult_i])

    # Compute last body's distance
    last_link_distance = ((state.x[last_i] - endingx)**2 + \
                          (state.y[last_i] - endingy)**2).sqrt().squeeze(-1)

    # x2 to prevent 0-len segments
    extend_needed = last_link_distance > params.half_len * 2

    # Compute location of new seg
    last_link_theta = torch.atan2(state.y[last_i] - state.y[penult_i], state.x[last_i] - state.x[penult_i])

    new_seg_x = endingx + params.half_len * torch.cos(last_link_theta)
    new_seg_y = endingy + params.half_len * torch.sin(last_link_theta)
    new_seg_theta = last_link_theta.squeeze()

    # Copy last body one forward
    state.x[new_i] = torch.where(extend_needed, state.x[last_i], state.x[new_i])
    state.y[new_i] = torch.where(extend_needed, state.y[last_i], state.y[new_i])
    state.theta[new_i] = torch.where(extend_needed, state.theta[last_i], state.theta[new_i])

    # Copy last body vel too
    # dstate.x[new_i] = torch.where(extend_needed, dstate.x[last_i], dstate.x[new_i])
    # dstate.y[new_i] = torch.where(extend_needed, dstate.y[last_i], dstate.y[new_i])
    # dstate.theta[new_i] = torch.where(extend_needed, dstate.theta[last_i], dstate.theta[new_i])

    dstate.x[new_i] = 0
    dstate.y[new_i] = 0
    dstate.theta[new_i] = 0

    # Set the new segment position
    state.x[last_i] = torch.where(extend_needed, new_seg_x, state.x[last_i])
    state.y[last_i] = torch.where(extend_needed, new_seg_y, state.y[last_i])
    state.theta[last_i] = torch.where(extend_needed, new_seg_theta, state.theta[last_i])

    # Set the new segment to have velocity of the former tip
    # dstate.x[last_i] = torch.where(extend_needed, dstate.x[penult_i], dstate.x[last_i])
    # dstate.y[last_i] = torch.where(extend_needed, dstate.y[penult_i], dstate.y[last_i])
    # dstate.theta[last_i] = torch.where(extend_needed, dstate.theta[penult_i], dstate.theta[last_i])

    bodies = bodies + extend_needed

    zero_out(state.tensor, bodies)
    zero_out(dstate.tensor, bodies)
    
    # FIXME return copies, don't mutate
    return bodies


def growth_rate(params: VineParams, state, dstate, bodies):

    state = StateTensor(state)
    dstate = StateTensor(dstate)
    id1 = bodies - 2
    id2 = bodies - 1

    assert state.tensor.shape == dstate.tensor.shape

    # Now return the constraints for growing the last segment
    x1 = state.x[id1]
    y1 = state.y[id1]
    vx1 = dstate.x[id1]
    vy1 = dstate.y[id1]

    x2 = state.x[id2]
    y2 = state.y[id2]
    vx2 = dstate.x[id2]
    vy2 = dstate.y[id2]

    constraint = ((x2 - x1) * (vx2 - vx1) + (y2 - y1) * (vy2 - vy1)) / \
                  torch.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return constraint


def forward_batched_part(params: VineParams, init_heading, init_x, init_y, state, dstate, bodies):
    '''
    Compute some jacobians about this state wrt forces, growth, sdf_now
    
    This function is separated from the QP solving stuff in forward() because it later
    gets compiled into a batched function using vmap, but we can't also compile the QP stuff
    '''
    
    bodies = extend(params, state, dstate, bodies)

    # Jacobian of SDF with respect to x and y
    L, contact_forces = torch.func.jacrev(partial(sdf, params), has_aux = True)(state, bodies)
    sdf_now, contact_forces = sdf(params, state, bodies)

    # Jacobian of joint deviation wrt state
    J = torch.func.jacrev(partial(joint_deviation, params, init_x, init_y))(state, bodies)
    deviation_now = joint_deviation(params, init_x, init_y, state, bodies)

    # Jacobian of growth rate wrt state
    growth_wrt_state, growth_wrt_dstate = torch.func.jacrev(partial(growth_rate, params), argnums=(0, 1))(state, dstate, bodies)
    growth = growth_rate(params, state, dstate, bodies)

    # Stiffness forces
    theta_rel = finite_changes(StateTensor(state).theta, init_heading)
    dtheta_rel = finite_changes(StateTensor(dstate).theta, 0.0)

    bend_energy = bending_energy(params, theta_rel, dtheta_rel, bodies)

    forces = StateTensor(torch.zeros_like(state))

    # Apply unbending forces
    forces.theta[:] += -bend_energy        # Apply bend energy as torque to own joint
    forces.theta[:-1] += bend_energy[1:]   # Apply bend energy as torque to joint before

    # FIXME sign flipped somewhere
    forces.x[:] += params.vel_damping * StateTensor(dstate).x
    forces.y[:] += params.vel_damping * StateTensor(dstate).y

    forces = forces.tensor

    return bodies, forces, growth, sdf_now, deviation_now, L, J, growth_wrt_state, growth_wrt_dstate


forward_batched = torch.func.vmap(forward_batched_part, in_dims = (None, 0, 0, 0, 0, 0, 0))


def solve(
        params: VineParams,
        dstate,
        forces,
        growth,
        sdf_now,
        deviation_now,
        L,
        J,
        growth_wrt_state,
        growth_wrt_dstate
    ):
    '''
    Given some physics data (batched) about the current vine state, find the next state by reexpressing the problem as a QP
    '''
    # Convert the values to a shape that qpth can understand
    N = params.max_bodies * 3
    dt = params.dt
    M = create_M(params.m.abs(), params.I.abs(), params.max_bodies)

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


def forward(params: VineParams, init_headings, init_x, init_y, state, dstate, bodies):
    '''
    Convenience function that takes an batched input state/dstate, and returns the next state/dstate (also batched)
    '''
    # Assert all types are float32
    assert state.dtype == torch.float32
    assert dstate.dtype == torch.float32

    bodies, forces, growth, sdf_now, deviation_now, L, J, growth_wrt_state, growth_wrt_dstate \
        = forward_batched(params, init_headings, init_x, init_y, state, dstate, bodies)

    next_dstate_solution = solve(
        params, dstate, forces, growth, sdf_now, deviation_now, L, J, growth_wrt_state, growth_wrt_dstate
        )

    # Update state and dstate
    new_state = state + next_dstate_solution * params.dt
    new_dstate = next_dstate_solution

    return new_state, new_dstate, bodies
