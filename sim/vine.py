import math
import torch
from torch.autograd.functional import jacobian
import cvxpy as cp
from functools import partial

# def jacobian_batched(func, input);
#     return functorch.vmap(functorch.jacrev(func))(inputs)
    
def finite_changes(h, init_val):
    h_rel = torch.empty_like(h)
    h_rel[0] = h[0] - init_val
    h_rel[1:] = h[1:] - h[:-1] 
    
    return h_rel

def dist2seg(x, y, start, end):
    '''
    Given any shape x and y points, return closest dist to the seg and normal
    '''
    points = torch.stack([x, y], dim=-1) # [..., 2]
    
    # Segment start and end as tensors
    p1 = torch.tensor(start, dtype=torch.float32)
    p2 = torch.tensor(end, dtype=torch.float32)
    
    # Vector from p1 to p2 (the segment direction)
    segment = p2 - p1                   # [2]
    
    # Vector from p1 to each point
    p1_to_points = points - p1          # [..., 2]
    
    # Project the point onto the segment, calculate the parameter t
    segment_length_squared = torch.dot(segment, segment) # [1]
    t = torch.clamp(torch.sum(p1_to_points * segment, dim=-1) / segment_length_squared, 0.0, 1.0) #[...]
    
    # print('p1_to_points * segment', (p1_to_points * segment).shape)
    # print('t', t.shape)
    
    # Closest point on the segment to the point
    closest_point_on_segment = p1 + t[..., None] * segment # [..., 2]
        
    # Distance from each point to the closest point on the segment
    distance_vectors = closest_point_on_segment - points  # [..., 2]
    distances = torch.norm(distance_vectors, dim=-1)      # [...]
    
    # Normalize the distance vectors to get normal vectors
    # normals = torch.nn.functional.normalize(distance_vectors, dim=1)
    normals = distance_vectors
    
    return distances, closest_point_on_segment
    
def dist2rect(x, y, rect):
    '''
    Given any shape x and y points, return closest dist to the rect and normal
    '''
    x1, y1, x2, y2 = rect
        
    # Initialize minimum distances and normals
    distances = torch.full(x.shape + (4,), float('inf'), dtype=torch.float32)
    contact_points = torch.zeros(x.shape + (4, 2), dtype=torch.float32)
    
    # FIXME. If broken, compute the contact points to debug
    
    # Iterate through each segment and compute the distance and normal
    distances[..., 0], contact_points[..., 0, :] = dist2seg(x, y, (x1, y1), (x2, y1))
    distances[..., 1], contact_points[..., 1, :] = dist2seg(x, y, (x2, y1), (x2, y2))
    distances[..., 2], contact_points[..., 2, :] = dist2seg(x, y, (x2, y2), (x1, y2))
    distances[..., 3], contact_points[..., 3, :] = dist2seg(x, y, (x1, y2), (x1, y1))

    result = torch.min(distances, dim=-1)

    return result.values, None

class StateTensor:
    '''
    Helpful wrapper around a position or velocity state vector, providing
    convenience functions for getting x, y, theta data
    '''
    def __init__(self, tensor):
        self.tensor = tensor
        if tensor.shape[-1] % 3 != 0:
            raise ValueError("Tensor last dim must be a multiple of 3 for x, y, theta extraction.")

    # Note the following are views
    @property
    def x(self): return self.tensor[..., 0::3]
    
    @property
    def y(self): return self.tensor[..., 1::3]
    
    @property
    def theta(self): return self.tensor[..., 2::3]
    
class VineParams:
    '''
    Time indepedent parameters.
    TODO make tensors and differentiable
    '''
    
    def __init__(self, batch_size, max_bodies, init_bodies, init_heading_deg=45, obstacles=[], grow_rate=10.0):
        self.max_bodies = max_bodies
        self.batch_size = batch_size
        self.nbodies = torch.full((batch_size,), fill_value=init_bodies)  # Number of bodies
        self.dt = 1 / 50 # 1/90  # Time step
        self.radius = 15.0 / 2  # Half-length of each body
        self.init_heading = math.radians(init_heading_deg)
        self.grow_rate = grow_rate # Length grown per unit time
        
        # Robot parameters
        self.m = 0.01 # 0.002  # Mass of each body
        self.I = 10  # Moment of inertia of each body
        self.half_len = 9
        
        # Dist from the center of each body to its end
        # Since the last body is connected via sliding joint, it is
        # not included in this tensor
        # self.d = torch.full((self.nbodies - 1,), fill_value=self.default_body_half_len, dtype=torch.float) 
        
        # Stiffness and damping coefficients
        self.stiffness = 15_000.0 # 30_000.0  # Stiffness coefficient
        self.damping = 50.0       # Damping coefficient

        # Environment obstacles (rects only for now)
        self.obstacles = obstacles
        
        # Make sure x < x2 y < y2 for rectangles
        for i in range(len(obstacles)):
            # Swap x
            if self.obstacles[i][0] > self.obstacles[i][2]:
                self.obstacles[i][0], self.obstacles[i][2] = self.obstacles[i][2], self.obstacles[i][0]
            # Swap y
            if self.obstacles[i][1] > self.obstacles[i][3]:
                self.obstacles[i][1], self.obstacles[i][3] = self.obstacles[i][3], self.obstacles[i][1]
        
        self.M = self.create_M()
        # This converts n-size torques to 3n size dstate
        # self.torque_to_maximal = torch.zeros((3 * self.nbodies, self.nbodies))    
        # for i in range(self.nbodies):            
        #     rot_idx = self.nbodies*2 + i
            
        #     if i != 0:
        #         self.torque_to_maximal[rot_idx - 1, i] = 1
                
        #     self.torque_to_maximal[rot_idx, i] = -1
            
        #     # if i != self.nbodies - 1:
        #     #     self.torque_to_maximal[rot_idx + 3, i] = 1
        
    def create_M(self):
        # Update mass matrix M (block diagonal)
        diagonal_elements = torch.Tensor([self.m, self.m, self.I]).repeat(self.max_bodies)
        return torch.diag(diagonal_elements)  # Shape: (nq, nq))

def create_state(batch_size, max_bodies):
    state = torch.zeros(batch_size, max_bodies * 3)
    dstate = torch.zeros(batch_size, max_bodies * 3)
    return state, dstate

def init_state(params: VineParams, state, dstate) -> StateTensor:
    '''
    Create vine state vectors from params
    '''
    state = StateTensor(state)
    
    # Init first body position
    state.theta[..., :] = torch.linspace(params.init_heading, params.init_heading + 0, params.max_bodies)
    state.x[..., 0] = params.half_len * torch.cos(state.theta[..., 0])
    state.y[..., 0] = params.half_len * torch.sin(state.theta[..., 0])
    
    # Init all body positions
    for b in range(params.batch_size):
        for i in range(1, params.max_bodies):
            if i == params.nbodies[b]: break
                
            lastx = state.x[b, i - 1]
            lasty = state.y[b, i - 1]
            lasttheta = state.theta[b, i - 1]
            thistheta = state.theta[b, i]
            
            state.x[b, i] = lastx + params.half_len * torch.cos(lasttheta) + params.half_len * torch.cos(thistheta)
            state.y[b, i] = lasty + params.half_len * torch.sin(lasttheta) + params.half_len * torch.sin(thistheta)

def zero_out(ten, nbodies):
    select_all = torch.arange(ten.shape[0])
    
    indices = torch.zeros_like(ten)
    indices[:, :] = torch.arange(0, ten.shape[-1])
    zeros = indices >= nbodies.unsqueeze(-1).repeat(1, ten.shape[-1]) 
    ten[zeros] = 0
    
def sdf(params: VineParams, state):
    '''
    TODO radius
    Given Nx1 x and y points, and list of rects, returns
    Nx1 min dist and normals (facing out of rect)
    '''    
    print(state.shape[:-1])
    min_dist =       torch.full(state.shape[:-1] + (params.max_bodies,), fill_value=torch.inf) # FIXMWE
    min_contactpts = torch.full(state.shape[:-1] + (params.max_bodies, 2), fill_value=torch.inf)
    
    state = StateTensor(state)
    
    for rect in params.obstacles:
        dist, contactpts = dist2rect(state.x, state.y, rect)
        dist -= params.radius
        
        update_min = dist < min_dist
        
        # check insideness, flip normal to face out
        isinside = (rect[0] < state.x) & (state.x < rect[2]) & (rect[1] < state.y) & (state.y < rect[3])
        
        print(state.x)
        dist = torch.where(isinside, -dist, dist) 
        # contactpts = torch.where(isinside[:, None], contactpts, contactpts) # FIXME
                                
        min_dist[update_min] = dist[update_min]
        # min_contactpts[update_min] = contactpts[update_min]
    
    zero_out(min_dist, params.nbodies)
    
    params.dbg_dist = min_dist
    # params.dbg_contactpts = min_contactpts
    return min_dist # min_normal

def joint_deviation(params: VineParams, state):
    
    # Vector of deviation per-joint, [x y x2 y2 x3 y3],
    # where each coordinate pair is the deviation with the last body
    constraints = torch.zeros(params.batch_size, params.max_bodies * 2 + 1)
    
    state = StateTensor(state)
    
    x = state.x
    y = state.y
    theta = state.theta
    
    constraints[..., 0] = x[..., 0] - params.half_len * torch.cos(theta[..., 0])
    constraints[..., 1] = y[..., 0] - params.half_len * torch.sin(theta[..., 0])
    
    constraints[..., 2:-1:2] = (x[..., 1:] - x[..., :-1]) - params.half_len * torch.cos(theta[..., :-1] 
                                                        + params.half_len * torch.cos(theta[..., 1:]))
    
    constraints[..., 3:-1:2] = (y[..., 1:] - y[..., :-1]) - params.half_len * torch.sin(theta[..., :-1] 
                                                        + params.half_len * torch.sin(theta[..., 1:]))
    
    # Last body is special. It has a sliding AND rotation joint with the second-last body
    select_all = torch.arange(state.x.shape[0])
    
    endx = x[select_all, params.nbodies-2] + params.half_len * torch.cos(theta[select_all, params.nbodies-2])
    endy = y[select_all, params.nbodies-2] + params.half_len * torch.sin(theta[select_all, params.nbodies-2])
    
    constraints[select_all, params.nbodies * 2 - 1] = torch.atan2(y[select_all, params.nbodies-1] - endy, 
                                                                  x[select_all, params.nbodies-1] - endx) \
                                                        - theta[select_all, params.nbodies-1]
    
    zero_out(constraints, params.nbodies * 2)
    
    return constraints

def bending_energy(params: VineParams, theta, dtheta):
    # Compute the response (like potential energy of bending)
    # Can think of the system as always wanting to get rid of potential 
    # Generally, \tau = - stiffness * benderino - damping * d_benderino
    
    # return -(self.K @ theta) - self.C @ dtheta
    bend = -params.stiffness * theta - params.damping * dtheta
    zero_out(bend, params.nbodies)
    
    return bend
    # return -self.stiffness * torch.where(torch.abs(theta) < 0.3, theta * 2, theta * 0.5) - self.damping * dtheta
    # return torch.sign(theta) * -(torch.sin((torch.abs(theta) + 0.1) / 0.3) + 1.4)
    
def extend(params: VineParams, state, dstate):     
    print('state', state.shape)
    state = StateTensor(state)
    dstate = StateTensor(dstate)
       
    # Compute position of second last seg
    ending_index = params.nbodies - 1
    penult_index = params.nbodies - 2
    new_index = params.nbodies
    
    print('ending_index', ending_index)
    
    select_all = torch.arange(state.x.shape[0])
    endingx = state.x[select_all, penult_index] + params.half_len * torch.cos(state.theta[select_all, penult_index])
    endingy = state.y[select_all, penult_index] + params.half_len * torch.sin(state.theta[select_all, penult_index])
    print('endingx', endingx)
        
    # Compute last body's distance 
    last_link_distance = ((state.x[select_all, ending_index] - endingx)**2 + \
                          (state.y[select_all, ending_index] - endingy)**2).sqrt().squeeze(-1)
    
    extend_mask = last_link_distance > params.half_len * 2 # x2 to prevent 0-len segments
    
    if not torch.any(extend_mask):
        return
    
    select_extends = torch.where(extend_mask)
    print('select_extends', select_extends)
       
    params.nbodies[extend_mask] += 1
    
    if torch.any(params.nbodies > params.max_bodies):
        raise RuntimeError('More bodies than array supports')
            
    # Compute location of new seg
    last_link_theta = torch.atan2(state.y[extend_mask][ending_index] - state.y[extend_mask][penult_index], 
                                  state.x[extend_mask][ending_index] - state.x[extend_mask][penult_index])
        
    new_seg_x = endingx + params.half_len * torch.cos(last_link_theta)
    new_seg_y = endingy + params.half_len * torch.sin(last_link_theta)
    new_seg_theta = last_link_theta.squeeze()
    new_seg_x.squeeze_(-1)
    new_seg_y.squeeze_(-1)
    
    print('new_seg_x', new_seg_x.shape)
    
    # Copy last body one forward (yes you can mix boolean masks and indexes)
    state.x[select_extends, new_index] = state.x[select_extends, ending_index]
    state.y[select_extends, new_index] = state.y[select_extends, ending_index]
    state.theta[select_extends, new_index] = state.theta[select_extends, ending_index]
    
    # Set the new segment position
    state.x[select_extends, ending_index] = new_seg_x[select_extends]
    state.y[select_extends, ending_index] = new_seg_y[select_extends]
    state.theta[select_extends, ending_index] = new_seg_theta[select_extends]
    
    # Set the new segment to have veleocity ofthe one before
    # FIXME should be tip vel
    dstate.x[select_extends, ending_index] = dstate.x[select_extends, penult_index]
    dstate.y[select_extends, ending_index] = dstate.y[select_extends, penult_index]
    dstate.theta[select_extends, ending_index] = dstate.theta[select_extends, penult_index]

def grow2(params: VineParams, stateanddstate):   
    
    half = stateanddstate.shape[-1] // 2
    
    print('stateanddstate', stateanddstate.shape)
    state = StateTensor(stateanddstate[..., :half])
    dstate = StateTensor(stateanddstate[..., half:])
    
    assert state.tensor.shape == dstate.tensor.shape
    
    select_all = torch.arange(state.tensor.shape[0])
    
    # Now return the constraints for growing the last segment        
    x1 = state.x[select_all, params.nbodies-2]
    y1 = state.y[select_all, params.nbodies-2]
    vx1 = dstate.x[select_all, params.nbodies-2]
    vy1 = dstate.y[select_all, params.nbodies-2]
    
    x2 = state.x[select_all, params.nbodies-1]
    y2 = state.y[select_all, params.nbodies-1]
    vx2 = dstate.x[select_all, params.nbodies-1]
    vy2 = dstate.y[select_all, params.nbodies-1]
        
    constraint = ((x2 - x1) * (vx2 - vx1) + (y2 - y1) * (vy2 - vy1)) / \
                  torch.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
    return constraint

def grow(self, stateanddstate):
    constraints = torch.zeros(self.nbodies - 1)
    
    state = stateanddstate[:self.nbodies*3]
    dstate = stateanddstate[self.nbodies*3:]
    
    x = state[0:self.nbodies]
    y = state[self.nbodies:self.nbodies*2]
    dx = dstate[0:self.nbodies]
    dy = dstate[self.nbodies:self.nbodies*2]
    
    for i in range(self.nbodies-1):
        x1 = x[i]
        y1 = y[i]
        x2 = x[i + 1]
        y2 = y[i + 1]
        
        vx1 = dx[i]
        vy1 = dy[i]
        vx2 = dx[i + 1]
        vy2 = dy[i + 1]
        
        constraints[i] = ((x2-x1)*(vx2-vx1) + (y2-y1)*(vy2-vy1)) / \
                            torch.sqrt((x2-x1)**2 + (y2-y1)**2)
        
    return constraints
        
def solve(params: VineParams, state, dstate, forces, growth, sdf_now, deviation_now, G, L, J):
    # Solve target
    next_dstate = cp.Variable((params.max_bodies * 3,)) 
    
    # Minimization objective
    # TODO Why subtract forces. Paper and code don't agree
    objective = cp.Minimize(0.5 * cp.quad_form(next_dstate, params.M) - 
                            next_dstate.T @ (params.M @ dstate - forces * params.dt))

    # Constrains for non-collision and joint connectivity
    
    growth_wrt_state = G[:params.max_bodies*3]
    growth_wrt_dstate = G[params.max_bodies*3:]
    
    g_con = growth - params.grow_rate - growth_wrt_dstate @ dstate
    g_coeff = growth_wrt_state * params.dt + growth_wrt_dstate
    
    # TODO in the julia they divide sdf_now and deviation_now by dt
    # technically equivalent, but why? numerical stability?
    constraints = [ sdf_now + (L @ next_dstate) * params.dt >= 0,
                    deviation_now + (J @ next_dstate) * params.dt == 0,
                    g_con + g_coeff @ next_dstate == 0]
    
    # TODO For backprop, need to set Parameters
    problem = cp.Problem(objective, constraints)
    
    # Well formedness
    assert problem.is_dpp()
    
    problem.solve(solver=cp.OSQP) # MOSEK is fast requires_grad=True)
    
    if problem.status != cp.OPTIMAL:
        print("status:", problem.status)
                                    
    next_dstate_solution = torch.tensor(next_dstate.value, dtype=torch.float)
    return next_dstate_solution
        
def forward(params: VineParams, state, dstate):
    extend(params, state, dstate)
        
    # Jacobian of SDF with respect to x and y
    L = jacobian(partial(sdf, params), state, vectorize=True)
    sdf_now = sdf(params, state)
    
    print(f'The jacobian of state {state.shape} -> sdf {sdf_now.shape} is', L.shape)
    
    # Jacobian of joint deviation wrt configuration
    J = jacobian(partial(joint_deviation, params), state, vectorize=True)
    deviation_now = joint_deviation(params, state)
    
    G = jacobian(partial(grow2, params), torch.cat([state, dstate], dim=-1), vectorize=True)
    growth = grow2(params, torch.cat([state, dstate], dim=-1))
    
    # Stiffness forces
    theta_rel = finite_changes(StateTensor(state).theta, params.init_heading)
    dtheta_rel = finite_changes(StateTensor(dstate).theta, 0.0)
    
    bend_energy = bending_energy(params, theta_rel, dtheta_rel)
    
    forces = StateTensor(torch.zeros(params.batch_size, params.max_bodies * 3))
    forces.theta[:] += -bend_energy       # Apply bend energy as torque to own joint
    forces.theta[:-1]  += bend_energy[1:] # Apply bend energy as torque to joint before
    forces = forces.tensor
    
    next_dstate_solution = torch.zeros((params.batch_size, params.max_bodies*3))
    
    for i in range(params.batch_size):
        next_dstate_solution[i] = solve(params, state[i], dstate[i], forces[i], growth, sdf_now[i], deviation_now[i], G[i], L[i], J[i])
    
    # Evolve
    new_state = state + next_dstate_solution * params.dt
    
    new_dstate = next_dstate_solution
    
    return new_state, new_dstate