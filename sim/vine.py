import math
import torch
from torch.autograd.functional import jacobian
import cvxpy as cp
from functools import partial

def finite_changes(h, init_val):
    h_rel = torch.empty_like(h)
    h_rel[0] = h[0] - init_val
    h_rel[1:] = h[1:] - h[:-1] 
    
    return h_rel

def dist2seg(x, y, start, end):
    '''
    Given Nx1 x and y points, return closest dist to the seg and normal
    '''
    points = torch.stack([x, y], dim=1)
    
    # Segment start and end as tensors
    p1 = torch.tensor(start, dtype=torch.float32)
    p2 = torch.tensor(end, dtype=torch.float32)
    
    # Vector from p1 to p2 (the segment direction)
    segment = p2 - p1
    
    # Vector from p1 to each point
    p1_to_points = points - p1
    
    # Project the point onto the segment, calculate the parameter t
    segment_length_squared = torch.dot(segment, segment)
    t = torch.clamp(torch.sum(p1_to_points * segment, dim=1) / segment_length_squared, 0.0, 1.0)
    
    # Closest point on the segment to the point
    closest_point_on_segment = p1 + t[:, None] * segment
        
    # Distance from each point to the closest point on the segment
    distance_vectors = closest_point_on_segment - points 
    distances = torch.norm(distance_vectors, dim=1)
    
    # Normalize the distance vectors to get normal vectors
    # normals = torch.nn.functional.normalize(distance_vectors, dim=1)
    normals = distance_vectors
    
    return distances, closest_point_on_segment
    
def dist2rect(x, y, rect):
    '''
    Given Nx1 x and y points, return closest dist to the rect and normal
    '''
    x1, y1, x2, y2 = rect
    
    # Define the four segments of the rectangle
    segments = [
        ((x1, y1), (x2, y1)),  # Bottom edge
        ((x2, y1), (x2, y2)),  # Right edge
        ((x2, y2), (x1, y2)),  # Top edge
        ((x1, y2), (x1, y1))   # Left edge
    ]
        
    # Initialize minimum distances and normals
    min_distances = torch.full((x.shape[0],), float('inf'), dtype=torch.float32)
    min_contact_points = torch.zeros((x.shape[0], 2), dtype=torch.float32)
    
    # Iterate through each segment and compute the distance and normal
    for seg_start, seg_end in segments:
        distances, contact_points = dist2seg(x, y, seg_start, seg_end)
        
        # Update the minimum distances and normals where applicable
        update_mask = distances < min_distances
        min_distances = torch.where(update_mask, distances, min_distances)
        min_contact_points = torch.where(update_mask[:, None], contact_points, min_contact_points)
    
    return min_distances, min_contact_points

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
    
    def __init__(self, nbodies, init_heading_deg=45, obstacles=[], grow_rate=10.0):
        self.nbodies = nbodies  # Number of bodies
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
        diagonal_elements = torch.Tensor([self.m, self.m, self.I]).repeat(self.nbodies)
        return torch.diag(diagonal_elements)  # Shape: (nq, nq))

def create_state(params: VineParams) -> StateTensor:
    '''
    Create vine state vectors from params
    '''
    state = StateTensor(torch.zeros(params.nbodies * 3))
    dstate = StateTensor(torch.zeros(params.nbodies * 3))
            
    # Init first body position
    state.theta[:] = torch.linspace(params.init_heading, params.init_heading + 0, params.nbodies)
    state.x[0] = params.half_len * torch.cos(state.theta[0])
    state.y[0] = params.half_len * torch.sin(state.theta[0])
    
    # Init all body positions
    for i in range(1, params.nbodies):
        lastx = state.x[i - 1]
        lasty = state.y[i - 1]
        lasttheta = state.theta[i - 1]
        thistheta = state.theta[i]
        
        state.x[i] = lastx + params.half_len * torch.cos(lasttheta) + params.half_len * torch.cos(thistheta)
        state.y[i] = lasty + params.half_len * torch.sin(lasttheta) + params.half_len * torch.sin(thistheta)
    
    return state.tensor, dstate.tensor
        
def sdf(params: VineParams, state):
    '''
    TODO radius
    Given Nx1 x and y points, and list of rects, returns
    Nx1 min dist and normals (facing out of rect)
    '''
    
    state = StateTensor(state)
    
    min_dist =       torch.full((params.nbodies,), fill_value=torch.inf)
    min_contactpts = torch.full((params.nbodies, 2), fill_value=torch.inf)
    
    for rect in params.obstacles:
        dist, contactpts = dist2rect(state.x, state.y, rect)
        dist -= params.radius
        
        update_min = dist < min_dist
        
        # check insideness, flip normal to face out
        isinside = (rect[0] < state.x) & (state.x < rect[2]) & (rect[1] < state.y) & (state.y < rect[3])
        
        dist = torch.where(isinside, -dist, dist) 
        contactpts = torch.where(isinside[:, None], contactpts, contactpts) # FIXME
                                
        min_dist[update_min] = dist[update_min]
        min_contactpts[update_min] = contactpts[update_min]
    
    params.dbg_dist = min_dist
    params.dbg_contactpts = min_contactpts
    return min_dist # min_normal

def joint_deviation(params: VineParams, state):
    
    # Vector of deviation per-joint, [x y x2 y2 x3 y3],
    # where each coordinate pair is the deviation with the last body
    constraints = torch.zeros(params.nbodies * 2 - 1)
    
    state = StateTensor(state)
    
    x = state.x
    y = state.y
    theta = state.theta
    
    constraints[0] = x[0] - params.half_len * torch.cos(theta[0])
    constraints[1] = y[0] - params.half_len * torch.sin(theta[0])
    
    for j2 in range(1, params.nbodies - 1):
        j1 = j2 - 1
        constraints[j2 * 2] = (x[j2] - x[j1]) - (params.half_len * torch.cos(theta[j1]) + params.half_len * torch.cos(theta[j2]))
        
        constraints[j2 * 2 + 1] = (y[j2] - y[j1]) - (params.half_len * torch.sin(theta[j1]) + params.half_len * torch.sin(theta[j2]))
    
    # Last body is special. It has a sliding AND rotation joint with the second-last body
    endx = x[-2] + params.half_len * torch.cos(theta[-2])
    endy = y[-2] + params.half_len * torch.sin(theta[-2])
    
    constraints[-1] = torch.atan2(y[-1] - endy, x[-1] - endx) - theta[-1]
    
    return constraints

def bending_energy(params: VineParams, theta, dtheta):
    # Compute the response (like potential energy of bending)
    # Can think of the system as always wanting to get rid of potential 
    # Generally, \tau = - stiffness * benderino - damping * d_benderino
    
    # return -(self.K @ theta) - self.C @ dtheta
    return -params.stiffness * theta - params.damping * dtheta
    # return -self.stiffness * torch.where(torch.abs(theta) < 0.3, theta * 2, theta * 0.5) - self.damping * dtheta
    # return torch.sign(theta) * -(torch.sin((torch.abs(theta) + 0.1) / 0.3) + 1.4)
    
def extend(params: VineParams, state, dstate):     
    
    state = StateTensor(state)
    dstate = StateTensor(dstate)
       
    # Compute position of second last seg
    endingx = state.x[-2] + params.half_len * torch.cos(state.theta[-2])
    endingy = state.y[-2] + params.half_len * torch.sin(state.theta[-2])
        
    # Compute last body's distance 
    last_link_distance = ((state.x[-1] -endingx)**2 + (state.y[-1] - endingy)**2).sqrt()
    last_link_theta = torch.atan2(state.y[-1] - state.y[-2], state.x[-1] - state.x[-2])

    if last_link_distance > params.half_len * 2: # x2 to prevent 0-len segments
        print(f'Extending {params.nbodies} -> {params.nbodies + 1}')
        
        params.nbodies += 1
        new_state = StateTensor(torch.zeros(params.nbodies * 3))
        new_dstate = StateTensor(torch.zeros(params.nbodies * 3))
                
        # Compute location of new seg
        new_seg_x = endingx + params.half_len * torch.cos(last_link_theta)
        new_seg_y = endingy + params.half_len * torch.sin(last_link_theta)
        new_seg_theta = last_link_theta
        
        # Extend state variables
        params.M = params.create_M()
        
        # Clone position
        new_state.x[:-1] = state.x
        new_state.y[:-1] = state.y
        new_state.theta[:-1] = state.theta
        
        new_dstate.x[:-1] = dstate.x
        new_dstate.y[:-1] = dstate.y
        new_dstate.theta[:-1] = dstate.theta
        
        # Copy last body one index forward
        new_state.x[-1] = state.x[-1]
        new_state.y[-1] = state.y[-1]
        new_state.theta[-1] = state.theta[-1]
        
        new_dstate.x[-1] = 0
        new_dstate.y[-1] = 0
        new_dstate.theta[-1] = 0
        
        # Set the new segment position (at -2)
        new_state.x[-2] = new_seg_x
        new_state.y[-2] = new_seg_y
        new_state.theta[-2] = new_seg_theta
        new_dstate.x[-2] = dstate.x[-2]
        new_dstate.y[-2] = dstate.y[-2] 
        new_dstate.theta[-2] = dstate.theta[-2] 
        
        return new_state.tensor, new_dstate.tensor
    
    return state.tensor, dstate.tensor

def grow2(params: VineParams, stateanddstate):        
    state = StateTensor(stateanddstate[:params.nbodies*3])
    dstate = StateTensor(stateanddstate[params.nbodies*3:])
    
    # Now return the constraints for growing the last segment        
    x1 = state.x[-2]
    y1 = state.y[-2]
    vx1 = dstate.x[-2]
    vy1 = dstate.y[-2]
    
    x2 = state.x[-1]
    y2 = state.y[-1]
    vx2 = dstate.x[-1]
    vy2 = dstate.y[-1]
        
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
        
def forward(params: VineParams, state, dstate):
    state, dstate = extend(params, state, dstate)
        
    # Jacobian of SDF with respect to x and y
    L = jacobian(partial(sdf, params), state)
    sdf_now = sdf(params, state)
    
    # Jacobian of joint deviation wrt configuration
    J = jacobian(partial(joint_deviation, params), state)
    deviation_now = joint_deviation(params, state)
    
    G = jacobian(partial(grow2, params), torch.cat([state, dstate]))
    growth = grow2(params, torch.cat([state, dstate]))
    
    # Stiffness forces
    theta_rel = finite_changes(StateTensor(state).theta, params.init_heading)
    dtheta_rel = finite_changes(StateTensor(dstate).theta, 0.0)
    
    bend_energy = bending_energy(params, theta_rel, dtheta_rel)
    
    forces = StateTensor(torch.zeros(params.nbodies * 3))
    forces.theta[:] += -bend_energy       # Apply bend energy as torque to own joint
    forces.theta[:-1]  += bend_energy[1:] # Apply bend energy as torque to joint before
    forces = forces.tensor
    
    # Solve target
    next_dstate = cp.Variable((params.nbodies * 3,)) 
    
    # Minimization objective
    # TODO Why subtract forces. Paper and code don't agree
    objective = cp.Minimize(0.5 * cp.quad_form(next_dstate, params.M) - 
                            next_dstate.T @ (params.M @ dstate - forces * params.dt))

    # Constrains for non-collision and joint connectivity
    
    growth_wrt_state = G[:params.nbodies*3]
    growth_wrt_dstate = G[params.nbodies*3:]
    
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
    
    # Evolve
    new_state = state + next_dstate_solution * params.dt
    
    new_dstate = next_dstate_solution
    
    return new_state, new_dstate