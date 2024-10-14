import math
import torch
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

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
    distance_vectors = points - closest_point_on_segment
    distances = torch.norm(distance_vectors, dim=1)
    
    # Normalize the distance vectors to get normal vectors
    normals = torch.nn.functional.normalize(distance_vectors, dim=1)
    
    return distances, normals
    
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
    min_normals = torch.zeros((x.shape[0], 2), dtype=torch.float32)
    
    # Iterate through each segment and compute the distance and normal
    for seg_start, seg_end in segments:
        distances, normals = dist2seg(x, y, seg_start, seg_end)
        
        # Update the minimum distances and normals where applicable
        update_mask = distances < min_distances
        min_distances = torch.where(update_mask, distances, min_distances)
        min_normals = torch.where(update_mask[:, None], normals, min_normals)
    
    return min_distances, min_normals

class Vine:
    def __init__(self, nbodies, init_heading_deg=45):
        self.nbodies = nbodies  # Number of bodies
        self.dt = 1/90  # Time step
        self.radius = 15.0 / 2  # Half-length of each body
        self.init_heading = math.radians(init_heading_deg)
        
        # Robot parameters
        self.m = 0.002  # Mass of each body
        self.I = 400  # Moment of inertia of each body
        self.default_body_half_len = 3
        
        # Dist from body center (distal) to proximal conenction 
        self.d = torch.full((self.nbodies,), fill_value=self.default_body_half_len) 

        # State variables
        self.x = torch.zeros(self.nbodies) 
        self.y = torch.zeros(self.nbodies) 
        self.theta = torch.zeros(self.nbodies) 
        
        # Init positions
        self.theta[:] = torch.linspace(self.init_heading, self.init_heading + 1, self.nbodies)
        self.x[0] = self.d[0] * torch.cos(self.theta[0])
        self.y[0] = self.d[0] * torch.sin(self.theta[0])
        
        for i in range(1, self.nbodies):
            lastx = self.x[i - 1]
            lasty = self.y[i - 1]
            lasttheta = self.theta[i - 1]
            thistheta = self.theta[i]
            self.x[i] = lastx + self.d[i-1] * torch.cos(lasttheta) + self.d[i] * torch.cos(thistheta)
            self.y[i] = lasty + self.d[i-1] * torch.sin(lasttheta) + self.d[i] * torch.sin(thistheta)
        
        # Init vels
        self.dx = torch.zeros(self.nbodies)
        self.dy = torch.zeros(self.nbodies)
        self.dtheta = torch.zeros(self.nbodies) 
                
        # Mass matrix M (block diagonal)
        # TODO mass of off-center rotation
        M_blocks = [torch.diag(torch.tensor([self.m, self.m, self.I])) for _ in range(nbodies)]
        self.M = torch.block_diag(*M_blocks)  # Shape: (nq, nq)
        
        # Stiffness and damping coefficients
        self.stiffness = 30000  # Stiffness coefficient
        self.damping = 10   # Damping coefficient

        # Environment obstacles (rects only for now)
        self.obstacles = [
            (2, 0, 3, 3) # x, y, x2, y2
        ]
        
        # This converts n-size torques to 3n size dstate
        self.torque_to_maximal = torch.zeros((3 * self.nbodies, self.nbodies))    
        for i in range(self.nbodies):            
            rot_idx = 3 * i + 2
            
            if i != 0:
                self.torque_to_maximal[rot_idx - 3, i] = 1
                
            self.torque_to_maximal[rot_idx, i] = -1
            
            # if i != self.nbodies - 1:
            #     self.torque_to_maximal[rot_idx + 3, i] = 1
                
            self.torque_to_maximal[-1, i - 1] = -1.1
        
    def sdf(self, unified_state):
        '''
        TODO radius
        Given Nx1 x and y points, and list of rects, returns
        Nx1 min dist and normals (facing out of rect)
        '''
        n = self.nbodies
        x = unified_state[0:self.nbodies]
        y = unified_state[self.nbodies:2*self.nbodies]
        
        min_dist = torch.full((n,), fill_value=torch.inf, dtype=torch.float32, device=x.device)
        min_normal = torch.full((n, 2), fill_value=torch.inf, dtype=torch.float32, device=x.device)
        
        for rect in self.obstacles:
            dist, normal = dist2rect(x, y, rect)
           
            update_min = dist < min_dist
            
            # check insideness, flip normal to face out
            isinside = (rect[0] < x) & (x < rect[2]) & (rect[1] < y) & (y < rect[3])
            
            dist = torch.where(isinside, -dist, dist)
            normal = torch.where(isinside[:, None], -normal, normal)
                        
            min_dist[update_min] = dist[update_min]
            min_normal[update_min] = normal[update_min]
        
        return min_dist # min_normal

    def joint_deviation(self, unified_state):
        
        x = unified_state[0:self.nbodies]
        y = unified_state[self.nbodies:2*self.nbodies]
        theta = unified_state[2*self.nbodies:3*self.nbodies]
        
        # Vector of deviation per-joint, [x y x2 y2 x3 y3],
        # where each coordinate pair is the deviation with the last body
        constraints = torch.zeros(self.nbodies * 2)
        
        constraints[0] = x[0] - self.d[0] * torch.cos(theta[0])
        constraints[1] = y[0] - self.d[0] * torch.sin(theta[0])
        
        for j2 in range(1, self.nbodies):
            j1 = j2 - 1
            constraints[j2 * 2] = (x[j2] - x[j1]) - (self.d[j1] * torch.cos(theta[j1]) + self.d[j2] * torch.cos(theta[j2]))
            
            constraints[j2 * 2 + 1] = (y[j2] - y[j1]) - (self.d[j1] * torch.sin(theta[j1]) + self.d[j2] * torch.sin(theta[j2]))
            
            # print('x disparity')
            # print(f'\t diff{(x[j2] - x[j1]):.3f}')
            # print(f'\t comp { (self.d[j1] * torch.cos(theta[j1]) + self.d[j2] * torch.cos(theta[j2])):.3f}')
        return constraints

    def bending_energy(self, theta, dtheta):
        # Compute the response (like potential energy of bending)
        # Can think of the system as always wanting to get rid of potential 
        # Generally, \tau = - stiffness * benderino - damping * d_benderino
        
        # return -(self.K @ theta) - self.C @ dtheta
        return -self.stiffness * theta - self.damping * dtheta
    
    def evolve(self):        
        state = torch.concat((self.x, self.y, self.theta), dim=0)
        dstate = torch.concat((self.dx, self.dy, self.dtheta), dim=0)
        
        # Jacobian of SDF with respect to x and y
        L = torch.autograd.functional.jacobian(self.sdf, state)
        sdf_now = self.sdf(state)
        
        # Jacobian of joint constraints (ie distance deviation at the joint points)
        # with respect to configuration
        J = torch.autograd.functional.jacobian(self.joint_deviation, state)
        deviation_now = self.joint_deviation(state)
        
        # Stiffness forces
        theta_rel = finite_changes(self.theta, self.init_heading)
        dtheta_rel = finite_changes(self.dtheta, torch.tensor(0, dtype=torch.float))
        
        forces = self.torque_to_maximal @ self.bending_energy(theta_rel, dtheta_rel)
        # forces = torch.zeros((self.nbodies * 3,) )
        # forces[2::3] = 1000
        # forces[:-1] = 0
        # forces[-1] = 100000
        
        print('theta_rel', theta_rel)
        print('bending energy', self.bending_energy(self.theta, self.dtheta))
        print('forces', forces)
        # print('joint deviation', deviation_now)
        
        # Solve target
        next_dstate = cp.Variable((self.nbodies * 3,)) 
        
        # Minimization objective
        # TODO Why subtract forces. Paper and code don't agree
        objective = cp.Minimize(0.5 * cp.quad_form(next_dstate, self.M) - 
                                next_dstate.T @ (self.M @ dstate - forces * self.dt))

        # Constrains for non-collision and joint connectivity
        # TODO in the julia they divide sdf_now and deviation_now by dt
        # technically equivalent, but why? numerical stability?
        constraints = [ sdf_now / self.dt + L @ next_dstate >= 0,
                        deviation_now / self.dt + J @ next_dstate == 0]
        
        # TODO For backprop, need to set Parameters
        problem = cp.Problem(objective, constraints)
        
        # Well formedness
        assert problem.is_dpp()
        
        problem.solve(solver=cp.OSQP) # requires_grad=True)
        
        if problem.status != cp.OPTIMAL:
            print("status:", problem.status)
                            
        next_dstate_solution = torch.tensor(next_dstate.value, dtype=torch.float)
        
        print('solved deviation', deviation_now + J @ next_dstate_solution * self.dt)
        
        # Evolve
        self.x += next_dstate_solution[0:self.nbodies] * self.dt
        self.y += next_dstate_solution[self.nbodies:2*self.nbodies] * self.dt
        self.theta += next_dstate_solution[2*self.nbodies:3*self.nbodies] * self.dt


ww = 30
def vis_init():
    plt.figure(1, figsize=(10, 10))
    
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim(-ww, ww)
    ax.set_ylim(-ww, ww)

def draw(vine: Vine):
    
    ax = plt.gca()
    ax.cla()
    
    ax.set_aspect('equal')
    ax.set_xlim(-ww, ww)
    ax.set_ylim(-ww, ww)
    
    # Draw the obstacles
    for obstacle in vine.obstacles:
        obstacle_patch = Rectangle((obstacle[0], obstacle[1]),
                                    obstacle[2] - obstacle[0], obstacle[1] - obstacle[3],
                                    linewidth=1, edgecolor='r', facecolor='gray')
        ax.add_patch(obstacle_patch)

    # Draw each body
    for i in range(vine.nbodies):
        # Body endpoints
        x_start = vine.x[i] - vine.d[i] * torch.cos(vine.theta[i])
        y_start = vine.y[i] - vine.d[i] * torch.sin(vine.theta[i])
        x_end = vine.x[i] + vine.d[i] * torch.cos(vine.theta[i])
        y_end = vine.y[i] + vine.d[i] * torch.sin(vine.theta[i])

        plt.plot([x_start, x_end], [y_start, y_end], 'b-', linewidth=10)
        plt.scatter(vine.x[i], vine.y[i], c='red', linewidths=10)        
        
if __name__ == '__main__':
    vine = Vine(nbodies=5, init_heading_deg=55)
    
    vis_init()
    
    draw(vine)
    plt.pause(0.001)
    
    for _ in range(1000):
        vine.evolve()
        
        draw(vine)
        # plt.show()
        plt.pause(0.001)
        
    plt.show()