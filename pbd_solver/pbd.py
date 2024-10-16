import math
import torch
from torch.autograd.functional import jacobian
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

min_term_hist = []
joint_constraint = []

class Vine:
    def __init__(self, nbodies, init_heading_deg=45, obstacles=[]):
        self.nbodies = nbodies  # Number of bodies
        self.dt = 1/90  # Time step
        self.radius = 15.0 / 2  # Half-length of each body
        self.init_heading = math.radians(init_heading_deg)
        self.grow_rate = 3 # Length grown per unit time
        
        # Robot parameters
        self.m = 0.002  # Mass of each body
        self.I = 200  # Moment of inertia of each body
        self.default_body_half_len = 3
        
        # Dist from body center (distal) to proximal conenction 
        self.d = torch.full((self.nbodies,), fill_value=self.default_body_half_len) 

        # State variables
        self.state = torch.zeros(self.nbodies * 3)
        self.dstate = torch.zeros(self.nbodies * 3)
        
        # Init positions
        self.thetaslice[:] = torch.linspace(self.init_heading, self.init_heading + 1, self.nbodies)
        self.xslice[0] = self.d[0] * torch.cos(self.thetaslice[0])
        self.yslice[0] = self.d[0] * torch.sin(self.thetaslice[0])
        
        for i in range(1, self.nbodies):
            lastx = self.xslice[i - 1]
            lasty = self.yslice[i - 1]
            lasttheta = self.thetaslice[i - 1]
            thistheta = self.thetaslice[i]
            self.xslice[i] = lastx + self.d[i-1] * torch.cos(lasttheta) + self.d[i] * torch.cos(thistheta)
            self.yslice[i] = lasty + self.d[i-1] * torch.sin(lasttheta) + self.d[i] * torch.sin(thistheta)
                
        # Mass matrix M (block diagonal)
        m_block = torch.tensor([self.m] * self.nbodies)  # nbodies of self.m
        i_block = torch.tensor([self.I] * self.nbodies)  # nbodies of self.I

        diagonal_elements = torch.cat([m_block, m_block, i_block])
        self.M = torch.diag(diagonal_elements)  # Shape: (nq, nq)
        
        # Stiffness and damping coefficients
        self.stiffness = 30000.0  # Stiffness coefficient
        self.damping = 10.0       # Damping coefficient

        # Environment obstacles (rects only for now)
        self.obstacles = obstacles
        
        # This converts n-size torques to 3n size dstate
        self.torque_to_maximal = torch.zeros((3 * self.nbodies, self.nbodies))    
        for i in range(self.nbodies):            
            rot_idx = self.nbodies*2 + i
            
            if i != 0:
                self.torque_to_maximal[rot_idx - 1, i] = 1
                
            self.torque_to_maximal[rot_idx, i] = -1
            
            # if i != self.nbodies - 1:
            #     self.torque_to_maximal[rot_idx + 3, i] = 1
    
    # Convience functions for getting x, y, theta
    @property
    def xslice(self): return self.state[0:self.nbodies]
    @property
    def yslice(self): return self.state[self.nbodies:self.nbodies*2]
    @property
    def thetaslice(self): return self.state[self.nbodies*2:self.nbodies*3]
    @property
    def dxslice(self): return self.dstate[0:self.nbodies]
    @property
    def dyslice(self): return self.dstate[self.nbodies:self.nbodies*2]
    @property
    def dthetaslice(self): return self.dstate[self.nbodies*2:self.nbodies*3]
    
    def sdf(self, state):
        '''
        TODO radius
        Given Nx1 x and y points, and list of rects, returns
        Nx1 min dist and normals (facing out of rect)
        '''
        x = state[0:self.nbodies]
        y = state[self.nbodies:self.nbodies*2]
        
        min_dist = torch.full((self.nbodies,), fill_value=torch.inf)
        min_contactpts = torch.full((self.nbodies, 2), fill_value=torch.inf)
        
        for rect in self.obstacles:
            dist, contactpts = dist2rect(x, y, rect)
            dist -= self.radius
           
            update_min = dist < min_dist
            
            # check insideness, flip normal to face out
            isinside = (rect[0] < x) & (x < rect[2]) & (rect[1] < y) & (y < rect[3])
            
            dist = torch.where(isinside, dist, dist) # FIXME
            contactpts = torch.where(isinside[:, None], contactpts, contactpts) # FIXME
                                    
            min_dist[update_min] = dist[update_min]
            min_contactpts[update_min] = contactpts[update_min]
        
        self.dbg_dist = min_dist
        self.dbg_contactpts = min_contactpts
        return min_dist # min_normal

    def joint_deviation(self, state):
        
        x = state[0:self.nbodies]
        y = state[self.nbodies:self.nbodies*2]
        theta = state[self.nbodies*2:self.nbodies*3]
        
        # Vector of deviation per-joint, [x y x2 y2 x3 y3],
        # where each coordinate pair is the deviation with the last body
        constraints = torch.zeros(self.nbodies * 2)
        
        constraints[0] = x[0] - self.d[0] * torch.cos(theta[0])
        constraints[1] = y[0] - self.d[0] * torch.sin(theta[0])
        
        for j2 in range(1, self.nbodies):
            j1 = j2 - 1
            constraints[j2 * 2] = (x[j2] - x[j1]) - (self.d[j1] * torch.cos(theta[j1]) + self.d[j2] * torch.cos(theta[j2]))
            
            constraints[j2 * 2 + 1] = (y[j2] - y[j1]) - (self.d[j1] * torch.sin(theta[j1]) + self.d[j2] * torch.sin(theta[j2]))
            
        return constraints

    def bending_energy(self, theta, dtheta):
        # Compute the response (like potential energy of bending)
        # Can think of the system as always wanting to get rid of potential 
        # Generally, \tau = - stiffness * benderino - damping * d_benderino
        
        # return -(self.K @ theta) - self.C @ dtheta
        return -self.stiffness * theta - self.damping * dtheta
    
    # def grow(self, dt):        
    #     # check length of last link
    #     if self.d[-1] == self.default_body_half_len:
    #         # Create a new link
    #         endingx = self.xslice[-1] + self.d[-1] * torch.cos(self.thetaslice[-1])
    #         endingy = self.yslice[-1] + self.d[-1] * torch.sin(self.thetaslice[-1])
            
    #         self.x = torch.cat((self.x, torch.tensor(endingx)))
    #         self.y = torch.cat((self.y, torch.tensor(endingy)))
    
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
            
    def evolve(self):        
        
        # Jacobian of SDF with respect to x and y
        L = jacobian(self.sdf, self.state)
        sdf_now = self.sdf(self.state)
        
        # Jacobian of joint deviation wrt configuration
        J = jacobian(self.joint_deviation, self.state)
        deviation_now = self.joint_deviation(self.state)
        
        G = jacobian(self.grow, torch.cat([self.state, self.dstate]))
        
        # Stiffness forces
        theta_rel = finite_changes(self.thetaslice, self.init_heading)
        dtheta_rel = finite_changes(self.dthetaslice, 0.0)
        
        forces = self.torque_to_maximal @ self.bending_energy(theta_rel, dtheta_rel)
        
        # Solve target
        next_dstate = cp.Variable((self.nbodies * 3,)) 
        
        # Minimization objective
        # TODO Why subtract forces. Paper and code don't agree
        objective = cp.Minimize(0.5 * cp.quad_form(next_dstate, self.M) - 
                                next_dstate.T @ (self.M @ self.dstate - forces * self.dt))

        # Constrains for non-collision and joint connectivity
        # TODO in the julia they divide sdf_now and deviation_now by dt
        # technically equivalent, but why? numerical stability?
        constraints = [ sdf_now + (L @ next_dstate) * self.dt >= 0,
                        deviation_now + (J @ next_dstate) * self.dt == 0]
        
        # TODO For backprop, need to set Parameters
        problem = cp.Problem(objective, constraints)
        
        # Well formedness
        assert problem.is_dpp()
        
        problem.solve(solver=cp.OSQP, max_iter=1000) # requires_grad=True)
        
        if problem.status != cp.OPTIMAL:
            print("status:", problem.status)
                                        
        next_dstate_solution = torch.tensor(next_dstate.value, dtype=torch.float)
        
        # Evolve
        self.state += next_dstate_solution * self.dt
        
        self.dstate = next_dstate_solution
        
        # Debug plots
        min_term = 0.5 * next_dstate_solution @ self.M @ next_dstate_solution - \
                   next_dstate_solution @ (self.M @ self.dstate - forces * self.dt)
        
        min_term_hist.append(min_term)
        # joint_constraint.append(deviation_now / self.dt + J @ next_dstate_solution)
        optimzied_deviation = deviation_now / self.dt + J @ next_dstate_solution
        actual_deviation = self.joint_deviation(self.state)
        joint_constraint.append(optimzied_deviation)
        fig_ax.cla()
        # fig_ax.plot(list(range(len(min_term_hist))), min_term_hist, label='min term', c='b')
        fig_ax.plot(list(range(len(joint_constraint))), joint_constraint, label='joint contraint', c='g')
        
ww = 100
main_ax = None
fig_ax = None
def vis_init():
    plt.figure(1, figsize=(10, 10))
    
    global main_ax, fig_ax
    main_ax = plt.gca()
    main_ax.set_aspect('equal')
    main_ax.set_xlim(-ww, ww)
    main_ax.set_ylim(-ww, ww)
    
    plt.figure(2, figsize=(10, 5))
    fig_ax = plt.gca()

def draw(vine: Vine):
    global main_ax, fig_ax
    main_ax.cla()
    
    main_ax.set_aspect('equal')
    main_ax.set_xlim(-ww, ww)
    main_ax.set_ylim(-ww, ww)
    
    def radius2pt(radius):
        # https://stackoverflow.com/questions/33094509/correct-sizing-of-markers-in-scatter-plot-to-a-radius-r-in-matplotlib
        sqrt_rad = radius * 72 / 4  # 1 point = dpi / 72 pixels
        return sqrt_rad * sqrt_rad
    
    # Draw the obstacles
    for obstacle in vine.obstacles:
        obstacle_patch = Rectangle((obstacle[0], obstacle[1]),
                                    obstacle[2] - obstacle[0], obstacle[3] - obstacle[1],
                                    linewidth=1, edgecolor='r', facecolor='gray')
        main_ax.add_patch(obstacle_patch)

    # Draw each body
    for i in range(vine.nbodies):
        # Body endpoints
        x_start = vine.xslice[i] - vine.d[i] * torch.cos(vine.thetaslice[i])
        y_start = vine.yslice[i] - vine.d[i] * torch.sin(vine.thetaslice[i])
        x_end = vine.xslice[i] + vine.d[i] * torch.cos(vine.thetaslice[i])
        y_end = vine.yslice[i] + vine.d[i] * torch.sin(vine.thetaslice[i])

        main_ax.plot([x_start, x_end], [y_start, y_end], c='blue', linewidth=10)
        # main_ax.scatter(vine.xslice[i], vine.yslice[i], c='pink', s=radius2pt(vine.radius))        
        
        for x, y in zip(vine.xslice, vine.yslice):
            circle = plt.Circle((x, y), vine.radius, color='g', fill=False)
            main_ax.add_patch(circle)
        
        if hasattr(vine, 'dbg_dist'):
            for x, y, dist, contact in zip(vine.xslice, vine.yslice, vine.dbg_dist, vine.dbg_contactpts):
                # Distance text
                # main_ax.text(x + 1, y + 0, f'{dist:.3f}')
                # Contact point
                # main_ax.arrow(x, y, contact[0] - x, contact[1] - y)
                pass
        
if __name__ == '__main__':
    vine = Vine(nbodies=6, init_heading_deg=-35, obstacles = [
            (15, 0, 30, 30) # x, y, x2, y2
        ])
    
    vis_init()
    
    draw(vine)
    plt.pause(0.001)
    
    for frame in range(1000):
        vine.evolve()
        
        draw(vine)
        
        if frame % 1 == 0:
            plt.pause(0.001)
        
    plt.show()