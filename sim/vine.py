import math
import torch
from torch.autograd.functional import jacobian
import cvxpy as cp


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

class Vine:
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

        # State variables
        self.state = torch.zeros(self.nbodies * 3)
        self.dstate = torch.zeros(self.nbodies * 3)
        
        # Init positions
        self.thetaslice[:] = torch.linspace(self.init_heading, self.init_heading + 0, self.nbodies)
        self.xslice[0] = self.half_len * torch.cos(self.thetaslice[0])
        self.yslice[0] = self.half_len * torch.sin(self.thetaslice[0])
        
        for i in range(1, self.nbodies):
            lastx = self.xslice[i - 1]
            lasty = self.yslice[i - 1]
            lasttheta = self.thetaslice[i - 1]
            thistheta = self.thetaslice[i]
            
            self.xslice[i] = lastx + self.half_len * torch.cos(lasttheta) + self.half_len * torch.cos(thistheta)
            self.yslice[i] = lasty + self.half_len * torch.sin(lasttheta) + self.half_len * torch.sin(thistheta)
        
        # Stiffness and damping coefficients
        self.stiffness = 15_000.0 # 30_000.0  # Stiffness coefficient
        self.damping = 50.0       # Damping coefficient

        # Environment obstacles (rects only for now)
        self.obstacles = obstacles
        
        # Make sure x < x2 y < y2
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
    
    def create_M(self):
        # Update mass matrix M (block diagonal)
        m_block = torch.tensor([self.m] * self.nbodies)  # nbodies of self.m
        i_block = torch.tensor([self.I] * self.nbodies)  # nbodies of self.I

        diagonal_elements = torch.cat([m_block, m_block, i_block])
        return torch.diag(diagonal_elements)  # Shape: (nq, nq))
        
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
            
            dist = torch.where(isinside, -dist, dist) 
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
        constraints = torch.zeros(self.nbodies * 2 - 1)
        
        constraints[0] = x[0] - self.half_len * torch.cos(theta[0])
        constraints[1] = y[0] - self.half_len * torch.sin(theta[0])
        
        for j2 in range(1, self.nbodies - 1):
            j1 = j2 - 1
            constraints[j2 * 2] = (x[j2] - x[j1]) - (self.half_len * torch.cos(theta[j1]) + self.half_len * torch.cos(theta[j2]))
            
            constraints[j2 * 2 + 1] = (y[j2] - y[j1]) - (self.half_len * torch.sin(theta[j1]) + self.half_len * torch.sin(theta[j2]))
        
        # Last body is special. It has a sliding AND rotation joint with the second-last body
        endx = x[-2] + self.half_len * torch.cos(theta[-2])
        endy = y[-2] + self.half_len * torch.sin(theta[-2])
        
        constraints[-1] = torch.atan2(y[-1] - endy, x[-1] - endx) - theta[-1]
        
        return constraints

    def bending_energy(self, theta, dtheta):
        # Compute the response (like potential energy of bending)
        # Can think of the system as always wanting to get rid of potential 
        # Generally, \tau = - stiffness * benderino - damping * d_benderino
        
        # return -(self.K @ theta) - self.C @ dtheta
        return -self.stiffness * theta - self.damping * dtheta
        # return -self.stiffness * torch.where(torch.abs(theta) < 0.3, theta * 2, theta * 0.5) - self.damping * dtheta
        # return torch.sign(theta) * -(torch.sin((torch.abs(theta) + 0.1) / 0.3) + 1.4)
        
    def extend(self):        
        # Compute position of second last seg
        endingx = self.xslice[-2] + self.half_len * torch.cos(self.thetaslice[-2])
        endingy = self.yslice[-2] + self.half_len * torch.sin(self.thetaslice[-2])
            
        # Compute last body's distance 
        last_link_distance = ((self.xslice[-1] -endingx)**2 + (self.yslice[-1] - endingy)**2).sqrt()
        last_link_theta = torch.atan2(self.yslice[-1] - self.yslice[-2], self.xslice[-1] - self.xslice[-2])
    
        if last_link_distance > self.half_len * 2: # x2 to prevent 0-len segments
            print(f'Extending!')
            
            # Compute location of new seg
            new_seg_x = endingx + self.half_len * torch.cos(last_link_theta)
            new_seg_y = endingy + self.half_len * torch.sin(last_link_theta)
            new_seg_theta = last_link_theta
            
            # Cache old data
            oldx = self.xslice.clone()
            oldy = self.yslice.clone()
            oldtheta = self.thetaslice.clone()
            olddx = self.dxslice.clone()
            olddy = self.dyslice.clone()
            olddtheta = self.dthetaslice.clone()
            
            # Extend state variables
            self.state = torch.cat((self.state, torch.zeros(3)))
            self.dstate = torch.cat((self.dstate, torch.zeros(3)))
            self.nbodies += 1
            self.M = self.create_M()
            
            # Clone position
            self.xslice[:-1] = oldx
            self.xslice[-1] = oldx[-1].clone()
            self.yslice[:-1] = oldy
            self.yslice[-1] = oldy[-1].clone()
            self.thetaslice[:-1] = oldtheta
            self.thetaslice[-1] = oldtheta[-1].clone()
            
            # Clone velocity
            self.dxslice[:-1] = olddx
            self.dxslice[-1] = self.dxslice[-2].clone()
            self.dyslice[:-1] = olddy
            self.dyslice[-1] = self.dyslice[-2].clone()
            self.dthetaslice[:-1] = olddtheta
            self.dthetaslice[-1] = self.dthetaslice[-2].clone()
            
            # Set the new segment position
            self.xslice[-2] = new_seg_x
            self.yslice[-2] = new_seg_y
            self.thetaslice[-2] = new_seg_theta
            self.dxslice[-2] = 0 
            self.dyslice[-2] = 0 
            self.dthetaslice[-2] = 0 
            
            return True
        return False
    
    def grow2(self, stateanddstate):        
        state = stateanddstate[:self.nbodies*3]
        dstate = stateanddstate[self.nbodies*3:]
        
        # Now return the constraints for growing the last segment        
        x1 = state[self.nbodies - 2]
        y1 = state[self.nbodies * 2 - 2]
        vx1 = dstate[self.nbodies - 2]
        vy1 = dstate[self.nbodies * 2 - 2]
        
        x2 = state[self.nbodies - 1]
        y2 = state[self.nbodies * 2 - 1]
        vx2 = dstate[self.nbodies - 1]
        vy2 = dstate[self.nbodies * 2 - 1]
            
        constraint = ((x2-x1)*(vx2-vx1) + (y2-y1)*(vy2-vy1)) / \
                                torch.sqrt((x2-x1)**2 + (y2-y1)**2)
            
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
            
    def evolve(self):
        extended = self.extend()
        
        # if extended: return
        
        # Jacobian of SDF with respect to x and y
        L = jacobian(self.sdf, self.state)
        sdf_now = self.sdf(self.state)
        
        # Jacobian of joint deviation wrt configuration
        J = jacobian(self.joint_deviation, self.state)
        deviation_now = self.joint_deviation(self.state)
        
        G = jacobian(self.grow2, torch.cat([self.state, self.dstate]))
        growth = self.grow2(torch.cat([self.state, self.dstate]))
        
        # Stiffness forces
        theta_rel = finite_changes(self.thetaslice, self.init_heading)
        dtheta_rel = finite_changes(self.dthetaslice, 0.0)
        
        bend_energy = self.bending_energy(theta_rel, dtheta_rel)
        
        forces = torch.zeros(self.nbodies*3)
        forces[self.nbodies*2:] += -bend_energy # Apply bend energy as torque to own joint
        forces[self.nbodies*2:-1] += bend_energy[1:] # Apply bend energy as torque to joint before
        
        # Solve target
        next_dstate = cp.Variable((self.nbodies * 3,)) 
        
        # Minimization objective
        # TODO Why subtract forces. Paper and code don't agree
        objective = cp.Minimize(0.5 * cp.quad_form(next_dstate, self.M) - 
                                next_dstate.T @ (self.M @ self.dstate - forces * self.dt))

        # Constrains for non-collision and joint connectivity
        
        growth_wrt_state = G[:self.nbodies*3]
        growth_wrt_dstate = G[self.nbodies*3:]
        
        g_con = growth - self.grow_rate - growth_wrt_dstate @ self.dstate 
        g_coeff = growth_wrt_state * self.dt + growth_wrt_dstate
        
        # TODO in the julia they divide sdf_now and deviation_now by dt
        # technically equivalent, but why? numerical stability?
        constraints = [ sdf_now + (L @ next_dstate) * self.dt >= 0,
                        deviation_now + (J @ next_dstate) * self.dt == 0,
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
        self.state += next_dstate_solution * self.dt
        
        self.dstate = next_dstate_solution
