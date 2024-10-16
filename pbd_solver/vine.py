import math
import torch
from torch.autograd.functional import jacobian
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .util import *

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

min_term_hist = []
joint_constraint = []

class Vine:
    def __init__(self, nbodies, init_heading_deg=45, obstacles=[], grow_rate=10.0):
        self.nbodies = nbodies  # Number of bodies
        self.dt = 1/90  # Time step
        self.radius = 15.0 / 2  # Half-length of each body
        self.init_heading = math.radians(init_heading_deg)
        self.grow_rate = grow_rate # Length grown per unit time
        
        # Robot parameters
        self.m = 0.002 # 0.002  # Mass of each body
        self.I = 200  # Moment of inertia of each body
        self.default_body_half_len = 3
        
        # Dist from the center of each body to its end
        # Since the last body is connected via sliding joint, it is
        # not included in this tensor
        self.d = torch.full((self.nbodies - 1,), fill_value=self.default_body_half_len, dtype=torch.float) 

        # State variables
        self.state = torch.zeros(self.nbodies * 3)
        self.dstate = torch.zeros(self.nbodies * 3)
        
        # Init positions
        self.thetaslice[:] = torch.linspace(self.init_heading, self.init_heading + 0, self.nbodies)
        self.xslice[0] = self.d[0] * torch.cos(self.thetaslice[0])
        self.yslice[0] = self.d[0] * torch.sin(self.thetaslice[0])
        
        for i in range(1, self.nbodies):
            lastx = self.xslice[i - 1]
            lasty = self.yslice[i - 1]
            lasttheta = self.thetaslice[i - 1]
            thistheta = self.thetaslice[i]
            
            length = self.d[i] if i != self.nbodies - 1 else self.default_body_half_len
            self.xslice[i] = lastx + self.d[i-1] * torch.cos(lasttheta) + length * torch.cos(thistheta)
            self.yslice[i] = lasty + self.d[i-1] * torch.sin(lasttheta) + length * torch.sin(thistheta)
        
        # Stiffness and damping coefficients
        self.stiffness = 30000.0  # Stiffness coefficient
        self.damping = 10.0       # Damping coefficient

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
        constraints = torch.zeros(self.nbodies * 2 - 1)
        
        constraints[0] = x[0] - self.d[0] * torch.cos(theta[0])
        constraints[1] = y[0] - self.d[0] * torch.sin(theta[0])
        
        for j2 in range(1, self.nbodies - 1):
            j1 = j2 - 1
            constraints[j2 * 2] = (x[j2] - x[j1]) - (self.d[j1] * torch.cos(theta[j1]) + self.d[j2] * torch.cos(theta[j2]))
            
            constraints[j2 * 2 + 1] = (y[j2] - y[j1]) - (self.d[j1] * torch.sin(theta[j1]) + self.d[j2] * torch.sin(theta[j2]))
        
        # Last body is special. It has a sliding AND rotation joint with the second-last body
        constraints[-1] = torch.atan2(y[-1] - y[-2], x[-1] - x[-2]) - theta[-1]
        
        return constraints

    def bending_energy(self, theta, dtheta):
        # Compute the response (like potential energy of bending)
        # Can think of the system as always wanting to get rid of potential 
        # Generally, \tau = - stiffness * benderino - damping * d_benderino
        
        # return -(self.K @ theta) - self.C @ dtheta
        return -self.stiffness * theta - self.damping * dtheta
    
    def extend(self):        
        
        # Compute position of second last seg
        endingx = self.xslice[-2] + (self.d[-1]) * torch.cos(self.thetaslice[-2])
        endingy = self.yslice[-2] + (self.d[-1]) * torch.sin(self.thetaslice[-2])
            
        # Check last body's distance 
        last_link_distance = ((self.xslice[-1] -endingx)**2 + (self.yslice[-1] - endingy)**2).sqrt()
        last_link_theta = torch.atan2(self.yslice[-1] - self.yslice[-2], self.xslice[-1] - self.xslice[-2])
        
        # if last_link_distance/2 > self.default_body_half_len:
        #     print(f'{last_link_distance} is big, extend')
            
        # return
    
        if last_link_distance/2 > self.default_body_half_len:
            print(f'{last_link_distance} is big, extend')
            
            # Compute location of new seg
            new_seg_x = endingx + self.default_body_half_len * torch.cos(last_link_theta)
            new_seg_y = endingy + self.default_body_half_len * torch.sin(last_link_theta)
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
            self.d = torch.cat((self.d, torch.tensor([self.default_body_half_len])))
            self.nbodies += 1
            self.M = self.create_M()
            
            # Clone position
            self.xslice[:-1] = oldx
            self.xslice[-1] = oldx[-1].clone()
            self.yslice[:-1] = oldy
            self.yslice[-1] = oldy[-1].clone()
            print('a', oldx[-1], oldy[-1])
            print('b', self.xslice[-1], self.yslice[-1])
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
            self.dxslice[-2] = self.dxslice[-3]
            self.dyslice[-2] = self.dyslice[-3]
            self.dthetaslice[-2] = self.dthetaslice[-3]
            
            print(self.xslice[-1], self.yslice[-1])
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
        
        if extended: return
        
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
        print('rel bend', theta_rel)
        print('bend_energy', bend_energy)
        
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
        
        problem.solve(solver=cp.CLARABEL, max_iter=1000) # requires_grad=True)
        
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
        joint_constraint.append(optimzied_deviation.mean())
        
        # global fig_ax
        # fig_ax.cla()
        # fig_ax.plot(list(range(len(min_term_hist))), min_term_hist, label='min term', c='b')
        # fig_ax.plot(list(range(len(joint_constraint))), joint_constraint, label='joint contraint', c='g')
