import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

class VineRobot:
    def __init__(self, nb, dt):
        self.nb = nb  # Number of bodies
        self.dt = dt  # Time step
        self.d = 1.0  # Half-length of each body
        self.nq = 3 * nb  # Number of state variables

        # Robot parameters
        self.m = 1.0  # Mass of each body
        self.I = 1.0  # Moment of inertia of each body

        # State variables
        self.q = torch.zeros(self.nq, requires_grad=True)  # Positions: [x0, y0, theta0, x1, y1, theta1, ...]
        self.v = torch.zeros(self.nq)  # Velocities: [vx0, vy0, omega0, vx1, vy1, omega1, ...]

        # Mass matrix M (block diagonal)
        M_blocks = [torch.diag(torch.tensor([self.m, self.m, self.I])) for _ in range(nb)]
        self.M = torch.block_diag(*M_blocks)  # Shape: (nq, nq)

        # Stiffness and damping coefficients
        self.k = 10.0  # Stiffness coefficient
        self.c = 1.0   # Damping coefficient

        # Environment obstacles (for simplicity, a single rectangle)
        self.obstacle = {'x': 2.0, 'y': 3.0, 'width': 3.0, 'height': 1.0}

        # Set up the cvxpylayer
        self.setup_cvxpy_layer()

    def setup_cvxpy_layer(self):
        nq = self.nq
        nc = 2 + 2 * (self.nb - 1)  # Number of joint constraints

        # cvxpy variables and parameters
        v_var = cp.Variable(nq)
        v_k_param = cp.Parameter(nq)
        F_param = cp.Parameter(nq)
        c_qk_param = cp.Parameter(nc)
        J_param = cp.Parameter((nc, nq))

        M = self.M.detach().numpy()
        dt = self.dt

        # Objective function
        objective = cp.Minimize(0.5 * cp.quad_form(v_var, M) - (M @ v_k_param + F_param * dt).T @ v_var)

        # Constraints
        constraints = [c_qk_param + J_param @ v_var * dt == 0]

        # Contact constraints (simplified as non-penetration constraints)
        # For this example, we'll skip contact constraints for clarity

        # Set up the problem
        problem = cp.Problem(objective, constraints)

        # Create the cvxpylayer
        self.cvxpylayer = CvxpyLayer(problem, parameters=[v_k_param, F_param, c_qk_param, J_param], variables=[v_var])

    def compute_constraints(self, q):
        nb = self.nb
        d = self.d

        # Extract positions
        x = q[0::3]
        y = q[1::3]
        theta = q[2::3]

        c_list = []

        # Base pin joint between body 0 and the world frame origin
        c1 = x[0] - d * torch.cos(theta[0])
        c2 = y[0] - d * torch.sin(theta[0])
        c_list.extend([c1, c2])

        # Constraints for each joint
        for i in range(nb - 1):
            x_i = x[i]
            y_i = y[i]
            theta_i = theta[i]
            x_ip1 = x[i + 1]
            y_ip1 = y[i + 1]
            theta_ip1 = theta[i + 1]

            if i % 2 == 0:
                # Pin joint constraint
                c_pin1 = (x_i + d * torch.cos(theta_i)) - (x_ip1 - d * torch.cos(theta_ip1))
                c_pin2 = (y_i + d * torch.sin(theta_i)) - (y_ip1 - d * torch.sin(theta_ip1))
                c_list.extend([c_pin1, c_pin2])
            else:
                # Prismatic joint constraint
                b_i = torch.stack([-torch.sin(theta_i), torch.cos(theta_i)])
                b_ip1 = torch.stack([-torch.sin(theta_ip1), torch.cos(theta_ip1)])
                a = torch.stack([x_ip1 - x_i, y_ip1 - y_i])
                c_prismatic1 = torch.dot(b_i, a)
                c_prismatic2 = torch.dot(b_ip1, a)
                c_list.extend([c_prismatic1, c_prismatic2])

        c = torch.stack(c_list)
        return c

    def compute_c_and_J(self, q):
        # Compute c(q)
        c = self.compute_constraints(q)

        # Compute J = dc/dq using PyTorch autograd
        J = []
        for c_i in c:
            grad_c_i = torch.autograd.grad(c_i, q, retain_graph=True, create_graph=True)[0]
            J.append(grad_c_i)
        J = torch.stack(J)
        return c, J

    def compute_external_forces(self, q, v):
        # Compute external forces due to stiffness and damping
        nb = self.nb

        theta = q[2::3]
        omega = v[2::3]

        # Relative angles and angular velocities
        theta_rel = theta[1:] - theta[:-1]
        omega_rel = omega[1:] - omega[:-1]

        # Torques due to stiffness and damping
        tau = -self.k * theta_rel - self.c * omega_rel

        # Map torques to maximal coordinates
        F = torch.zeros(self.nq)
        for i in range(nb):
            if i > 0:
                F[3 * i + 2] += tau[i - 1]
            if i < nb - 1:
                F[3 * i + 2] -= tau[i]
        return F

    def simulate_step(self):
        q_k = self.q.detach().requires_grad_(True)
        v_k = self.v.detach()

        # Compute c(q_k) and J(q_k)
        c_qk, J = self.compute_c_and_J(q_k)

        # Compute external forces
        F = self.compute_external_forces(q_k, v_k)

        # Update cvxpy Parameters
        
        # v_k_param, F_param, c_qk_param, J_param = self.cvxpylayer.parameters()
        # v_k_param.value = v_k.numpy()
        # F_param.value = F
        # c_qk_param.value = c_qk
        # J_param.value = J

        # Solve the problem
        print(v_k.dtype, F.dtype, c_qk.dtype, J.dtype)
        # print(v_k, F, c_qk, J)
        
        v_sol, = self.cvxpylayer(v_k, F, c_qk, J)
        v_kp1 = torch.from_numpy(v_sol.value).float()
        self.v = v_kp1.detach()

        # Update positions
        self.q = q_k + self.v * self.dt
        self.q = self.q.detach().requires_grad_(True)

    def simulate(self, steps):
        q_history = []
        # Render initial state before first solve
        self.visualize(self.q.detach().numpy())
        plt.pause(1.0)  # Pause to view initial state

        for _ in range(steps):
            q_history.append(self.q.clone().detach().numpy())
            self.simulate_step()
            # Render current state
            self.visualize(self.q.detach().numpy())
            plt.pause(0.1)
        plt.show()
        return q_history

    def visualize(self, q):
        plt.clf()
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlim(-5, 10)
        ax.set_ylim(-5, 15)

        # Draw the obstacle
        obstacle_patch = Rectangle((self.obstacle['x'], self.obstacle['y']),
                                   self.obstacle['width'], self.obstacle['height'],
                                   linewidth=1, edgecolor='r', facecolor='gray')
        ax.add_patch(obstacle_patch)

        x = q[0::3]
        y = q[1::3]
        theta = q[2::3]

        # Draw each body
        for i in range(self.nb):
            x_i, y_i, theta_i = x[i], y[i], theta[i]
            dx = self.d * np.cos(theta_i)
            dy = self.d * np.sin(theta_i)

            # Body endpoints
            x_start = x_i - dx
            y_start = y_i - dy
            x_end = x_i + dx
            y_end = y_i + dy

            plt.plot([x_start, x_end], [y_start, y_end], 'b-', linewidth=2)
            plt.plot(x_i, y_i, 'ro')  # Center of mass

            # Draw orientation arrow
            plt.arrow(x_i, y_i, 0.5 * dx, 0.5 * dy, head_width=0.2, head_length=0.2, fc='k', ec='k')

        plt.draw()

if __name__ == '__main__':
    nb = 5  # Number of bodies
    dt = 0.1  # Time step
    steps = 50  # Number of simulation steps

    vine_robot = VineRobot(nb, dt)

    # Initialize the state
    for i in range(nb):
        vine_robot.q.data[3 * i] = 0.0  # x positions
        vine_robot.q.data[3 * i + 1] = i * 1.0  # y positions
        vine_robot.q.data[3 * i + 2] = 0.0  # orientations

    # Simulate and render
    q_history = vine_robot.simulate(steps)
