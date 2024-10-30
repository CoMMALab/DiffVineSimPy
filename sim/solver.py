import numpy as np
import scipy
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
#from qpth.qp import QPFunction, QPSolvers, SpQPFunction
from torch.autograd import Variable
# from lqp_py import box_qp_control
# from lqp_py import SolveBoxQP

from .sqrtm import MatrixSquareRoot, sqrtm

# Set up cvxpylayers constraints
cvxpylayer = None


def init_layers(sol_size, Q_size, p_size, G_size, h_size, A_size, b_size, vel_cap = float('inf')):
    '''
    Set up the cvxpylayer for the QP problem (inputs are unbatched sizes)
    '''

    global cvxpylayer
    if cvxpylayer is not None:
        return

    # Tell cvxpy what params can be differentiable
    next_dstate = cp.Variable(sol_size)
    Q_sqrt = cp.Parameter(Q_size)
    p = cp.Parameter(p_size)
    G = cp.Parameter(G_size)
    h = cp.Parameter(h_size)
    A = cp.Parameter(A_size)
    b = cp.Parameter(b_size)

    # print('Q size', Q_size, 'p size', p_size, 'G size', G_size, 'h size', h_size, 'A size', A_size, 'b size', b_size)

    objective = cp.Minimize(0.5 * cp.sum_squares(Q_sqrt @ next_dstate) + p @ next_dstate)
    # objective = cp.Minimize(0.5 * cp.quad_form(next_dstate, Q_sqrt) + p @ next_dstate)

    constraints = [A @ next_dstate == b, G @ next_dstate <= h]

    problem = cp.Problem(objective, constraints)

    print('is dcp?', problem.is_dcp())
    # assert problem.is_dpp()

    cvxpylayer = CvxpyLayer(problem, parameters = [Q_sqrt, p, G, h, A, b], variables = [next_dstate])


sqrtm_module = MatrixSquareRoot()


def solve_layers(Q, p, G, h, A, b):
    '''
    Batched QP solve, internally uses cvxpylayers
    '''
    global cvxpylayer

    # solver_args = {'ignore_dpp': True, 'max_iters': 100000}
    # Args to SCS, not cvxpy!

    batch_size = p.shape[0]
    Q_batched = sqrtm_module.apply(Q).unsqueeze(0).expand(batch_size, -1, -1)

    solver_args_scs = {'acceleration_lookback': 40_000, 'verbose': False, 'max_iters': 10_000}
    # solver_args_ecos = {'abstol': 1e-9, 'reltol': 1e-9, 'feastol': 1e-9, 'max_iters': 1000}
        
    solution = cvxpylayer(Q_batched, p, G, h, A, b, solver_args = solver_args_scs)

    return solution[0]


def solve_cvxpy(Q, p, G, h, A, b, **solver_kwargs):

    batch_size = Q.shape[0]
    max_bodies = Q.shape[1] // 3
    assert Q.shape[1] == Q.shape[2]

    next_dstate_solution = torch.zeros((batch_size, max_bodies * 3))

    for i in range(batch_size):
        # print('Q size', Q.shape[1:], 'p size', p.shape[1:], 'G size', G.shape[1:], 'h size', h.shape[1:], 'A size', A.shape[1:], 'b size', b.shape[1:])

        next_dstate = cp.Variable((max_bodies * 3, ))

        objective = cp.Minimize(0.5 * cp.quad_form(next_dstate, Q[i]) + p[i].unsqueeze(0) @ next_dstate)

        constraints = [A[i] @ next_dstate == b[i], G[i] @ next_dstate <= h[i]]

        problem = cp.Problem(objective, constraints)

        assert problem.is_dpp()

        problem.solve(**solver_kwargs)

        if problem.status != cp.OPTIMAL:
            print("status:", problem.status)

        next_dstate_solution[i] = torch.tensor(next_dstate.value, dtype = torch.float)

    return next_dstate_solution


# control = box_qp_control(max_iters = 1000, eps_rel = 1e-1, eps_abs = 1e-1, verbose = True)
# QP = SolveBoxQP(control = control)


def solve_lqp(Q, p, G, h, A, b):

    print('Rank of G', torch.linalg.matrix_rank(G[0]), 'Rank of A', torch.linalg.matrix_rank(A[0]))
    ub, residuals, rank, singular_values = torch.linalg.lstsq(G, h)

    ub = ub.unsqueeze(-1)
    p = p.unsqueeze(-1)
    b = b.unsqueeze(-1)

    print('Q shape', Q.shape, "p shape", p.shape)
    print('G shape', G.shape, "h shape", h.shape, 'ub shape', ub.shape)
    print('A shape', A.shape, "b shape", b.shape)

    lb = torch.full_like(ub, -1000)

    # Add some slack to A
    # The fuller rank is A, the better convergence
    diag = torch.arange(A.shape[1])
    A[:, diag, diag] += 10
    # A += torch.randn_like(A) * 1e-8

    solution = QP.forward(Q = Q, p = p, A = A, b = b, lb = lb, ub = ub)

    print('Solution shape', solution.shape)

    return solution.squeeze(-1)


# qp_layer = QPFunction(verbose = False, eps = 1e-6, solver = QPSolvers.PDIPM_BATCHED)


def solve_qpth(Q, p, G, h, A, b):
    return qp_layer(Q, p, G, h, A, b)


def solve_sqpth(Q, p, G, h, A, b):
    # self, Qi, Qsz, Gi, Gsz, Ai, Asz, eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20):
    solver2 = SpQPFunction(
        0, Q.shape, 0, G.shape, 0, A.shape, eps = 1e-6, verbose = False, notImprovedLim = 3, maxIter = 20
        )
    return solver2(Variable(Q), Variable(p), Variable(G), Variable(h), Variable(A), Variable(b))


def solve_cvxpy_combined(Q, p, G, h, A, b, **solver_kwargs):
    batch_size = Q.shape[0]
    n_vars = Q.shape[1]    # Number of variables per problem

    # Create a single CVXPY variable for all problems
    next_dstate_combined = cp.Variable(batch_size * n_vars)

    # Convert PyTorch tensors to NumPy arrays
    Q_np = Q.cpu().numpy()
    p_np = p.cpu().numpy()
    G_np = G.cpu().numpy()
    h_np = h.cpu().numpy()
    A_np = A.cpu().numpy()
    b_np = b.cpu().numpy()

    # Create block-diagonal Hessian matrix Q_combined
    Q_blocks = [scipy.sparse.csr_matrix(Q_np[i]) for i in range(batch_size)]
    Q_combined = scipy.sparse.block_diag(Q_blocks)

    # Concatenate linear term p_combined
    p_combined = p_np.flatten()

    # Initialize lists for constructing G_combined and h_combined
    G_data, G_rows, G_cols = [], [], []
    h_combined = []

    row_offset = 0
    col_offset = 0
    for i in range(batch_size):
        G_i = scipy.sparse.coo_matrix(G_np[i])
        n_constraints_i = G_i.shape[0]

        # Adjust row and column indices for block-diagonal structure
        G_rows.extend(G_i.row + row_offset)
        G_cols.extend(G_i.col + col_offset)
        G_data.extend(G_i.data)

        h_combined.extend(h_np[i])

        row_offset += n_constraints_i
        col_offset += n_vars

    total_G_rows = row_offset
    total_G_cols = batch_size * n_vars
    G_combined = scipy.sparse.coo_matrix((G_data, (G_rows, G_cols)), shape = (total_G_rows, total_G_cols))

    h_combined = np.array(h_combined)

    # Initialize lists for constructing A_combined and b_combined
    A_data, A_rows, A_cols = [], [], []
    b_combined = []

    row_offset = 0
    col_offset = 0
    for i in range(batch_size):
        A_i = scipy.sparse.coo_matrix(A_np[i])
        n_constraints_e = A_i.shape[0]

        # Adjust row and column indices for block-diagonal structure
        A_rows.extend(A_i.row + row_offset)
        A_cols.extend(A_i.col + col_offset)
        A_data.extend(A_i.data)

        b_combined.extend(b_np[i])

        row_offset += n_constraints_e
        col_offset += n_vars

    total_A_rows = row_offset
    total_A_cols = batch_size * n_vars
    A_combined = scipy.sparse.coo_matrix((A_data, (A_rows, A_cols)), shape = (total_A_rows, total_A_cols))

    b_combined = np.array(b_combined)

    # Define the objective function
    objective = cp.Minimize(
        0.5 * cp.quad_form(next_dstate_combined, Q_combined) + p_combined @ next_dstate_combined
        )

    # Define the constraints
    constraints = [
        A_combined @ next_dstate_combined == b_combined,
        G_combined @ next_dstate_combined <= h_combined,
        ]

    # Form and solve the problem
    problem = cp.Problem(objective, constraints)

    # Ensure the problem is differentiable
    assert problem.is_dpp()

    # Solve the problem using the specified solver
    problem.solve(**solver_kwargs)

    # Check if the problem was solved optimally
    if problem.status != cp.OPTIMAL:
        print("status:", problem.status)

    # Extract the solution and reshape it to match the original batch structure
    next_dstate_combined_value = next_dstate_combined.value # NumPy array
    next_dstate_solution = torch.from_numpy(next_dstate_combined_value).float().view(batch_size, n_vars)

    return next_dstate_solution
