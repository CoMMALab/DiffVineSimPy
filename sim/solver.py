import torch
import cvxpy as cp
from qpth.qp import QPFunction, QPSolvers
from torch.autograd import Variable

qp_layer = QPFunction(verbose=False, eps=1e-6, solver=QPSolvers.CVXPY)

def solve_qpth(Q, p, G, h, A, b):
    return qp_layer(Variable(Q), Variable(p), Variable(G), Variable(h), Variable(A), Variable(b))

def solve_cvxpy(Q, p, G, h, A, b):
    
    batch_size = Q.shape[0]
    max_bodies = Q.shape[1] // 3
    assert Q.shape[1] == Q.shape[2]
    
    next_dstate_solution = torch.zeros((batch_size, max_bodies * 3))
    
    for i in range(batch_size):
        next_dstate = cp.Variable((max_bodies * 3,)) 
        
        objective = cp.Minimize(0.5 * cp.quad_form(next_dstate, Q[i]) + 
                                p[i].unsqueeze(0) @ next_dstate)

        
        constraints = [ A[i] @ next_dstate == b[i],
                        G[i] @ next_dstate <= h[i]]
        
        problem = cp.Problem(objective, constraints)
        
        assert problem.is_dpp()
        
        problem.solve(solver=cp.SCS, verbose=False) # requires_grad=True)
        
        if problem.status != cp.OPTIMAL:
            print("status:", problem.status)
                                        
        next_dstate_solution[i] = torch.tensor(next_dstate.value, dtype=torch.float)
        
    return next_dstate_solution