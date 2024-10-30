import torch
import math
from sim.vine import VineParams
import matplotlib.pyplot as plt

def predict_moment(params, turning_angle):
    """
    Given a vector of turning angles in radians, pressure in the robot, and radius of the robot,
    compute the bending moment for each angle.
    """
    # Ensure that P and R are tensors
    P = params.sicheng
    R = 1

    full_moment = torch.pi * P * R**3  # Constant-moment model prediction at full wrinkling
    phiTrans = eval_phi_trans(P) # eval_phi_trans(P)       # Transition angle from linear to wrinkling-based model
    eps_crit = eval_eps(P)             # Critical strain leading to wrinkling

    # Separate angles based on whether they are in the wrinkling regime or linear regime
    wrinkling_mask = turning_angle > phiTrans
    
    # Initialize alp tensor for all angles
    alp = torch.zeros_like(turning_angle)
    
    # Wrinkling-based model for angles greater than phiTrans
    phi2_wrinkling = turning_angle / 2
    the_0_wrinkling = torch.arccos(2 * eps_crit / torch.sin(phi2_wrinkling) - 1)
    alp_wrikle = (torch.sin(2 * the_0_wrinkling) + 2 * torch.pi - 2 * the_0_wrinkling) / \
                          (4 * (torch.sin(the_0_wrinkling) + torch.cos(the_0_wrinkling) * (torch.pi - the_0_wrinkling)))
    
    # Linear elastic model for angles less than or equal to phiTrans
    phi2_linear = phiTrans / 2
    the_0_linear = torch.arccos(2 * eps_crit / torch.sin(phi2_linear) - 1)
    alp_trans = (torch.sin(2 * the_0_linear) + 2 * torch.pi - 2 * the_0_linear) / \
                (4 * (torch.sin(the_0_linear) + torch.cos(the_0_linear) * (torch.pi - the_0_linear)))
    alp_nowrinkle = alp_trans / phiTrans * turning_angle
    
    alp = torch.where(wrinkling_mask, alp_wrikle, alp_nowrinkle)
    
    # Calculate the bending moment for each angle
    M = alp * full_moment *params.sicheng2
    return M

def eval_eps(P):
    """
    Calculate critical strain that causes wrinkling at different pressures.
    """
    P_scaled = (P - 6167) / 5836
    eps_crit = (0.002077863400343 * P_scaled**3 +
                0.009091543113141 * P_scaled**2 +
                0.014512785114617 * P_scaled +
                0.007656015122415)
    return torch.tensor(eps_crit, dtype=torch.float32)
    

def eval_phi_trans(P):
    """
    Calculate the bending angle that sees the transition from the linear model to the wrinkling-based model.
    """
    P_scaled = (P - 8957) / 5149
    phiTrans = (0.003180574067535 * P_scaled**3 +
                0.020924128997619 * P_scaled**2 +
                0.048366932757916 * P_scaled +
                0.037544481890778)
    return torch.tensor(phiTrans, dtype=torch.float32)
    
def main():
    params = VineParams(
        max_bodies = 90,
        obstacles = [[0, 0, 0, 0]],
        grow_rate = -1,
        stiffness_mode = 'real',
        stiffness_val = torch.tensor([30_000.0 / 100_000.0], dtype = torch.float32)
        )
    params.sicheng=torch.tensor(20_335, dtype=torch.float32)
    params.sicheng2=torch.tensor(1/160_000, dtype=torch.float32)
    moments=[]
    for angle in range(90):
        moment = predict_moment(params, torch.tensor(math.radians(angle), dtype=torch.float32))
        moments.append(moment)
    plt.plot(range(90), moments)
    plt.show()
if __name__ == '__main__':
    main()