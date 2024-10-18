        
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from .vine import StateTensor, VineParams

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

ww = 400
main_ax = None
fig_ax = None

def vis_init():
    plt.figure(1, figsize=(10, 10))
    
    global main_ax, fig_ax
    main_ax = plt.gca()
    main_ax.set_aspect('equal')
    # main_ax.set_xlim(-ww, ww)
    # main_ax.set_ylim(-ww, ww)
    
    # plt.figure(2, figsize=(10, 5))
    # fig_ax = plt.gca()

def draw(params: VineParams, state, dstate):
    global main_ax, fig_ax
    main_ax.cla()
    
    main_ax.set_aspect('equal')
    main_ax.set_xlim(-30, 400)
    main_ax.set_ylim(-490, 30)
    
    state = StateTensor(state)
    dstate = StateTensor(dstate)
    
    # Draw the obstacles
    for obstacle in params.obstacles:
        obstacle_patch = Rectangle((obstacle[0], obstacle[1]),
                                    obstacle[2] - obstacle[0], obstacle[3] - obstacle[1],
                                    linewidth=1, edgecolor='black', facecolor='moccasin')
        main_ax.add_patch(obstacle_patch)

    # Draw each body
    for i in range(params.nbodies - 1):
        # Body endpoints
        x_start = state.x[i] - params.half_len * torch.cos(state.theta[i])
        y_start = state.y[i] - params.half_len * torch.sin(state.theta[i])
        x_end = state.x[i] + params.half_len * torch.cos(state.theta[i])
        y_end = state.y[i] + params.half_len * torch.sin(state.theta[i])

        main_ax.plot([x_start, x_end], [y_start, y_end], c='blue', linewidth=10)
        # main_ax.scatter(state.x[i], state.y[i], c='pink', s=radius2pt(vine.radius))       
    
    # Draw last body
    x_start = state.x[-2] + params.half_len * torch.cos(state.theta[-2])
    y_start = state.y[-2] + params.half_len * torch.sin(state.theta[-2])
    x_end = state.x[-1]
    y_end = state.y[-1]
    main_ax.plot([x_start, x_end], [y_start, y_end], c='blue', linewidth=10)
        
    for x, y in zip(state.x, state.y):
        circle = plt.Circle((x, y), params.radius, color='g', fill=False)
        main_ax.add_patch(circle)
    
    if hasattr(params, 'dbg_dist'):
        for x, y, dist, contact in zip(state.x, state.y, params.dbg_dist, params.dbg_contactpts):
            # Distance text
            # main_ax.text(x + 1, y + 0, f'{dist:.3f}')
            # Contact point
            # main_ax.arrow(x, y, contact[0] - x, contact[1] - y)
            pass 