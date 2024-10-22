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
    plt.figure(1, figsize = (10, 10))

    global main_ax, fig_ax
    main_ax = plt.gca()
    main_ax.set_aspect('equal')
    # main_ax.set_xlim(-ww, ww)
    # main_ax.set_ylim(-ww, ww)

    # plt.figure(2, figsize=(10, 5))
    # fig_ax = plt.gca()


def draw_one_vine(x, y, theta, params):
    # Draw each body
    for i in range(x.shape[-1] - 1):
        # Body endpoints
        x_start = x[i] - params.half_len * torch.cos(theta[i])
        y_start = y[i] - params.half_len * torch.sin(theta[i])
        x_end = x[i] + params.half_len * torch.cos(theta[i])
        y_end = y[i] + params.half_len * torch.sin(theta[i])

        main_ax.plot([x_start, x_end], [y_start, y_end], c = 'blue', linewidth = 5)
        # main_ax.scatter(state.x[i], state.y[i], c='pink', s=radius2pt(vine.radius))

    # Draw last body
    x_start = x[-2] + params.half_len * torch.cos(theta[-2])
    y_start = y[-2] + params.half_len * torch.sin(theta[-2])
    x_end = x[-1]
    y_end = y[-1]
    main_ax.plot([x_start, x_end], [y_start, y_end], c = 'blue', linewidth = 5)

    # Draw circle colliders
    for x, y in zip(x, y):
        circle = plt.Circle((x, y), params.radius, color = 'g', fill = False)
        main_ax.add_patch(circle)


def draw(params: VineParams, state, dstate, bodies):
    global main_ax, fig_ax
    main_ax.cla()

    main_ax.set_aspect('equal')
    main_ax.set_xlim(-30, 400)
    main_ax.set_ylim(-490, 30)

    # Draw the obstacles
    for obstacle in params.obstacles:
        obstacle_patch = Rectangle(
            (obstacle[0], obstacle[1]),
            obstacle[2] - obstacle[0],
            obstacle[3] - obstacle[1],
            linewidth = 1,
            edgecolor = 'black',
            facecolor = 'moccasin'
            )
        main_ax.add_patch(obstacle_patch)

    for i in range(state.shape[0]):
        state_item = StateTensor(state[i])
        len = bodies[i]

        draw_one_vine(state_item.x[:len], state_item.y[:len], state_item.theta[:len], params)

    # if hasattr(params, 'dbg_dist'):
    #     for x, y, dist, contact in zip(state.x, state.y, params.dbg_dist, params.dbg_contactpts):
    #         # Distance text
    #         # main_ax.text(x + 1, y + 0, f'{dist:.3f}')
    #         # Contact point
    #         # main_ax.arrow(x, y, contact[0] - x, contact[1] - y)
    #         pass
