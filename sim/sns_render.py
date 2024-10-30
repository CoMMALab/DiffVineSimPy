import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import signal
from .vine import StateTensor, VineParams

# Set signal handler
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Set up Seaborn theme
sns.set_theme(style="white")  # Use "white" to disable gridlines

# Plot parameters
ww = 400
main_ax = None
fig_ax = None

def vis_init():
    global main_ax, fig_ax
    plt.ion()  # Enable interactive mode
    plt.figure(1, figsize=(10, 10))
    main_ax = plt.gca()
    main_ax.set_aspect('equal')
    main_ax.axis('off')  # Remove axis numbers

def draw_one_vine(x, y, theta, params, **kwargs):
    # Draw each body segment in the vine with thicker blue line
    for i in range(x.shape[-1] - 1):
        x_start = x[i] - params.half_len * torch.cos(theta[i])
        y_start = y[i] - params.half_len * torch.sin(theta[i])
        x_end = x[i] + params.half_len * torch.cos(theta[i])
        y_end = y[i] + params.half_len * torch.sin(theta[i])
        main_ax.plot([x_start, x_end], [y_start, y_end], linewidth=params.radius * 2, alpha=0.5, **kwargs)
    
    # Draw the last body with thicker blue line
    x_start = x[-2] + params.half_len * torch.cos(theta[-2])
    y_start = y[-2] + params.half_len * torch.sin(theta[-2])
    x_end = x[-1]
    y_end = y[-1]
    main_ax.plot([x_start, x_end], [y_start, y_end], linewidth=params.radius * 2, alpha=0.5, **kwargs)

def draw_batched(params: VineParams, state, bodies, lims=False, clear=True, obstacles=True, **kwargs):
    global main_ax, fig_ax

    if clear:
        main_ax.cla()
        main_ax.set_aspect('equal')
        main_ax.axis('off')  # Remove axis numbers again after clearing

    if lims:
        main_ax.set_xlim(-30, 400)
        main_ax.set_ylim(-490, 30)

    # Draw the obstacles with Seaborn styling
    if obstacles:
        for obstacle in params.obstacles:
            obstacle_patch = Rectangle(
                (obstacle[0], obstacle[1]),
                obstacle[2] - obstacle[0],
                obstacle[3] - obstacle[1],
                linewidth=1,
                edgecolor='black',
                facecolor=sns.color_palette("pastel")[1]
            )
            main_ax.add_patch(obstacle_patch)

    # Draw each vine in the batch without red circles
    for i in range(state.shape[0]):
        state_item = StateTensor(state[i])
        leng = bodies[i]
        draw_one_vine(state_item.x[:leng], state_item.y[:leng], state_item.theta[:leng], params, **kwargs)

    plt.draw()  # Draws without blocking
    plt.pause(0.001)