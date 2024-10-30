import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import torchvision
import io

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


def draw_one_vine(x, y, theta, params, **kwargs):
    # Draw each body
    for i in range(x.shape[-1] - 1):
        # Body endpoints
        x_start = x[i] - params.half_len * torch.cos(theta[i])
        y_start = y[i] - params.half_len * torch.sin(theta[i])
        x_end = x[i] + params.half_len * torch.cos(theta[i])
        y_end = y[i] + params.half_len * torch.sin(theta[i])

        main_ax.plot([x_start, x_end], [y_start, y_end], linewidth = 5, **kwargs)
        # main_ax.scatter(state.x[i], state.y[i], c='pink', s=radius2pt(vine.radius))

    # Draw last body
    x_start = x[-2] + params.half_len * torch.cos(theta[-2])
    y_start = y[-2] + params.half_len * torch.sin(theta[-2])
    x_end = x[-1]
    y_end = y[-1]
    main_ax.plot([x_start, x_end], [y_start, y_end], linewidth = 5, **kwargs)

    # Draw circle colliders
    for x, y in zip(x, y):
        circle = plt.Circle((x, y), params.radius, color = 'g', alpha = 0.7, fill = False)
        main_ax.add_patch(circle)


def draw_batched(params: VineParams, state, bodies, lims=False, clear=True, obstacles=True, **kwargs):
    global main_ax, fig_ax
    
    if clear:
        main_ax.cla()

        main_ax.set_aspect('equal')
    
    if lims:
        main_ax.set_xlim(-30, 400)
        main_ax.set_ylim(-490, 30)

    # Draw the obstacles
    if obstacles and params.obstacles is not None:
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
    
    if obstacles and params.obstacles is None:
        # Draw params.segments (Nx4), as x1 y1 x2 y2
        for segment in params.segments:
            main_ax.plot([segment[0], segment[2]], [segment[1], segment[3]], color = 'black', linewidth = 1)
        
        
    for i in range(state.shape[0]):
        state_item = StateTensor(state[i])
        leng = bodies[i]
        
        draw_one_vine(state_item.x[:leng], state_item.y[:leng], state_item.theta[:leng], params, **kwargs)

    # if hasattr(params, 'dbg_dist'):
    #     for x, y, dist, contact in zip(state.x, state.y, params.dbg_dist, params.dbg_contactpts):
    #         # Distance text
    #         # main_ax.text(x + 1, y + 0, f'{dist:.3f}')
    #         # Contact point
    #         # main_ax.arrow(x, y, contact[0] - x, contact[1] - y)
    #         pass


stiff_fig, stiff_ax = plt.subplots()

def log_stiffness_func(writer, stiffness_func, iter):
    """
    Logs the weights, biases, gradients, and output plot of stiffness_func to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard writer instance.
        stiffness_func (torch.nn.Module): Stiffness function model.
        iter (int): Current iteration step for logging.
    """
    
    # Log weights, biases, and gradients for each layer in stiffness_func
    for i, layer in enumerate(stiffness_func):
        if isinstance(layer, torch.nn.Linear):
            # Log weights and biases
            writer.add_histogram(f'Stiffness_func/layer_{i}_weights', layer.weight, iter)
            writer.add_histogram(f'Stiffness_func/layer_{i}_biases', layer.bias, iter)

            # Log gradients if they exist
            if layer.weight.grad is not None:
                writer.add_histogram(f'Stiffness_func/layer_{i}_weights_grad', layer.weight.grad, iter)
            if layer.bias.grad is not None:
                writer.add_histogram(f'Stiffness_func/layer_{i}_biases_grad', layer.bias.grad, iter)

    # Generate output plot for stiffness_func from inputs -1.5 to 1.5
    inputs = torch.arange(-1.5, 1.6, 0.1).unsqueeze(1)  # Shape [N, 1]
    with torch.no_grad():
        outputs = stiffness_func(inputs).squeeze()  # Shape [N]
    
    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=inputs.numpy().flatten(), y=outputs.numpy(), mode='lines', name='Stiffness Output'))
    fig.update_layout(title="Stiffness Function Output", xaxis_title="Input", yaxis_title="Output")

    #convert a Plotly fig to  a RGBA-array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = torchvision.transforms.functional.pil_to_tensor(Image.open(buf))

    # Log the image to TensorBoard
    writer.add_image("Stiffness_func/output_plot", img, iter)
    