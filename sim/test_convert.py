import torch
import matplotlib.pyplot as plt

from sim.fitting_good import convert_to_valid_state

# Define the convert_to_valid_state function as provided in the previous message
# Place the entire function definition here

# Test function to plot the original and computed points
def test_and_plot():
    B = 1  # Batch size of 1 for simplicity in visualization
    max_size = 20
    d = 0.2  # Desired distance between computed points

    # Starting points for each batch item
    start_x = torch.tensor([0.0])
    start_y = torch.tensor([0.0])

    # Define a sequence of points in the shape of [B, N]
    state = torch.tensor([
        [0.0, 0.0, 0.0, 1.5, 0.5, 0.0, 2.5, 1.5, 0.0, 3.5, 0.5, 0.0, 4.5, 1.5, 0.0]
    ], dtype=torch.float32)

    # Run the convert_to_valid_state function
    points_tensor, bodies = convert_to_valid_state(start_x, start_y, state, d, max_size)
    
    # Plot the results
    plt.figure(figsize=(10, 6))

    # Extract original points from the state tensor
    x_coords = state[0, ::3]
    y_coords = state[0, 1::3]
    
    # Plot original points
    plt.plot(x_coords, y_coords, 'bo-', label="Original Path", markersize=5)
    
    # Extract computed points
    print(points_tensor)
    computed_x = points_tensor[0, 0:bodies[0, 0]*3:3]
    computed_y = points_tensor[0, 1:bodies[0, 0]*3:3]
    computed_theta = points_tensor[0, 2:bodies[0, 0]*3:3]

    # Plot computed points that are d units apart
    # plt.plot(computed_x, computed_y, 'ro-', label="Computed Points (d units apart)", markersize=8)
    
    for x,y,theta in zip(computed_x, computed_y, computed_theta):
        startx = x - d * torch.cos(theta)
        starty = y - d * torch.sin(theta)
        endx = x + d * torch.cos(theta)
        endy = y + d * torch.sin(theta)
        plt.plot([startx, endx], [starty, endy], 'g--', linewidth=1, markersize=8)

    # Annotate the points for clarity
    for i, (x, y) in enumerate(zip(computed_x, computed_y)):
        plt.text(x, y, f"{i}", fontsize=12, ha='right')

    # Add plot labels and title
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Original Points and Computed Points (d = {d} units apart)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")  # Keep the aspect ratio equal for accurate distance representation
    plt.show()

# Run the test and plot
test_and_plot()
