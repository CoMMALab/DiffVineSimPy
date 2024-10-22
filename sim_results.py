import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)

ipm = 39.3701 / 1000   # inches per mm


def load_vine_robot_csv(filepath):
    """Load vine robot configuration from a CSV file."""
    data = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0: continue    # skip header

            # Convert each row to floats and append to the data list
            data.append([float(value) for value in row])

    # Convert the list to a numpy array for easier manipulation
    data = np.array(data)

    # Load rectangle data from the rect file (x, y, w, h format)
    rect_dir = Path(*os.path.split(filepath)[:-2]) / 'rects'
    rect_file = rect_dir / Path(filepath).stem
    print('rect_file', rect_file)
    rectangles = []
    with open(rect_file, 'r') as file:
        for line in file:
            x, y, w, h = map(float, line.strip().split())
            rectangles.append((x / ipm / 1000, y / ipm / 1000, w / ipm / 1000, h / ipm / 1000))

    return data, rectangles


def plot_vine_robot(data, rects, step = 1, title = 'Vine Robot Simulation', save_to = None):
    """Visualize vine robot movement over time."""
    # plt.clf()
    # plt.figure(figsize=(8, 8))

    # Loop through each time step (each row in data)
    q_size = data.shape[1] // 2

    for timestep in range(0, data.shape[0], step):
        plt.clf()  # Clear figure before each frame

        # Draw obstacles
        # Plot rectangles
        for rect in rects:
            x, y, w, h = rect
            rectangle = plt.Rectangle((x, y), w, h, fill = True, edgecolor = 'red', linewidth = 2)
            plt.gca().add_patch(rectangle)

        # Separate the x and y coordinates
        x_values = []
        y_values = []

        d = 3.0 / 1000
        links = 25

        # For each link
        for i in range(1, links):
            idx = 6 * (i - 1)

            # Proximal endpoint
            x1 = data[timestep, idx + 0]
            y1 = data[timestep, idx + 1]
            θ1 = data[timestep, idx + 2]
            X1 = x1 - d * math.cos(θ1)
            Y1 = y1 - d * math.sin(θ1)

            # Append proximal endpoint of the first link
            if i == 1:
                x_values.append(X1)
                y_values.append(Y1)

            # Distal endpoint
            x2 = data[timestep, idx + 3]
            y2 = data[timestep, idx + 4]
            θ2 = data[timestep, idx + 5]
            X2 = x2 + d * math.cos(θ2)
            Y2 = y2 + d * math.sin(θ2)

            x_values.append(X2)
            y_values.append(Y2)

        # Plot the robot's configuration as a line plot connecting x,y pairs
        plt.plot(x_values, y_values, '-o', color = 'blue', label = f'Time Step {timestep}')

        # Set plot limits, assuming your vine robot stays in a fixed area
        # plt.xlim([-0.25, 0.25])
        # plt.ylim([-0.25, 0.25])

        # Meters
        plt.xlim([-0.1, 1])
        plt.ylim([-1, 0.1])

        plt.title(f"{title} - Time Step {timestep}")
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True)

        plt.legend()
        plt.pause(0.01)    # Pause to simulate real-time visualization

    # If a save path is provided, save the final figure
    if save_to:
        plt.savefig(save_to)


if __name__ == '__main__':
    # Directory containing the CSV files
    directory = './sim_output'     # Replace with your actual directory

    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            print(f"Reading file: {filename}")

            # Load vine robot data from CSV
            data, rects = load_vine_robot_csv(filepath)

            # Visualize the vine robot configurations
            plot_vine_robot(data, rects, title = filename, step = 30)

    # Load vine robot data from CSV
    # data, rects = load_vine_robot_csv('sim_output/rectangles_200.csv')

    # # Visualize the vine robot configurations
    # plot_vine_robot(data, rects, step=30)
