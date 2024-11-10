import cv2
import numpy as np
import heapq
#import classifier
import matplotlib.pyplot as plt
import os
import copy
import re
from PIL import Image


# takes sim output and reverses the transformation to match original video
def output_irl():
    folder = '../sim/sim_out3/'
    irlframes = ['0123', '0237', '0390', '0435', '0489']
    first = 57
    images = []    # List to store images with drawn points
    final_pts = []
    for frame in irlframes:
        file = f'data/frames/viddemo/frame_{frame}.jpg'
        img = cv2.imread(file)

        # Process the frame number to get corresponding points file
        outnum = (int(frame) - first) / 3
        outnum = int(outnum / 1.5)
        points = f'{folder}points{outnum}.npy'

        # Load points and apply reverse transformation
        outpts = reverse_transformation(np.load('data/frames/vid3/walls.npy'), np.load(points))
        final_pts.append(outpts)
        images.append(img)     # Add the processed image to the list
    return final_pts, images

def output_manual():
    folder = '../sim/sim_out2/'
    irlframes = ['0087', '0207', '0399', '0519', '0627']
    outframes = [169, 461, 840, 940, 1157]
    first = 57
    images = []    # List to store images with drawn points

    for i, frame in enumerate(irlframes):
        file = f'data/frames/viddemo/frame_{frame}.jpg'
        img = cv2.imread(file)
        points = f'{folder}points{outframes[i]}.npy'

        # Load points and apply reverse transformation
        outpts = reverse_transformation(np.load('data/frames/vid3/walls.npy'), np.load(points))

        # Draw points on the image
        for point in outpts:
            cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)

        images.append(img)     # Add the processed image to the list

    # Display all images in separate windows
    for i, image in enumerate(images):
        cv2.imshow(f'Image {i + 1}', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Step 2  
def output_irl_drawer():
    folder = '../sim/sim_out_real/'
    irlframes = ['0123', '0237', '0390', '0435', '0489']
    # irlframes = ['0500', '0700', '1000', '1200']
    first = 57
    images = []    # List to store images with drawn points

    for frame in irlframes:
        file = f'data/frames/viddemo/frame_{frame}.jpg'
        img = cv2.imread(file)

        outnum = #
        points = f'{folder}points{outnum}.npy'

        # Load points and apply reverse transformation
        outpts = reverse_transformation(np.load('data/frames/vid5/walls.npy'), np.load(points))

        # Draw points on the image
        for point in outpts:
            cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)

        images.append(img)     # Add the processed image to the list

    # Display all images in separate windows
    for i, image in enumerate(images):
        cv2.imshow(f'Image {i + 1}', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Step 1
def output_drawer():
    folder = '../sim/sim_out_real/'
    #filename = '../sim/sim_out/points50.npy'
    #points = np.load(filename)
    oldpoints = np.load('data/frames/vid5/points.npy')
    fpsconversion = 0.105
    #print(points)
    #return
    for i in range(0, 5000, 10):
        print(i)
        
        filename = f'points{i}.npy'
        path = os.path.join(folder, filename)
        points = np.load(path)
        #points = points.reshape(-1, 2)
        walls = np.load('./data/frames/vid5/walls.npy')
        points = reverse_transformation(walls, points)
        #print(points)
        for point in points:
            plt.plot(point[0], point[1], 'go')
        for wall in walls:
            plt.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], marker = 'o', color = 'b')

        oldind = int(i * fpsconversion)
        if oldind < len(oldpoints):
            oldframe = oldpoints[oldind]
        else:
            oldframe = oldpoints[len(oldpoints) - 1]
            oldframe = oldframe[1:] * [1, -1]
            oldframe = reverse_transformation(walls, oldframe)
            for point in oldframe:
                plt.plot(point[0], point[1], 'ro')
            break
        oldframe = oldframe[1:] * [1, -1]
        oldframe = reverse_transformation(walls, oldframe)
        for point in oldframe:
            plt.plot(point[0], point[1], 'ro')
        plt.grid(True)
        plt.pause(0.0001)
        plt.clf()
    plt.show()


def reverse_transform_point(x, R, p):
    #print(x)
    return R.dot([x[0], x[1]]) + p


def reverse_transformation(walls, points):
    """
    Applies a reverse transformation to a set of points based on the provided walls.

    This function takes a list of walls and a list of points, and applies a reverse
    transformation to the points. The transformation is based on the orientation and
    position of the first wall in the list.

    Args:
        walls (list): A list of walls, where each wall is represented by a pair of points.
                      Each point is a list or array of two coordinates [x, y].
        points (list): A list of points to be transformed, where each point is a list or
                       array of two coordinates [x, y].

    Returns:
        numpy.ndarray: A numpy array of the transformed points.
    """
    points = np.array(points)
    p = copy.deepcopy(walls[0][1])
    v = p - copy.deepcopy(walls[0][0])
    theta = np.arctan2(v[1], v[0])
    flipped = points
    #flipped = [[y, x] for x, y in points]

    R = np.linalg.inv(np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]))
    for point in flipped:
        point[:] = reverse_transform_point(point, R, p)
    return flipped


def test_transformation(walls, points):
    for wall in walls:
        plt.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], marker = 'o', color = 'b')
    for point in points:
        plt.plot(point[0], point[1], 'go')
    plt.grid(True)
    plt.show()


def transform_point(x, R, p):
    #return x-p
    return R.dot(x - p) * [1, -1]


def transform_vine_base(walls, points):
    #walls = np.array(walls)
    points = np.array(points)
    p = copy.deepcopy(walls[0][1])
    v = p - copy.deepcopy(walls[0][0])
    theta = np.arctan2(v[1], v[0])
    flipped = [[y, x] for x, y in points]

    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    for wall in walls:
        for point in wall:
            point[:] = transform_point(point, R, p)
    for point in flipped:
        point[:] = transform_point(point, R, p)
    return walls, flipped


def find_nearest(binary_array, start):
    rows = len(binary_array)
    cols = len(binary_array[0]) if rows else 0
    start_x, start_y = start

    # Priority queue stores tuples (distance, x, y)
    priority_queue = [(0, start_x, start_y)]
    visited = set()
    visited.add((start_x, start_y))

    while priority_queue:
        dist, x, y = heapq.heappop(priority_queue)

        # Return the position as soon as a cell with value 1 is found
        if binary_array[x][y] == 1:
            return (x, y)

        # Explore all adjacent cells (8 directions)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    # Calculate Euclidean distance from the start point
                    new_dist = np.sqrt((nx - start_x)**2 + (ny - start_y)**2)
                    heapq.heappush(priority_queue, (new_dist, nx, ny))

    return None


def travel(binary_array, start):
    rows, cols = len(binary_array), len(binary_array[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] # 8 possible moves
    current_pos = start
    prev_pos = None
    visited = []
    distance = 0
    while True:
        visited.append(current_pos)
        x, y = current_pos
                                                                                        #print(f"Visiting: ({x}, {y})")
        count_ones = 0
        potential_moves = []

        # Examine all neighbors within the 3x3 kernel
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and binary_array[nx][ny] == 1:
                count_ones += 1
                neighbor_pos = (nx, ny)
                potential_moves.append(neighbor_pos)

        # Check termination and move conditions
        if count_ones == 0:
            #print("Stopping: No further moves possible or only self is 1.")
            break
        elif count_ones == 1 and prev_pos is not None:
            #print("Stopping: The only 1 is the previous step.")
            break
        elif count_ones == 1 and prev_pos is None:
            prev_pos = current_pos
            current_pos = potential_moves[0]
        elif count_ones == 2 and prev_pos is None:
            #print("Stopping: Reached a split with multiple new moves.")
            break
        elif count_ones > 2:
            #print("Stopping: Reached a split with multiple new moves.")
            break
        elif count_ones == 2 and prev_pos is not None:
            potential_moves = [pos for pos in potential_moves if pos != prev_pos]
            prev_pos = current_pos
            current_pos = potential_moves[0]
        else:
            #print("No valid move condition met. Check algorithm logic.")
            break
        if current_pos[0] != prev_pos[0] and current_pos[1] != prev_pos[1]:
            distance += 1.414
        else:
            distance += 1

    return current_pos, visited, distance


def find_midpoints(line, split, n):
    distance = 0
    prev_pos = line[0]
    points = []
    for current_pos in line:
        if current_pos == line[0]:
            continue
        if current_pos[0] != prev_pos[0] and current_pos[1] != prev_pos[1]:
            distance += 1.414
        else:
            distance += 1
        if abs(distance - split) < 0.76:
            distance -= split
            points.append(current_pos)
            if len(points) == n:
                break
        prev_pos = current_pos
    return points


def find_points(img, walls, n):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)
    base = walls[0][1]
    start = find_nearest(img, (base[1], base[0]))
    if start is None:
        return []
    end, line, dist = travel(img, start)
    return line
    split = dist / (n - 1)     # n-1 segments given n points including endpoints
    midpoints = find_midpoints(line, split, n - 2)
    img = (img * 255).astype(np.uint8)
                               #_, img = cv2.threshold(img, 256, 1, cv2.THRESH_BINARY)
    img[start[0]][start[1]] = 128
    img[end[0]][end[1]] = 128
    for point in midpoints:
        img[point[0]][point[1]] = 128
                               #classifier.display_large(img)
    midpoints.insert(0, start)
    midpoints.append(end)
    return midpoints


def collection(folder):
    wall_og = np.load(os.path.join(folder, 'walls.npy'))
    data = []
    #min_frame =
    max_frame = 580
    past_length = 0
    for entry in os.listdir(folder):
        # Construct full file path
        if entry == 'frame_ref.jpg' or entry == 'walls.npy' or entry == 'points.npy' or entry == 'tf_walls.npy':
            continue
        match = re.search(r'frame_(\d+)\.jpg', entry)
        if match:
            # Convert the matched group to an integer
            frame_number = int(match.group(1))
            if frame_number > max_frame:
                continue
        full_path = os.path.join(folder, entry)
        if os.path.isfile(full_path):
            img = cv2.imread(full_path)
            walls = copy.deepcopy(wall_og)
            line = find_points(img, walls, 10)
            _, line = transform_vine_base(walls, line)
            if len(line) > 0:
                if len(line) > past_length * 0.9:
                    past_length = len(line)
                    data.append(line)
    max_len = max(len(sublist) for sublist in data)

    # Pad sublists with a specified tuple (e.g., (0, 0)) and convert to array
    padded_arrays = np.array(
        [[(len(sublist), 0)] + sublist + [(0, 0)] * (max_len - len(sublist)) for sublist in data]
        )
    np.save(os.path.join(folder, 'points.npy'), padded_arrays)
    tf_walls, _ = transform_vine_base(walls, [])
    np.save(os.path.join(folder, 'tf_walls.npy'), tf_walls)


def main():
    #folder = './data/frames/vid6/'
    #collection(folder)
    img_path = './data/frames/vid3/frame_0477.jpg'
    #wall_path = './data/frames/vid3/walls.npy'
    #walls = np.load(wall_path)
    #img = cv2.imread(img_path)
    
    
    # output_drawer()
    output_irl_drawer()
    # pts, images = output_manual()
    
    
    # pts, images = output_irl()
    # for i, (image, pt) in enumerate(zip(images, pts)):
    #     pt = pt.astype(np.int32)
    #     cv2.polylines(image, [pt], False, (0, 255, 255), thickness = 3)
    #     im = Image.fromarray(image[...,::-1])
    #     print(f"saved image to i{i}.png")
    #     im.save(f'i{i}.png')
    #     print(pt)
        
        
    #output_irl()
    #print(img.shape)
    #points = find_points(img, walls, 10)
    #print(walls, points)
    #walls, points = transform_vine_base(walls, points)
    #print(walls, points)
    #test_transformation(walls, points)


if __name__ == '__main__':
    main()
