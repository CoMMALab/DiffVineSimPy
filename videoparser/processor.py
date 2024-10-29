import cv2
import numpy as np
import heapq
import classifier
import matplotlib.pyplot as plt

def test_transformation(walls, points):
    for wall in walls:
        for point in wall:
            plt.plot(point[1], -point[0], 'bo')
    for point in points:
        plt.plot(point[1], -point[0], 'go')
    plt.grid(True)
    plt.show()
def transform_point(x, R, p):
    return R.dot(x - p)
def transform_vine_base(walls, points):
    p = walls[0][1]
    v = walls[0][0] - walls[0][1]
    theta = np.arctan2(v[1], v[0])

    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    for wall in walls:
        for point in wall:
            point = transform_point(point, R, p)
    for point in points:
        point = transform_point(point, R, p)
    return walls, points


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
                    new_dist = np.sqrt((nx - start_x) ** 2 + (ny - start_y) ** 2)
                    heapq.heappush(priority_queue, (new_dist, nx, ny))

    return None

def travel(binary_array, start):
    rows, cols = len(binary_array), len(binary_array[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8 possible moves
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
        elif count_ones >2:
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
    split = dist/(n-1) # n-1 segments given n points including endpoints
    midpoints = find_midpoints(line, split, n-2)
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
    
def main():
    folder = './data/frames/vid3/'
    img_path = './data/frames/vid3/frame_0477.jpg'
    wall_path = './data/frames/vid3/walls.npy'
    walls = np.load(wall_path)
    img = cv2.imread(img_path)
    #print(img.shape)
    points = find_points(img, walls, 10)
    #walls, points = transform_vine_base(walls, points)
    test_transformation(walls, points)

if __name__ == '__main__':
    main()