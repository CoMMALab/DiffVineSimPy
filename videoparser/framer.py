import cv2
import os
import shutil
import numpy as np
#from sklearn.cluster import KMeans
#import classifier
#from skimage.morphology import skeletonize

def click_event(event, x, y, flags, param):
    frame = param['frame']
    points = param['points']
    all_points = param['all_points']
    initial_group = param['initial_group']
    check = param['4check']

    if event == cv2.EVENT_LBUTTONDOWN:
        # Adding points conditionally based on the groups
        if (not check and len(points) < 4) or (check and len(points) < 2):
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)  # Draw all points green
            cv2.imshow('image', frame)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Check conditions before locking in the points
        if not check and len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            points.append((x1, y2))
            points.append((x2, y1))
            #print('initial filled')
            param['4check'] = True
            initial_group.extend(points)  # Save the initial group separately
            for point in points:
                cv2.circle(frame, point, 5, (0, 0, 0), -1)
            points.clear()
        elif not check and len(points) == 4:
            #print('initial filled')
            param['4check'] = True
            initial_group.extend(points)  # Save the initial group separately
            for point in points:
                cv2.circle(frame, point, 5, (0, 0, 0), -1)
            points.clear()
        elif check and len(points) == 2:
            all_points.append(points.copy())  # Save subsequent groups
            for point in points:
                cv2.circle(frame, point, 5, (255, 0, 0), -1)
            cv2.line(frame, points[0], points[1], (255, 0, 0), 2)
            points.clear()
        
        cv2.imshow('image', frame)

def marker(frame):
    points = []  # Current points being marked
    initial_group = []  # Separate the initial group of 4 points
    all_points = []  # All subsequent groups of 2 points

    param = {'frame': frame.copy(), 'points': points, 'all_points': all_points, 'initial_group': initial_group, '4check': False}
    cv2.imshow('image', frame)
    cv2.setMouseCallback('image', click_event, param)

    while True:
        if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
            break
        key = cv2.waitKey(1) & 0xFF
        if key == 8:  # Backspace key to undo the last point
            if points:
                points.pop()
                frame_copy = frame.copy()
                for point in points:
                    cv2.circle(frame_copy, point, 5, (0, 255, 255), -1)
                for group in all_points:
                    for point in group:
                        cv2.circle(frame_copy, point, 5, (255, 0, 0), -1)
                for point in initial_group:
                    cv2.circle(frame_copy, point, 5, (0, 0, 0), -1)
                cv2.imshow('image', frame_copy)
                param['frame'] = frame_copy
        elif key == 13:  # Enter key to exit
            break

    cv2.destroyAllWindows()
    #base = all_points[0]
    return initial_group, all_points  # Return separate lists for initial group and subsequent groups


def order_points(pts):
    # Initial sorting based on the x-coordinates
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Now differentiate between top-left and bottom-right by y-coordinates
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect
def four_point_transform(image, pts, walls=None, w=256):
    # Obtain a consistent order of the points and unpack them individually
    pts = np.array(pts)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width and height of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Calculate the new height to maintain the aspect ratio
    aspect_ratio = maxHeight / maxWidth
    new_height = int(w * aspect_ratio)

    # Set up the destination points for perspective transform, adjusted to new dimensions
    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, new_height - 1],
        [0, new_height - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix and warp the perspective
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (w, new_height))
    #print(M, M.shape)
    #print(walls[0])
    # Transform each wall using the matrix M
    transformed_walls = []
    if walls is not None:
        walls = np.array(walls)
        for wall in walls:
            wall_pts = np.array([wall[0], wall[1]], dtype="float32")
            wall_pts = np.concatenate((wall_pts, np.ones((2, 1))), axis=1)  # Add ones for homogeneous coordinates
            transformed_pts = M.dot(wall_pts.T).T
            transformed_pts = transformed_pts[:, :2] / transformed_pts[:, [2]]  # Convert back to 2D
            transformed_walls.append(transformed_pts.astype(int))
    #print(transformed_walls[0])
    return warped, transformed_walls

#not needed resize in 4pt transform

def resize_image(img, walls, new_width, new_height):
    # Check if image is loaded
    if img is None:
        print("Error: Image not found.")
        return None, None
    
    # Get original dimensions
    original_height, original_width = img.shape[:2]

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Calculate the scale factors
    x_scale = new_width / original_width
    y_scale = new_height / original_height
    
    # Resize wall coordinates
    resized_walls = []
    for wall in walls:
        # Rescale each point in the wall
        resized_wall = [
            (int(point[0] * x_scale), int(point[1] * y_scale)) for point in wall
        ]
        resized_walls.append(resized_wall)

    return resized_img, resized_walls

def convert_to_grayscale(img):
    # Load the image from file
    if img is None:
        print("Error: Image not found.")
        return
    
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

# use a diff tolerance NEED TO FIX OVERFLOW
def pixdifftol(ref, frame):
    tolerance = 5
    floor_color = [198/255, 169/255, 140/255]
    #floor_color = [140/255, 169/255, 198/255]
    if ref.shape != frame.shape:
        raise ValueError("Images must have the same dimensions and number of channels.")

    # Compute the absolute difference between the images
    diff = cv2.absdiff(ref, frame)

    # Calculate the magnitude of differences
    # For colored images, you can calculate the Euclidean distance across color channels
    if len(ref.shape) == 3:
        diff_magnitude = np.sqrt(np.sum(diff**2, axis=-1))
    else:
        diff_magnitude = diff  # For grayscale images

    # Generate a binary output: 1 if differences are above the tolerance, 0 otherwise
    diff_binary = (diff_magnitude > tolerance).astype(np.uint8)
    img = (diff_binary * 255).astype(np.uint8)

    return img

# use a reference color
def pixdiffref(ref, img):

    ref_color = [255, 255, 255]
    floor_color = [140, 169, 198]

    ref_color = np.array(ref_color, dtype=np.uint8)
    img = np.array(img, dtype=float)
    ref = np.array(ref, dtype=float)
    alpha = 0.07
    beta = 0.2
    scale = [2, 0, 0]
    scale = np.array(scale)
    # Calculate the Euclidean distance from each pixel in img2 to the reference color
    dist_to_ref = np.sqrt(np.sum((img - ref_color) ** 2, axis=-1))
    # Calculate the Euclidean distance from each pixel in img2 to the corresponding pixel in img1
    dist_to_img1 = np.sqrt(np.sum((img - ref) ** 2, axis=-1))
    dist_to_floor = np.sqrt(np.sum((img * scale - floor_color * scale) ** 2, axis=-1))
    diff_binary1 = (alpha * dist_to_ref < dist_to_img1).astype(np.uint8)
    #diff_binary2 = (beta * dist_to_ref < dist_to_floor).astype(np.uint8)
    diff_binary2 = (dist_to_floor > 40).astype(np.uint8)

    diff_binary = (diff_binary1 * diff_binary2).astype(np.uint8)
    img = (diff_binary1 * 255).astype(np.uint8)

    return img

# combination
def pixdiffreftol(ref, img):
    tolerance = 5

    # Compute the absolute difference between the images
    diff = cv2.absdiff(ref, img)

    # Calculate the magnitude of differences
    # For colored images, you can calculate the Euclidean distance across color channels
    if len(ref.shape) == 3:
        diff_magnitude = np.sqrt(np.sum(diff**2, axis=-1))
    else:
        diff_magnitude = diff  # For grayscale images
    ref_color = [255, 255, 255]
    ref_color = np.array(ref_color, dtype=np.uint8)
    alpha = 0.7
    # Calculate the Euclidean distance from each pixel in img2 to the reference color
    dist_to_ref = np.sqrt(np.sum((img - ref_color) ** 2, axis=-1))

    # Calculate the Euclidean distance from each pixel in img2 to the corresponding pixel in img1
    dist_to_img1 = np.sqrt(np.sum((img - ref) ** 2, axis=-1))
    diff_binary = (alpha * dist_to_ref < dist_to_img1).astype(np.uint8)
    img = (diff_binary * 255).astype(np.uint8)

    return img

# essentially detects the white highlight
def pixwhite(ref, img):
    ref_color = [255, 255, 255]
    img = np.array(img, dtype=float)

    dist_to_ref = np.sqrt(np.sum((img - ref_color) ** 2, axis=-1))
    diff_binary = (dist_to_ref < 100).astype(np.uint8)
    img = (diff_binary * 255).astype(np.uint8)

    return img

def skel(img):
    skeleton = skeletonize(img // 255)  # skimage expects a binary image

# Convert skeleton to a format suitable for display
    skeleton = (skeleton * 255).astype(np.uint8)
    return skeleton

def extract_frames(video_path, output_folder, outfold=None):
    # Create the output folder if it does not exist, clear if it does
    if os.path.exists(output_folder):
        # Check if the directory is not empty
        if os.listdir(output_folder):  
            # Remove all contents of the directory
            shutil.rmtree(output_folder)
            # Create an empty directory again
            os.makedirs(output_folder)
    else:
        # If the directory doesn't exist, create it
        os.makedirs(output_folder)
    if outfold is not None:
        if os.path.exists(outfold):
            # Check if the directory is not empty
            if os.listdir(outfold):  
                # Remove all contents of the directory
                shutil.rmtree(outfold)
                # Create an empty directory again
                os.makedirs(outfold)
        else:
            # If the directory doesn't exist, create it
            os.makedirs(outfold)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_count = 0
    # only save every framerate'th frame
    framerate = 3
    counter = 0
    mod = False
    ref = True
    while True:
        counter += 1
        frame_count += 1
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no frames left

        # if video is flipped, uncomment this
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        if mod:
            frame5, _ = four_point_transform(frame, bound)
            if ref:
                ref = False
                reference_frame = frame
            else:
                frame1 = pixdiffref(reference_frame, frame)
                #frame = pixwhite(reference_frame, frame)
                frame1 = cv2.medianBlur(frame, 11)
                kernel = np.ones((11, 11), np.uint8)
                _, binary = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
                closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                frame1 = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
                #frame1 = skel(frame)
        # Save the frame as an image
        if counter == framerate:
            if frame_count == 5*framerate:
                # on the 5*framerate'th frame, run script to manually mark corners
                bound, walls = marker(frame)
                frame, walls = four_point_transform(frame, bound, walls)
                frame_filename = os.path.join(output_folder, f"frame_ref.jpg")
                wall_filename = os.path.join(output_folder, f"walls")
                np.save(wall_filename, walls)
                cv2.imwrite(frame_filename, frame)

                mod = True
            if frame_count > 5*framerate:
                frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame5)
                if outfold is not None:
                    frame_filename1 = os.path.join(outfold, f"frame_{frame_count:04d}.jpg")
                    cv2.imwrite(frame_filename1, frame1)
            counter = 0
    
    # When everything done, release the video capture object
    cap.release()
    print("Released video resource.")
    print(f"Total frames extracted: {frame_count}")

# Usage
def main():
    video_path = './data/videos/vid5.mp4'
    output_folder1 = './data/frames/viddemo2'
    output_folder2 = './data/frames/vid01'
    #extract_frames(video_path, output_folder1, output_folder2)
    extract_frames(video_path, output_folder1)

main()