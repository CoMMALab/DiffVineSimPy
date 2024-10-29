import cv2
import os
import numpy as np
import processor

def draw_walls(frame, walls):
    # Loop over each wall in the list
    #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    for wall in walls:
        # Ensure each wall has exactly two points
        if len(wall) == 2:
            start_point, end_point = wall[0], wall[1]
            # Draw the start and end points
            cv2.circle(frame, start_point, 5, (255, 0, 0), -1)
            cv2.circle(frame, end_point, 5, (255, 0, 0), -1)
            # Draw the line between points
            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        else:
            print("Warning: Each wall should have exactly two points.")
    return frame

def draw_points(img, points):
    for point in points:
        cv2.circle(img, (point[1], point[0]), 3, (0, 255, 0), -1)
    return img
def merge_images(image1, image2):
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    if h1 != h2 or w1 != w2:
        print('not same size')
        return
    same = image1 == image2

    merged_image = np.where(same, image2, 128).astype(np.uint8)
    #cv2.imshow('image3', merged_image)
    return merged_image
# doesnt work when resize, make sure they come from same batch when framing
def merge_images_resize(image1, image2):
    """ Merge two images where non-matching parts are gray. """
    # Ensure both images are of the same size
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Determine the size to resize to (smallest dimensions)
    new_height = min(h1, h2)
    new_width = min(w1, w2)
    new_size = (new_width, new_height)

    # Resize images
    resized_image1 = cv2.resize(image1, new_size, interpolation=cv2.INTER_AREA)
    resized_image2 = cv2.resize(image2, new_size, interpolation=cv2.INTER_AREA)
    #cv2.imshow('image', resized_image1)
    #cv2.imshow('image2', resized_image2)
    #resized_image2 = cv2.medianBlur(resized_image2, 3)
    #resized_image1 = cv2.medianBlur(resized_image1, 3) 
    _, binary_image1 = cv2.threshold(resized_image1, 127, 255, cv2.THRESH_BINARY)
    _, binary_image2 = cv2.threshold(resized_image2, 127, 255, cv2.THRESH_BINARY)
    # Create a mask where both images are the same
    same = binary_image2 == binary_image1
    same = (same * 255).astype(np.uint8)
    cv2.imshow('same', same)
    # Create a merged image: if same set white (255), else set gray (128)
    merged_image = np.where(same, binary_image2, 128).astype(np.uint8)
    #cv2.imshow('image3', merged_image)
    return merged_image

def play_video_with_treated_frames(video_path, treated_images_folder, overlay=None):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    walls = np.load(treated_images_folder + 'walls.npy')
    #walls = np.load('./data/frames/vid0/walls.npy')
    # Frame index initialization
    frame_index = 0
    treated_frame_interval = 3
    first_treated_frame = 18
    last_treated_image = None  # To store the last displayed treated image

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # if video is flipped, uncomment this
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        # Load new treated image if it's the correct frame, and update last_treated_image
        if frame_index >= first_treated_frame and (frame_index - first_treated_frame) % treated_frame_interval == 0:
            image_path = os.path.join(treated_images_folder, f'frame_{frame_index:04d}.jpg')
            if os.path.exists(image_path):
                treated_image = cv2.imread(image_path)
                if treated_image is not None:
                    # Resize and update the last treated image
                    h, w, _ = treated_image.shape
                    aspect_ratio = w/h
                    if overlay is None:
                        points = processor.find_points(treated_image, walls, 10)
                        treated_image = draw_points(treated_image, points)
                        treated_image = draw_walls(treated_image, walls)
                        last_treated_image = cv2.resize(treated_image, (int(frame.shape[0] * aspect_ratio), frame.shape[0]))
                    else:
                        over_path = os.path.join(overlay, f'frame_{frame_index:04d}.jpg')
                        over = cv2.imread(over_path)
                        merge = merge_images(treated_image, over)
                        #last_treated_image = over
                        last_treated_image = cv2.resize(merge, (frame.shape[0] * aspect_ratio, frame.shape[0]))

        # Use the last treated image if available
        if last_treated_image is not None:
            combined_frame = cv2.hconcat([frame, last_treated_image])
            new_width = int(combined_frame.shape[1] * 0.5)
            new_height = int(combined_frame.shape[0] * 0.5)
            scaled_frame = cv2.resize(combined_frame, (new_width, new_height))
            cv2.imshow('Video with Treated Frames', scaled_frame)
            #cv2.imwrite('./data/testframe.jpg', combined_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frame_index += 1

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()

# Usage
video_path = './data/videos/vid4.mp4'
treated_images_folder = './data/frames/vid4/'
#play_video_with_treated_frames(video_path, treated_images_folder, './data/frames/vid3/')
play_video_with_treated_frames(video_path, treated_images_folder)