Video processing code to extract vine coordinates from video frames
for the simulation fitting.

# Requirements:
cv2
scikit-learn
scipy

Note: The scripts in this folder require your working directory to be here before executing.

# framer.py
This file takes a video and populates a frames folder with images. Includes code
to manually select points to delimit boundaries, vine base position and wall positions.
Need to comment/uncomment lines to perform desired manipulation on frames. Also can edit
framerate and resolution in code. Capable of saving 2 versions to 2 separate folders for later
comparison (change output variable to frame1)

## processor:

`marker`: Marks workspace boundaries and walls. Use left click to place a point, and right click to save your selection. Workspace bounds are always a rectangle, but if you just want an aluigned one, select 2 points, otherwise 4 for 4-sided gon. Walls will only let you select 2 points. Walls are saved in walls.npy

`four_point_transform`: Does the homography transforms to warp the workspace limits (from previou step) into the corners of the frame. Also transforms the walls along. Not required to run but you should. Also changes resolution

`pixdiff_`: Different processes for doing optical flow from reference frame (usually the first one). `pixdiffref` works best by 
comparing the distance from old pixel to a reference color, which can be changed

`rotate`: if video is taken upside down, this will flip the frame before it is processed. make sure to uncomment it in the visualizer if you need it.

NOTE: For the next 3 steps, you need to adjust the kernel size along with resolution

`median blur`: cv2 algo that reduces noise. other blurs exist but use this because it ensures pixels are black and white (others will give grayscale but that can then be thresholded back into binary). 

`morphology_close`: fills in gaps in the vine (usually when a wall blocks part of the vine at the bend)

`morphology_open`: removes large noise while maintaining shape of vine. Good for getting rid of noise at the
tip of the vine where the shadows do funny things

`skeletonize`: Reduces a segmentation of the vine to a continuous 1 pixel thick centerline. This is needed later when we convert the vine into a sequence of points.

# classifier.py

Not used any morem but it was used in the earliest versions by segmenting wall colors with knn. The `display_large()` function is called in other code though.

# visualizer.py

Plays original video side by side with treated frames. Capable of taking 2 treated frame inputs and does 
an overlap (used to evaluate how well skeletonize works but there is weird noise). Can comment/uncomment
line that draws the walls. If framerate is changed in framer.py, need to change treated_frame_interval and
first_treated_frame (6*framerate)

# processor.py

For sim fitting. This takes a frame from the frames folder and finds the endpoint where the skeleton splits and
then splitting it into n equal parts. Can be run in visualizer to display points. Returns list of points

