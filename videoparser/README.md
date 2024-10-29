This is the video processing code to extract vine coordinates from video frames
for the simulation fitting.

# Requirements:
cv2
sci-kit learn
scipy

NOTE: all file paths hard coded for testing purposes are relative to the videoparser directory (i.e. you need to cd into this folder)

# framer.py
This file takes a video and populates a frames folder with images. Includes code
to manually select points to delimit boundaries, vine base position and wall positions.
Need to comment/uncomment lines to perform desired manipulation on frames. Also can edit
framerate and resolution in code. Capable of saving 2 versions to 2 separate folders for later
comparison (change output variable to frame1)

## processes:

marker: marks bounds and walls. left click to place point, right click to save group. bounds will 
autofill rectangle if only 2 points are selected, otherwise 4. Walls will only let you select 2 points.
Marker required to run; you don't need to select walls. walls are later saved in walls.npy

four_point_transform: transforms image to fit the bounding corners and also transforms the walls.
Not required to run but you should. Also changes resolution

pixdiff_: different processes for detecting changes from reference frame. pixdiffref works best by 
comparing distance from old pixel to a reference color, which can be changed

rotate: if video is taken upside down, this will flip the frame before it is processed. make sure to also uncomment
it in the visualizer

NOTE: for the next 3, if resolution is changed then kernel size needs to change as well

median blur: cv2 algo that reduces noise. other blurs exist but use this because it ensures pixels are
black and white (others will give grayscale but that can then be thresholded back into binary). 

morphology_close: fills in gaps in the vine (usually when a wall blocks part of the vine at the bend)

morphology_open: removes large noise while maintaining shape of vine. Good for getting rid of noise at the
tip of the vine where the shadows do funny things

skeletonize: reduces vine to 1 pixel thick

# classifier.py

basically useless since it was used for earliest versions by detecting walls with knn. the only thing
used is the display_large() function

# visualizer.py

Plays original video side by side with treated frames. Capable of taking 2 treated frame inputs and does 
an overlap (used to evaluate how well skeletonize works but there is weird noise). Can comment/uncomment
line that draws the walls. If framerate is changed in framer.py, need to change treated_frame_interval and
first_treated_frame (6*framerate)

# processor.py

For sim fitting. This takes a frame from the frames folder and finds the endpoint where the skeleton splits and
then splitting it into n equal parts. Can be run in visualizer to display points. Returns list of points

