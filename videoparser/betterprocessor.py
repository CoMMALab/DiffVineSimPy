
import math
import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import signal

# sig int kills matplotlib
signal.signal(signal.SIGINT, signal.SIG_DFL)

def resize_multiple(image, scale):
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    
def inverse_tranform(points, theta, translate):
    rot_matrix = np.array([[np.cos(theta), np.sin(theta)], 
                           [-np.sin(theta), np.cos(theta)]])
                           
    R = np.linalg.inv(rot_matrix)

    transformed_points = np.zeros_like(points)
    
    for i, point in enumerate(points):
        transformed_points[i] = R.dot(points[i]) + translate
        
    return transformed_points
    
def get_transform_from_walls(walls):
    # Walls are shape [N, 2, 2], where
    # N is the number of walls
    # 2 is the number of points per wall
    # 2 is the number of coordinates per point (x, y)
    
    anchor_point = walls[0][0]
    direction_point = walls[0][1]

    theta = np.arctan2(direction_point[1] - anchor_point[1], direction_point[0] - anchor_point[0])
    
    return theta, anchor_point

def draw_pair(sim_frame, real_frame, fig, ax):        
    print(f'Sim Frame: {sim_frame}, Real Frame: {real_frame}')
    
    file_index = 18 + 3 * real_frame
    video_frame = cv2.imread(f'{real_video_path}/frame_{file_index:04d}.jpg') 
    
    sim_vine = np.load(f'{sim_vine_path}/points{sim_frame}.npy')
    
    # Regular
    # sim_vine = inverse_tranform(sim_vine, theta, anchor_point)
    
    # real
    sim_vine = inverse_tranform(sim_vine, theta + math.radians(0), anchor_point + np.array([16, 0]))        
    
    
    # Draw points on the image 
    # Draw points on the image
    for pointa, pointb in zip(sim_vine[:-1], sim_vine[1:]):
        cv2.line(img=video_frame, 
                    pt1=(int(pointa[0]), int(pointa[1])), 
                    pt2=(int(pointb[0]), int(pointb[1])), 
                    color=(0, 255, 255), 
                    thickness=3)
    
    im = cv2.cvtColor(video_frame, 5, cv2.COLOR_BGR2RGB)
    ax.imshow(im)
    ax.set_title(f'Sim Frame: {sim_frame}, Real Frame: {real_frame}')
    # set plt.gcf()ure size
    fig.set_size_inches(18, 10)
    fig.canvas.draw_idle()
    # disable padding and gridlines
    # plt.gca().axis('off')
    
        
def interactive():
        
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.4)
    plt.gca().imshow(np.zeros((100, 100, 3), dtype=np.uint8))  # Placeholder imag
    
    curr_sim_frame = 0
    curr_real_frame = 0
    
    # (sim frame, real frame) pairs
    pair_frames = []
    
    # Buttons for simulation frame adjustment
    sim_buttons = {
        -100: Button(plt.axes([0.1, 0.05, 0.1, 0.075]), "-100 Sim"),
        -5: Button(plt.axes([0.22, 0.05, 0.1, 0.075]), "-10 Sim"),
        +5: Button(plt.axes([0.34, 0.05, 0.1, 0.075]), "+10 Sim"),
        +100: Button(plt.axes([0.46, 0.05, 0.1, 0.075]), "+100 Sim")
    }
    for step, button in sim_buttons.items():
        def doit(event, step):
            nonlocal curr_sim_frame
            curr_sim_frame += step
            try:
                draw_pair(curr_sim_frame, curr_real_frame, fig, ax)
            except Exception:
                curr_sim_frame -= step
            
        button.on_clicked(lambda event, s=step: doit(event, s))

    # Buttons for real frame adjustment
    # Adjusted positions for real frame buttons to be right below sim buttons
    real_buttons = {
        -30: Button(plt.axes([0.1, 0.15, 0.1, 0.075]), "-30 Real"),
        -3: Button(plt.axes([0.22, 0.15, 0.1, 0.075]), "-3 Real"),
        +3: Button(plt.axes([0.34, 0.15, 0.1, 0.075]), "+3 Real"),
        +30: Button(plt.axes([0.46, 0.15, 0.1, 0.075]), "+30 Real")
    }
    
    for step, button in real_buttons.items():
        def doit2(event, step):
            nonlocal curr_real_frame
            curr_real_frame += step
            try:
                draw_pair(curr_sim_frame, curr_real_frame,  fig, ax)
            except Exception:
                curr_real_frame -= step
            
        button.on_clicked(lambda event, s=step: doit2(event, s))

    # Button to save the current frame pair
    save_button = Button(plt.axes([0.45 + 0.3, 0.15, 0.1, 0.075]), "Save Pair")
    save_button.on_clicked(lambda event: print('SAVED', ((curr_sim_frame, curr_real_frame))))
    
    draw_pair(curr_sim_frame, curr_real_frame, fig, ax)
    plt.show()
        
        
if __name__ == '__main__':
    
    walls_path = 'data/frames/vid5/walls.npy'
    real_video_path = 'data/frames/viddemo2'
    sim_vine_path = '../sim/sim_out_nonlinear'
    
    walls_data = np.load(walls_path)
    theta, anchor_point = get_transform_from_walls(walls_data)
    
    # interactive()
    
    
    # just generate
    
    # pairs = [[240, 10],
    #         [460, 49],
    #         [780, 109],
    #         [980, 154],
    #         [1280, 244],]
    
    pairs = [[10, 6],
    [40, 48],
    [75, 108],
    [95, 150],
    [120, 210]]
    
    # pairs = [[10, 3],
    # [40, 51],
    # [70, 111],
    # [80, 162],
    # [100, 237]]
    
    plt.figure()
    plt.gcf().set_size_inches(18, 10)
    plt.gca().axis('off')
    
    
    for pair in pairs:
        draw_pair(pair[0], pair[1], plt.gcf(), plt.gca())
        plt.savefig(f'pair_{pair[0]}_{pair[1]}.png')
        plt.clf()