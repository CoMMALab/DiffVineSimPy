
import numpy as np
import torch
torch.set_printoptions(profile = 'full', linewidth = 900, precision = 2)

def read_yitian(file_path):

    data = np.load(file_path)
    
    T = data.shape[0]
    max_bodies = (data.shape[1] - 1)
    
    bodies = data[:, 0, 0]
    points = data[:, 1:]
    x = points[:, :, 0]
    y = points[:, :, 1]
        
    final = np.zeros((T, max_bodies * 3))
    final_body_count = np.zeros((T, 1))
    
    final[:, 0::3] = x
    final[:, 1::3] = y
    final[:, 2::3] = 0 
    
    final_body_count = bodies
    
    return torch.from_numpy(final), torch.from_numpy(final_body_count)

def read_yitian_walls(file_path):
    data = np.load(file_path)
    
    

if __name__ == '__main__':
    state, bodies = read_yitian('videoparser/data/frames/vid3/points.npy')
    walls = read_yitian_walls('videoparser/data/frames/vid3/walls.npy')
    
    # Print shape
    print(state.shape)
    print(bodies.shape)
    
    print(walls.shape)