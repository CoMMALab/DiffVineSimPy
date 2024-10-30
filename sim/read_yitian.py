
import numpy as np
import torch
torch.set_printoptions(profile = 'full', linewidth = 900, precision = 2)

def read_yitian_vine(file_path):

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
    final[:, 1::3] = -y
    final[:, 2::3] = 0 
    
    final_body_count = bodies
    
    return torch.from_numpy(final), torch.from_numpy(final_body_count).unsqueeze(-1)

def read_yitian_walls(file_path):
    data = np.load(file_path)
    return torch.from_numpy(data[1:])

def read_yitian(number):
    number = str(number)
    state, bodies = read_yitian_vine(f'videoparser/data/frames/vid{number}/points.npy')
    walls = read_yitian_walls(f'videoparser/data/frames/vid{number}/tf_walls.npy')
    return state, bodies, walls
    
    
if __name__ == '__main__':
    state, bodies, walls = read_yitian(1)
    
    # Print shape
    print(state)
    
    print(state.shape)
    print(walls.shape)