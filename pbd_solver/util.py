        
from matplotlib import pyplot as plt
import torch

ww = 100
main_ax = None
fig_ax = None


def finite_changes(h, init_val):
    h_rel = torch.empty_like(h)
    h_rel[0] = h[0] - init_val
    h_rel[1:] = h[1:] - h[:-1] 
    
    return h_rel

def dist2seg(x, y, start, end):
    '''
    Given Nx1 x and y points, return closest dist to the seg and normal
    '''
    points = torch.stack([x, y], dim=1)
    
    # Segment start and end as tensors
    p1 = torch.tensor(start, dtype=torch.float32)
    p2 = torch.tensor(end, dtype=torch.float32)
    
    # Vector from p1 to p2 (the segment direction)
    segment = p2 - p1
    
    # Vector from p1 to each point
    p1_to_points = points - p1
    
    # Project the point onto the segment, calculate the parameter t
    segment_length_squared = torch.dot(segment, segment)
    t = torch.clamp(torch.sum(p1_to_points * segment, dim=1) / segment_length_squared, 0.0, 1.0)
    
    # Closest point on the segment to the point
    closest_point_on_segment = p1 + t[:, None] * segment
        
    # Distance from each point to the closest point on the segment
    distance_vectors = closest_point_on_segment - points 
    distances = torch.norm(distance_vectors, dim=1)
    
    # Normalize the distance vectors to get normal vectors
    # normals = torch.nn.functional.normalize(distance_vectors, dim=1)
    normals = distance_vectors
    
    return distances, closest_point_on_segment
    
def dist2rect(x, y, rect):
    '''
    Given Nx1 x and y points, return closest dist to the rect and normal
    '''
    x1, y1, x2, y2 = rect
    
    # Define the four segments of the rectangle
    segments = [
        ((x1, y1), (x2, y1)),  # Bottom edge
        ((x2, y1), (x2, y2)),  # Right edge
        ((x2, y2), (x1, y2)),  # Top edge
        ((x1, y2), (x1, y1))   # Left edge
    ]
        
    # Initialize minimum distances and normals
    min_distances = torch.full((x.shape[0],), float('inf'), dtype=torch.float32)
    min_contact_points = torch.zeros((x.shape[0], 2), dtype=torch.float32)
    
    # Iterate through each segment and compute the distance and normal
    for seg_start, seg_end in segments:
        distances, contact_points = dist2seg(x, y, seg_start, seg_end)
        
        # Update the minimum distances and normals where applicable
        update_mask = distances < min_distances
        min_distances = torch.where(update_mask, distances, min_distances)
        min_contact_points = torch.where(update_mask[:, None], contact_points, min_contact_points)
    
    return min_distances, min_contact_points

