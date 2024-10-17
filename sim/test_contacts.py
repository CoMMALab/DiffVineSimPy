import unittest
import torch

from pbd_solver.vine import dist2seg, dist2rect, finite_changes

class TestDistanceFunctions(unittest.TestCase):

    def test_dist2seg(self):
        # Test case for dist2seg
        x = torch.tensor([1.0, 3.0])
        y = torch.tensor([1.0, 1.0])
        start = (0.0, 0.0)
        end = (3.0, 0.0)
        
        distances, normals = dist2seg(x, y, start, end)
        
        # Expected distances: 1.0 for (1.0, 1.0) and 3.0 for (3.0, 1.0)
        expected_distances = torch.tensor([1.0, 1.0])
        expected_normals = torch.tensor([[0.0, 1.0], [0.0, 1.0]])  # Normalized vectors pointing away
        
        # Check if distances are close to the expected values
        self.assertTrue(torch.allclose(distances, expected_distances, atol=1e-5))
        
        # Check if normals are as expected
        self.assertTrue(torch.allclose(normals, expected_normals, atol=1e-5))

    def test_dist2rect(self):
        # Test case for dist2rect
        # Square of 3x3
        # One is at 2,1, and the other at 4,1
        x = torch.tensor([2.0, 4.0])
        y = torch.tensor([1.5, 1.0])
        rect = (0.0, 0.0, 3.0, 3.0)
        
        min_distances, min_normals = dist2rect(x, y, rect)
        
        # Expected distances and normals
        expected_distances = torch.tensor([1.0, 1.0]) 
        expected_normals = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])  # No normal for inside point, right normal
        
        # Check distances
        self.assertTrue(torch.allclose(min_distances, expected_distances, atol=1e-5))
        
        # Check normals
        self.assertTrue(torch.allclose(min_normals, expected_normals, atol=1e-5))

    # def test_dist2rects(self):
    #     # Test case for dist2rects
    #     x = torch.tensor([1.5, 3.0, 6.0])
    #     y = torch.tensor([1.0, 1.0, 1.0])
    #     rects = [(0.0, 0.0, 2.0, 2.0), (4.0, 0.0, 6.0, 2.0)]
        
    #     min_distances, min_normals = dist2rects(x, y, rects)
        
    #     # Expected results
    #     expected_distances = torch.tensor([0.5, 1.0, 0.0])  # (1,1) is inside the first rect, (5,1) is inside second rect
    #     expected_normals = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])  # Inside: normal face out, otherwise normals
        
    #     # Check distances
    #     self.assertTrue(torch.allclose(min_distances, expected_distances, atol=1e-5))
        
    #     # Check normals
    #     self.assertTrue(torch.allclose(min_normals, expected_normals, atol=1e-5))
    def test_diffs(self):
        # Test case for dist2rect
        # Square of 3x3
        # One is at 2,1, and the other at 4,1
        x = torch.tensor([1, 2, 3, 5, 6, 1006])
        res = finite_changes(x, init_val=-1)

        self.assertTrue(torch.allclose(res, torch.tensor([2, 1, 1, 2, 1, 1000]), atol=1e-5))
        
if __name__ == '__main__':
    unittest.main()