import unittest
import torch

from .vine import finite_changes, dist2segments, generate_segments_from_rectangles


class TestDistanceFunctions(unittest.TestCase):

    def test_dist2seg(self):
        # Test case for dist2seg
        x = torch.tensor([[1.0, 3.0, 2.0]])
        y = torch.tensor([[1.0, 1.0, 1.5]])
        start = (3.0, 0.0)
        end = (3.0, 3.0)

        distances, contacts = dist2seg(x, y, start, end)

        # Expected distances: 1.0 for (1.0, 1.0) and 3.0 for (3.0, 1.0)
        expected_distances = torch.tensor([[2.0, 0.0, 1.0]])
        expected_contacts = torch.tensor(
            [[[3.0, 1.0], [3.0, 1.0], [3.0, 1.5]]]
            )                                      # Normalized vectors pointing away

        # print('expected_contact', expected_contacts)
        # print('distances', contacts)
        # Check if distances are close to the expected values
        self.assertTrue(torch.allclose(distances, expected_distances, atol = 1e-5))

        # Check if contacts are as expected
        self.assertTrue(torch.allclose(contacts, expected_contacts, atol = 1e-5))

    def test_dist2rect(self):
        # Test case for dist2rect
        # Square of 3x3
        # One is at 2,1.5 and the other at 4,1
        x = torch.tensor([[2.0, 4.0]])
        y = torch.tensor([[1.5, 1.0]])
        rect = (0.0, 0.0, 3.0, 3.0)

        min_distances, min_contacts = dist2rect(x, y, rect)

        # Expected distances and contacts
        expected_distances = torch.tensor([[1.0, 1.0]])
        expected_contacts = torch.tensor(
            [[[-1.0, 0.0], [1.0, 0.0]]]
            )                              # No normal for inside point, right normal

        # Check distances
        self.assertTrue(torch.allclose(min_distances, expected_distances, atol = 1e-5))

        # Check contacts
        print("Not checking contacts")
        # self.assertTrue(torch.allclose(min_contacts, expected_contacts, atol=1e-5))

    # def test_dist2rects(self):
    #     # Test case for dist2rects
    #     x = torch.tensor([1.5, 3.0, 6.0])
    #     y = torch.tensor([1.0, 1.0, 1.0])
    #     rects = [(0.0, 0.0, 2.0, 2.0), (4.0, 0.0, 6.0, 2.0)]

    #     min_distances, min_contacts = dist2rects(x, y, rects)

    #     # Expected results
    #     expected_distances = torch.tensor([0.5, 1.0, 0.0])  # (1,1) is inside the first rect, (5,1) is inside second rect
    #     expected_contacts = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])  # Inside: normal face out, otherwise contacts

    #     # Check distances
    #     self.assertTrue(torch.allclose(min_distances, expected_distances, atol=1e-5))

    #     # Check contacts
    #     self.assertTrue(torch.allclose(min_contacts, expected_contacts, atol=1e-5))
    def test_diffs(self):
        # Test case for dist2rect
        # Square of 3x3
        # One is at 2,1, and the other at 4,1
        x = torch.tensor([1, 2, 3, 5, 6, 1006])
        res = finite_changes(x, init_val = -1)

        self.assertTrue(torch.allclose(res, torch.tensor([2, 1, 1, 2, 1, 1000]), atol = 1e-5))


if __name__ == '__main__':
    # unittest.main()

    # 4          5
    # |          |
    # 1          2
    #  1 -- 3     5 -- 7
    obstacles = torch.tensor([[1.0, 1.0, 3.0, 4.0], [5.0, 2.0, 7.0, 5.0]], dtype = torch.float32)

    # Generate segments from obstacles
    segments = generate_segments_from_rectangles(obstacles)

    # Query points
    points = torch.tensor([[2.0, 2.0], [6.0, 3.0]], dtype = torch.float32)

    # Compute distances
    min_distances, closest_points = dist2segments(points, segments)

    print("Minimum Distances:", min_distances)
    print("Closest Points on Segments:", closest_points)
