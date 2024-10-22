import torch
# import torch.autograd.functional as F

# # Define a sample function
# def my_function(x):
#     # Some arbitrary differentiable function
#     return x ** 2 + 3 * x

# # Input tensor
# x = torch.tensor([1.0, 2.0, 3.0, 4.0])

# # Define a mask
# mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)

# # Apply the mask to the input tensor
# masked_x = torch.masked.masked_tensor(x, mask=mask)

# # Compute the Jacobian using masked tensor
# jacobian = F.jacobian(my_function, masked_x)

# print("Jacobian:\n", jacobian)

# Create some example tensors
a = torch.tensor([1, 2, 4, 5])
b = torch.tensor([6, 7, 8, 9])

# Stack them along a new dimension at the end
stacked_tensor = torch.stack([a, b], dim = -1)

print(stacked_tensor)
