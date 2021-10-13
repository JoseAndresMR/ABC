import torch
import torch.nn as nn

m = nn.Conv2d(16, 33, 2, stride=2)
input = torch.randn(20, 16, 50, 100)
output = m(input)
print(output.shape)