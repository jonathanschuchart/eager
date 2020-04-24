from typing import List

import torch


class MLP(torch.nn.Module):
    def __init__(self, layer_dims: List[int]):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(d1, d2) for d1, d2 in zip(layer_dims, layer_dims[1:])]
        )

    def forward(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        z = torch.cat((x1, x2), dim=1)
        for i, layer in enumerate(self.layers):
            z = layer(z)
            z = torch.relu(z) if i < len(self.layers) - 1 else torch.nn.functional.softmax(z)
        return z
