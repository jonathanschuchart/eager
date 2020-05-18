from typing import List

import torch


class MLP(torch.nn.Module):
    def __init__(self, layer_dims: List[int]):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(d1, d2) for d1, d2 in zip(layer_dims, layer_dims[1:])]
        )
        self.final_layer = torch.nn.Linear(layer_dims[-1], 2)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = torch.relu(x)
        return torch.nn.functional.softmax(self.final_layer(x))
