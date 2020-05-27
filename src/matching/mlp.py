from typing import List

import torch


class MLP(torch.nn.Module):
    def __init__(self, layer_dims: List[int]):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(d1, d2)
                for d1, d2 in zip(layer_dims, layer_dims[1:] + [2])
            ]
        )
        self.dropouts = torch.nn.ModuleList([torch.nn.Dropout(0.3) for _ in layer_dims])
        self.norms = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(n) for n in layer_dims[1:] + [2]]
        )
        self.activations = [torch.relu for _ in layer_dims[:-1]] + [torch.sigmoid]

    def forward(self, x):
        for dropout, layer, norm, act in zip(
            self.dropouts, self.layers, self.norms, self.activations
        ):
            x = dropout(x)
            x = layer(x)
            x = norm(x)
            x = act(x)
        return x
