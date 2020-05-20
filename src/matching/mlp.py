from typing import List

import torch


class MLP(torch.nn.Module):
    def __init__(self, layer_dims: List[int]):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(d1, d2) for d1, d2 in zip(layer_dims, layer_dims[1:])]
        )
        self.dropouts = torch.nn.ModuleList([torch.nn.Dropout(0.5) for _ in layer_dims])
        self.final_layer = torch.nn.Linear(layer_dims[-1], 1)

    def forward(self, x):
        for i, (layer, dropout) in enumerate(zip(self.layers, self.dropouts)):
            x = dropout(x)
            x = layer(x)
            x = torch.relu(x)
        return torch.sigmoid(self.final_layer(x))
