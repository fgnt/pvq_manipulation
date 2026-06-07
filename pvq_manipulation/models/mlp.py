import torch
from padertorch.ops.mappings import ACTIVATION_FN_MAP


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_channels,
        output_activation=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        current_dim = input_dim
        layers = []
        for hidden_dim in hidden_channels:
            layers.append(torch.nn.Linear(current_dim, hidden_dim))
            layers.append(torch.nn.Dropout(p=0.2))
            layers.append(torch.nn.ReLU())
            current_dim = hidden_dim

        layers.append(torch.nn.Linear(current_dim, self.output_dim))
        if output_activation is not None:
            if output_activation == 'softmax':
                layers.append(ACTIVATION_FN_MAP[output_activation](dim=-1))
            else:
                layers.append(ACTIVATION_FN_MAP[output_activation]())

        self.net = torch.nn.Sequential(*layers)
        self.output_activation = output_activation

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x
