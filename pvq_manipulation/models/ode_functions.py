"""
Implementation of Δz = f(t, z, labels)
f() is a neural network with the architecture defined in StyleFlow
StyleFlow: Attribute-conditioned Exploration of StyleGAN-Generated Images using Conditional Continuous Normalizing Flows
"""
import torch
from padertorch.ops.mappings import ACTIVATION_FN_MAP


class CNFNN(torch.nn.Module):
    def __init__(
            self,
            input_dim,
            condition_dim,
            hidden_channels,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        hidden_dims = hidden_channels + [input_dim]
        self.input_dim = input_dim

        for idx, hidden_dim in enumerate(hidden_dims):
            self.layers.append(CNFBlock(
                input_dim=input_dim,
                condition_dim=condition_dim,
                output_dim=hidden_dim,
                output_layer=False if idx < len(hidden_dims) - 1 else True,
            ))
            input_dim = hidden_dim

    def forward(self, t, z, labels):
        """
        This function computes: Δz = f(t, z, labels)

        Args:
            t (torch.Tensor): () Time step of the ODE
            z (torch.Tensor): (Batch_size, Input_dim) Intermediate value
            labels (torch.Tensor): (Batch_size, condition_dim) Speaker attributes 

        Returns:
            Δz (torch.Tensor): : (Batch_size, Input_dim) Computed delta
        """
        for layer in self.layers:
            z = layer(t, z, labels)
        return z


class CNFBlock(torch.nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            condition_dim,
            output_layer,
    ):
        super().__init__()
        self._layer = torch.nn.Linear(input_dim, output_dim)
        self._hyper_bias = torch.nn.Linear(
            1 + condition_dim,
            output_dim,
            bias=False
        )
        self._hyper_gate = torch.nn.Linear(
            1 + condition_dim,
            output_dim
        )
        self.output_layer = output_layer

    def forward(self, t, z, labels):
        """
        Args:
            t (torch.Tensor): () Time step of the ODE
            z (torch.Tensor): (Batch_size, Input_dim) Intermediate value
            labels (torch.Tensor): (Batch_size, condition_dim) Speaker attributes 

        Returns:
            z (torch.Tensor): : (Batch_size, Output_dim) Intermediate value
        """
        if labels.dim() == 1:
            labels = labels[:, None]
        elif labels.dim() == 3:
            labels = labels.squeeze(1)

        tz_cat = torch.cat((t.expand(z.shape[0], 1), labels), dim=1)

        gate = torch.sigmoid(self._hyper_gate(tz_cat))
        bias = self._hyper_bias(tz_cat)

        if z.dim() == 3:
            gate = gate.unsqueeze(1)
            bias = bias.unsqueeze(1)

        z = self._layer(z) * gate + bias

        if not self.output_layer:
            z = torch.tanh(z)
        return z


class MLP(torch.nn.Module):
    def __init__(
            self,
            input_dim,
            condition_dim,
            hidden_channels,
            activation='relu'
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.input_dim = input_dim
        hidden_channels = hidden_channels + [input_dim]

        self.layers.append(torch.nn.Linear(
            input_dim + condition_dim + 1,
            hidden_channels[0]
        ))
        self.layers.append(ACTIVATION_FN_MAP[activation]())

        for idx in range(len(hidden_channels) - 1):
            self.layers.append(
                torch.nn.Linear(
                    hidden_channels[idx],
                    hidden_channels[idx + 1]
                )
            )
            if idx < len(hidden_channels) - 2:
                self.layers.append(ACTIVATION_FN_MAP[activation]())

    def forward(self, t, z, labels):
        """
        This function computes: Δz = f(t, z, labels)

        Args:
            t (torch.Tensor): () Time step of the ODE
            z (torch.Tensor): (Batch_size, Input_dim) Intermediate value
            labels (torch.Tensor): (Batch_size, condition_dim) Speaker attributes

        Returns:
            Δz (torch.Tensor): : (Batch_size, Input_dim) Computed delta
        """
        t = t.expand(z.shape[0], 1)
        z = torch.cat((z, labels, t), dim=1)
        for layer in self.layers:
            z = layer(z)
        return z
