from typing import Callable, Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F


class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pooled_q_seq, q_his, batch_size):
        return torch.mean(
            torch.softmax(torch.bmm(pooled_q_seq, q_his.permute(0, 2, 1)), -1).view(
                batch_size, -1, 1
            )
            * q_his,
            1,
            keepdim=True,
        )


class UserDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dp_attn = DotProductAttention()

    def forward(self, q_msg, user_features):
        q_msg = q_msg.view(q_msg.shape[0], 6, -1)

        h_reprs = []

        for q_repr_idx in range(q_msg.shape[1]):
            h_reprs.append(
                self.dp_attn(
                    q_msg[:, q_repr_idx].unsqueeze(1),
                    user_features,
                    user_features.shape[0],
                )
            )
        return torch.cat(h_reprs, 2)


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.
    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0

    https://github.com/pytorch/vision/blob/ce257ef78b9da0430a47d387b8e6b175ebaf94ce/torchvision/ops/misc.py#L263
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = True,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
