# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Positionwise feed forward layer definition."""

import torch

from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode


class MS_MLP(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = True,
        node: str = "plif",
        T: int = 4,
        *dummy_args,
        **dummy_kwargs,
    ):
        """Construct a PositionwiseFeedForward object."""
        super(MS_MLP, self).__init__()

        self.T = T
        if node == "plif":
            self.fc1_lif = MultiStepParametricLIFNode(detach_reset=True)
        elif node == "lif":
            self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.fc1_linear = torch.nn.Linear(idim, hidden_units, bias=bias)
        self.fc1_bn = torch.nn.BatchNorm1d(hidden_units)

        if node == "plif":
            self.fc2_lif = MultiStepParametricLIFNode(detach_reset=True)
        elif node == "lif":
            self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.fc2_linear = torch.nn.Linear(hidden_units, idim, bias=bias)
        self.fc2_bn = torch.nn.BatchNorm1d(idim)

        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """

        # (T, b/T, t, d)

        # x = self.fc1_lif(x)  # (T, b/T, t, d)
        x = self.fc1_linear(xs.flatten(0, 1))  # (Tb, t, d)
        x = self.fc1_bn(x.transpose(1,
                                    2).contiguous()).transpose(1,
                                                               2).contiguous()
        x = self.fc1_lif(
            x.reshape(self.T, int(x.size(0) / self.T), x.size(1),
                      x.size(2)).contiguous())

        x = self.dropout(x)

        # x = self.fc2_lif(x.reshape(self.T, int(x.size(0)/self.T), x.size(1), x.size(2)).contiguous())
        x = self.fc2_linear(x.flatten(0, 1))
        x = self.fc2_bn(x.transpose(1,
                                    2).contiguous()).transpose(1,
                                                               2).contiguous()
        x = self.fc2_lif(
            x.reshape(self.T, int(x.size(0) / self.T), x.size(1),
                      x.size(2)).contiguous())

        return x


class SpikePositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = True,
        node: str = "plif",
        T: int = 4,
        *dummy_args,
        **dummy_kwargs,
    ):
        """Construct a PositionwiseFeedForward object."""
        super(SpikePositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units, bias=bias)
        self.T = T
        if node == "plif":
            self.activation = MultiStepParametricLIFNode(detach_reset=True)
        elif node == "lif":
            self.activation = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.activation = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim, bias=bias)
        # print("init,self.T:", self.T)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (T * B, L, D)
        Returns:
            output tensor, (T * B, L, D)
        """
        # print("xs:", xs.shape)
        # print("self.T:", self.T)
        xs = self.w_1(xs)  # (Tb, L, D)
        xs = xs.reshape(self.T,
                        xs.size(0) // self.T, xs.size(1),
                        xs.size(2)).contiguous()
        xs = self.activation(xs)
        xs = xs.flatten(0, 1)
        xs = self.dropout(xs)
        xs = self.w_2(xs)
        # print("xs:", xs.shape)
        return xs
        # return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class SpikeLengthPositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = True,
        node: str = "plif",
        T: int = 4,
        *dummy_args,
        **dummy_kwargs,
    ):
        """Construct a PositionwiseFeedForward object."""
        super(SpikeLengthPositionwiseFeedForward, self).__init__()
        # print("initT:",T)
        self.w_1 = torch.nn.Linear(idim, hidden_units, bias=bias)
        self.T = T
        if node == "plif":
            self.activation = MultiStepParametricLIFNode(detach_reset=True)
        elif node == "lif":
            self.activation = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.activation = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim, bias=bias)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        # print("mlpT:",self.T)
        xs = self.w_1(xs)  # (B, L, D)
        xs = xs.reshape(xs.size(0), self.T,
                        xs.size(1) // self.T,
                        xs.size(2)).contiguous().transpose(
                            0, 1)  # (T, B, L//T, D)
        xs = self.activation(xs)
        xs = xs.transpose(0, 1).flatten(1, 2)  # (B, L, D)
        xs = self.dropout(xs)
        xs = self.w_2(xs)
        # print("xs:", xs.shape)
        return xs
        # return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class SpikeAudioLengthPositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = True,
        node: str = "plif",
        T: int = 4,
        *dummy_args,
        **dummy_kwargs,
    ):
        """Construct a PositionwiseFeedForward object."""
        super(SpikeAudioLengthPositionwiseFeedForward, self).__init__()
        # print("initT:",T)
        self.w_1 = torch.nn.Linear(idim, hidden_units, bias=bias)
        self.T = T
        if node == "plif":
            self.activation = MultiStepParametricLIFNode(detach_reset=True)
        elif node == "lif":
            self.activation = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.activation = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim, bias=bias)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        # print("mlpT:",self.T)
        xs = self.w_1(xs)  # (B, L, D)
        xs = xs.transpose(0, 1)  # (L, B, D)
        xs = self.activation(xs)
        xs = xs.transpose(0, 1)  # (B, L, D)
        xs = self.dropout(xs)
        xs = self.w_2(xs)
        # print("xs:", xs.shape)
        return xs
        # return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = True,
        *dummy_args,
        **dummy_kwargs,
    ):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units, bias=bias)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim, bias=bias)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class MoEFFNLayer(torch.nn.Module):
    """
    Mixture of expert with Positionwise feed forward layer
    See also figure 1 in https://arxiv.org/pdf/2305.15663.pdf
    The output dim is same with the input dim.

    Modified from https://github.com/Lightning-AI/lit-gpt/pull/823
                  https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
    Args:
        n_expert: number of expert.
        n_expert_activated: The actual number of experts used for each frame
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = False,
        n_expert: int = 8,
        n_expert_activated: int = 2,
    ):
        super(MoEFFNLayer, self).__init__()
        self.gate = torch.nn.Linear(idim, n_expert, bias=False)
        self.experts = torch.nn.ModuleList(
            PositionwiseFeedForward(
                idim, hidden_units, dropout_rate, activation, bias=bias)
            for _ in range(n_expert))
        self.n_expert = n_expert
        self.n_expert_activated = n_expert_activated

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Foward function.
        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)

        """
        B, L, D = xs.size(
        )  # batch size, sequence length, embedding dimension (idim)
        xs = xs.view(-1, D)  # (B*L, D)
        router = self.gate(xs)  # (B*L, n_expert)
        logits, selected_experts = torch.topk(
            router, self.n_expert_activated
        )  # probs:(B*L, n_expert_activated), selected_exp: (B*L, n_expert_activated)
        weights = torch.nn.functional.softmax(
            logits, dim=1,
            dtype=torch.float).to(dtype=xs.dtype)  # (B*L, n_expert_activated)
        output = torch.zeros_like(xs)  # (B*L, D)
        for i, expert in enumerate(self.experts):
            mask = selected_experts == i
            token_ids, ith_expert = torch.where(mask)
            output[token_ids] += weights[token_ids, ith_expert, None] * expert(
                xs[token_ids])
        return output.view(B, L, D)


class GatedVariantsMLP(torch.nn.Module):
    """ https://arxiv.org/pdf/2002.05202.pdf
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.GELU(),
        bias: bool = True,
        *dummy_args,
        **dummy_kwargs,
    ):
        """Construct a PositionwiseFeedForward object."""
        super(GatedVariantsMLP, self).__init__()
        self.gate = torch.nn.Linear(idim, hidden_units, bias=False)
        self.activation = activation
        # w_1 as up proj
        self.w_1 = torch.nn.Linear(idim, hidden_units, bias=bias)
        self.dropout = torch.nn.Dropout(dropout_rate)
        # w_2 as down proj
        self.w_2 = torch.nn.Linear(hidden_units, idim, bias=bias)

    def forward(self, x) -> torch.Tensor:
        """Foward function.
        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)

        """
        gate = self.activation(self.gate(x))
        up = self.w_1(x)
        fuse = gate * up
        return self.w_2(self.dropout(fuse))
