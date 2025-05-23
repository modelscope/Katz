# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
import torch.nn.functional as F
from torch import nn
import fast_kernel

from ..utils import deprecate


ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


class GELU(nn.Module):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate, approximate=self.approximate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


# Original GEGLU
# class GEGLU(nn.Module):
#     r"""
#     A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

#     Parameters:
#         dim_in (`int`): The number of channels in the input.
#         dim_out (`int`): The number of channels in the output.
#         bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
#     """

#     def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
#         super().__init__()
#         self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

#     def gelu(self, gate: torch.Tensor) -> torch.Tensor:
#         if gate.device.type != "mps":
#             return F.gelu(gate)
#         # mps: gelu is not implemented for float16
#         return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

#     def forward(self, hidden_states, *args, **kwargs):
#         if len(args) > 0 or kwargs.get("scale", None) is not None:
#             deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
#             deprecate("scale", "1.0.0", deprecation_message)

#         hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
#         return hidden_states * self.gelu(gate)


# Optimized GEGLU
class GEGLU(nn.Module):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.enable_fused_geglu = os.getenv("ENABLE_FUSED_GEGLU", "0") == "1"
        if os.getenv("VERBOSE", "0") == "1":
            print(f"enable_fused_geglu: {self.enable_fused_geglu}")
        if self.enable_fused_geglu:
            self.proj_1 = nn.Linear(dim_in, dim_out, bias=bias)
            self.proj_2 = nn.Linear(dim_in, dim_out, bias=bias)
            self._register_load_state_dict_pre_hook(self.load_state_dict_pre_hook)
        else:
            self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states, *args, **kwargs):
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        if self.enable_fused_geglu:
            return torch.ops.fast_kernel.fast_gelu_mul_fp16(self.proj_2(hidden_states), self.proj_1(hidden_states))
            # return self.proj_1(hidden_states) * self.gelu(self.proj_2(hidden_states))
        else:
            hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
            return hidden_states * self.gelu(gate)

    def load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        proj.weight --> proj_1.weight, proj_2.weight
        proj.bias --> proj_1.bias, proj_2.bias
        """
        # print(f"GEGLU test, prefix: {prefix}")
        if prefix + "proj.weight" not in state_dict:
            return
        weight = state_dict[prefix + "proj.weight"]
        bias = state_dict.get(prefix + "proj.bias")

        state_dict.pop(prefix + "proj.weight")
        weight_1, weight_2 = weight.chunk(2, dim=0)
        state_dict[prefix + "proj_1.weight"] = weight_1.contiguous()
        state_dict[prefix + "proj_2.weight"] = weight_2.contiguous()

        state_dict.pop(prefix + "proj.bias")
        bias_1, bias_2 = bias.chunk(2, dim=0)
        state_dict[prefix + "proj_1.bias"] = bias_1.contiguous()
        state_dict[prefix + "proj_2.bias"] = bias_2.contiguous()


class ApproximateGELU(nn.Module):
    r"""
    The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
    [paper](https://arxiv.org/abs/1606.08415).

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)
