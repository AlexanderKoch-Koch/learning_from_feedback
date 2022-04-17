import numpy as np
from typing import Any, List, Tuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.framework import TensorType

torch, nn = try_import_torch()
from torch import distributions as td
from ray.rllib.agents.dreamer.utils import TanhBijector
from torch.nn import Conv2d, ConvTranspose2d, GRUCell, Linear


# Represents dreamer policy
class ActionModel(nn.Module):
    """ActionDecoder is the policy module in Dreamer. It outputs a distribution
      parameterized by mean and std, later to be transformed by a custom
      TanhBijector in utils.py for Dreamer.
      """

    def __init__(self,
                 input_size: int,
                 action_size: int,
                 layers: int,
                 units: int,
                 action_space,
                 dist: str = "tanh_normal",
                 act=None,
                 min_std: float = 1e-4,
                 init_std: float = 5.0,
                 mean_scale: float = 5.0):
        """Initializes Policy

          Args:
            input_size (int): Input size to network
            action_size (int): Action space size
            layers (int): Number of layers in network
            units (int): Size of the hidden layers
            dist (str): Output distribution, with tanh_normal implemented
            act (Any): Activation function
            min_std (float): Minimum std for output distribution
            init_std (float): Intitial std
            mean_scale (float): Augmenting mean output from FC network
          """
        super().__init__()
        self.layrs = layers
        self.units = units
        self.dist = dist
        self.act = act
        if not act:
            self.act = nn.ELU
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.action_size = action_size

        self.layers = []
        self.softplus = nn.Softplus()

        action_range = action_space.high[0] - action_space.low[0]
        self.action_scale = action_range /2 # scale after tanh
        self.action_loc = 0.5 * (action_space.high[0] + action_space.low[0])
        self.max_action = action_space.high[0]
        self.min_action = action_space.low[0]

        # MLP Construction
        cur_size = input_size
        for _ in range(self.layrs):
            self.layers.extend([Linear(cur_size, self.units), self.act()])
            cur_size = self.units
        if self.dist in ["tanh_normal", 'normal', 'beta']:
            self.layers.append(Linear(cur_size, 2 * action_size))
        elif self.dist == "onehot":
            self.layers.append(Linear(cur_size, action_size))
        self.model = nn.Sequential(*self.layers)

    # Returns distribution
    def forward(self, x):
        raw_init_std = np.log(np.exp(self.init_std) - 1)
        x = self.model(x)
        if self.dist == 'normal':
            mean, std = torch.chunk(x, 2, dim=-1)
            # mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
            std = self.softplus(std) + self.min_std
            print(f'action mean {mean} std {std}')
            dist = td.Normal(mean, std)
            dist.mode = mean
        elif self.dist == "tanh_normal":
            mean, std = torch.chunk(x, 2, dim=-1)
            scaled_mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
            # std = self.softplus(std + raw_init_std) + self.min_std
            std = self.softplus(std + raw_init_std) + self.min_std
            dist = td.Normal(scaled_mean, std)
            transforms = [TanhBijector(),
                    td.transforms.AffineTransform(self.action_loc, self.action_scale)]
            entropy = dist.entropy().sum(dim=-1)
            dist = td.transformed_distribution.TransformedDistribution(
                dist, transforms)
            dist = td.Independent(dist, 1)
            # dist.mode = self.action_scale * torch.tanh(mean) + self.action_loc
            dist.mode = self.action_scale * scaled_mean + self.action_loc
            dist._entropy = entropy
            dist.scale = std
        elif self.dist == "onehot":
            dist = td.OneHotCategorical(logits=x)
        elif self.dist == 'beta':
            params = 10 * self.softplus(x) + 1
            alpha, beta = torch.chunk(params, 2, dim=-1)
            dist = td.Beta(alpha, beta)
            entropy = dist.entropy()
            dist = td.transformed_distribution.TransformedDistribution(dist,
                    [td.transforms.AffineTransform(self.min_action,
                              self.max_action - self.min_action)])
            mode = (alpha - 1) / (alpha + beta - 2)
            mode = (self.max_action - self.min_action) * mode + self.min_action
            dist.mode = mode
            dist.scale = torch.mean(alpha + beta)
            dist._entropy = entropy
        else:
            raise NotImplementedError("action distribution '" + self.dist + "' not implemented yet!")
        return dist
