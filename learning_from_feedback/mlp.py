from torch.nn.functional import softplus
from torch import distributions as td
import torch


class MlpModel(torch.nn.Module):
    """Multilayer Perceptron with last layer linear.

    Args:
        input_size (int): number of inputs
        hidden_sizes (list): can be empty list for none (linear model).
        output_size: linear layer at output, or if ``None``, the last hidden size will be the output size and will have nonlinearity applied
        nonlinearity: torch nonlinearity Module (not Functional).
    """

    def __init__(
            self,
            input_size,
            hidden_sizes,  # Can be empty list or None for none.
            output_size=None,  # if None, last layer has nonlinearity applied.
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
    ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        elif hidden_sizes is None:
            hidden_sizes = []
        hidden_layers = [torch.nn.Linear(n_in, n_out) for n_in, n_out in
                         zip([input_size] + hidden_sizes[:-1], hidden_sizes)]
        sequence = list()
        for layer in hidden_layers:
            sequence.extend([layer, nonlinearity()])
        if output_size is not None:
            last_size = hidden_sizes[-1] if hidden_sizes else input_size
            sequence.append(torch.nn.Linear(last_size, output_size))
        self.model = torch.nn.Sequential(*sequence)
        self._output_size = (hidden_sizes[-1] if output_size is None
                             else output_size)

    def forward(self, input):
        """Compute the model on the input, assuming input shape [B,input_size]."""
        return self.model(input)

    @property
    def output_size(self):
        """Retuns the output size of the model."""
        return self._output_size


class StochMlpModel(MlpModel):
    def __init__(self, input_size, hidden_sizes, output_size, min_std=0.1, fixed_std=None, squeeze=True, dist='normal', **kwargs):
        if dist == 'normal' and fixed_std is None:
            output_size *= 2
        super().__init__(input_size, hidden_sizes, output_size=output_size, **kwargs)
        self.min_std = min_std
        self.squeeze = squeeze
        self.dist = dist
        self.fixed_std = fixed_std

    def forward(self, x):
        x = self.model(x)
        if self.dist == 'normal':
            if self.fixed_std:
                if self.squeeze:
                    x = x.squeeze(-1)
                dist = td.Normal(x, self.fixed_std)
            else:
                mean, std = torch.chunk(x, 2, dim=-1)
                std = softplus(std) + self.min_std
                if self.squeeze:
                    mean = mean.squeeze(-1)
                    std = std.squeeze(-1)
                dist = td.Normal(mean, std)
        elif self.dist == "binary":
            if self.squeeze:
                x = x.squeeze(-1)
            dist = td.Bernoulli(logits=x)

        return dist
