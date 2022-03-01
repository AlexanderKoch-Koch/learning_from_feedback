import gym
from typing import Any, List, Tuple
from ray.rllib.agents.dreamer.utils import FreezeParameters
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

ActFunc = Any

class Mlp(nn.Module):
    """
    A simple feedforward network with skip connections and ELU activations
    """
    def __init__(self, num_layers, input_size, hidden_size, output_size):
        nn.Module.__init__(self)
        self.encoding_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = nn.functional.elu(self.encoding_layer(x))
        for hidden_layer in self.hidden_layers:
            x =  x + nn.functional.elu(hidden_layer(x))

        return self.output_layer(x)


class SingleStepDreamer(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.hidden_size = model_config["hidden_size"]
        self.action_size = action_space.shape[0]
        self.obs_size = obs_space.shape[0]
        if isinstance(self.action_space, gym.spaces.Box):
            self.action_model = Mlp(1, self.obs_size, self.hidden_size, self.action_size)
            self.pred_model = Mlp(2, self.obs_size + self.action_size, self.hidden_size, self.obs_size)
            self.reward_model = Mlp(0, 2 * self.obs_size + self.action_size, self.hidden_size, 1)
        else:
            self.action_model = Mlp(1, self.obs_size, self.hidden_size, 2 * self.action_size)
            self.pred_model = Mlp(2, self.obs_size + 2 * self.action_size, self.hidden_size, self.obs_size)
            self.reward_model = Mlp(0, 2 * self.obs_size + 2 * self.action_size, self.hidden_size, 1)

        self.standard_Gumbel = torch.distributions.Gumbel(0, 1)
        self.tau = 1

    def policy(self, obs, explore=False):
        self.action_model.eval()
        if isinstance(self.action_space, gym.spaces.Box):
            action = torch.tanh(self.action_model(obs))
        else:
            if explore:
                probs = torch.nn.functional.gumbel_softmax(
                    self.action_model(obs).reshape(-1, self.action_size, 2), tau=self.tau, hard=False)
                action = torch.distributions.Categorical(probs=probs).sample()
            else:
                logits = self.action_model(obs).reshape(-1, self.action_size, 2)
                action = torch.argmax(logits, dim=-1)

        return action

    def loss(self, obs, action, reward):
        if isinstance(self.action_space, gym.spaces.Box):
            action = action[:, 1]
        else:
            action = nn.functional.one_hot(action[:, 1].long(), num_classes=2).flatten(start_dim=1)

        obs_pred = self.pred_model(torch.cat((obs[:, 0], action), dim=-1))
        pred_loss = torch.mean((obs_pred - obs[:, 1]) ** 2)
        reward_pred = self.reward_model(torch.cat((obs[:, 0], obs[:, 1], action), dim=-1))
        # reward_target_one_hot = torch.zeros(reward_pred.shape, device=obs.device)
        # reward_target_one_hot[reward[:, 1] < 0.5, 0] = 1
        # reward_target_one_hot[reward[:, 1] > 0.5, 1] = 1
        # reward_loss = torch.nn.functional.binary_cross_entropy_with_logits(reward_pred, reward_target_one_hot)
        reward_loss = torch.mean((reward_pred.squeeze(-1) - reward[:, 1]) ** 2)
        model_loss = pred_loss + reward_loss

        if isinstance(self.action_space, gym.spaces.Box):
            action = torch.tanh(self.action_model(obs[:, 0]))
        else:
            logits = self.action_model(obs[:, 0]).reshape(-1, self.action_size, 2)
            action = torch.nn.functional.gumbel_softmax(logits, tau=self.tau, hard=True)
            action = action.flatten(start_dim=1)

        with FreezeParameters(list(self.pred_model.parameters())):
            obs_pred_on_policy = self.pred_model(torch.cat((obs[:, 0], action), dim=-1))
        with FreezeParameters(list(self.reward_model.parameters())):
            reward_pred_on_policy = self.reward_model(torch.cat((obs[:, 0], obs_pred_on_policy, action), dim=-1))

        actor_loss = -reward_pred_on_policy.mean() + (action ** 2).mean()
        # actor_loss = (reward_pred_on_policy[:, 0] - reward_pred_on_policy[:, 1]).mean() + (action ** 2).mean()
        info = dict(
            pred_loss=pred_loss,
            pred_min=obs_pred.min(dim=-1)[0].mean(),
            pred_max=obs_pred.max(dim=-1)[0].mean(),
            on_policy_pred_min=obs_pred_on_policy.min(),
            on_policy_pred_max=obs_pred_on_policy.max(),
            reward_loss=reward_loss,
            reward_pred_mean=reward_pred_on_policy.mean(),
            reward_pred_max=reward_pred_on_policy.max(),
            action_magnitude=action.abs().mean(),
            action_min=action.min(),
            action_max=action.max(),
        )
        return model_loss, actor_loss, info
