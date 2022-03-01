import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
from ray.rllib.utils.framework import TensorType
from ray.rllib.agents.dreamer.utils import FreezeParameters
from typing import Any, List, Tuple
from learning_from_feedback.mlp import MlpModel, StochMlpModel


def pad(tensor, desired_size, dim=-1):
    """
    simple function to add zero padding
    :param tensor: tensor that should be padded
    :param desired_size: desired size of specified dim after padding
    :param dim: padding dimension
    :return: tensor with zero padding in last dimension so that shape[-1] == desired_size
    """
    padding_shape = list(tensor.shape)
    padding_shape[dim] = desired_size - tensor.shape[dim]
    return torch.cat((tensor, torch.zeros(padding_shape, device=tensor.device)), dim=dim)


class PositionalEncoding(torch.nn.Module):
    """
    Learnable positional encoding for Transformer
    """
    def __init__(self, d_model, max_len=300):
        """

        :param d_model: transformer size
        :param max_len: maximum expected sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = nn.Parameter(torch.randn(max_len, 1, self.d_model))

    def forward(self, x):
        x = x + self.pe[:x.shape[0]]
        return x


class OpenLoopDreamerModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.action_size = action_space.shape[0]
        self.state_size = state_size = model_config['state_size']
        self.hidden_size = hidden_size = model_config['hidden_size']
        self.memory_length = model_config['memory_length']
        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        transformer_layer = nn.TransformerEncoderLayer(state_size, 4, hidden_size, dropout=0.0, activation='gelu')
        policy_transformer_layer = nn.TransformerEncoderLayer(state_size * 2, 4, hidden_size * 2, dropout=0.0,
                                                              activation='gelu')
        self.policy_transformer = nn.TransformerEncoder(policy_transformer_layer, 3)
        self.policy_positional_encoding = PositionalEncoding(state_size * 2)
        self.reward_transformer = nn.TransformerEncoder(transformer_layer, 2)
        self.reward_positional_encoding = PositionalEncoding(state_size)
        self.reward_model = StochMlpModel(self.state_size, [], 1, fixed_std=1, squeeze=False)
        self.obs_model = nn.TransformerEncoder(transformer_layer, 3)
        self.obs_model_pos_encoding = PositionalEncoding(state_size)
        self.forward_mask = torch.triu(torch.full((300, 300), float('-inf')), diagonal=1).to(self.device)
        self.discount = model_config['discount']

    def policy(self,
               obs: TensorType,
               state: List[TensorType],
               explore=True
               ) -> Tuple[TensorType, List[TensorType], List[TensorType]]:
        observations = torch.cat((state[0], obs.unsqueeze(1)), dim=1).transpose(0, 1)[1:]
        prev_actions = state[1].transpose(0, 1)
        T = observations.shape[0]
        S = self.memory_length
        policy_input = pad(torch.cat((observations[-S:], 0 * prev_actions[-S:]), dim=-1), self.state_size * 2)
        policy_input = pad(policy_input, T, dim=0)
        state = self.policy_transformer(self.policy_positional_encoding(policy_input))
        action = self.squash_action(state[S:, :, :self.action_size])[0]

        state = [observations.transpose(0, 1),
                 torch.cat((prev_actions, action.unsqueeze(0)), dim=0)[1:].transpose(0, 1)]
        return action, state, None

    def loss(self, observations, actions, rewards, valid, done):
        """observations and actions in shape T x B x ..."""
        T, B = observations.shape[:2]
        S = self.memory_length
        model_input = pad(torch.cat((observations[:S], actions[:S]), dim=-1), self.state_size)
        model_input = torch.cat((model_input, pad(actions[S:], self.state_size)), dim=0)
        obs_pred = self.obs_model(self.obs_model_pos_encoding(model_input), mask=self.forward_mask[:T, :T])
        obs_pred = obs_pred[S:, :, :self.obs_space.shape[0]]

        model_input = torch.cat((torch.cat((observations[:S], obs_pred.detach()), dim=0), actions), dim=-1)
        # model_input = torch.cat((observations, actions), dim=-1)
        model_input = pad(model_input, self.state_size)
        reward_pred = self.reward_transformer(self.reward_positional_encoding(model_input),
                                              mask=self.forward_mask[:T, :T])#[S:, :, 0]
        # model_reward_pred_loss = ((reward_pred - rewards[S:]) ** 2).mean()
        reward_pred = self.reward_model(reward_pred[S:])
        model_reward_pred_loss = -reward_pred.log_prob(rewards[S:].unsqueeze(-1)).mean()
        obs_abs_error = (obs_pred - observations[S:]).abs().mean()
        reward_abs_error = (reward_pred.mean - rewards[S:].unsqueeze(-1)).abs().mean()
        # reward_abs_error = (reward_pred - rewards[S:]).abs().mean()
        model_obs_pred_loss = ((obs_pred - observations[S:]) ** 2).mean()
        # model_obs_pred_loss = -obs_pred.log_prob(observations[S:]).mean()
        model_loss = model_obs_pred_loss + model_reward_pred_loss

        policy_input = pad(torch.cat((observations[:S], 0 * actions[:S]), dim=-1), self.state_size * 2)
        # policy_input = torch.cat((policy_input, pad(noise * 0, self.state_size * 2)), dim=0)
        state = self.policy_transformer(self.policy_positional_encoding(pad(policy_input, T, dim=0)))
        policy_actions = self.squash_action(state[S:, :, :self.action_size])

        with FreezeParameters(list(self.obs_model.parameters()) + list(self.obs_model_pos_encoding.parameters())
                              + list(self.reward_positional_encoding.parameters())
                              + list(self.reward_model.parameters())
                              + list(self.reward_transformer.parameters())):
            model_input = pad(torch.cat((observations[:S], actions[:S]), dim=-1), self.state_size)
            model_input = torch.cat((model_input, pad(policy_actions, self.state_size)), dim=0)
            model_state = self.obs_model(self.obs_model_pos_encoding(model_input), mask=self.forward_mask[:T, :T])

            obs_pred = model_state[S:, :, :self.obs_space.shape[0]]

            observations = torch.cat((observations[:S], obs_pred), dim=0)
            model_input = pad(torch.cat((observations, torch.cat((actions[:S], policy_actions), dim=0)), dim=-1), self.state_size)
            reward_pred = self.reward_transformer(self.reward_positional_encoding(model_input),
                                                  mask=self.forward_mask[:T, :T])#[S:, :, 0]
            reward_pred = self.reward_model(reward_pred[S:]).mean[:, :, 0]

        # policy_loss = -reward_pred.mean()
        policy_loss = -torch.mean(torch.cumprod(self.discount * torch.ones(T - S, device=self.device), dim=0).unsqueeze(-1) * reward_pred)
        loss = model_loss + policy_loss

        info_dict = dict(
            model_obs_pred_loss=model_obs_pred_loss,
            model_reward_pred_loss=model_reward_pred_loss,
            reward_abs_error=reward_abs_error,
            obs_abs_error=obs_abs_error,
            mean_reward_pred=reward_pred.mean(),
            true_reward_mean=rewards[S:].mean()
        )
        return loss, info_dict

    def squash_action(self, action):
        high = torch.tensor(self.action_space.high, device=action.device)
        low = torch.tensor(self.action_space.low, device=action.device)
        return torch.sigmoid(action) * (high - low) + low

    def get_initial_state(self, batch_size=1, sequence_length=2) -> List[TensorType]:
        state = [
            torch.zeros((batch_size, sequence_length, self.obs_space.shape[0]), device=self.device),
            torch.zeros((batch_size, sequence_length, self.action_space.shape[0]), device=self.device),
        ]
        return state

    def get_model_weights(self):
        return list(self.reward_model.parameters()) + \
               list(self.state_transformer.parameters()) + \
               list(self.state_positional_encoding.parameters()) + \
               list(self.termination_pred_model.parameters())

    def get_random_action(self, shape):
        return self.squash_action(torch.randn(shape).to(self.device))

    def get_state_dims(self):
        return 3

    def value_function(self) -> TensorType:
        return None

    def generate_mask(self, sz: int):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
