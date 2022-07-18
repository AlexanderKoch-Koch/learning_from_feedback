import gym.spaces
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
from ray.rllib.utils.framework import TensorType
from ray.rllib.agents.dreamer.utils import FreezeParameters
from typing import Any, List, Tuple
from feedback_learning.mlp import MlpModel, StochMlpModel
from torch import distributions as td
from learning_from_feedback.transformer_tools import PositionalEncoding, pad
import math


class VAEDreamerModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.action_size = action_size = math.prod(action_space.shape)
        self.obs_size = obs_size = obs_space.shape[0]
        self.stoch_size = stoch_size = model_config['stoch_size']
        self.state_size = state_size = model_config['state_size']
        self.hidden_size = hidden_size = model_config['hidden_size']
        self.discount = 0.98
        self.pred_horizon = 3

        transformer_layer = nn.TransformerEncoderLayer(state_size, 4, hidden_size, dropout=0.0, activation='gelu')

        # self.action_model = StochMlpModel(obs_size, 2 * [hidden_size,], stoch_size, squeeze=False)
        self.action_model = MlpModel(obs_size, 2 * [hidden_size,], stoch_size)
        self.encoder = nn.TransformerEncoder(transformer_layer, 2)
        self.decoder = nn.TransformerEncoder(transformer_layer, 2)
        self.reward_model = nn.TransformerEncoder(transformer_layer, 2)
        self.pos_enc = PositionalEncoding(self.state_size)

    def policy(self,
               obs: TensorType,
               state: List[TensorType],
               explore=True
               ) -> Tuple[TensorType, List[TensorType], List[TensorType]]:
        observations = torch.cat((state[0], obs.unsqueeze(1)), dim=1).transpose(0, 1)[1:]
        prev_actions = state[1].transpose(0, 1)
        T, B, D = observations.shape

        # latent_action = self.action_model(obs).sample() if explore else self.action_model(obs).mean
        latent_action = torch.tanh(self.action_model(obs))
        action_obs =torch.cat((prev_actions[-1], obs), dim=-1)
        decoder_input = torch.cat((action_obs, latent_action), dim=-1)
        decoder_input = pad(decoder_input.reshape(1, B, -1).repeat(self.pred_horizon, 1, 1), self.state_size)
        reconstruction = self.decoder(self.pos_enc(decoder_input))[:, :, :action_obs.shape[-1]]
        action = reconstruction[0, :, :self.action_size]

        if explore:
            action = td.Normal(action, 0.3).sample()

        self.squash_action(action, clip=False)
        state = [observations.transpose(0, 1),
                 torch.cat((prev_actions, action.reshape(1, B, self.action_size)), dim=0)[1:].transpose(0, 1)]
        return action, state, None

    def loss(self, observations, actions, rewards, valid, done):
        """observations and actions in shape T x B x ..."""
        T, B = observations.shape[:2]
        action_obs = torch.cat((actions, observations), dim=-1)

        encoding = self.encoder(self.pos_enc(pad(action_obs, self.state_size)))[0, :, :2 * self.stoch_size]
        mean, std = torch.chunk(encoding, 2, dim=-1)
        latent_dist = td.Normal(mean, torch.nn.functional.softplus(std))
        kl_loss = td.kl_divergence(latent_dist, td.Normal(torch.zeros_like(mean), 1)).mean()

        decoder_input = torch.cat((action_obs[0], latent_dist.rsample()), dim=-1)
        decoder_input = pad(decoder_input.reshape(1, B, -1).repeat(T-1, 1, 1), self.state_size)
        reconstruction = self.decoder(self.pos_enc(decoder_input))[:, :, :action_obs.shape[-1]]
        reconstruction_loss = ((reconstruction - action_obs[1:]) ** 2).mean()

        reward_pred = self.reward_model(self.pos_enc(pad(action_obs, self.state_size)))[1:, :, 0]
        reward_pred_loss = ((reward_pred - rewards[1:]) ** 2).mean()

        latent_action = torch.tanh(self.action_model(observations[0]))
        # action_kl = td.kl_divergence(latent_action, td.Normal(torch.zeros_like(latent_action.mean), 1))#.mean()

        with FreezeParameters(self.get_model_weights()):
            # decoder_input = torch.cat((action_obs[0], latent_action.rsample()), dim=-1)
            decoder_input = torch.cat((action_obs[0], latent_action), dim=-1)
            decoder_input = pad(decoder_input.reshape(1, B, -1).repeat(T - 1, 1, 1), self.state_size)
            reconstruction = self.decoder(self.pos_enc(decoder_input))[:, :, :action_obs.shape[-1]]
            reward_input = pad(reconstruction[:, :, :self.action_size + self.obs_size], self.state_size)
            reward_input = pad(torch.cat((action_obs[:1], reconstruction), dim=0), self.state_size)
            policy_reward_pred = self.reward_model(self.pos_enc(reward_input))[1:, :, 0]

        # discounted_return = torch.cumprod(torch.tensor(self.discount, device=self.device).repeat(T//2), dim=0) * policy_reward_pred
        discounted_return = policy_reward_pred
        loss = 1 * reconstruction_loss + 1 * kl_loss - 1 * discounted_return.mean() + 1 * reward_pred_loss \
                # + (latent_action ** 4).mean()
                # + action_kl.clamp(min=1).mean() \
                # + 10 * action_kl.mean()

        info_dict = dict(
            reward_pred_loss=reward_pred_loss,
            reconstruction_loss=reconstruction_loss,
            kl_loss=kl_loss,
            # action_kl=action_kl.mean(),
            max_abs_action=latent_action.abs().max(),
            reward_pred_abs_error=(reward_pred - rewards[1:]).abs().mean(),
            mean_reward_pred=policy_reward_pred.mean(),
            true_reward_mean=rewards[1:].mean()
        )
        return loss, info_dict

    def squash_action(self, action, clip=True):
        if isinstance(self.action_space, gym.spaces.Box):
            high = torch.tensor(self.action_space.high, device=action.device)
            low = torch.tensor(self.action_space.low, device=action.device)
            if clip:
                return torch.clip(action, min=low, max=high)
            else:
                return torch.sigmoid(action) * (high - low) + low
        elif isinstance(self.action_space, gym.spaces.Discrete):
            return action.argmax(dim=-1)

    def action_pred_loss(self, action_pred, true_actions):
        if isinstance(self.action_space, gym.spaces.Discrete):
            # action_pred = torch.nn.functional.one_hot(action_pred, self.action_size).shape
            # true_actions = torch.nn.functional.one_hot(true_actions.argmax(dim=-1), self.action_size).float().reshape(-1, self.action_size)
            action_pred = action_pred.reshape(-1, self.action_size)
            action_pred_loss = torch.nn.functional.cross_entropy(action_pred, true_actions.flatten().long())
        elif isinstance(self.action_space, gym.spaces.Box):
            action_pred = self.squash_action(action_pred)
            action_pred_loss = ((action_pred - true_actions) ** 2).mean()

        return action_pred_loss

    def get_initial_state(self, batch_size=1, sequence_length=2) -> List[TensorType]:
        state = [
            torch.zeros((batch_size, sequence_length // 2, self.obs_space.shape[0]), device=self.device),
            torch.zeros((batch_size, sequence_length // 2, self.action_size), device=self.device),
        ]
        return state

    def get_model_weights(self):
        return list(self.reward_model.parameters()) + \
               list(self.encoder.parameters()) + \
               list(self.decoder.parameters()) + \
               list(self.pos_enc.parameters())

    def get_actor_weights(self):
        return list(self.action_model.parameters())

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
