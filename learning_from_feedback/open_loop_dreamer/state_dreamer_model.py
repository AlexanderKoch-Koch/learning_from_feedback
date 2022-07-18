g import List, Tuple
import numpy as np
import torch
import torch.distributions as td
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import TensorType
from torch import nn
from torch.nn import GRUCell

from learning_from_feedback.mlp import MlpModel, StochMlpModel
from learning_from_feedback.transformer_tools import PositionalEncoding, pad, generate_mask
from learning_from_feedback.open_loop_dreamer.action_model import ActionModel


class FreezeParameters:
    def __init__(self, parameters):
        self.parameters = parameters
        self.param_states = [p.requires_grad for p in self.parameters]

    def __enter__(self):
        for param in self.parameters:
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(self.parameters):
            param.requires_grad = self.param_states[i]


# class StateDreamerModel(TorchModelV2, nn.Module):
class StateDreamerModel(nn.Module, TorchModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.horizon, self.discount = model_config['horizon'], model_config['discount']
        self.action_size = action_size = action_space.shape[0]
        stoch_size, deter_size = model_config['stoch_size'], model_config['deter_size']
        self.stoch_size, self.deter_size = stoch_size, deter_size
        self.state_size = state_size = deter_size + stoch_size
        hidden_size = model_config['hidden_size']
        self.transformer_dim = transformer_dim = 128
        # self.post_model = StochMlpModel(obs_space.shape[0], [hidden_size, ] * 2, stoch_size)
        self.post_model = StochMlpModel(obs_space.shape[0] + deter_size, [hidden_size, ] * 2, stoch_size)

        # self.decoder = StochMlpModel(state_size, 2 * [hidden_size, ],
        #                              output_size=obs_space.shape[0], squeeze=False, fixed_std=1)
        self.obs_pred_model = MlpModel(2 * obs_space.shape[0] + 2 * action_size, [hidden_size, hidden_size], 2 * obs_space.shape[0])
        self.decoder = MlpModel(state_size, 4 * [hidden_size, ], output_size=obs_space.shape[0], nonlinearity=torch.nn.ELU)

        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # self.reward_model = StochMlpModel(state_size, [hidden_size,], 1, fixed_std=1)
        # self.reward_model = MlpModel(state_size, [hidden_size,], 1)
        self.reward_model = MlpModel(obs_space.shape[0], [hidden_size, ] * 1, 1)
        transformer_layer = nn.TransformerEncoderLayer(transformer_dim, 4, hidden_size, dropout=0.0)
        self.reward_model = nn.TransformerEncoder(transformer_layer, 1)
        self.positional_encoding = PositionalEncoding(transformer_dim)
        self.mask = generate_mask(size=16, device=self.device)

        self.termination_pred_model = StochMlpModel(state_size, [hidden_size], 1, dist='binary')
        self.prior_model = StochMlpModel(deter_size, [hidden_size], stoch_size)
        # self.action_model = ActionModel(deter_size + stoch_size, action_size, 2, hidden_size,
        #                                 action_space, min_std=0.001, init_std=0.3)
        self.action_model = MlpModel(deter_size + stoch_size, [hidden_size, ] * 2
                                     , action_size, nonlinearity=torch.nn.ELU)

        self.gru_cell = GRUCell(hidden_size, hidden_size=deter_size)
        self.gru = torch.nn.GRU(hidden_size, deter_size)
        self.img1 = MlpModel(stoch_size + action_size, 1 * [hidden_size], )

    def policy(self,
               obs: TensorType,
               state: List[TensorType],
               explore=True
               ) -> Tuple[TensorType, List[float], List[TensorType]]:
        stoch_state, gru_state, prev_action = state
        # posterior = self.post_model(torch.cat((gru_state, obs), dim=-1)).rsample()
        # posterior = self.post_model(obs).rsample()
        # stoch_state = posterior
        # state = torch.cat((stoch_state, gru_state), dim=-1)

        # gru_inputs = self.img1(torch.cat((stoch_state, prev_action), dim=-1))
        # gru_state = self.gru_cell(gru_inputs, gru_state)
        # post_dist = self.post_model(torch.cat((obs, gru_state * 0), dim=-1))
        # stoch_state = post_dist.rsample()
        # state = torch.cat((stoch_state, gru_state), dim=-1)

        if explore:
            # action = torch.tanh(self.action_model(state))  # .rsample()
            # action = td.Normal(action, torch.zeros_like(action) + 0.3).sample().clamp(min=-1, max=1)
            # action = torch.tensor(np.random.choice([-0.5, 0, 0.5], size=action.shape))
            # if torch.rand(1) > 0.5:
            action = torch.rand(prev_action.shape, device=self.device) * 2 - 1
        else:
            # action = torch.tanh(self.action_model(state))
            action = torch.rand(prev_action.shape, device=self.device) * 2 - 1
            # action = self.action_model(state).mode

        # x = self.img1(torch.cat([stoch_state, action], dim=-1))
        # _, gru_state = self.gru(x.unsqueeze(0), gru_state.unsqueeze(0))
        # gru_state = gru_state[0]

        state = [stoch_state, gru_state, action]
        return action, state, None

    def loss(self, observations, actions, rewards, valid, done):
        T, B = actions.shape[:2]
        stoch_state, gru_state, _ = self.get_initial_state(batch_size=B)
        stoch_states, gru_states = [], []
        kl_loss = 0
        mean_kl_div = 0

        for t in range(T):
            if t > 0:
                gru_inputs = self.img1(torch.cat((stoch_state, actions[t]), dim=-1))
                gru_state = self.gru_cell(gru_inputs, gru_state)
                prior_dist = self.prior_model(gru_state)
                prior = prior_dist.rsample()

            post_dist = self.post_model(torch.cat((observations[t], gru_state * 0), dim=-1))
            stoch_state = post_dist.rsample()
            # stoch_state = post_dist.mean
            if t > 0:
                kl_loss += (1 / T) * torch.mean(self.kl_loss(prior_dist, post_dist) * valid[t])
                mean_kl_div += (1 / T) * (td.kl_divergence(prior_dist, post_dist).mean(dim=-1) * valid[t]).mean()
                # stoch_state *= 0
            stoch_states.append(stoch_state)
            gru_states.append(gru_state)

        stoch_states, gru_states = torch.stack(stoch_states), torch.stack(gru_states)
        states = torch.cat((stoch_states, gru_states), dim=-1)


        obs_reconstruction = self.decoder(states)
        obs_input = torch.cat((observations[0:1], torch.zeros_like(observations[1:])), dim=0)
        obs_reconstruction = self.obs_pred_model(torch.cat((obs_input, actions), dim=-1).permute(1, 0, 2).reshape(B, -1)).reshape(B, T, -1).permute(1, 0, 2)
        reward_model_input = torch.cat((obs_reconstruction.detach(), actions), dim=-1)
        reward_model_input = self.positional_encoding(pad(reward_model_input, self.transformer_dim))
        # reward_model_input = self.positional_encoding(
        #     pad(torch.cat((observations, actions), dim=-1), self.transformer_dim))
        model_reward_pred = self.reward_model(reward_model_input, mask=self.mask[:T, :T])[:, :, 0]
        termination_pred = self.termination_pred_model(states)

        # obs_loss = (((obs_reconstruction - observations)[0, :, :] ** 2).mean(dim=-1)).mean()
        obs_loss = (valid[1:] * ((obs_reconstruction - observations)[1:, :, -4:-2] ** 2).mean(dim=-1)).mean()
        reward_loss = (valid * ((model_reward_pred - rewards) ** 2)).mean()
        termination_loss = -(valid * termination_pred.log_prob(done)).mean()
        model_loss = 0 * kl_loss + 1 * obs_loss + 1 * reward_loss + 0 * termination_loss
        print(f'reward Loss: {reward_loss} obs_loss {obs_loss}')

        with FreezeParameters(list(self.get_model_weights())):
            imagined_states, new_actions, action_stds = self.dream(stoch_states[0].detach(), gru_states[0].detach(),
                                                                   self.horizon)
            states = torch.cat((states[:1].detach(), imagined_states), dim=0)
            obs_pred = self.decoder(states)
            # obs_pred = self.decoder(imagined_states)  # .mean
            # reward_model_input = self.positional_encoding(pad(
            # torch.cat((obs_reconstruction.mean[:-1].detach(), obs_pred), dim=0), 32))
            # obs = torch.cat((obs_reconstruction[:1].detach(), obs_pred), dim=0)
            policy_actions = torch.cat((actions[:1], new_actions), dim=0)
            # obs_pred = torch.cat((observations[:1], obs_pred[1:]), dim=0)
            reward_model_input = torch.cat((obs_pred, policy_actions), dim=-1)
            reward_model_input = self.positional_encoding(pad(reward_model_input, self.transformer_dim))
            reward_pred = self.reward_model(reward_model_input, mask=self.mask[:obs_pred.shape[0], :obs_pred.shape[0]])[1:, :, 0]
            # reward_pred = self.reward_model(obs_pred)
            # reward_pred = self.reward_model(imagined_states)#.mean
            termination_pred = self.termination_pred_model(imagined_states).mean
        termination_pred = torch.cat((torch.zeros_like(termination_pred[0:1]), termination_pred), dim=0)[:-1]
        discount = torch.cumprod(1 - termination_pred, dim=0)
        policy_loss = -torch.mean(discount.detach() * reward_pred) + 10 * (
                    (new_actions.abs() - 0.9).clamp(min=0) ** 2).mean()
        # policy_loss = -torch.mean(discount.detach() * reward_pred) + (new_actions ** 2).mean()
        info = dict(
            reward_loss=reward_loss,
            # abs_action_value=torch.mean(torch.abs(self.action_model(torch.cat((stoch_state, gru_state), dim=-1)))),
            abs_action_value=torch.mean(torch.abs(new_actions)),
            action_dist_std=torch.mean(action_stds),
            termination_pred_loss=termination_loss,
            avg_termination_pred=termination_pred.mean(),
            kl_loss=kl_loss,
            reconstruction_loss=obs_loss,
            mean_kl_div=mean_kl_div,
            mean_dreamed_reward=(discount * reward_pred).sum(dim=0).mean(),
        )
        return model_loss + 0 * policy_loss, info

    def loss_no_loop(self, observations, actions, rewards, valid, done):
        """observations and actions in shape T x B x ..."""
        T, B = actions.shape[:2]
        stoch_state, gru_state, _ = self.get_initial_state(batch_size=B)
        post_dists = self.post_model(observations)
        stoch_states = post_dists.rsample()

        gru_inputs = self.img1(torch.cat((stoch_states[:-1], actions[1:]), dim=-1))
        gru_states, _ = self.gru(gru_inputs, gru_state.unsqueeze(0))

        gru_states = torch.cat((gru_state.unsqueeze(0), gru_states), dim=0)
        prior_dists = self.prior_model(gru_states[1:])
        target_dist = td.Normal(post_dists.mean[1:], post_dists.stddev[1:])
        kl_loss = (valid[1:] * self.kl_loss(prior_dists, target_dist)).sum() / valid[1:].sum()
        # kl_loss = 0.2 * td.kl_divergence(target_dist, prior_dists).sum(dim=-1).clamp(min=2).mean()
        # kl_loss = torch.mean(((td.kl_divergence(target_dist, prior_dists).sum(dim=-1) - 1).clamp(min=0).detach() ** 2) * td.kl_divergence(target_dist, prior_dists).sum(dim=-1))
        mean_kl_div = (valid[1:] * td.kl_divergence(target_dist, prior_dists).sum(dim=-1)).sum() / valid[1:].sum()
        states = torch.cat((stoch_states, gru_states), dim=-1)

        obs_reconstruction = self.decoder(states)
        # obs_loss = -obs_reconstruction.log_prob(observations)[:, :, -1].mean()
        obs_loss = (valid * ((obs_reconstruction - observations)[:, :, :] ** 2).mean(dim=-1)).sum() / valid.sum()
        # reward_pred = self.reward_model(states)
        # reward_pred = self.reward_model(obs_reconstruction.mean.detach())
        # reward_model_input = torch.cat((obs_reconstruction.detach(), actions), dim=-1)
        # reward_model_input = self.positional_encoding(pad(reward_model_input, self.transformer_dim))
        reward_model_input = self.positional_encoding(
            pad(torch.cat((observations, actions), dim=-1), self.transformer_dim))
        model_reward_pred = self.reward_model(reward_model_input, mask=self.mask[:T, :T])[:, :, 0]
        # reward_loss = -torch.mean(valid * reward_pred.log_prob(rewards))
        # reward_loss = -torch.mean(reward_pred.log_prob(rewards))
        reward_loss = (valid * ((model_reward_pred - rewards) ** 2)).sum() / valid.sum()
        termination_pred = self.termination_pred_model(states)
        termination_loss = -(valid * termination_pred.log_prob(done)).sum() / valid.sum()
        model_loss = 1 * kl_loss + 1 * obs_loss + reward_loss + termination_loss

        # stoch_state = stoch_states[:-1].reshape(-1, self.stoch_size)
        # gru_state = gru_states[:-1].reshape(-1, self.deter_size)
        stoch_state = stoch_states[:1].reshape(-1, self.stoch_size)
        gru_state = gru_states[:1].reshape(-1, self.deter_size)
        # samples_indexes = torch.randint(low=0, high=stoch_state.shape[0], size=(B,))
        # gru_state = gru_state[samples_indexes]
        # stoch_state = stoch_state[samples_indexes]
        with FreezeParameters(list(self.get_model_weights())):
            imagined_states, new_actions, action_stds = self.dream(stoch_state.detach(), gru_state.detach(),
                                                                   self.horizon)
            obs_pred = self.decoder(imagined_states)  # .mean
            # reward_model_input = self.positional_encoding(pad(
            # torch.cat((obs_reconstruction.mean[:-1].detach(), obs_pred), dim=0), 32))
            # torch.cat((obs_reconstruction[:-1].detach(), obs_pred), dim=0), 32))
            actions = torch.cat((actions[:1], new_actions), dim=0)
            obs = torch.cat((observations[:1], obs_pred), dim=0)
            reward_model_input = torch.cat((obs, actions), dim=-1)
            reward_model_input = self.positional_encoding(pad(reward_model_input, self.transformer_dim))
            reward_pred = self.reward_model(reward_model_input, mask=self.mask[:T, :T])[1:, :, 0]
            # reward_pred = self.reward_model(obs_pred)
            # reward_pred = self.reward_model(imagined_states)#.mean
            termination_pred = self.termination_pred_model(imagined_states).mean
        termination_pred = torch.cat((torch.zeros_like(termination_pred[0:1]), termination_pred), dim=0)[:-1]
        discount = torch.cumprod(1 - termination_pred, dim=0)
        policy_loss = -torch.mean(discount.detach() * reward_pred) + 10 * (
                    (actions.abs() - 0.9).clamp(min=0) ** 2).mean()
        # policy_loss = -torch.mean(discount.detach() * reward_pred) + 0.1 * (actions ** 2).mean()

        info = dict(
            reward_loss=reward_loss,
            # abs_action_value=torch.mean(torch.abs(self.action_model(torch.cat((stoch_state, gru_state), dim=-1)))),
            abs_action_value=torch.mean(torch.abs(new_actions)),
            action_dist_std=torch.mean(action_stds),
            termination_pred_loss=termination_loss,
            avg_termination_pred=termination_pred.mean(),
            kl_loss=kl_loss,
            reconstruction_loss=obs_loss,
            reward_obs_abs_error=torch.mean((observations[-1, :, -1] - obs_reconstruction[-1, :, -1]).abs()),
            mean_kl_div=mean_kl_div,
            mean_dreamed_reward=(discount * reward_pred).mean(),
        )
        print(f'mean valid {valid.mean()} reward loss {reward_loss}')
        return model_loss + 1 * policy_loss, info

    def dream(self, stoch_state, gru_state, horizon: int, actions=None) -> TensorType:
        """Given a batch of states, rolls out more state of length horizon."""
        imagined_states = []
        used_actions = torch.zeros((horizon, stoch_state.shape[0], self.action_size), device=self.device)
        action_stds = []
        for t in range(horizon):
            state = torch.cat((stoch_state, gru_state), dim=-1)
            if actions is None:
                action = torch.tanh(self.action_model(state.detach()))
                # action_dist = self.action_model(state.detach())
                # action = action_dist.rsample()
                # action_stds.append(action_dist.scale)
                action_stds.append(torch.tensor(0.0))
                used_actions[t] = action
            else:
                used_actions[t] = actions[t]

            action = used_actions[t]
            x = self.img1(torch.cat((stoch_state, action), dim=-1))
            gru_state = self.gru_cell(x, gru_state)
            # gru_state, _ = self.gru(x.unsqueeze(0), gru_state.unsqueeze(0))
            # gru_state = gru_state[0] # remove time dim
            stoch_state = self.prior_model(gru_state).rsample() # * 0
            imagined_states.append(torch.cat((stoch_state, gru_state), dim=-1))

        action_stds = torch.stack(action_stds) if len(action_stds) > 0 else None
        return torch.stack(imagined_states), used_actions, action_stds

    def get_initial_state(self, batch_size=1, sequence_length=None) -> List[TensorType]:
        state = [
            torch.zeros(batch_size, self.stoch_size, device=self.device),
            torch.zeros(batch_size, self.deter_size, device=self.device),
            torch.zeros(batch_size, self.action_size, device=self.device),
        ]
        return state

    def get_state_dims(self):
        return 2

    def value_function(self) -> TensorType:
        return None

    def get_model_weights(self):
        return list(self.prior_model.parameters()) \
             + list(self.obs_pred_model.parameters()) \
             + list(self.post_model.parameters()) \
             + list(self.decoder.parameters()) \
             + list(self.img1.parameters()) \
             + list(self.gru_cell.parameters()) \
             + list(self.positional_encoding.parameters()) \
             + list(self.gru.parameters()) \
             + list(self.termination_pred_model.parameters()) \
             + list(self.reward_model.parameters())

    def get_actor_weights(self):
        return list(self.action_model.parameters())

    def kl_loss(self, prior_dist, post_dist):
        post_dist_detached = td.Normal(post_dist.mean.detach(), post_dist.stddev.detach())
        prior_dist_detached = td.Normal(prior_dist.mean.detach(), prior_dist.stddev.detach())

        prior_kl = td.kl_divergence(post_dist_detached, prior_dist).mean(dim=-1)
        post_kl = td.kl_divergence(post_dist, prior_dist_detached).mean(dim=-1)
        # kl_loss = 1 * prior_kl + 0.2 * post_kl
        kl_loss = 1 * prior_kl + 0.5 * post_kl

        # div = torch.mean(torch.distributions.kl_divergence(target_dist, prior_dist).sum(dim=-1))
        # kl_loss = torch.clamp(div.mean(), min=1.5)
        return kl_loss

    def logs(self, obs, actions, rewards, valid):
        T, B = actions.shape[:2]
        stoch_state, gru_state, _ = self.get_initial_state(batch_size=B)
        states = []
        num_conditioning_obs = 1

        post_dist = self.post_model(torch.cat((obs[0], gru_state * 0), dim=-1))
        stoch_state = post_dist.rsample()
        # stoch_state = post_dist.mean
        states = torch.cat((stoch_state, gru_state), dim=-1).unsqueeze(0)
        imagined_states, new_actions, action_stds = self.dream(stoch_state, gru_state, self.horizon,
                                                               actions=actions[1:])
        states = torch.cat((states, imagined_states), dim=0)

        obs_pred = self.decoder(states)
        obs_input = torch.cat((obs[0:1], torch.zeros_like(obs[1:])), dim=0)
        obs_pred = self.obs_pred_model(torch.cat((obs_input, actions), dim=-1).permute(1, 0, 2).reshape(B, -1)).reshape(B, T, -1).permute(1, 0, 2)

        reward_model_input = torch.cat((obs_pred, actions[:1 + self.horizon]), dim=-1)
        reward_model_input = self.positional_encoding(pad(reward_model_input, self.transformer_dim))
        # reward_model_input = self.positional_encoding(
        #     pad(torch.cat((obs, actions), dim=-1), self.transformer_dim))
        reward_pred = self.reward_model(reward_model_input, mask=self.mask[:T, :T])[:, :, 0]

        info = dict(
            obs_reconstruction_abs_error=torch.mean((obs_pred[:1] - obs[:1]).abs()[:]),
            obs_prediction_abs_error=torch.mean(
                valid[1:, :, None] * (obs_pred[1:1 + self.horizon] - obs[1:1 + self.horizon]).abs()),
            reward_prediction_abs_error=torch.mean(
                valid[1:] * (reward_pred[1:1 + self.horizon] - rewards[1:1 + self.horizon]).abs()),
            reward_reconstruction_abs_error=torch.mean((reward_pred[:1] - rewards[:1]).abs()),
            correct_reward_pred_proportion=(valid[1:] * (
                        reward_pred[1:1 + self.horizon] - rewards[1:1 + self.horizon]).abs() < 0.1).float().mean(),
            feedback_pre_abs_error=torch.mean(
                valid[1:, :, None] * (obs_pred[1:1 + self.horizon] - obs[1:1 + self.horizon])[:, :, -4:-2].abs()),
            feedback_pred_correct=((obs_pred[1:1 + self.horizon] - obs[1:1 + self.horizon])[:, :, -4:-2].abs() < 0.05).float().mean(),
        )
        return info

    def logs_old(self, obs, actions, rewards):
        T, B = actions.shape[:2]
        stoch_state, gru_state, _ = self.get_initial_state(batch_size=B)
        states = []
        num_conditioning_obs = 1
        # for t in range(num_conditioning_obs):
        #     post_dist = self.post_model(obs[t])
        #     # stoch_state = post_dist.rsample()
        #     x = self.img1(torch.cat([stoch_state, actions[t]], dim=-1))
        #     _, gru_state = self.gru(x.unsqueeze(0), gru_state.unsqueeze(0))
        #     gru_state = gru_state[0]
        #     # gru_state = self.cell(x, gru_state)
        #     # prior_dist = self.prior_model(gru_state)
        #     # prior = prior_dist.rsample()
        #     # post_dist = self.post_model(torch.cat((gru_state, obs[t]), dim=-1))
        #     stoch_state = post_dist.rsample()
        #     states.append(torch.cat((stoch_state, gru_state), dim=-1))

        post_dists = self.post_model(obs[:1])
        stoch_states = post_dists.rsample()
        # gru_inputs = self.img1(torch.cat((stoch_states[:-1], actions[1:num_conditioning_obs]), dim=-1))
        gru_inputs = self.img1(torch.cat((stoch_states, actions[:1]), dim=-1))
        # if gru_inputs.shape[0] > 0:
        #     gru_states, gru_state = self.gru(gru_inputs, gru_state.unsqueeze(0))
        #     gru_state = gru_state[0]
        #     gru_states = torch.cat((gru_state.unsqueeze(0), gru_states), dim=0)
        # else:
        gru_states = gru_state.unsqueeze(0)
        states = torch.cat((stoch_states, gru_states), dim=-1)

        # gru_states = torch.cat((gru_state, gru_states), dim=0)[:-1]
        # states = torch.cat((post_samples, gru_states), dim=-1)

        # states = torch.stack(states)
        obs_reconstruction = self.decoder(states)  # .mean
        # imagined_states = self.dream(stoch_states[-1], gru_states[-1], horizon=self.horizon, actions=actions[num_conditioning_obs:])
        imagined_states, _, _ = self.dream(stoch_states[-1], gru_states[-1],
                                           horizon=self.horizon,
                                           actions=actions[1:])
        obs_prediction = self.decoder(imagined_states)  # .mean
        # reward_prediction = self.reward_model(imagined_states)[:, :, 0]#.mean
        # reward_reconstruction = self.reward_model(states)#.mean
        # reward_prediction = self.reward_model(obs_prediction)[:, :, 0]#.mean
        # reward_reconstruction = self.reward_model(obs_reconstruction)#.mean

        reward_model_input = torch.cat((obs_reconstruction, actions[:1]), dim=-1)
        reward_model_input = self.positional_encoding(pad(reward_model_input, self.transformer_dim))
        reward_reconstruction = self.reward_model(reward_model_input)[:, :, 0]

        # reward_model_input = self.positional_encoding(pad(torch.cat((obs_reconstruction, obs_prediction), dim=0), 32))
        obs_input = torch.cat((obs[:1], obs_prediction), dim=0)
        reward_model_input = torch.cat((obs_input, actions), dim=-1)
        reward_model_input = self.positional_encoding(pad(reward_model_input, self.transformer_dim))
        reward_prediction = self.reward_model(reward_model_input, mask=self.mask[:T, :T])[1:, :, 0]

        info = dict(
            obs_reconstruction_abs_error=torch.mean((obs_reconstruction - obs[:num_conditioning_obs]).abs()[:]),
            obs_prediction_abs_error=torch.mean(
                (obs_prediction - obs[num_conditioning_obs:num_conditioning_obs + self.horizon]).abs()),
            reward_prediction_abs_error=torch.mean(
                (reward_prediction - rewards[num_conditioning_obs:num_conditioning_obs + self.horizon]).abs()),
            reward_reconstruction_abs_error=torch.mean(
                (reward_reconstruction - rewards[:num_conditioning_obs]).abs()),
            correct_reward_pred_proportion=((reward_prediction -
                                             rewards[
                                             num_conditioning_obs:num_conditioning_obs + self.horizon]).abs() < 0.1).float().mean(),
        )
        return info
