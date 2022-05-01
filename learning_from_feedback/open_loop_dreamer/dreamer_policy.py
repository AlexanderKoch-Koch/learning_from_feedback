"""
based on https://github.com/ray-project/ray/blob/master/rllib/agents/dreamer/dreamer_torch_policy.py

"""

import logging
import gym.spaces
import learning_from_feedback
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.framework import try_import_torch
from feedback_learning.transformer_dreamer.torch_policy import apply_grad_clipping, dreamer_stats, preprocess_episode
from ray.rllib.utils.timer import TimerStat

torch, nn = try_import_torch()
logger = logging.getLogger(__name__)


def compute_dreamer_loss(policy,
                         obs,
                         action,
                         reward,
                         valid,
                         done,
                         model,
                         log=False):
    """Constructs loss for the Dreamer objective
        all tensor args in shape T x B x ...

        Args:
            obs (TensorType): Observations (o_t)
            action (TensorType): Actions (a_(t-1))
            reward (TensorType): Rewards (r_(t-1))
            model (TorchModelV2): DreamerModel, encompassing all other models
            log (bool): If log, log absolute weight mean
        """
    # model.train()
    # if not hasattr(policy, 'jit_loss_function'):
    #     policy.jit_loss_function = torch.jit.trace(model.forward, (obs, action, reward, valid, done))#, strict=False)
    with policy.model_observe_timer:
        loss, info_dict = model.loss(obs, action, reward, valid, done)
        # loss, info_dict = policy.jit_loss_function(obs, action, reward, valid, done)
        # loss = policy.jit_loss_function(obs, action, reward, valid, done)
        # info_dict = dict()
    if log:
        weight_magnitudes = torch.stack([p.abs().mean() for p in model.parameters()])
        info_dict['mean_abs_weight'] = weight_magnitudes.mean()
        info_dict['max_abs_weight'] = weight_magnitudes.max()
        if hasattr(model, "logs"):
            info_dict.update(model.logs(obs, action, reward, valid))
    return loss, info_dict


def dreamer_loss(policy, model, dist_class, train_batch):
    if isinstance(policy.action_space, gym.spaces.Discrete):
        train_batch['actions'] = train_batch['actions'].unsqueeze(-1).float()

    log_gif = "log_gif" in train_batch

    if len(train_batch['obs'].shape) == 2:
        # RLLib tests the loss function automatically before training with fake data
        # But the batch dimensions are invalid. This is fixed here
        # train_batch['obs'] = train_batch['obs'].unsqueeze(1).repeat(1, 32, 1)
        # train_batch["actions"] = train_batch["actions"].unsqueeze(1).repeat(1, 32, 1)
        # train_batch["rewards"] = train_batch["rewards"].unsqueeze(1).repeat(1, 32)
        # train_batch["valid"] = train_batch["valid"].unsqueeze(1).repeat(1, 32)
        # train_batch["dones"] = train_batch["dones"].unsqueeze(1).repeat(1, 32)
        T, B = policy.config['batch_length'], policy.config['batch_size']
        device = train_batch['obs'].device

        train_batch['obs'] = torch.zeros(B, T, train_batch['obs'].shape[-1], device=device)
        train_batch["actions"] = torch.zeros(B, T, train_batch['actions'].shape[-1], device=device)
        train_batch["rewards"] = torch.zeros(B, T, device=device)
        train_batch["valid"] = torch.zeros(B, T, device=device)
        train_batch["dones"] = torch.zeros(B, T, device=device)

    loss, policy.stats_dict = compute_dreamer_loss(
        policy,
        train_batch["obs"].transpose(0, 1),  # batch in shape T x B x .. as input
        train_batch["actions"].transpose(0, 1),
        train_batch["rewards"].transpose(0, 1),
        train_batch["valid"].transpose(0, 1),
        train_batch["dones"].transpose(0, 1),
        policy.model,
        log_gif,
    )

    policy.stats_dict['model_observe_ms'] = 1000 * policy.model_observe_timer.mean
    return loss


def build_dreamer_model(policy, obs_space, action_space, config):
    policy.steps_sampled = 0
    policy.model_observe_timer = TimerStat()

    policy.model = config['dreamer_model']['custom_model'](
        obs_space,
        action_space,
        1,
        config["dreamer_model"],
        name="ParallelDreamerModel")

    policy.model_variables = policy.model.variables()
    return policy.model


def dreamer_optimizer_fn(policy, config):
    # optim = torch.optim.Adam([
    #     dict(params=policy.model.parameters()),
    #     # dict(params=policy.model.get_model_weights()),
    #     # dict(params=policy.model.action_model.parameters(), lr=5e-5)
    # ], lr=policy.config['td_model_lr'])

    optim = torch.optim.Adam([
        dict(params=policy.model.get_actor_weights(), lr=policy.config['actor_lr']),
        dict(params=policy.model.get_model_weights(), lr=policy.config['td_model_lr']),
    ])
    return optim


def action_sampler_fn(policy, model, input_dict, state, explore, timestep):
    """Action sampler function has two phases. During the prefill phase,
    actions are sampled uniformly [-1, 1]. During training phase, actions
    are evaluated through DreamerPolicy and an additive gaussian is added
    to incentivize exploration.
    """
    obs = input_dict["obs"]
    logp = torch.tensor([0.0]) # necessary for RLLib for unknown reason

    # Weird RLLib Handling, this happens when env rests
    if len(state) == 0 or len(state[0].shape) != model.get_state_dims():
        # print('ressetting state')
        # Very hacky, but works on all envs
        state = model.get_initial_state(batch_size=obs.shape[0], sequence_length=policy.config['batch_length'])

    action, state, _ = model.policy(obs, state, explore)

    policy.global_timestep += policy.config["action_repeat"]
    policy.steps_sampled += policy.config['action_repeat']
    # print(f'steps sampled: {policy.steps_sampled}')

    return action, logp, state


OpenLoopDreamerTorchPolicy = build_torch_policy(
    name="OpenLoopDreamerTorchPolicy",
    get_default_config=lambda: learning_from_feedback.open_loop_dreamer.open_loop_dreamer.DEFAULT_CONFIG,
    action_sampler_fn=action_sampler_fn,
    postprocess_fn=preprocess_episode,
    loss_fn=dreamer_loss,
    stats_fn=dreamer_stats,
    make_model=build_dreamer_model,
    optimizer_fn=dreamer_optimizer_fn,
    extra_grad_process_fn=apply_grad_clipping)
