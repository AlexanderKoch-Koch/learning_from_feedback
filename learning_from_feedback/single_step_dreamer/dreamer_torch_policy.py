import logging
import gym
import ray
from ray.rllib.agents.dreamer.utils import FreezeParameters
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import apply_grad_clipping

torch, nn = try_import_torch()
if torch:
    from torch import distributions as td

logger = logging.getLogger(__name__)


# This is the computation graph for workers (inner adaptation steps)
def compute_dreamer_loss(obs,
                         action,
                         reward,
                         model):
    """Constructs loss for the Dreamer objective

        Args:
            obs (TensorType): Observations (o_t)
            action (TensorType): Actions (a_(t-1))
            reward (TensorType): Rewards (r_(t-1))
            model (TorchModelV2): DreamerModel, encompassing all other models
        """
    if len(obs.shape) == 2:
        obs = obs.unsqueeze(1).repeat(1, 2, 1)
        action = action.unsqueeze(1).repeat(1, 2, 1)
        reward = reward.unsqueeze(1).repeat(1, 2)
    model_loss, actor_loss, info = model.loss(obs, action, reward)

    return_dict = {
        "model_loss": model_loss,
        "actor_loss": actor_loss,
    }
    return_dict.update(info)

    return return_dict


def dreamer_loss(policy, model, dist_class, train_batch):
    policy.stats_dict = compute_dreamer_loss(
        train_batch["obs"],
        train_batch["actions"],
        train_batch["rewards"],
        policy.model,
    )
    loss_dict = policy.stats_dict

    return (loss_dict["model_loss"], loss_dict["actor_loss"])


def build_dreamer_model(policy, obs_space, action_space, config):
    policy.model = ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        1,
        config["dreamer_model"],
        name=config['dreamer_model']['custom_model'],
        framework="torch")
    policy.model_variables = policy.model.variables()
    return policy.model


def action_sampler_fn(policy, model, input_dict, state, explore, timestep):
    """Action sampler function has two phases. During the prefill phase,
    actions are sampled uniformly [-1, 1]. During training phase, actions
    are evaluated through DreamerPolicy and an additive gaussian is added
    to incentivize exploration.
    """
    obs = input_dict["obs"]

    # Custom Exploration
    logp = torch.tensor([0.0])
    if timestep <= policy.config["prefill_timesteps"]:
        # Random action in space [-1.0, 1.0]
        if isinstance(model.action_space, gym.spaces.Box):
            action = 2.0 * torch.rand(1, model.action_space.shape[0]) - 1.0
        else:
            action = td.Bernoulli(torch.zeros(model.action_space.shape[0]) + 0.5).sample().unsqueeze(0)
        state = model.get_initial_state()
    else:
        action = model.policy(obs, explore)
        if explore:
            if isinstance(model.action_space, gym.spaces.Box):
                action = td.Normal(action, policy.config["explore_noise"]).sample()
            else:
                action = td.Bernoulli(action * 0.8 + 0.1).sample()

        action = torch.clamp(action, min=-1.0, max=1.0)

    policy.global_timestep += policy.config["action_repeat"]

    return action, logp, state


def dreamer_stats(policy, train_batch):
    return policy.stats_dict


def dreamer_optimizer_fn(policy, config):
    model = policy.model
    actor_weights = list(model.action_model.parameters())
    model_weights = list(model.reward_model.parameters()) + list(model.pred_model.parameters())
    model_opt = torch.optim.Adam(
        model_weights,
        lr=config["td_model_lr"])  # , weight_decay=1e-4)
    actor_opt = torch.optim.Adam(actor_weights, lr=config["actor_lr"])
    # policy.actor_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(actor_opt, gamma=0.9999)

    return (model_opt, actor_opt)


def apply_grad_clipping(policy, optimizer, loss):
    """Applies gradient clipping to already computed grads inside `optimizer`.

    Args:
        policy (TorchPolicy): The TorchPolicy, which calculated `loss`.
        optimizer (torch.optim.Optimizer): A local torch optimizer object.
        loss (torch.Tensor): The torch loss tensor.
    """
    info = {}
    if policy.config['td_model_lr'] == optimizer.defaults['lr']:
        grad_clip = policy.config['pred_model_grad_clip']
        name = 'pred_model'
    elif policy.config['actor_lr'] == optimizer.defaults['lr']:
        grad_clip = policy.config['action_model_grad_clip']
        name = 'action_model'
    else:
        return dict()

    for param_group in optimizer.param_groups:
        # Make sure we only pass params with grad != None into torch
        # clip_grad_norm_. Would fail otherwise.
        params = list(
            filter(lambda p: p.grad is not None, param_group["params"]))
        if params:
            grad_gnorm = nn.utils.clip_grad_norm_(params, grad_clip)
            if isinstance(grad_gnorm, torch.Tensor):
                grad_gnorm = grad_gnorm.cpu().numpy()
            # info["grad_gnorm"] = float(grad_gnorm)
            info[name + "_grad_gnorm"] = float(grad_gnorm)
    return info


DreamerTorchPolicy = build_policy_class(
    name="DreamerTorchPolicy",
    framework="torch",
    get_default_config=lambda: ray.rllib.agents.dreamer.dreamer.DEFAULT_CONFIG,
    action_sampler_fn=action_sampler_fn,
    loss_fn=dreamer_loss,
    stats_fn=dreamer_stats,
    make_model=build_dreamer_model,
    optimizer_fn=dreamer_optimizer_fn,
    extra_grad_process_fn=apply_grad_clipping)
