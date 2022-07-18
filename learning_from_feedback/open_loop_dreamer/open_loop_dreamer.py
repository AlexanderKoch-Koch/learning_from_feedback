"""
code based on https://github.com/ray-project/ray/blob/master/rllib/agents/dreamer/dreamer.py
Added support for parallel sampling and learning
"""


import ray
import collections
from ray.rllib.agents import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.execution.common import STEPS_SAMPLED_COUNTER, _get_shared_metrics
from learning_from_feedback.open_loop_dreamer.model import OpenLoopDreamerModel
from learning_from_feedback.open_loop_dreamer.dreamer_policy import OpenLoopDreamerTorchPolicy
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from learning_from_feedback.open_loop_dreamer.learner_thread import LearnerThread
from ray.rllib.execution.concurrency_ops import Concurrently, Enqueue, Dequeue
from learning_from_feedback.open_loop_dreamer.episode_buffer import EpisodeBuffer
from ray.rllib.execution.common import STEPS_TRAINED_COUNTER, \
    _get_global_vars, _get_shared_metrics

DEFAULT_CONFIG = with_common_config({
    "framework": "torch",
    # "parallel_dreamer": False,
    "policy_class": OpenLoopDreamerTorchPolicy,
    # PlaNET Model LR
    "td_model_lr": 6e-4,
    # Actor LR
    "actor_lr": 8e-5,
    # Critic LR
    "critic_lr": 8e-5,
    # Grad Clipping
    "grad_clip": 100.0,
    # Lambda
    "lambda": 0.95,
    # Training iterations per data collection from real env
    "dreamer_train_iters": 100,
    "training_steps_per_env_step": 0.2,
    "buffer_size": 1000, # maximum number of recent episodes saved for sampling
    # Horizon for Enviornment (1000 for Mujoco/DMC)
    "horizon": 200,
    # Number of episodes to sample for Loss Calculation
    "batch_size": 50,
    # Length of each episode to sample for Loss Calculation
    "batch_length": 50,
    # Imagination Horizon for Training Actor and Critic
    "imagine_horizon": 15,
    # Free Nats
    "free_nats": 3.0,
    # KL Coeff for the Model Loss
    "kl_coeff": 1.0,
    # Distributed Dreamer not implemented yet
    "num_workers": 0,

    "rollout_fragment_length": 1,  # only one episode at once
    # Batch mode
    "batch_mode": "complete_episodes",
    'min_iter_time_s': 20,

    # Prefill Timesteps
    "prefill_timesteps": 5000,
    "prefill_episodes": 5,
    "clip_actions": True,

    # max queue size for train batches feeding into the learner
    "learner_queue_size": 16,
    "broadcast_interval": 1,
    "metrics_smoothing_episodes": 10,

    # This should be kept at 1 to preserve sample efficiency
    "num_envs_per_worker": 1,
    # Exploration Gaussian
    "explore_noise": 0.3,
    # Custom Model
    "dreamer_model": {
        "custom_model": OpenLoopDreamerModel,
        "memory_length": None,
        "horizon": 8,
        # General Network Parameters
        "state_size": 256,
        "stoch_size": 32,
        "deter_size": 256,
        "hidden_size": 256,
        # Discount
        "discount": 0.99,
    },

    # "env_config": {
    # Repeats action send by policy for frame_skip times in env
    # "frame_skip": 2,
    # },
    "evaluation_config": dict(explore=False)
})


class BroadcastUpdateLearnerWeights:
    def __init__(self, learner_thread, workers, broadcast_interval):
        self.learner_thread = learner_thread
        self.workers = workers
        self.steps_since_update = collections.defaultdict(int)
        self.max_weight_sync_delay = broadcast_interval
        self.weights = None

    def __call__(self, item: ("ActorHandle", SampleBatchType)):
        actor, batch = item
        self.weights = ray.put(self.workers.local_worker().get_weights())
        actor.set_weights.remote(self.weights, _get_global_vars())
        # Update metrics.
        metrics = _get_shared_metrics()
        metrics.counters["num_weight_syncs"] += 1


class LearnerStepsReceiver:
    def __init__(self, env_action_repeat, episode_buffer):
        self.env_action_repeat = env_action_repeat
        self.episode_buffer = episode_buffer

    def record_steps_trained(self, item):
        num_steps_sampled, num_steps_trained, fetches = item
        metrics = _get_shared_metrics()
        metrics.counters[STEPS_TRAINED_COUNTER] = num_steps_trained
        metrics.counters[STEPS_SAMPLED_COUNTER] = self.episode_buffer.timesteps * self.env_action_repeat
        print(f'updated steps sampled counter to {metrics.counters[STEPS_SAMPLED_COUNTER]}')
        return item


def execution_plan(workers, config):
    """
    Sampling (real environment interactions) and Learning (updating neural network weights) are done in parallel in
    different processes if num_workers > 0. At every iteration, the new environment samples are saved in the episode buffer.
    This buffer then sends a randomly sampled batch to the learner via a queue.
    After dreamer_train_iters weight updates in the learner, it sends logging info to the learner_stats_receiver.
    :param workers:
    :param config:
    :return:
    """
    train_batches = ParallelRollouts(workers, mode="async", num_async=1)

    # Start the learner thread.
    learner_thread = LearnerThread(
        workers.local_worker(),
        learner_queue_size=config["learner_queue_size"],
        config=config)
    learner_thread.start()

    episode_buffer = EpisodeBuffer(learner_thread.inqueue,
                                   config['batch_size'],
                                   length=config['batch_length'],
                                   memory_length=config['dreamer_model']['memory_length'],
                                   prefill_timesteps=config['prefill_timesteps'],
                                   max_length=config['buffer_size'],
                                   training_steps_per_env_steps=config['training_steps_per_env_step'])
    # This sub-flow sends experiences to the learner.
    enqueue_op = train_batches.for_each(episode_buffer)

    # Only need to update workers if there are remote workers.
    if workers.remote_workers():
        enqueue_op = enqueue_op.zip_with_source_actor() \
            .for_each(BroadcastUpdateLearnerWeights(learner_thread, workers,
                                                    broadcast_interval=config["broadcast_interval"]))

    # This sub-flow updates the steps trained counter based on learner output.
    learner_stats_receiver = LearnerStepsReceiver(episode_buffer=episode_buffer,
                                                  env_action_repeat=config['action_repeat'])

    dequeue_op = Dequeue(learner_thread.outqueue, check=learner_thread.is_alive) \
        .for_each(learner_stats_receiver.record_steps_trained)

    merged_op = Concurrently(
        [enqueue_op, dequeue_op], output_indexes=[1], mode='async')

    return StandardMetricsReporting(merged_op, workers, config) \
        .for_each(learner_thread.add_learner_metrics)


def get_policy_class(config):
    return config['policy_class']


def validate_config(config):
    if 'frame_skip' in config['env_config'].keys():
        config["action_repeat"] = config["env_config"]["frame_skip"]
    else:
        config['action_repeat'] = 1
    if config["framework"] != "torch":
        raise ValueError("Dreamer not supported in Tensorflow yet!")
    if config["batch_mode"] != "complete_episodes":
        raise ValueError("truncate_episodes not supported")
    if config["action_repeat"] > 1:
        config["horizon"] = config["horizon"] / config["action_repeat"]


OpenLoopDreamerTrainer = build_trainer(
    name="TransformerDreamer",
    default_config=DEFAULT_CONFIG,
    default_policy=OpenLoopDreamerTorchPolicy,
    get_policy_class=get_policy_class,
    execution_plan=execution_plan,
    validate_config=validate_config)
