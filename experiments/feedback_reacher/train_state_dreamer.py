from ray import tune
from learning_from_feedback.training import train
from learning_from_feedback.open_loop_dreamer.open_loop_dreamer import OpenLoopDreamerTrainer
from learning_from_feedback.envs.reacher_2d import Reacher2D
from learning_from_feedback.envs.feedback_reacher import FeedbackReacher
from learning_from_feedback.open_loop_dreamer.state_dreamer_model import StateDreamerModel

tune.register_env('feedback_reacher', lambda kwargs: FeedbackReacher(action_repeat=8))


config = dict(
    env='feedback_reacher',
    framework='torch',
    num_workers=2,
    num_gpus=1,
    prefill_timesteps=2000,
    dreamer_train_iters=1000,
    training_steps_per_env_step=0.2,
    batch_size=64,
    batch_length=14,
    explore_noise=0.3,
    td_model_lr=3e-4,
    min_iter_time_s=5,
    grad_clip=100,
    evaluation_interval=1,
    evaluation_num_episodes=10,
    evaluation_config=dict(
        explore=False
    ),
    dreamer_model=dict(
        custom_model=StateDreamerModel,
        hidden_size=512,
        stoch_size=32,
        deter_size=256,
        discount=0.99,
        horizon=14,
),
)
train(OpenLoopDreamerTrainer,
      config,
      default_logdir='logs/state_dreamer', # directory where to save training progress and checkpoints
      debug=False, # runs in single thread when debug=True
      max_iterations=10000,
      num_samples=1, # number of training runs with different random seeds
      default_checkpoint_freq=10, # save network weights every x iterations
)
