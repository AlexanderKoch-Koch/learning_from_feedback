from ray import tune
from learning_from_feedback.training import train
from learning_from_feedback.open_loop_dreamer.open_loop_dreamer import OpenLoopDreamerTrainer
from learning_from_feedback.envs.simple_feedback_reacher import SimpleFeedbackReacher
from learning_from_feedback.open_loop_dreamer.state_dreamer_model import StateDreamerModel

tune.register_env('simple_feedback_reacher',
                  lambda kwargs: SimpleFeedbackReacher(num_objects=32,
                                                       visible_objects=16))


config = dict(
    env='simple_feedback_reacher',
    framework='torch',
    num_workers=2,
    num_gpus=1,
    prefill_timesteps=2000,
    dreamer_train_iters=100,
    training_steps_per_env_step=1,
    batch_size=128,
    batch_length=2,
    explore_noise=0.3,
    td_model_lr=3e-4,
    min_iter_time_s=5,
    grad_clip=1000,
    evaluation_interval=1,
    evaluation_num_episodes=100,
    evaluation_config=dict(
        explore=False
    ),
    dreamer_model=dict(
        custom_model=StateDreamerModel,
        hidden_size=512,
        stoch_size=64,
        deter_size=512,
        discount=0.99,
        horizon=1,
),
)
train(OpenLoopDreamerTrainer,
      config,
      default_logdir='logs/state_dreamer', # directory where to save training progress and checkpoints
      debug=False, # runs in single thread when debug=True
      max_iterations=10000,
      num_samples=1, # number of training runs with different random seeds
      default_checkpoint_freq=50, # save network weights every x iterations
      # restore="/home/alex/learning_from_feedback/experiments/simple_feedback_reacher/logs/state_dreamer/TransformerDreamer_2022-04-16_20-12-17/TransformerDreamer_simple_feedback_reacher_c035e_00000_0_2022-04-16_20-12-17/checkpoint_000400/checkpoint-400"
)
