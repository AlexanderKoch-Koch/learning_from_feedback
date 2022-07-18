from ray import tune
from learning_from_feedback.training import train
from learning_from_feedback.open_loop_dreamer.open_loop_dreamer import OpenLoopDreamerTrainer
from learning_from_feedback.vae_dreamer.vae_dreamer_model import VAEDreamerModel

config = dict(
    env='Pendulum-v0',
    framework='torch',
    num_workers=2,
    num_gpus=1,
    prefill_timesteps=5000,
    buffer_size=int(1e6),
    dreamer_train_iters=100,
    training_steps_per_env_step=0.2,
    batch_size=128,
    batch_length=32,
    explore_noise=0.3,
    td_model_lr=3e-4,
    actor_lr=3e-4,
    # min_iter_time_s=5,
    grad_clip=100,
    evaluation_interval=2,
    evaluation_num_episodes=4,
    evaluation_config=dict(
        explore=False
    ),
    dreamer_model=dict(
        custom_model=VAEDreamerModel,
        hidden_size=256,
        stoch_size=64,
        state_size=128,
        deter_size=512,
        discount=0.99,
        horizon=4,
        # memory_length=0,
    ),
)
train(OpenLoopDreamerTrainer,
      config,
      default_logdir='~/training_logs/learning_from_feedback/vae_dreamer',
      # directory where to save training progress and checkpoints
      debug=False,  # runs in single thread when debug=True
      max_iterations=10000,
      num_samples=1,  # number of training runs with different random seeds
      default_checkpoint_freq=50,  # save network weights every x iterations
      # restore="/home/alex/training_logs/learning_from_feedback/state_dreamer/TransformerDreamer_2022-05-01_09-25-18/TransformerDreamer_simple_feedback_reacher_da4a9_00000_0_2022-05-01_09-25-18/checkpoint_000050/checkpoint-50"
      )
