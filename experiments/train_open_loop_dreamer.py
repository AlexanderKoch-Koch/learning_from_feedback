from training import train
from learning_from_feedback.open_loop_dreamer.open_loop_dreamer import OpenLoopDreamerTrainer

config = dict(
    env='Pendulum-v0',
    framework='torch',
    horizon=200,
    num_gpus=1,
    prefill_timesteps=5000,
    batch_size=512,
    batch_length=50 + 8,
    explore_noise=0.3,
    td_model_lr=1e-4,
    min_iter_time_s=5,
    grad_clip=4,
    evaluation_interval=1,
    evaluation_num_episodes=4,
    evaluation_config=dict(
        explore=False
    ),
    dreamer_model=dict(
        state_size=64,
        hidden_size=256,
        memory_length=8,
    ),
)
train(OpenLoopDreamerTrainer,
      config,
      default_logdir='logs/open_loop_dreamer', # directory where to save training progress and checkpoints
      debug=False, # runs in single thread when debug=True
      max_iterations=100,
      num_samples=10, # number of training runs with different random seeds
      default_checkpoint_freq=100, # save network weights every x iterations
)
