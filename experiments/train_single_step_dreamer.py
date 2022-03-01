from ray import tune
from training import train
from learning_from_feedback.single_step_dreamer.dreamer import DREAMERTrainer
from learning_from_feedback.envs.sequence_guessing_env import SequenceGuessingEnv
from ray.rllib.models import ModelCatalog
from learning_from_feedback.single_step_dreamer.single_step_dreamer_model import SingleStepDreamer

tune.register_env('sequence_guessing', lambda kwargs: SequenceGuessingEnv(discrete_action_space=False,
                                                                          num_sequences=32,
                                                                          sequence_length=64,
                                                                          **kwargs))

ModelCatalog.register_custom_model('single_step_dreamer', SingleStepDreamer)

state_dreamer_config = dict(
    env='sequence_guessing',
    framework='torch',
    num_gpus=1,
    prefill_timesteps=4000,
    batch_size=512,
    buffer_size=100_000,
    imagine_horizon=1,
    explore_noise=0.3, # standard deviation of Gaussian noise added to actions
    grad_clip=100,
    evaluation_interval=1,
    evaluation_num_episodes=100,
    td_model_lr=1e-4,
    actor_lr=1e-6,
    evaluation_config=dict(
        # don't add random noise for evaluation
        explore=False
    ),
    # Change here error_obs to run experiment with or without feedback observation
    env_config=dict(random_seed=tune.randint(0, 1000), error_obs=True),
    batch_length=2,
    dreamer_train_iters=100,
    preprocessor_pref=None,
    dreamer_model=dict(
        custom_model='single_step_dreamer',
        # General Network Parameters
        hidden_size=4096,
    ),
)
train(DREAMERTrainer, state_dreamer_config,
      default_logdir='logs/sequence_guessing_without_feedback', # directory where to save training progress and checkpoints
      debug=False, # runs in single thread when debug=True
      max_iterations=1000,
      num_samples=10, # number of training runs with different random seeds
      default_checkpoint_freq=500, # save network weights every x iterations
)
