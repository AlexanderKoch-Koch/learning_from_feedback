
from ray import tune
from learning_from_feedback.training import train
from learning_from_feedback.single_step_dreamer.dreamer import DREAMERTrainer
from learning_from_feedback.envs.sequence_guessing_env import SequenceGuessingEnv
from ray.rllib.models import ModelCatalog
from learning_from_feedback.single_step_dreamer.single_step_dreamer_model import SingleStepDreamer
from learning_from_feedback.open_loop_dreamer.open_loop_dreamer import OpenLoopDreamerTrainer
from learning_from_feedback.envs.simple_feedback_reacher import SimpleFeedbackReacher
from learning_from_feedback.open_loop_dreamer.state_dreamer_model import StateDreamerModel
from learning_from_feedback.open_loop_dreamer.single_step_dreamer import SingleStepDreamerModel

tune.register_env('simple_feedback_reacher', lambda kwargs: SimpleFeedbackReacher(num_objects=32,
                                                                                  visible_objects=4,
                                                                                  object_code_length=4,
                                                                                  task_instruction_size=4))


ModelCatalog.register_custom_model('single_step_dreamer', SingleStepDreamer)

state_dreamer_config = dict(
    env='simple_feedback_reacher',
    framework='torch',
    num_workers=16,
    num_gpus=1,
    prefill_timesteps=2000,
    batch_size=512,
    buffer_size=100000,
    imagine_horizon=1,
    explore_noise=0.3, # standard deviation of Gaussian noise added to actions
    grad_clip=100,
    evaluation_interval=4,
    evaluation_num_episodes=100,
    training_steps_per_env_step=.1,
    td_model_lr=3e-4,
    actor_lr=3e-4,
    evaluation_config=dict(
        # don't add random noise for evaluation
        explore=False
    ),
    # Change here error_obs to run experiment with or without feedback observation
    # env_config=dict(random_seed=tune.randint(0, 1000), error_obs=True),
    batch_length=2,
    dreamer_train_iters=100,
    preprocessor_pref=None,
    dreamer_model=dict(
        custom_model=SingleStepDreamerModel,
        # General Network Parameters
        hidden_size=512,
        # hidden_size=1024,
    ),
)
train(OpenLoopDreamerTrainer, state_dreamer_config,
      default_logdir='~/training_logs/learning_from_feedback/single_step_dreamer', # directory where to save training progress and checkpoints
      debug=False, # runs in single thread when debug=True
      max_iterations=100000,
      num_samples=1, # number of training runs with different random seeds
      default_checkpoint_freq=5, # save network weights every x iterations
      # restore='/home/alex/training_logs/learning_from_feedback/single_step_dreamer/TransformerDreamer_2022-04-24_19-50-05/TransformerDreamer_simple_feedback_reacher_f9699_00000_0_2022-04-24_19-50-05/checkpoint_000070/checkpoint-70'
)
