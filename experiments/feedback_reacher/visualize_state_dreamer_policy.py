import time
import argparse
from ray import tune
from learning_from_feedback.open_loop_dreamer.open_loop_dreamer import OpenLoopDreamerTrainer
import gym
from learning_from_feedback.envs.feedback_reacher import FeedbackReacher
from learning_from_feedback.open_loop_dreamer.state_dreamer_model import StateDreamerModel

tune.register_env('feedback_reacher', lambda kwargs: FeedbackReacher(action_repeat=8))

config = dict(
    env='feedback_reacher',
    framework='torch',
    num_gpus=1,
    prefill_timesteps=2000,
    batch_size=64,
    batch_length=32,
    explore_noise=0.3,
    td_model_lr=3e-4,
    min_iter_time_s=5,
    grad_clip=100,
    evaluation_interval=1,
    evaluation_num_episodes=4,
    evaluation_config=dict(
        explore=False
    ),
    dreamer_model=dict(
        custom_model=StateDreamerModel,
        hidden_size=512,
        stoch_size=32,
        deter_size=256,
        discount=0.99,
        horizon=16,
    ),
)

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint_path', help='log dir to restore')
parser.add_argument('--headless', action='store_true', help='log dir to restore')
args = parser.parse_args()

trainer = OpenLoopDreamerTrainer(config)
trainer.load_checkpoint(args.checkpoint_path)
env = trainer.workers.local_worker().env
returns_ = []
while True:
    done = False
    state = []
    obs = env.reset()
    return_ = 0
    while not done:
        s = time.time()
        # action = trainer.compute_action(obs, explore=False)
        # action = env.action_space.sample() * 0
        action, state, _ = trainer.compute_single_action(obs, state, explore=False)
        # print(f'action {action}  took {time.time() - s}')
        s = time.time()
        obs, reward, done, info = env.step(action)
        # if not args.headless:
        #     env.render()
        # print(f'step took {time.time() - s}')
        return_ += reward
        env.render()
        time.sleep(env.action_repeat * 0.01)

    returns_.append(return_)
    print(f'avg return: {sum(returns_)/len(returns_)}return_: {return_}')

