import time
import argparse
from ray import tune
from learning_from_feedback.open_loop_dreamer.open_loop_dreamer import OpenLoopDreamerTrainer
import gym
from learning_from_feedback.envs.reacher_2d import Reacher2D

tune.register_env('reacher2d', lambda kwargs: Reacher2D())

config = dict(
    env='reacher2d',
    framework='torch',
    horizon=50,
    num_gpus=1,
    prefill_timesteps=5000,
    batch_size=512,
    batch_length=16 + 8,
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
        print(f'action {action}  took {time.time() - s}')
        s = time.time()
        obs, reward, done, info = env.step(action)
        # if not args.headless:
        #     env.render()
        # print(f'step took {time.time() - s}')
        return_ += reward
        env.render()

    returns_.append(return_)
    print(f'avg return: {sum(returns_)/len(returns_)}return_: {return_}')

