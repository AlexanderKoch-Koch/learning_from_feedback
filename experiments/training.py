from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune import Callback
import subprocess

"""
file contains generic train function that uses ray tune
"""


class EnvInfoCallback(DefaultCallbacks):
    """
    RLLib callback that saves the env info from the last episode step as a custom metric
    """

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        for key in episode._agent_to_last_info['agent0'].keys():
            episode.custom_metrics[key] = episode._agent_to_last_info['agent0'][key]


class TuneGitLogCallback(Callback):
    """
    Tune callback that creates a git_status.json file in the trial directory which contains the last git commit and
    git diff at the time of the trial start.
    """

    def on_trial_start(self, iteration, trials, trial, **info):
        with open(trial.logdir + '/git_status.json', "a") as f:
            f.write('\n\nlast commit: ' + subprocess.getoutput('git log --max-count=1') + '\n\n')
            f.write(subprocess.getoutput('git diff'))


def train(trainer, config, default_logdir='./logs', debug=False, default_checkpoint_freq=50, restore=None,
          max_iterations=None, **kwargs):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', help='log dir to restore', default=restore)
    parser.add_argument('--logdir', help='directory to save logs', default=default_logdir)
    parser.add_argument('--checkpoint_freq', help='directory to save logs', default=default_checkpoint_freq)
    args = parser.parse_args()

    config['callbacks'] = EnvInfoCallback

    if debug:
        config['num_workers'] = 0
        trainer = trainer(config)
        if args.restore is not None:
            trainer.restore(args.restore)
        while True:
            print(trainer.step())
    else:
        stopper = None if max_iterations is None else tune.stopper.MaximumIterationStopper(max_iterations)
        tune.run(trainer, config=config, local_dir=args.logdir, checkpoint_freq=args.checkpoint_freq,
                 callbacks=[TuneGitLogCallback()], restore=args.restore, stop=stopper, **kwargs)
