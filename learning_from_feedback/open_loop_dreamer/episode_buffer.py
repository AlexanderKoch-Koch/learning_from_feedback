"""
code based on https://github.com/ray-project/ray/blob/master/rllib/agents/dreamer/dreamer.py
Added queues to send batches to the learner process
"""

import numpy as np
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.policy.sample_batch import SampleBatch
import time


class EpisodeBuffer(object):
    def __init__(self, learner_inqueue,
                 batch_size,
                 prefill_timesteps=2000,
                 training_steps_per_env_steps=0.2,
                 max_length: int = 1000,
                 length: int = 50,
                 memory_length=None,
                 max_episode_length=1000):
        """Data structure that stores episodes and samples chunks
        of size length from episodes

        Args:
            max_length: Maximum episodes it can store
            length: Episode chunking lengh in sample()
        """
        # Stores all episodes into a list: List[SampleBatchType]
        self.episodes = []
        self.max_length = max_length
        self.max_episode_length = max_episode_length
        self.timesteps = 0
        self.length = length
        self.learner_inqueue = learner_inqueue
        self.batch_size = batch_size
        self.prefill_timesteps = prefill_timesteps
        self.training_steps_per_env_steps = training_steps_per_env_steps
        self.memory_length = memory_length or 1

    def __call__(self, samples):
        """
        Saves new samples to buffer and sends a random sample from buffer to learner
        :param samples:
        :return: samples
        """
        num_new_episodes = self.add(samples)
        print(f'sampled {len(self.episodes)} episodes and {samples.count} steps; num new {num_new_episodes}')
        if self.timesteps >= self.prefill_timesteps:
            while not self.learner_inqueue.qsize() < 10:  # 1 * self.train_iters:
                time.sleep(0.1)
            print(f'training for {int(samples.count * self.training_steps_per_env_steps)}')
            for _ in range(int(samples.count * self.training_steps_per_env_steps)):
                self.learner_inqueue.put(self.sample(self.batch_size))

        return samples

    def add(self, batch: SampleBatchType):
        """Splits a SampleBatch into episodes and adds episodes
        to the episode buffer

        Args:
            batch: SampleBatch to be added
        """
        self.timesteps += batch.count
        episodes = batch.split_by_episode()

        self.episodes.extend(episodes)

        if len(self.episodes) > self.max_length:
            delta = len(self.episodes) - self.max_length
            # Drop oldest episodes
            self.episodes = self.episodes[delta:]

        return len(episodes)

    def preprocess_episode(self, episode: SampleBatchType):
        """Batch format should be in the form of (s_t, a_(t-1), r_(t-1))
        When t=0, the resetted obs is paired with action and reward of 0.

        Args:
            episode: SampleBatch representing an episode
        """
        obs = episode["obs"]
        new_obs = episode["new_obs"]
        action = episode["actions"]
        reward = episode["rewards"]

        act_shape = action.shape
        act_reset = np.array([0.0] * act_shape[-1])[None]
        rew_reset = np.array(0.0)[None]
        obs_end = np.array(new_obs[act_shape[0] - 1])[None]

        batch_obs = np.concatenate([obs, obs_end], axis=0)
        batch_action = np.concatenate([act_reset, action], axis=0)
        batch_rew = np.concatenate([rew_reset, reward], axis=0)
        batch_obs = np.concatenate((np.zeros((self.length - 1, *batch_obs.shape[1:])), batch_obs), axis=0)
        batch_action = np.concatenate((np.zeros((self.length - 1, *batch_action.shape[1:])), batch_action), axis=0)
        batch_rew = np.concatenate((np.zeros((self.length - 1, *batch_rew.shape[1:])), batch_rew), axis=0)
        valid = np.ones_like(batch_rew)
        done = np.zeros_like(batch_rew)
        # valid[:self.length - 1] = 0
        done[-1] = 1

        new_batch = {
            "obs": batch_obs,
            "rewards": batch_rew,
            "actions": batch_action,
            'valid': valid,
            'done': done
        }
        return SampleBatch(new_batch)

    def sample(self, batch_size: int):
        """Samples [batch_size, length] from the list of episodes

        Args:
            batch_size: batch_size to be sampled
        """

        episodes_buffer = []
        while len(episodes_buffer) < batch_size:
            rand_index = np.random.randint(0, len(self.episodes))
            episode = self.episodes[rand_index]
            # if episode.count < self.length:
            #     continue
            available = episode.count - self.length
            assert available >= 0, f"episode length is just {episode.count} but training batch length is set to {self.length}"
            index = np.random.randint(-self.memory_length + 1, available + 1)
            # index = np.random.randint(0, available + 1)
            valid_steps = episode[max(0, index):index + self.length]
            batch_obs = np.concatenate((np.zeros((max(0, -index), *valid_steps['obs'].shape[1:])), valid_steps['obs']),
                                       axis=0)
            batch_action = np.concatenate(
                (np.zeros((max(0, -index), *valid_steps['actions'].shape[1:])), valid_steps['actions']), axis=0)
            batch_rew = np.concatenate(
                (np.zeros((max(0, -index), *valid_steps['rewards'].shape[1:])), valid_steps['rewards']), axis=0)
            valid = np.ones_like(batch_rew)
            done = np.zeros_like(batch_rew)
            if index < 0:
                valid[:-index] = 0
            done[-1] = (episode.count == index + self.length)
            episode = SampleBatch({
                SampleBatch.OBS: batch_obs,
                SampleBatch.REWARDS: batch_rew,
                SampleBatch.ACTIONS: batch_action,
                SampleBatch.DONES: done,
                'valid': valid
            })
            episodes_buffer.append(episode)

        batch = {}
        for k in episodes_buffer[0].keys():
            batch[k] = np.stack([e[k] for e in episodes_buffer], axis=0)
        return SampleBatch(batch)
