import gym
import os
import time
import numpy as np

dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.npy')


class SequenceGuessingEnv(gym.Env):

    def __init__(self, num_sequences=32, sequence_length=16, log_info=False, error_obs=True,
                 discrete_action_space=False, one_hot_id=True, random_seed=0):
        print(f'starting with random seed {random_seed}')
        self.error_obs = error_obs
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.log_info = log_info
        rng = np.random.default_rng(random_seed)
        self.sequences = rng.choice([0, 1], (num_sequences, sequence_length))
        self.sequences_lengths = rng.integers(1, sequence_length, (num_sequences,))
        self.sequences_start = rng.integers(0, sequence_length - self.sequences_lengths, (num_sequences,))

        if self.log_info:
            print(self.sequences)
        if discrete_action_space:
            self.action_space = gym.spaces.MultiDiscrete([2 for _ in range(self.sequence_length)])
        else:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.sequence_length,))
        # self.action_space = gym.spaces.MultiDiscrete([2 for _ in range(self.sequence_length)])
        if one_hot_id:
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_sequences + self.sequence_length,))
        else:
            self.id_len = np.ceil(np.log2(self.num_sequences)).astype(np.int)
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.id_len + self.sequence_length,))

        self.one_hot_id = one_hot_id

    def reset(self):
        self.id = int(np.random.choice(self.num_sequences, 1))
        if self.one_hot_id:
            self.id_obs = np.zeros(self.num_sequences)
            self.id_obs[self.id] = 1
        else:
            self.id_obs = np.array(list(np.binary_repr(self.id).zfill(self.id_len))).astype(np.int8)

        self.obs_step = 0
        return np.concatenate((self.id_obs, np.zeros(self.sequence_length)))

    def step(self, action: np.array):
        self.obs_step += 1
        # action = np.random.normal(action, 0.01)
        binary_action = (action > 0)# - 0.5
        # binary_action = np.random.binomial(1, (action + 1) / 2)
        # start = self.sequence_length - self.sequences_lengths[self.id]
        # end = None
        start = self.sequences_start[self.id]
        end = start + self.sequences_lengths[self.id]
        binary_error = np.abs(binary_action[start:end] - self.sequences[int(self.id)][start:end])
        # error = np.abs(action[start:end] - self.sequences[int(self.id)][start:end]) / 2

        if self.obs_step == 1:
            reward = 1 if np.abs(binary_error).sum() < 0.01 else 0
            self.correct_proportion = 1 - binary_error.mean()
            # reward -= (action ** 4).mean()
        else:
            reward = 0
        # reward -= (action ** 2).mean()
        # reward = -error.mean() if self.obs_step == 1 else 0
        # reward = -binary_error.mean() if self.obs_step == 1 else 0
        if self.obs_step == 1 and self.log_info:
            if binary_error.sum() < 0.01:
                print(f'sequence {self.id} SUCCESS length {self.sequences_lengths[self.id]} action {action[start:end]}'
                      f' target {self.sequences[int(self.id)][start:end]} reward {reward}')
            else:
                print(f'sequence {self.id} FAIL length {self.sequences_lengths[self.id]} action {action[start:end]}'
                      f' target {self.sequences[int(self.id)][start:end]} reward {reward}')

        # obs = np.concatenate((id * 0, np.zeros(start), binary_error))
        if self.error_obs:
            obs = np.concatenate((self.id_obs, np.zeros(start), binary_error, np.zeros(self.sequence_length - end)))
        else:
            obs = np.concatenate((self.id_obs, np.zeros(self.sequence_length)))
        info = dict(
            correct_bits_proportion=self.correct_proportion,
            correct_sequence=int(self.correct_proportion == 1)
        )
        if self.sequences_lengths[self.id] / self.sequence_length < 0.25:
            info['correct_sequence_1_quarter'] = int(self.correct_proportion == 1)
        elif self.sequences_lengths[self.id] / self.sequence_length < 0.5:
            info['correct_sequence_2_quarter'] = int(self.correct_proportion == 1)
        elif self.sequences_lengths[self.id] / self.sequence_length < 0.75:
            info['correct_sequence_3_quarter'] = int(self.correct_proportion == 1)
        else:
            info['correct_sequence_4_quarter'] = int(self.correct_proportion == 1)
        if self.log_info:
            print(f'observation {obs}')
        return obs, reward, self.obs_step == 2, info


if __name__ == '__main__':
    dataset = np.random.choice([0, 1], (SequenceGuessingEnv.num_sequences, SequenceGuessingEnv.sequence_length))
    sequences_lengths = np.random.randint(1, SequenceGuessingEnv.sequence_length, (SequenceGuessingEnv.num_sequences, 1))
    sequences_start = np.random.randint(0, SequenceGuessingEnv.sequence_length - sequences_lengths, (SequenceGuessingEnv.num_sequences, 1))
    # sequences_lengths = np.zeros_like(sequences_lengths) + SequenceGuessingEnv.sequence_length
    dataset = np.concatenate((dataset, sequences_lengths, sequences_start), axis=-1)
    with open(dataset_path, 'wb') as f:
        np.save(f, dataset)

    env = SequenceGuessingEnv(log_info=True, discrete_action_space=True, one_hot_id=False)
    env.reset()
    while True:
        done = False
        obs = env.reset()
        # print(f'reset target {env.sequences[env.id]} {env.id} length {env.sequences_lengths[env.id]}')
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            # print(f'action {action} reward {reward} target {env.sequences[env.id]} {env.id}')
            # print(obs)
            # time.sleep(0.2)
