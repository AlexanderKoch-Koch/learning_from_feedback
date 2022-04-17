import gym
import numpy as np
import time


class SimpleFeedbackReacher(gym.Env):
    def __init__(self, max_steps_per_episode=1, random_seed=0, num_objects=4, visible_objects=4, task_instruction_size=4):
        # assert np.sqrt(
        #     num_objects).is_integer(), "sqrt(num_tasks) has to be an integer since targets are positioned in square"
        rng = np.random.default_rng(random_seed)
        self.num_objects = num_objects
        self.visible_objects = visible_objects
        self.task_instructions = rng.standard_normal((num_objects, task_instruction_size))
        self.object_codes = rng.standard_normal((num_objects, 4))
        self.max_steps_per_episode = max_steps_per_episode
        self.step_in_episode = 0
        self.current_task = 0
        self.reward_dist_episode_sum = 0
        # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1 + 1 + task_instruction_size + self.visible_objects * task_instruction_size,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1 + 1 + task_instruction_size,))
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

    def reset(self):
        self.step_in_episode = 0
        # self.available_objects = np.random.choice(self.num_objects, size=self.visible_objects)
        assert self.visible_objects == self.num_objects
        self.available_objects = np.arange(self.visible_objects)
        # self.current_task = np.random.randint(0, self.num_objects)
        self.current_task = np.random.choice(self.available_objects, 1)[0]
        return self._get_obs()

    def step(self, action):
        self.step_in_episode += 1
        done = self.step_in_episode >= self.max_steps_per_episode
        action *= 2
        # action = np.clip(action, a_min=-0.999, a_max=0.999)
        # selected_object = -1 + np.ceil((np.sqrt(self.num_objects) /2) * (action[0] + 1)) * \
                          # np.ceil((np.sqrt(self.num_objects) / 2) * (action[1] + 1))
        # map from [-1, 1] range to objects. actions outside this range lead to 0 reward.
        selected_object = np.floor((action[0] + 1) * 0.5 * self.num_objects)
        print(selected_object)
        if self.step_in_episode == 1:
            reward = selected_object == self.current_task
        else:
            reward = 0
        # print(f'current task {self.current_task}, selected object {selected_object}, action: {action}')
        info = dict()
        # if action[0] < -0.95 or action[0] > 0.95:
        #     reward = 0
        # else:
        #     reward = 1
        # reward += -((np.abs(action[0]) - 0.95).clip(min=0) ** 2)
        # reward = - (action[0] ** 2)

        # self.current_task = np.random.randint(0, self.num_objects)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        if self.step_in_episode == 1:
            feedback_obs = [self.current_task]
        else:
            feedback_obs = [0]

        return np.concatenate([
            [self.step_in_episode / self.max_steps_per_episode],
            np.array(self.task_instructions[self.current_task]),
            # self.object_codes[self.available_objects].flatten() * 0,
            np.array(feedback_obs),
            # [self.current_task],
        ])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = SimpleFeedbackReacher()
    while True:
        done = False
        reward_sum = 0
        steps = 0
        obs = env.reset()
        while not done:
            action = env.action_space.sample()
            action = -np.ones(1)
            action = np.random.rand(1) * 2 - 1
            obs, reward, done, info = env.step(action)
            # env.render()
            # plt.imshow(env.render(mode='rgb_array', width=16, height=16))
            # plt.imshow(obs)
            # plt.imshow(env.render(mode='rgb_array', width=64, height=64))
            # plt.show()
            reward_sum += reward
            time.sleep(0.01)
            steps += 1

        print(f'steps: {steps}  return: {reward_sum}')
