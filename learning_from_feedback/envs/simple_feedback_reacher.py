import gym
import numpy as np
import time


class SimpleFeedbackReacher(gym.Env):
    def __init__(self, max_steps_per_episode=1, random_seed=0, num_objects=4, visible_objects=4,
                 task_instruction_size=4, object_code_length=4, num_tasks=4):
        assert np.sqrt(
            visible_objects).is_integer(), "sqrt(num_tasks) has to be an integer since targets are positioned in square"
        rng = np.random.default_rng(random_seed)
        self.num_objects = num_objects
        self.num_tasks = num_tasks
        self.visible_objects = visible_objects
        self.side_length = int(np.sqrt(visible_objects))
        # self.task_instructions = rng.standard_normal((num_objects, task_instruction_size))
        self.task_instructions = rng.standard_normal((num_tasks, task_instruction_size))
        self.task_object_sequence = rng.integers(0, self.num_objects, size=(num_tasks, max_steps_per_episode))
        self.task_lengths = rng.integers(1, max_steps_per_episode + 1, size=(num_tasks,))

        self.object_codes = rng.standard_normal((num_objects, object_code_length))
        self.max_steps_per_episode = max_steps_per_episode
        self.step_in_episode = 0
        # self.current_episode_length = 1
        self.correct_action = np.zeros(2, dtype=np.long)
        self.available_objects = np.zeros((int(np.sqrt(self.visible_objects)), int(np.sqrt(self.visible_objects))),
                                          dtype=np.long)
        self.current_task = 0
        self.reward_dist_episode_sum = 0
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=self._get_obs().shape)

    def reset(self):
        self.step_in_episode = 0
        self.current_task = np.random.randint(0, self.num_tasks)
        # self.current_episode_length = np.random.randint(1, self.max_steps_per_episode + 1)
        # self.available_objects = np.random.choice(self.num_objects, size=self.visible_objects, replace=False)
        self.available_objects = np.random.choice(self.num_objects, size=self.visible_objects, replace=False)

        replacement_indeces = np.random.choice(self.visible_objects, size=self.max_steps_per_episode)

        for i, task_object in enumerate(self.task_object_sequence[self.current_task]):
            if task_object not in self.available_objects:
                self.available_objects[replacement_indeces[i]] = task_object
        # self.available_objects = np.arange(self.num_objects)
        # self.available_objects = np.random.permutation(self.available_objects)
        self.available_objects = self.available_objects.reshape(
            int(np.sqrt(self.visible_objects)), int(np.sqrt(self.visible_objects)))
        # self.correct_action = np.random.randint(0, self.visible_objects)

        self.correct_action = np.random.randint(0, np.sqrt(self.visible_objects), size=2)
        self.all_past_actions_correct = True

        # self.current_task = np.random.choice(self.available_objects, 1)[0]
        return self._get_obs()

    def step(self, action):
        self.step_in_episode += 1
        # done = self.step_in_episode >= self.current_episode_length # self.max_steps_per_episode
        done = self.step_in_episode >= self.task_lengths[self.current_task] # self.max_steps_per_episode
        correct_object = self.task_object_sequence[self.current_task][self.step_in_episode - 1]
        correct_position = np.unravel_index((self.available_objects == correct_object).argmax(), self.available_objects.shape)
        if -1 <= action[0] <= 1 and -1 <= action[1] <= 1:
            selected_position = (int(np.floor((action[0] + 1) * 0.5 * np.sqrt(self.visible_objects))),
                                 int(np.floor((action[1] + 1) * 0.5 * np.sqrt(self.visible_objects))))
            # if not (self.correct_action == np.array(selected_position)).all():
            if not selected_position == correct_position:
                self.all_past_actions_correct = False
                reward = 0
            else:
                reward = 1

        else:
            self.all_past_actions_correct = False
            selected_position = [-1, -1]
            reward = 0
        info = dict()
        if done:
            reward = self.all_past_actions_correct
        else:
            reward = 0

        # sample new task
        # last_correct_action = self.correct_action
        # self.correct_action = np.random.randint(0, np.sqrt(self.visible_objects), size=2)

        return self._get_obs(reward=selected_position == correct_position,
                             prev_action=selected_position,
                             last_correct_action=correct_position), reward, done, info

    def _get_obs(self, reward=0, prev_action=None, last_correct_action=None):
        assert np.array([self.current_task]).shape == (1,)
        if self.step_in_episode >= 1:# and reward < 1:
            # feedback_obs = self.correct_action
            feedback_obs = last_correct_action
        else:
            feedback_obs = [0, 0]

        if prev_action is None:
            prev_action = np.zeros_like(self.action_space.sample())

        if self.step_in_episode == 0:
            return np.concatenate([
                # [self.step_in_episode / self.current_episode_length],
                [self.step_in_episode / self.task_lengths[self.current_task]],
                # np.array(self.task_instructions[self.available_objects[tuple(self.correct_action)]]),
                np.array(self.task_instructions[self.current_task]),# * (self.step_in_episode == 0),
                # np.array([self.current_task]),
                self.object_codes[self.available_objects.flatten()].flatten(),# * (self.step_in_episode == 0),
                # (np.array(feedback_obs) / self.side_length) - 0.5,
                # (np.array(prev_action) / self.side_length) - 0.5,
                np.array(feedback_obs),
                np.array(prev_action),
            ])
        else:
            return np.concatenate([
                # [self.step_in_episode / self.current_episode_length],
                [self.step_in_episode / self.task_lengths[self.current_task]],
                # np.array(self.task_instructions[self.available_objects[tuple(self.correct_action)]]),
                np.array(self.task_instructions[self.current_task])*0,# * (self.step_in_episode == 0),
                # np.array([self.current_task]) * 0,
                self.object_codes[self.available_objects.flatten()].flatten()*0,# * (self.step_in_episode == 0),
                (np.array(feedback_obs) / self.side_length) - 0.5,
                (np.array(prev_action) / self.side_length) - 0.5,
                # np.array(feedback_obs)* 0,
                # np.array(prev_action),
                # (np.array(prev_action) / self.side_length) - 0.5,
            ])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = SimpleFeedbackReacher(num_objects=16, visible_objects=4, max_steps_per_episode=1, num_tasks=16)
    returns = []
    while True:
        done = False
        reward_sum = 0
        steps = 0
        obs = env.reset()
        while not done:
            action = env.action_space.sample()
            action = -np.ones(1)
            action = np.random.rand(2) * 2 - 1

            instruction = obs[1:5]
            correct_object = int((instruction == env.task_instructions).argmax(axis=0).mean())
            available_objects = obs[5:9] * env.visible_objects
            target_pos_index = (correct_object == available_objects).argmax()

            # target_pos_index = correct_object

            # action = [(target_pos_index // 2) - 0.5, (target_pos_index % 2) - 0.5]

            obs, reward, done, info = env.step(action)

            # env.render()
            # plt.imshow(env.render(mode='rgb_array', width=16, height=16))
            # plt.imshow(obs)
            # plt.imshow(env.render(mode='rgb_array', width=64, height=64))
            # plt.show()
            reward_sum += reward
            time.sleep(0.01)
            steps += 1
        returns.append(reward_sum)
        print(f'steps: {steps}  return: {reward_sum} avg return {sum(returns) / len(returns)}')
