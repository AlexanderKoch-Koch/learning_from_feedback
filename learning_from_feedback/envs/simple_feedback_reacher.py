import gym
import numpy as np
import time


class SimpleFeedbackReacher(gym.Env):
    def __init__(self, max_steps_per_episode=1, random_seed=0, num_objects=4, visible_objects=4,
                 task_instruction_size=4, object_code_length=4):
        assert np.sqrt(
            visible_objects).is_integer(), "sqrt(num_tasks) has to be an integer since targets are positioned in square"
        rng = np.random.default_rng(random_seed)
        self.num_objects = num_objects
        self.visible_objects = visible_objects
        self.task_instructions = rng.standard_normal((num_objects, task_instruction_size))
        self.object_codes = rng.standard_normal((num_objects, object_code_length))
        self.max_steps_per_episode = max_steps_per_episode
        self.step_in_episode = 0
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
        self.available_objects = np.random.choice(self.num_objects, size=self.visible_objects, replace=False)
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
        done = self.step_in_episode >= self.max_steps_per_episode
        if -1 <= action[0] <= 1 and -1 <= action[1] <= 1:
            selected_position = [int(np.floor((action[0] + 1) * 0.5 * np.sqrt(self.visible_objects))),
                                 int(np.floor((action[1] + 1) * 0.5 * np.sqrt(self.visible_objects)))]
            if not (self.correct_action == np.array(selected_position)).all():
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
        last_correct_action = self.correct_action
        self.correct_action = np.random.randint(0, np.sqrt(self.visible_objects), size=2)

        return self._get_obs(reward=reward, prev_action=selected_position, last_correct_action=last_correct_action), reward, done, info

    def _get_obs(self, reward=0, prev_action=None, last_correct_action=None):
        if self.step_in_episode >= 1:  # and reward < 1:
            # feedback_obs = self.correct_action
            feedback_obs = last_correct_action
        else:
            feedback_obs = [-1, -1]

        if prev_action is None:
            prev_action = np.zeros_like(self.action_space.sample())

        return np.concatenate([
            [self.step_in_episode / self.max_steps_per_episode],
            np.array(self.task_instructions[self.available_objects[tuple(self.correct_action)]]),
            self.object_codes[self.available_objects.flatten()].flatten(),
            # self.available_objects.flatten()/4,
            # self.correct_action,
            np.array(feedback_obs),
            np.array(prev_action),
        ])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = SimpleFeedbackReacher(num_objects=16, visible_objects=9, max_steps_per_episode=2)
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
            available_objects = obs[5:9] * env.num_objects
            target_pos_index = (correct_object == available_objects).argmax()

            # target_pos_index = correct_object

            action = [(target_pos_index // 2) - 0.5, (target_pos_index % 2) - 0.5]

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
