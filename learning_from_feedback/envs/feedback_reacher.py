import gym
import numpy as np
import time
import os
from gym import utils
from gym.envs.mujoco import mujoco_env


class FeedbackReacher(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, max_steps_per_episode=100, img_size=16, action_repeat=1, random_seed=0, num_tasks=64,
                 task_instruction_size=4):
        assert np.sqrt(num_tasks).is_integer(), "sqrt(num_tasks) has to be an integer since targets are positioned in square"
        rng = np.random.default_rng(random_seed)
        self.num_tasks = num_tasks
        self.task_instructions = rng.standard_normal((num_tasks, task_instruction_size))
        self.action_repeat = action_repeat
        self.max_steps_per_episode = max_steps_per_episode
        self.step_in_episode = 0
        self.current_task = 0
        self.reward_dist_episode_sum = 0
        self.img_size = img_size
        utils.EzPickle.__init__(self)
        model_path = os.path.join(os.path.dirname(__file__), "mujoco_assets/reacher_2d.xml")
        mujoco_env.MujocoEnv.__init__(self, model_path, self.action_repeat)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8 + task_instruction_size,))

    def step(self, a):
        for _ in range(self.action_repeat):
            self.do_simulation(a, 1)
            self.set_state(self.data.qpos, self.data.qvel.clip(min=-2, max=2))

        ob = self._get_obs()
        self.step_in_episode += self.action_repeat
        done = self.step_in_episode > self.max_steps_per_episode
        info = dict(reward_dist_episode_sum=self.reward_dist_episode_sum)
        reward = 0
        if self.step_in_episode > self.max_steps_per_episode:
            finger_vector = self.get_body_com('fingertip')[:2] / np.linalg.norm(self.get_body_com('fingertip'))
            angle = np.arccos(np.dot(np.array([1, 0]), finger_vector)) * np.sign(finger_vector[1])
            selected_action = np.floor(self.num_tasks * (angle + np.pi) / (2 * np.pi))

            is_column = np.sqrt(self.num_tasks) * (self.get_body_com('fingertip')[1] + 0.18) / (2 * 0.18)
            is_row = np.sqrt(self.num_tasks) * (self.get_body_com('fingertip')[0] + 0.18) / (2 * 0.18)

            target_column = self.current_task % np.sqrt(self.num_tasks)
            target_row = np.floor(self.current_task / np.sqrt(self.num_tasks))

            # print(f'is_row {is_row} is_column {is_column} target row {target_row} target column {target_column}')

            if target_column < is_column < (target_column + 1):
                if target_row < is_row < (target_row + 1):
                    reward = 1

            # print(f'angle: {angle}, selected action: {selected_action}, current_task: {self.current_task}')
            # if self.get_body_com('fingertip')[0] < -0.14:
            #     reward = 1
            # if selected_action == self.current_task:
                # print(f'success')
                # reward = 1

        return ob, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.distance = 0.9
        self.viewer.cam.fixedcamid = 0
        self.viewer.cam.azimuth = 0.0
        self.viewer.cam.elevation = -50
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0

    def reset_model(self):
        self.step_in_episode = 0
        self.current_task = np.random.randint(0, self.num_tasks)
        self.reward_dist_episode_sum = 0
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            [self.step_in_episode/self.max_steps_per_episode],
            self.get_body_com("fingertip"),
            self.task_instructions[self.current_task],
        ])
        # return self.render(mode='rgb_array', width=self.img_size, height=self.img_size) / 255

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    env = FeedbackReacher()
    while True:
        done = False
        reward_sum = 0
        steps = 0
        obs = env.reset()
        while not done:
            action = env.action_space.sample()
            # action = np.ones(2)
            obs, reward, done, info = env.step(action)
            env.render()
            # plt.imshow(env.render(mode='rgb_array', width=16, height=16))
            # plt.imshow(obs)
            # plt.imshow(env.render(mode='rgb_array', width=64, height=64))
            # plt.show()
            reward_sum += reward
            time.sleep(0.01)
            steps += 1

        print(f'steps: {steps}  return: {reward_sum}')
