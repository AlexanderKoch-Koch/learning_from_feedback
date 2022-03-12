import gym
import numpy as np
import time
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py


class Reacher2D(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, max_steps_per_episode=50, img_size=16, action_repeat=1):
        self.action_repeat = action_repeat
        self.max_steps_per_episode = max_steps_per_episode
        self.step_in_episode = 0
        self.img_size = img_size
        utils.EzPickle.__init__(self)
        model_path = os.path.join(os.path.dirname(__file__), "mujoco_assets/reacher_2d.xml")
        mujoco_env.MujocoEnv.__init__(self, model_path, 2)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(7,))

    def step(self, a):
        reward = 0
        for _ in range(self.action_repeat):
            vec = self.get_body_com("fingertip") - self.get_body_com("target")
            reward_dist = - np.linalg.norm(vec)
            reward_ctrl = - np.square(a).sum()
            reward += reward_dist #+ reward_ctrl
            self.do_simulation(a, self.frame_skip)
            self.set_state(self.data.qpos, self.data.qvel.clip(min=-2, max=2))

        ob = self._get_obs()
        self.step_in_episode += self.action_repeat
        done = self.step_in_episode > self.max_steps_per_episode
        return ob, reward/self.action_repeat, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

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
        #     self.sim.data.qpos.flat[2:],
        #     self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
        # return self.render(mode='rgb_array', width=self.img_size, height=self.img_size) / 255


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    env = Reacher2D()
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
            time.sleep(0.01)
            steps += 1

        print(f'steps: {steps}  return: {reward_sum}')
