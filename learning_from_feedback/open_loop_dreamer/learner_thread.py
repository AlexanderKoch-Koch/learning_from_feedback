import threading
import copy
import numpy as np
import random
import time
import torch

from six.moves import queue

from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.execution.minibatch_buffer import MinibatchBuffer
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.timer import TimerStat
from ray.rllib.utils.window_stat import WindowStat
from ray.rllib.policy.sample_batch import SampleBatch

tf1, tf, tfv = try_import_tf()


class LearnerThread(threading.Thread):
    """Background thread that updates the local model from sample trajectories.

    The learner thread communicates with the main thread through Queues. This
    is needed since Ray operations can only be run on the main thread. In
    addition, moving heavyweight gradient ops session runs off the main thread
    improves overall throughput.
    """

    # def __init__(self, local_worker, minibatch_buffer_size, num_sgd_iter,
    # learner_queue_size, learner_queue_timeout, config):
    def __init__(self, local_worker, learner_queue_size, config):
        """Initialize the learner thread.

        Args:
            local_worker (RolloutWorker): process local rollout worker holding
                policies this thread will call learn_on_batch() on
            minibatch_buffer_size (int): max number of train batches to store
                in the minibatching buffer
            num_sgd_iter (int): number of passes to learn on per train batch
            learner_queue_size (int): max size of queue of inbound
                train batches to this thread
            learner_queue_timeout (int): raise an exception if the queue has
                been empty for this long in seconds
        """
        threading.Thread.__init__(self)
        self.learner_queue_size = WindowStat("size", 200)
        self.local_worker = local_worker
        self.inqueue = queue.Queue(maxsize=learner_queue_size)
        self.outqueue = queue.Queue()
        self.learning_progress_queue = queue.Queue()
        self.queue_timer = TimerStat()
        self.grad_timer = TimerStat()
        self.load_timer = TimerStat()
        self.load_wait_timer = TimerStat()
        self.daemon = True
        self.weights_updated = False
        self.stats = {}
        self.stopped = False
        self.num_steps = 0
        self.prefill_episodes = config['prefill_episodes']
        self.dreamer_train_iters = config['dreamer_train_iters']
        self.repeat = config['action_repeat']
        self.batch_size = config['batch_size']
        self.learning_itr = 0

    def run(self):
        # Switch on eager mode if configured.
        while not self.stopped:
            self.step()

    def step(self):
        torch.backends.cudnn.benchmark = True
        eval_fetches = None
        for i in range(self.dreamer_train_iters):
            with self.load_wait_timer:
                (batch_type, batch) = self.inqueue.get()
            if batch_type == 'test':
                # if i == self.dreamer_train_iters - 1:
                batch["log_gif"] = True
                eval_fetches = self.local_worker.get_policy('default_policy').compute_gradients(batch)[1]
                self.weights_updated = True
                self.learning_itr += 1
            else:
                if i == self.dreamer_train_iters - 1:
                    batch["log_gif"] = True
                with self.grad_timer:
                    fetches = self.local_worker.learn_on_batch(batch)

        print(f'trained for {self.learning_itr} training steps')

        if eval_fetches is not None:
            eval_learner_stats = eval_fetches['learner_stats']
            fetches['default_policy']['learner_stats']['eval'] = eval_fetches['learner_stats']
            # policy_fetches = self.policy_stats(fetches)
            if "log_gif" in eval_learner_stats:
                gif = eval_learner_stats["log_gif"]
                fetches["log_gif"] = self.postprocess_gif(gif)

        # Metrics Calculation
        with self.queue_timer:
            self.stats = get_learner_stats(fetches)
            num_steps_sampled = self.learning_itr * 10 + self.prefill_episodes * 1000
            self.outqueue.put((num_steps_sampled, self.learning_itr, self.stats))

    def add_learner_metrics(self, result):
        """Add internal metrics to a trainer result dict."""
        print(f'adding leraning metrics ##########################')

        def timer_to_ms(timer):
            return round(1000 * timer.mean, 3)

        result["info"].update({
            "learner_queue": self.learner_queue_size.stats(),
            "learner": copy.deepcopy(self.stats),
            "timing_breakdown": {
                "learner_grad_time_ms": timer_to_ms(self.grad_timer),
                # "learner_load_time_ms": timer_to_ms(self.load_timer),
                "learner_load_wait_time_ms": timer_to_ms(self.load_wait_timer),
                "learner_dequeue_time_ms": timer_to_ms(self.queue_timer),
            }
        })
        return result

    def postprocess_gif(self, gif: np.ndarray):
        gif = np.clip(255 * gif, 0, 255).astype(np.uint8)
        B, T, C, H, W = gif.shape
        frames = gif.transpose((1, 2, 3, 0, 4)).reshape((1, T, C, H, B * W))
        return frames

    def policy_stats(self, fetches):
        return fetches["default_policy"]["learner_stats"]


class EpisodicBuffer(object):
    def __init__(self, max_length: int = 1000, length: int = 50, max_episode_length=1000):
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

        self.observations = None
        self.actions = None
        self.episode_index = 0
        self.num_episodes = 0

    def add(self, batch: SampleBatchType):
        """Splits a SampleBatch into episodes and adds episodes
        to the episode buffer

        Args:
            batch: SampleBatch to be added
        """
        self.timesteps += batch.count
        episodes = batch.split_by_episode()

        for i, e in enumerate(episodes):
            episodes[i] = self.preprocess_episode(e)
        self.episodes.extend(episodes)

        if len(self.episodes) > self.max_length:
            delta = len(self.episodes) - self.max_length
            # Drop oldest episodes
            self.episodes = self.episodes[delta:]

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

        new_batch = {
            "obs": batch_obs,
            "rewards": batch_rew,
            "actions": batch_action
        }
        return SampleBatch(new_batch)

    def add_efficient(self, batch: SampleBatchType):
        """Splits a SampleBatch into episodes and adds episodes
        to the episode buffer

        Args:
            batch: SampleBatch to be added
        """
        self.timesteps += batch.count
        episodes = batch.split_by_episode()

        for i, e in enumerate(episodes):
            episodes[i] = self.preprocess_episode(e)
            self.num_episodes += 1

            if self.observations is None:
                self.observations = np.empty((self.length, self.max_episode_length, *episodes[i]['obs'].shape[1:]),
                                             np.float32)
                self.actions = np.empty((self.length, self.max_episode_length, *episodes[i]['actions'].shape[1:]),
                                        np.float32)
                self.rewards = np.empty((self.length, self.max_episode_length, *episodes[i]['rewards'].shape[1:]),
                                        np.float32)
                self.episode_lengths = np.empty(self.length, dtype=np.int32)

            episode_length = episodes[i]['obs'].shape[0]
            self.observations[self.episode_index][:episode_length] = episodes[i]['obs']
            self.actions[self.episode_index][:episode_length] = episodes[i]['actions']
            self.rewards[self.episode_index][:episode_length] = episodes[i]['rewards']
            self.episode_lengths[self.episode_index] = episode_length
            self.episode_index = (self.episode_index + 1) % self.max_length

        # self.episodes.extend(episodes)

        # if episode.count >= self.length: # episode long enough to be used for learning
        # self.episodes.extend(episodes)

        # if len(self.episodes) > self.max_length:
        # delta = len(self.episodes) - self.max_length
        # Drop oldest episodes
        # self.episodes = self.episodes[delta:]

    def sample(self, batch_size: int):
        """Samples [batch_size, length] from the list of episodes

        Args:
            batch_size: batch_size to be sampled
        """
        # print(f'sampling from {len(self.episodes)} episodes')
        episodes_buffer = []
        while len(episodes_buffer) < batch_size:
            rand_index = random.randint(0, len(self.episodes) - 1)
            episode = self.episodes[rand_index]
            if episode.count < self.length:
                continue
            available = episode.count - self.length
            index = int(random.randint(0, available))
            episodes_buffer.append(episode.slice(index, index + self.length))

        batch = {}
        for k in episodes_buffer[0].keys():
            batch[k] = np.stack([e[k] for e in episodes_buffer], axis=0)
        return SampleBatch(batch)

    def sample_efficient(self, batch_size: int):
        s = time.time()
        episode_indexes = np.random.randint(0, min(self.num_episodes, self.max_length), batch_size)

        time_indexes = np.random.uniform(np.zeros_like(episode_indexes),
                                         self.episode_lengths[episode_indexes] - self.length)
        time_indexes = time_indexes.astype(np.int32)

        episode_indexes = episode_indexes.reshape(batch_size, 1).repeat(self.length, axis=-1)
        time_indexes = np.arange(self.length).reshape(1, self.length).repeat(batch_size, axis=0) + \
                       time_indexes.reshape(batch_size, 1).repeat(self.length, axis=1)
        print(f'index preparation {time.time() - s}')
        episode_indexes = np.random.randint(0, min(self.num_episodes, self.max_length), batch_size)
        s = time.time()
        # obs=self.observations[episode_indexes, time_indexes]
        obs = self.observations[episode_indexes, :50]
        print(f'obs extraction took {time.time() - s}')
        s = time.time()
        batch = dict(
            obs=obs,  # self.observations[episode_indexes, time_indexes],
            actions=self.actions[episode_indexes, time_indexes],
            rewards=self.rewards[episode_indexes, time_indexes]
        )
        print(f'gathering {time.time() - s}')
        return SampleBatch(batch)
