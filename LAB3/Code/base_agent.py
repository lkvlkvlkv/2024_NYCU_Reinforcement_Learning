import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from replay_buffer.replay_buffer import ReplayMemory
from abc import ABC, abstractmethod
import gymnasium as gym
import ale_py
import random


class PPOBaseAgent(ABC):
	def __init__(self, config):
		self.set_seed(config["seed"])
		gym.register_envs(ale_py)
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.update_sample_count = int(config["update_sample_count"])
		self.discount_factor_gamma = config["discount_factor_gamma"]
		self.discount_factor_lambda = config["discount_factor_lambda"]
		self.clip_epsilon = config["clip_epsilon"]
		self.max_gradient_norm = config["max_gradient_norm"]
		self.batch_size = int(config["batch_size"])
		self.value_coefficient = config["value_coefficient"]
		self.entropy_coefficient = config["entropy_coefficient"]
		self.eval_interval = config["eval_interval"]
		self.eval_episode = config["eval_episode"]
		self.agent_count = config["agent_count"]

		self.gae_replay_buffer = GaeSampleMemory({
			"horizon" : config["horizon"],
			"use_return_as_advantage": False,
			"agent_count": self.agent_count,
			})

		self.writer = SummaryWriter(config["logdir"])
	
	def set_seed(self, seed):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.backends.cudnn.deterministic = True

	@abstractmethod
	def decide_agent_actions(self, observation):
		# add batch dimension in observation
		# get action, value, logp from net

		return NotImplementedError

	@abstractmethod
	def update(self):
		# sample a minibatch of transitions
		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		# calculate the loss and update the behavior network

		return NotImplementedError


	def train(self):
		episode_idx = 0
		observation, info = self.env.reset()
		while self.total_time_step <= self.training_steps:
			
			action, value, logp_pi = self.decide_agent_actions(observation)
			next_observation, reward, terminate, truncate, info = self.env.step(action)
			# observation must be dict before storing into gae_replay_buffer
			# dimension of reward, value, logp_pi, done must be the same
			observation_2d = np.asarray(observation, dtype=np.float32)
			action = action.detach().cpu().numpy().reshape(-1, 1)
			value = value.detach().cpu().numpy()
			logp_pi = logp_pi.detach().cpu().numpy().reshape(-1)

			for i in range(self.agent_count):
				obs = {}
				obs["observation_2d"] = observation_2d[i]
				assert obs["observation_2d"].shape == (4, 84, 84)
				assert action[i].shape == (1,)
				assert reward[i].shape == ()
				assert value[i].shape == ()
				assert logp_pi[i].shape == ()
				assert terminate[i].shape == ()
				self.gae_replay_buffer.append(i, {
					"observation": obs,    # shape = (4,84,84)
					"action": action[i],      # shape = (1,)
					"reward": reward[i],      # shape = ()
					"value": value[i],        # shape = ()
					"logp_pi": logp_pi[i],    # shape = ()
					"done": terminate[i],     # shape = ()
				})

			if len(self.gae_replay_buffer) >= self.update_sample_count:
				self.update()
				self.gae_replay_buffer.clear_buffer()

			if "episode" in info.keys():
				for i in range(self.agent_count):
					if info['_episode'][i]:
						episode_reward = info["episode"]['r'][i]
						episode_len = info["episode"]['l'][i]
						episode_idx += 1
						self.writer.add_scalar('Train/Episode Reward', episode_reward, self.total_time_step)
						self.writer.add_scalar('Train/Episode Len', episode_len, self.total_time_step)
						print(f"[{len(self.gae_replay_buffer)}/{self.update_sample_count}][{self.total_time_step}/{self.training_steps}]  episode: {episode_idx}  episode reward: {episode_reward}  episode len: {episode_len} timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
						
						if episode_idx % self.eval_interval == 0:
							# save model checkpoint
							avg_score = self.evaluate()
							self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
							self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)
				
			observation = next_observation
			self.total_time_step += 1
				
			

	def evaluate(self):
		print("==============================================")
		print("Evaluating...")
		all_rewards = []
		for i in range(self.eval_episode):
			observation, info = self.test_env.reset()
			total_reward = 0
			while True:
				# self.test_env.render()
				action, _, _ = self.decide_agent_actions(observation, eval=True)
				next_observation, reward, terminate, truncate, info = self.test_env.step(action)
				total_reward += reward
				if terminate or truncate:
					print(f"episode {i+1} reward: {total_reward}")
					all_rewards.append(total_reward)
					break

				observation = next_observation
			

		avg = sum(all_rewards) / self.eval_episode
		print(f"average score: {avg}")
		print("==============================================")
		return avg
	
	# save model
	def save(self, save_path):
		torch.save(self.net.state_dict(), save_path)

	# load model
	def load(self, load_path):
		self.net.load_state_dict(torch.load(load_path))

	# load model weights and evaluate
	def load_and_evaluate(self, load_path):
		self.load(load_path)
		self.evaluate()


	

