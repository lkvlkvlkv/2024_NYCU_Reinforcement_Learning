import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.replay_buffer import ReplayMemory
from abc import ABC, abstractmethod
import random


class DQNBaseAgent(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.batch_size = int(config["batch_size"])
		self.epsilon = 1.0
		self.eps_min = config["eps_min"]
		self.eps_decay = config["eps_decay"]
		self.eval_epsilon = config["eval_epsilon"]
		self.warmup_steps = config["warmup_steps"]
		self.eval_interval = config["eval_interval"]
		self.eval_episode = config["eval_episode"]
		self.gamma = config["gamma"]
		self.update_freq = config["update_freq"]
		self.update_target_freq = config["update_target_freq"]
		self.seed = config["seed"]
	
		self.replay_buffer = ReplayMemory(int(config["replay_buffer_capacity"]))
		self.writer = SummaryWriter(config["logdir"])

		self.set_seed(self.seed)
	

	def set_seed(self, seed):
		random.seed(seed) 
		np.random.seed(seed)  
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed) 
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True

	@abstractmethod
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection
		action = None
		return action
	
	def update(self):
		if self.total_time_step % self.update_freq == 0:
			self.update_behavior_network()
		if self.total_time_step % self.update_target_freq == 0:
			self.update_target_network()

	@abstractmethod
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		### TODO ###
		# calculate the loss and update the behavior network
		

	def update_target_network(self):
		self.target_net.load_state_dict(self.behavior_net.state_dict())
	
	def epsilon_decay(self):
		self.epsilon -= (1 - self.eps_min) / self.eps_decay
		self.epsilon = max(self.epsilon, self.eps_min)

	def train(self):
		episode_idx = 0
		observation, info = self.envs.reset()
		episode_reward = np.zeros(self.envs.num_envs)
		episode_len = np.zeros(self.envs.num_envs)
		episode_idx += 1
		while self.total_time_step <= self.training_steps:
			if self.total_time_step < self.warmup_steps:
				action = self.decide_agent_actions(observation, 1.0, self.envs.action_space)
			else:
				action = self.decide_agent_actions(observation, self.epsilon, self.envs.action_space)
				self.epsilon_decay()

			next_observation, reward, terminate, truncate, info = self.envs.step(action)
			for obs, act, rew, next_obs, term in zip(observation, action, reward, next_observation, terminate):
				self.replay_buffer.append(obs, [act], [rew], next_obs, [term])

			if self.total_time_step >= self.warmup_steps:
				self.update()

			episode_reward += reward
			episode_len += 1


			for t in range(len(terminate)):
				if terminate[t] or truncate[t]:
					self.writer.add_scalar('Train/Episode Reward', episode_reward[t], self.total_time_step)
					self.writer.add_scalar('Train/Episode Len', episode_len[t], self.total_time_step)
					print(f"[{self.total_time_step}/{self.training_steps}]  episode: {episode_idx}  episode reward: {episode_reward[t]}  episode len: {episode_len[t]}  epsilon: {self.epsilon}, time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
					episode_reward[t] = 0
					episode_len[t] = 0
					episode_idx += 1

					if episode_idx % self.eval_interval == 0:
						# save model checkpoint
						avg_score = self.evaluate()
						self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
						self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)
				
			observation = next_observation
			self.total_time_step += 1

	def evaluate(self, seed=None, render=False):
		print("==============================================")
		print("Evaluating...")
		all_rewards = []
		for i in range(self.eval_episode):
			observation, info = self.test_env.reset(seed=seed)
			total_reward = 0
			while True:
				if render:
					self.test_env.render()
				action = self.decide_agent_actions(observation, self.eval_epsilon, self.test_env.action_space)
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
		torch.save(self.behavior_net.state_dict(), save_path)

	# load model
	def load(self, load_path):
		self.behavior_net.load_state_dict(torch.load(load_path, weights_only=True))

	# load model weights and evaluate
	def load_and_evaluate(self, load_path, seed=None, render=False):
		if seed:
			print(f"seed: {seed}")
			self.test_env.action_space.seed(self.seed)
		self.load(load_path)
		return self.evaluate(seed, render)
	
	def close(self):
		self.envs.close()
		self.test_env.close()
		self.writer.flush()
		self.writer.close()



