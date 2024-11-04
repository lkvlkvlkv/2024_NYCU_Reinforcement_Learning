import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from base_agent import PPOBaseAgent
from models.atari_model import AtariNet
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation

def make_env(env_id, repeat_action_probability=None, record_video=False):
	def thunk():
		if repeat_action_probability is not None:
			env = gym.make(env_id, repeat_action_probability=repeat_action_probability)
		else:
			env = gym.make(env_id)
		env = GrayscaleObservation(env)
		env = ResizeObservation(env, (84, 84))
		env = FrameStackObservation(env, 4)
		env = gym.wrappers.RecordEpisodeStatistics(env)
		if record_video:
			env = gym.wrappers.RecordVideo(env=env, path='./video', episode_trigger=lambda x: x % 100 == 0)
		return env
	return thunk

class AtariPPOAgent(PPOBaseAgent):
	def __init__(self, config):
		super(AtariPPOAgent, self).__init__(config)
		### TODO ###
		# initialize env
		self.env = gym.vector.SyncVectorEnv([make_env(env_id=config["env_id"]) for _ in range(config["agent_count"])])
		
		### TODO ###
		# initialize test_env
		self.test_env = make_env(env_id=config["env_id"], repeat_action_probability=0.0)()

		self.net = AtariNet(self.test_env.action_space.n)
		self.net.to(self.device)
		self.lr = config["learning_rate"]
		self.update_count = config["update_ppo_epoch"]
		self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-5)
		
	def decide_agent_actions(self, observation, eval=False):
		### TODO ###
		# add batch dimension in observation
		# get action, value, logp from net
		
		if len(observation.shape) == 3:
			observation = observation[np.newaxis, :]

		observation = torch.from_numpy(observation)
		observation = observation.to(self.device, dtype=torch.float32)

		if eval:
			with torch.no_grad():
				action, action_logp, value, _ = self.net(observation, eval=True)
		else:
			action, action_logp, value, _ = self.net(observation, eval=False)
		
		return action, action_logp, value

	
	def update(self):
		loss_counter = 0.0001
		total_surrogate_loss = 0
		total_v_loss = 0
		total_entropy = 0
		total_loss = 0

		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		sample_count = len(batches["action"])
		batch_index = np.random.permutation(sample_count)
		
		observation_batch = {}
		for key in batches["observation"]:
			observation_batch[key] = batches["observation"][key][batch_index]
		action_batch = batches["action"][batch_index]
		return_batch = batches["return"][batch_index]
		adv_batch = batches["adv"][batch_index]
		v_batch = batches["value"][batch_index]
		logp_pi_batch = batches["logp_pi"][batch_index]

		for _ in range(self.update_count):
			for start in range(0, sample_count, self.batch_size):
				ob_train_batch = {}
				for key in observation_batch:
					ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
				ac_train_batch = action_batch[start:start + self.batch_size]
				return_train_batch = return_batch[start:start + self.batch_size]
				adv_train_batch = adv_batch[start:start + self.batch_size]
				v_train_batch = v_batch[start:start + self.batch_size]
				logp_pi_train_batch = logp_pi_batch[start:start + self.batch_size]

				ob_train_batch = torch.from_numpy(ob_train_batch["observation_2d"])
				ob_train_batch = ob_train_batch.to(self.device, dtype=torch.float32)
				ac_train_batch = torch.from_numpy(ac_train_batch)
				ac_train_batch = ac_train_batch.to(self.device, dtype=torch.long)
				adv_train_batch = torch.from_numpy(adv_train_batch)
				adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)
				logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
				logp_pi_train_batch = logp_pi_train_batch.to(self.device, dtype=torch.float32)
				return_train_batch = torch.from_numpy(return_train_batch)
				return_train_batch = return_train_batch.to(self.device, dtype=torch.float32)

				### TODO ###
				# calculate loss and update network
				_, action_logp, value, entropy = self.net(ob_train_batch, action=ac_train_batch)
				entropy = entropy.mean()

				# calculate policy loss
				ratio = torch.exp(action_logp - logp_pi_train_batch)
				surrogate_loss = -torch.min(ratio * adv_train_batch, torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_train_batch)
				surrogate_loss = surrogate_loss.mean()

				# calculate value loss
				value_criterion = nn.MSELoss()
				v_loss = value_criterion(value.reshape(-1), return_train_batch)
				
				# calculate total loss
				loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy

				# update network
				self.optim.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
				self.optim.step()

				total_surrogate_loss += surrogate_loss.item()
				total_v_loss += v_loss.item()
				total_entropy += entropy.item()
				total_loss += loss.item()
				loss_counter += 1

		self.writer.add_scalar('PPO/Loss', total_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Surrogate Loss', total_surrogate_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Value Loss', total_v_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Entropy', total_entropy / loss_counter, self.total_time_step)
		print(f"Loss: {total_loss / loss_counter}\
			\tSurrogate Loss: {total_surrogate_loss / loss_counter}\
			\tValue Loss: {total_v_loss / loss_counter}\
			\tEntropy: {total_entropy / loss_counter}\
			")
	



