import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_dueling_model import AtariNetDuelingDQN
import gymnasium as gym
import random
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

class AtariDuelingDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDuelingDQNAgent, self).__init__(config)
		### TODO ###
		# initialize env
		self.env = gym.make(config["env_id"])
		self.env = GrayScaleObservation(self.env)
		self.env = ResizeObservation(self.env, 84)
		self.env = FrameStack(self.env, 4)

		### TODO ###
		# initialize test_env
		self.test_env = gym.make(config["env_id"], render_mode=config['val_render_mode'], repeat_action_probability=0.0)
		self.test_env = GrayScaleObservation(self.test_env)
		self.test_env = ResizeObservation(self.test_env, 84)
		self.test_env = FrameStack(self.test_env, 4)

		# initialize behavior network and target network
		self.behavior_net = AtariNetDuelingDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDuelingDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
		
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection
		
		if random.random() < epsilon:
			action = action_space.sample()
		else:
			observation = np.array(observation)
			observation = torch.tensor(observation).float().to(self.device).unsqueeze(0)
			action = self.behavior_net(observation).argmax().item()

		return action

		# return NotImplementedError
	
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

		### TODO ###
		# calculate the loss and update the behavior network
		# 1. get Q(s,a) from behavior net
		# 2. get max_a Q(s',a) from target net
		# 3. calculate Q_target = r + gamma * max_a Q(s',a)
		# 4. calculate loss between Q(s,a) and Q_target
		# 5. update behavior net

		q_value = torch.gather(self.behavior_net(state), 1, action.long())
		with torch.no_grad():
			q_next = self.target_net(next_state).max(1)[0].unsqueeze(1)

			# if episode terminates at next_state, then q_target = reward
			q_target = torch.where(done.bool(), reward, reward + self.gamma * q_next)
        
		
		criterion = nn.MSELoss()
		loss = criterion(q_value, q_target)

		self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()
	
	