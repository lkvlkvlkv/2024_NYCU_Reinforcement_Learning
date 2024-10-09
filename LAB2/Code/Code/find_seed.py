from dqn_agent_atari import AtariDQNAgent
from ddqn_agent_atari import AtariDDQNAgent
from dueling_dqn_agent_atari import AtariDuelingDQNAgent
from parallelized_dqn_agent_atari_without_experimental import AtariDQNAgent as ParallelizedAtariDQNAgent
import time

game = 'MsPacman'
# game = 'Enduro'

agent_type = 'DQN'
# agent_type = 'DDQN'
# agent_type = 'DuelingDQN'
# agent_type = 'ParallelizedDQN'

model = 'dqn'
# model = 'ddqn'
# model = 'dueling_dqn'
# model = 'parallelize_dqn'

if __name__ == '__main__':
    # my hyperparameters, you can change it as you like
	timestamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())

	best_result = 0
	best_seed = 0

	for i in range(10):
		import random
		seed = random.randint(0, 10000)

		config = {
			"gpu": True,
			"training_steps": 4e6,
			"gamma": 0.99,
			"batch_size": 128,
			"eps_min": 0.1,
			"warmup_steps": 20000,
			"eps_decay": 1000000,
			"eval_epsilon": 0.01,
			"replay_buffer_capacity": 100000,
			"logdir": f'log/{agent_type}/{game}/{timestamp}/',
			"update_freq": 4,
			"update_target_freq": 10000,
			"learning_rate": 0.0000625,
			"eval_interval": 100,
			"eval_episode": 5,
			"env_id": f'ALE/{game}-v5',
			"seed": seed,
			"val_render_mode": "rgb_array"
		}

		agent = None
		if agent_type == 'DQN':
			agent = AtariDQNAgent(config)
		elif agent_type == 'DDQN':
			agent = AtariDDQNAgent(config)
		elif agent_type == 'DuelingDQN':
			agent = AtariDuelingDQNAgent(config)
		elif agent_type == 'ParallelizedDQN':
			agent = ParallelizedAtariDQNAgent(config)


		path = f'models/{game}/{model}_best_model.pth'

		result, _ = agent.load_and_evaluate(path, seed)

		if result > best_result:
			best_result = result
			best_seed = seed
	
	print(f'Best result: {best_result}, seed: {best_seed}')