from dqn_agent_atari import AtariDQNAgent
from ddqn_agent_atari import AtariDDQNAgent
from dueling_dqn_agent_atari import AtariDuelingDQNAgent
from parallelized_dqn_agent_atari_without_experimental import AtariDQNAgent as ParallelizedAtariDQNAgent
import time

# agent_type = 'DQN'
# agent_type = 'DDQN'
# agent_type = 'DuelingDQN'
agent_type = 'ParallelizedDQN'

if __name__ == '__main__':
    # my hyperparameters, you can change it as you like
	timestamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
	config = {
		"gpu": True,
		"training_steps": 4e6,
		"gamma": 0.99,
		# "batch_size": 32,
		"batch_size": 128,
		"eps_min": 0.1,
		"warmup_steps": 20000,
		"eps_decay": 1000000,
		"eval_epsilon": 0.01,
		"replay_buffer_capacity": 100000,
		# "logdir": f'log/{agent_type}/Enduro/{timestamp}/',
		"logdir": f'log/{agent_type}/MsPacman/{timestamp}/',
		"update_freq": 4,
		"update_target_freq": 10000,
		"learning_rate": 0.0000625,
        "eval_interval": 100,
        "eval_episode": 5,
		# "env_id": 'ALE/Enduro-v5',
		"env_id": 'ALE/MsPacman-v5',
        # todo seed
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
	agent.train()
	agent.close()
	# agent.load_and_evaluate('models/model_307096_796.pth')