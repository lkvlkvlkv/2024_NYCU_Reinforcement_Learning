from dqn_agent_atari import AtariDQNAgent
from ddqn_agent_atari import AtariDDQNAgent
from dueling_dqn_agent_atari import AtariDuelingDQNAgent
from parallelized_dqn_agent_atari_without_experimental import AtariDQNAgent as ParallelizedAtariDQNAgent
import time

game, agent_type, model, seed = 'MsPacman', 'DQN', 'dqn', 9639
# game, agent_type, model, seed = 'MsPacman', 'DDQN', 'ddqn', 3300
# game, agent_type, model, seed = 'MsPacman', 'DuelingDQN', 'dueling_dqn', 3
# game, agent_type, model, seed = 'MsPacman', 'ParallelizedDQN', 'parallelize_dqn', 2421

# game, agent_type, model, seed = 'Enduro', 'DQN', 'dqn', 3356
# game, agent_type, model, seed = 'Enduro', 'DuelingDQN', 'dueling_dqn', 9826

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

	score, frames = agent.load_and_evaluate(path, seed, True)


	for idx, frame_list in enumerate(frames):
		import moviepy.editor as mpy
		clip = mpy.ImageSequenceClip(frame_list, fps=30)
		import os
		os.makedirs(f'demo/{game}/{agent_type}', exist_ok=True)
		clip.write_videofile(f'demo/{game}/{agent_type}/video_{idx}.mp4')