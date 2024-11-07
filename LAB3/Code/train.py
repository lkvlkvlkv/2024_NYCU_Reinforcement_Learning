from ppo_agent_atari import AtariPPOAgent
import time

if __name__ == '__main__':
	timestamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
	config = {
		"gpu": True,
		"training_steps": 1e7,
		# "training_steps": 5e6,
		# "training_steps": 1e6,
		"update_sample_count": 10000,
		"discount_factor_gamma": 0.99,
		"discount_factor_lambda": 0.95,
		"clip_epsilon": 0.2,
		"max_gradient_norm": 0.5,
		"batch_size": 256,
		# "batch_size": 128,
		"logdir": f'log/Enduro/{timestamp}/',
		"update_ppo_epoch": 3,
		"learning_rate": 2.5e-4,
		# "learning_rate": 1e-5,
		# "learning_rate": 1e-7,
		"value_coefficient": 0.5,
		"entropy_coefficient": 0.01,
		"horizon": 128,
		"env_id": 'ALE/Enduro-v5',
		"eval_interval": 100,
		"eval_episode": 1,
		"agent_count": 8,
		"seed": 0,
	}
	agent = AtariPPOAgent(config)
	agent.train()



