from ppo_agent_atari import AtariPPOAgent
import time

if __name__ == '__main__':
	timestamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
	config = {
		"gpu": True,
		"training_steps": 1e8,
		"update_sample_count": 10000,
		"discount_factor_gamma": 0.99,
		"discount_factor_lambda": 0.95,
		"clip_epsilon": 0.2,
		"max_gradient_norm": 0.5,
		"batch_size": 128,
		"logdir": f'log/Enduro/{timestamp}/',
		"update_ppo_epoch": 3,
		"learning_rate": 2.5e-4,
		"value_coefficient": 0.5,
		"entropy_coefficient": 0.01,
		"horizon": 128,
		"env_id": 'ALE/Enduro-v5',
		"eval_interval": 100,
		"eval_episode": 5,
		"agent_count": 1,
		"seed": 0,
	}
	agent = AtariPPOAgent(config)
	load_path = 'log/Enduro/best/model_4698222_2346.pth'
	frame_list = agent.load_and_evaluate(load_path)
	import moviepy.editor as mpy
	for i, frames in enumerate(frame_list):
		print(f"Saving video of episode {i+1}...")
		clip = mpy.ImageSequenceClip(frames, fps=30)
		clip.write_videofile(f'log/Enduro/best/video_{i}.mp4')



