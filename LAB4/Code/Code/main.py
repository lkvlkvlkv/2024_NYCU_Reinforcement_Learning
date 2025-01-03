from td3_agent_CarRacing import CarRacingTD3Agent
import time

if __name__ == '__main__':
	timestamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
	# my hyperparameters, you can change it as you like
	config = {
		"gpu": True,
		"training_steps": 1e8,
		"gamma": 0.99,
		"tau": 0.005,
		"batch_size": 32,
		"warmup_steps": 1000,
		"total_episode": 100000,
		"lra": 4.5e-5,
		"lrc": 4.5e-5,
		"replay_buffer_capacity": 5000,
		"logdir": f'log/CarRacing/td3_test/{timestamp}/',
		"update_freq": 1,
		"update_target_freq": 2,
		"eval_interval": 10,
		"eval_episode": 1,
		"policy_noise": 0.05,
		"noise_clip_ratio": 0.1,
		"twin_q_network": True,
		"target_policy_smoothing": True,
		"delayed_policy_update": True,
		"brake_rate": 0.015
	}
	agent = CarRacingTD3Agent(config)
	# agent.train()
	load_path = 'log/CarRacing/best/model_2214072_915.pth'
	seeds = [3128, 6727, 8843, 7021, 2712]
	frame_list = agent.load_and_evaluate(load_path, seeds=seeds)
	import moviepy.editor as mpy
	for i, frames in enumerate(frame_list):
		clip = mpy.ImageSequenceClip(frames, fps=30)
		clip.write_videofile(f'log/CarRacing/best/seed_{seeds[i]}.mp4')
