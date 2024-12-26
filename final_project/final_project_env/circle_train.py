import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, VecTransposeImage, VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from racecar_gym.my_env import RaceEnv
from stable_baselines3 import PPO
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation
import time


def make_env(scenario='austria_competition', random_start=True, **kwargs):
    def thunk():
        env = RaceEnv(
            scenario=scenario,
            render_mode='rgb_array_birds_eye',
            random_start=random_start,
            reset_when_collision=False,
            **kwargs
        )
        env = ResizeObservation(env, shape=(84, 84))
        env = GrayScaleObservation(env, keep_dim=True)
        return env
    return thunk


class ScoreEvalCallback(EvalCallback):
    def __init__(self, eval_env, verbose=1, **kwargs):
        super(ScoreEvalCallback, self).__init__(eval_env, verbose=verbose, **kwargs)
        self.best_mean_score = -np.inf
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.eval_freq == 0 or self.n_calls % self.eval_freq != 0:
            return True

        obs = self.eval_env.reset()
        total_rewards = np.zeros(self.eval_env.num_envs)
        episode_lengths = np.zeros(self.eval_env.num_envs)
        dones = [False] * self.eval_env.num_envs
        scores = np.zeros(self.eval_env.num_envs)
        while not all(dones):
            actions, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, rewards, done_flags, infos = self.eval_env.step(actions)

            total_rewards += rewards
            for idx, done in enumerate(done_flags):
                if done:
                    info = infos[idx]
                    episode_lengths[idx] = info["episode"]["l"]
                    if "lap" in info and "progress" in info:
                        scores[idx] = info["lap"] + info["progress"] - 1

            dones = [d or done for d, done in zip(dones, done_flags)]

        mean_length = np.mean(episode_lengths)
        self.logger.record("eval/mean_length", mean_length)

        mean_reward = np.mean(total_rewards)
        self.logger.record("eval/mean_reward", mean_reward)

        mean_score = np.mean(scores)
        self.logger.record("eval/mean_progress", mean_score)

        if self.best_mean_reward < mean_reward:
            self.best_mean_reward = mean_reward
            if self.verbose > 0:
                print(f"New best reward: {self.best_mean_reward}")
            
            if self.best_model_save_path is not None:
                self.model.save(f"{self.best_model_save_path}/best_reward_model")
                if self.verbose > 0:
                    print(f"Saving new best reward model to {self.best_model_save_path}")

        if mean_score > self.best_mean_score:
            self.best_mean_score = mean_score
            if self.verbose > 0:
                print(f"New best score: {self.best_mean_score}")

            if self.best_model_save_path is not None:
                self.model.save(f"{self.best_model_save_path}/best_progress_model")
                if self.verbose > 0:
                    print(f"Saving new best progress model to {self.best_model_save_path}")

        return True


if __name__ == '__main__':
    scenario = 'circle_cw_competition_collisionStop_v3'
    # scenario = 'austria_competition_training_v13'
    total_timesteps = 1e7
    agent_count = 2
    eval_count = 5

    # train_env = SubprocVecEnv([make_env() for _ in range(agent_count)])
    train_env = DummyVecEnv([make_env(scenario=scenario, random_start=True) for _ in range(agent_count)])
    train_env = VecTransposeImage(train_env)
    train_env = VecFrameStack(train_env, n_stack=4, channels_order='first')
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=False)
    train_env = VecMonitor(train_env)

    test_env = SubprocVecEnv([make_env(scenario='circle_cw_competition_collisionStop',random_start=False) for _ in range(eval_count)])
    test_env = VecTransposeImage(test_env)
    test_env = VecFrameStack(test_env, n_stack=4, channels_order='first')
    test_env = VecNormalize(test_env, norm_obs=False, norm_reward=False)
    test_env = VecMonitor(test_env)

    score_callback = ScoreEvalCallback(
        eval_env=test_env,
        n_eval_episodes=1,
        eval_freq=10000,
        deterministic=True,
        best_model_save_path=f'models/{time.strftime("%m%d_%H%M")}/',
    )
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=f'models/{time.strftime("%m%d_%H%M")}/', name_prefix='freq_saving')


    model = PPO(
        'CnnPolicy',
        env=train_env,
        learning_rate=0.0001,
        verbose=1,
        tensorboard_log='logs/tensorboard/',
        batch_size=128,
    )

    model.load('models/circle/best_progress_model')
    model.learn(total_timesteps=total_timesteps, callback=[score_callback, checkpoint_callback])
    model.save('models/final_model')
