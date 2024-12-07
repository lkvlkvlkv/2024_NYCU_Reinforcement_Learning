import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, VecTransposeImage, VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from racecar_gym.my_env import RaceEnv
from sbx import PPO
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation


def make_env(random_start=True, **kwargs):
    def thunk():
        env = RaceEnv(
            scenario='austria_competition',
            render_mode='rgb_array_birds_eye',
            random_start=random_start,
            **kwargs
        )
        env = ResizeObservation(env, shape=(84, 84))
        env = GrayscaleObservation(env, keep_dim=True)
        return env
    return thunk


class ScoreEvalCallback(EvalCallback):
    def __init__(self, eval_env, verbose=1, **kwargs):
        super(ScoreEvalCallback, self).__init__(eval_env, verbose=verbose, **kwargs)
        self.best_mean_score = -np.inf

    def _on_step(self) -> bool:
        continue_training = super(ScoreEvalCallback, self)._on_step()

        if self.eval_freq == 0 or self.n_calls % self.eval_freq != 0:
            return continue_training

        print('calling _on_step')
        obs = self.eval_env.reset()
        dones = [False] * self.eval_env.num_envs
        scores = np.zeros(self.eval_env.num_envs)
        while not all(dones):
            actions, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, _, done_flags, infos = self.eval_env.step(actions)

            # 只提取剛完成的 episode 的資訊
            for idx, done in enumerate(done_flags):
                if done:
                    info = infos[idx]
                    if "lap" in info and "progress" in info:
                        scores[idx] = info["lap"] + info["progress"] - 1

            dones = [d or done for d, done in zip(dones, done_flags)]

        # 計算平均值
        mean_score = np.mean(scores)
        self.logger.record("eval/mean_progress", mean_score)

        # 更新最佳模型依據
        if mean_score > self.best_mean_score:
            self.best_mean_score = mean_score
            if self.verbose > 0:
                print(f"New best score: {self.best_mean_score}")
            # 保存最佳模型
            if self.best_model_save_path is not None:
                self.model.save(f"{self.best_model_save_path}/best_model")
                if self.verbose > 0:
                    print(f"Saving new best model to {self.best_model_save_path}")

        return continue_training


if __name__ == '__main__':
    total_timesteps = 1e7
    agent_count = 1

    # train_env = SubprocVecEnv([make_env() for _ in range(agent_count)])
    train_env = SubprocVecEnv([make_env() for _ in range(agent_count)])
    train_env = VecTransposeImage(train_env)
    train_env = VecFrameStack(train_env, n_stack=4, channels_order='first')
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    train_env = VecMonitor(train_env)

    test_env = SubprocVecEnv([make_env(random_start=False) for _ in range(5)])
    test_env = VecTransposeImage(test_env)
    test_env = VecFrameStack(test_env, n_stack=4, channels_order='first')
    test_env = VecNormalize(test_env, norm_obs=True, norm_reward=True)
    test_env = VecMonitor(test_env)

    score_callback = ScoreEvalCallback(
        eval_env=test_env,
        n_eval_episodes=1,
        # eval_freq=50000,
        eval_freq=1000,
        deterministic=True,
        best_model_save_path='models/',
    )

    model = PPO(
        'MlpPolicy',
        env=train_env,
        learning_rate=0.0001,
        verbose=1,
        tensorboard_log='logs/tensorboard/',
    )

    model.learn(total_timesteps=total_timesteps, callback=[score_callback])
    model.save('models/final_model')
