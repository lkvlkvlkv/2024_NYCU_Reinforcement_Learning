import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from stable_baselines3 import PPO
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import numpy as np
from collections import deque


class PPOAgent:
    def __init__(self, action_space, model_path):
        self.action_space = action_space
        self.model = PPO.load(model_path)
        self.buffer = deque(maxlen=4)

    def act(self, observation):
        observation = self.preprocessing(observation)
        stacked_obs = self.stack_obs(observation)
        return self.model.predict(stacked_obs)[0]

    def stack_obs(self, observation):
        self.buffer.append(observation)
        while len(self.buffer) < 4:
            self.buffer.append(observation)
        ret = np.array(self.buffer)
        return ret

    def preprocessing(self, obs):
        # transpose
        obs = np.transpose(obs, (1, 2, 0))
        # resize
        import cv2
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        # grayscale
        obs = np.sum(
            np.multiply(obs, np.array([0.2125, 0.7154, 0.0721])), axis=-1
        ).astype(np.uint8)
        return obs
