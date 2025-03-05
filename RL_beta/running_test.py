import time
start_time = time.time()

import os
import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np


from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3 import PPO, A2C

from stable_baselines_RL import ParkingGarageEnv

env = DummyVecEnv([lambda: ParkingGarageEnv()])
env = VecCheckNan(env, raise_exception=True)

model = A2C('MlpPolicy', env, verbose=0)


model.learn(total_timesteps=2000, log_interval=4, progress_bar=True)

save_path = os.path.join('Training', 'Saved Models', 'Betting_Model_A2C')
model.save(save_path)

end_time = time.time()
total_time = end_time - start_time

print(round(total_time / 60 / 60), ' Hours ', round(total_time / 60), ' Minutes')