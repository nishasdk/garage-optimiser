# Parking Garage environment(s)
#
# Created on May 2023
#
# Modified from Stable Baselines tutorials
#

from math import floor
from typing import Optional
import gymnasium as gym
from gymnasium import spaces
from config import time_arr, seed_no, floor_min, floor_max, space_per_floor, price, cost_ops, growth_factor, cost_land, cost_construction, rate_discount
from objective_funcs import demand_stochastic, cost_construction_initial
from decision_rule_funcs import expansion_cost
import numpy as np
from numpy_financial import npv
from stable_baselines3.common.env_checker import check_env

"""
seed_no = 123 # means script always selects the same N scenarios. N is defined by sims
np.random.seed(seed_no)

sims = 10
scenarios = np.random.choice(sims,size=sims,replace=False)
demand = np.zeros((sims,time_arr[-1]+1))
for i in range(sims):
    demand[i,:] = demand_stochastic(time_arr,scenarios[i])
    # demand[0,:] is the 20 year demand distribution for scenario 1
    # demand[:,0] is the demand at year 0 for all scenarios
    
def expansion_cost(current_capacity, action):
        return cost_construction * space_per_floor * (((1 + growth_factor) ** action - 1) / growth_factor * (1 + growth_factor) ** (current_capacity / space_per_floor - 1))
"""


class ParkingGarageEnv(gym.Env):
    
    metadata = {
        "render_modes": ["human","rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(self):
        super(ParkingGarageEnv, self).__init__()
        self.max_years = 20  # Total number of years
        self.year = 0  # Current year
        self.max_capacity = floor_max * space_per_floor
        self.min_capacity = floor_min * space_per_floor
        self.max_cashflow = self.max_capacity * price - self.max_capacity * cost_ops - cost_land
        self.min_cashflow = cost_land - cost_construction_initial(3)
        self.reward_arr = np.zeros(20)
        self.demand_arr = demand_stochastic(time_arr,1)
        
        self.action_space = spaces.Discrete(3)  # Actions: 0 - Expand by 0, 1 - Expand by 1, 2 - Expand by 2 ...

        # Observation space = current capacity, current demand, current cash flow
        # We start actions at year 1 so the first pairs will be years 0 to 1. Iterate over 20 years, the last pairs are years 19 to 20.
        self.observation_space = spaces.Box(
            low=np.array([self.min_capacity, np.amin(self.demand_arr), self.min_cashflow]), 
            high=np.array([self.max_capacity, np.amax(self.demand_arr), self.max_cashflow]), 
            shape=(3,), dtype=np.float32)  # Demand for parking

    def step(self, action):
        # sourcery skip: remove-unnecessary-else, remove-unreachable-code, swap-if-else-branches
        # assert self.action_space.contains(action), f"Invalid action: {action}"
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        assert self.state is not None, "Call reset before using step method."

        current_capacity, current_demand, current_cashflow = self.state
        current_demand = self.demand_arr[self.year]
        
        if current_capacity + action * space_per_floor < self.max_capacity:
            capacity = current_capacity + action * space_per_floor 
        else:
            capacity = self.max_capacity

        # Rewards based on action and demand

        if self.year == 0:
            init_cost = cost_construction_initial(capacity/space_per_floor)
            self.reward_arr[0] = - (cost_land + init_cost) / (10**6)
            reward = 0
            current_cashflow = 0
            self.year += 1

        elif self.year == self.max_years:
            current_cashflow = (min(capacity, current_demand)*price - capacity*cost_ops - cost_land) / (10**6)
            self.reward_arr[self.year] =current_cashflow
            reward = npv(rate_discount,self.reward_arr) 

        else:
            current_cashflow = min(capacity, current_demand)*price - capacity*cost_ops - cost_land - expansion_cost(current_capacity,action)
            self.reward_arr[self.year] = current_cashflow / (10**6)
            reward = current_cashflow / ((1+rate_discount) ** self.year)
            self.year += 1

        self.state = capacity, current_demand, current_cashflow
        done = self.year == self.max_years
        return np.array(self.state,dtype=np.float32), reward, done, False, {}

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        self.year = 0
        self.demand_arr = demand_stochastic(time_arr,seed)
        current_capacity = self.min_capacity
        current_demand = self.demand_arr[self.year]
        current_cashflow = 0
        self.state = current_capacity, current_demand, current_cashflow
        
        return np.array(self.state,dtype=np.float32), {}


    def render(self, mode='human'):
        pass
    

# # Check the environment with SB env checker
# env = ParkingGarageEnv()
# # If the environment don't follow the interface, an error will be thrown
# check_env(env, warn=True)

# from stable_baselines3 import A2C
# from stable_baselines3.common.env_util import make_vec_env

# env = ParkingGarageEnv()
# #use gym.make to create multiple environments
# vec_env = make_vec_env(lambda: env, n_envs=1)

# model = A2C("MlpPolicy", vec_env, learning_rate = 0.1, verbose=1,tensorboard_log="./a2c_pg/")
# model.learn(total_timesteps=25000, log_interval=4, progress_bar=True)


# reward_arr = np.zeros([20])

# n_steps = 100
# # Test the trained agent
# obs = vec_env.reset()
# for step in range(n_steps):
#     action, _ = model.predict(obs, deterministic=False)
#     print(f"Year {step + 1}")
#     print("Action: ", action)
#     obs, reward, done, truncated, info = env.step(int(action))
#     print('obs=', obs, 'reward=', reward, 'done=', done)
#     env.render(mode='console')
#     if done:
#         # Note that the VecEnv resets automatically
#         # when a done signal is encountered
#         print("Goal reached!", "reward=", reward)
#         break

# print(reward)