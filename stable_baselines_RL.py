# Parking Garage environment(s)
#
# Created on May 2023
#
# Modified from Stable Baselines tutorials
#

from math import floor
from typing import Optional
import gym
from gym import spaces
from config import time_arr, seed_no, floor_min, floor_max, space_per_floor, price, cost_ops, growth_factor, cost_land, cost_construction, rate_discount
from objective_funcs import demand_stochastic, cost_construction_initial
from decision_rule_funcs import expansion_cost
import numpy as np
from numpy_financial import npv
from stable_baselines3.common.env_checker import check_env

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



class ParkingGarageEnv(gym.Env):
    def __init__(self):
        super(ParkingGarageEnv, self).__init__()
        self.max_years = 20  # Total number of years
        self.year = 0  # Current year
        self.sim = 1
        self.floor_initial = 2
        self.initial_capacity = self.floor_initial * space_per_floor
        self.max_capacity = floor_max * space_per_floor
        self.reward_arr = np.zeros(20)
        self.reward_arr[0] = cost_construction_initial(self.floor_initial)
        
        self.action_space = spaces.Discrete(3)  # Actions: 0 - Expand by 0, 1 - Expand by 1, 2 - Expand by 2

        # Observation space = current capacity, current demand.
        # We start actions at year 1 so the first pairs will be years 0 and 1. Iterate over 20 years, the last pairs are years 19 and 20.
        self.observation_space = spaces.Box(low=np.array([self.initial_capacity, np.amin(demand)]), high=np.array([self.max_capacity, np.amax(demand)]), dtype=np.float32)  # Demand for parking
        self.state = None

    def step(self, action):
        # sourcery skip: remove-unnecessary-else, remove-unreachable-code, swap-if-else-branches
        # assert self.action_space.contains(action), f"Invalid action: {action}"
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        assert self.state is not None, "Call reset before using step method."

        current_capacity, current_demand = self.state  # Demand for parking in the current year
        current_demand = self.demand_arr[self.year]

        # Rewards based on action and demand

        if action == 0:
            reward = min(current_capacity, current_demand)*price - current_capacity*cost_ops - cost_land
            current_capacity = current_capacity
            current_demand = self.demand_arr[self.year+1]
        elif action in [1, 2]:
            if current_capacity + action*space_per_floor < self.max_capacity:
                reward = min(current_capacity, current_demand)*price - current_capacity*cost_ops - cost_land - expansion_cost(current_capacity,action)
                current_capacity += action * space_per_floor
            else:
                current_capacity = current_capacity
                reward = min(current_capacity, current_demand)*price - current_capacity*cost_ops - cost_land
            current_demand = self.demand_arr[self.year+1]

        self.reward_arr[self.year] = reward
        self.year += 1  # Increment the year
        
        self.state = current_capacity, current_demand
        reward = npv(rate_discount,self.reward_arr)
        
        # Check if the episode is done
        if self.year == self.max_years:
            self.sim += 1
            done = True
        else:
            done = False


        return np.array(self.state,dtype=np.float32), reward, done, {}

    def reset(self):
        self.year = 0
        self.demand_arr = demand[1,:]
        current_capacity = self.initial_capacity
        current_demand = self.demand_arr[self.year]
        self.state = current_capacity, current_demand
        
        return np.array(self.state,dtype=np.float32)


    def render(self, mode='human'):
        pass
    

# Check the environment with SB env checker
env = ParkingGarageEnv()
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env


n_steps = 20

env = ParkingGarageEnv()
#use gym.make to create multiple environments
vec_env = make_vec_env(lambda: env, n_envs=1)

model = A2C("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)


reward_arr = np.zeros([n_steps])

# Test the trained agent
obs = vec_env.reset()
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print(f"Year {step + 1}")
    print("Action: ", action)
    obs, reward, done, info = env.step(int(action))
    print('obs=', obs, 'reward=', reward, 'done=', done)
    env.render(mode='console')
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break
    reward_arr[step] = reward

print(reward_arr)
print(np.sum(reward_arr[:20]))