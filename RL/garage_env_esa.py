
"""
Created on Wed May  6 16:45:12 2020

Editted 3rd May 2023, @nishasdk

@author: cesa_
"""
import gym
from gym import spaces
from decision_rules.config import time_arr, seed_no, floor_min, floor_max, space_per_floor, price, cost_ops, rate_discount, cost_land
from decision_rules.objective_funcs import demand_stochastic, cost_construction_initial
from decision_rules.decision_rule_funcs import expansion_cost
import numpy as np


MAX_CAP = floor_max * space_per_floor
MIN_CAP = floor_min * space_per_floor


class GarageEnvFull(gym.Env):

    def __init__(self):
        # Demand and capacity data
        self.action_space = spaces.Discrete(4) # No expansion, expand 1,2, or 3 floors
        self.count = 0
        self.time_steps = 0
        self.current_capacity = 0 
        
        #set state
        self.current_demand = demand_stochastic(time_arr,seed_no) #start with static demand only
        self.state = self.current_capacity, self.current_demand

    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        current_capacity, current_demand = self.state
        # Capacity is restricted to max level even if agent chooses expansion
        if action == 0:
            possible_next_capacity = current_capacity
            E_cost = 0
        elif action == 1:
            if current_capacity + action * space_per_floor < MAX_CAP:         
                E_cost = expansion_cost(action,current_capacity)
                possible_next_capacity = current_capacity + 200
            else : 
                E_cost = 0
                possible_next_capacity = MAX_CAP
        elif action == 2:
            if current_capacity + action * space_per_floor < MAX_CAP:         
                E_cost = expansion_cost(action,current_capacity)
                possible_next_capacity = current_capacity + 400
            elif self.current_capacity == MAX_CAP:
                E_cost = 0
                possible_next_capacity = MAX_CAP
            else:
                E_cost = expansion_cost(1,current_capacity)
                possible_next_capacity = current_capacity + 200
        elif action == 3:
            if current_capacity + action * space_per_floor < MAX_CAP:         
                E_cost = expansion_cost(action,current_capacity)
                possible_next_capacity = current_capacity + 600
            elif self.current_capacity == MAX_CAP:
                E_cost = 0
                possible_next_capacity = MAX_CAP
            else:
                expansion = (MAX_CAP - self.current_capacity)/space_per_floor
                E_cost = expansion(expansion,current_capacity)
# Bring forward to next time step in episode and calculate reward
        self.time_steps +=1
        self.current_capacity = possible_next_capacity
        self.current_demand = demand_stochastic(time_arr,seed_no) # bring forward timestep
        self.state = (self.current_capacity, self.current_demand)

        # Termination conditions: true if end
        done = self.time_steps == time_arr[-1]
        
        # Include construction cost for initial f0 design decision    
        if self.time_steps == 0:
            revenue = -cost_construction_initial(self.current_capacity/MAX_CAP)
        # Imaginary reward penalty for violating max capacity contraint REMOVED in testing environment
        elif possible_next_capacity == MAX_CAP and action != 0:
            revenue = -1
        else:
            revenue = np.minimum(self.current_capacity, self.current_demand)*price

        reward = (revenue - self.current_capacity * cost_ops - cost_land -E_cost)/((1+rate_discount)**self.time_steps)

        return np.array(self.state), reward, done, {}
    
    def render(self, mode='human'):
        pass
        
    def reset(self):
        self.time_steps = 0
        self.current_capacity = 0
        #reinitialize random demand curve profile
        self.rD0s = np.random.random_sample()
        self.rD10s = np.random.random_sample()
        self.rDfs = np.random.random_sample()
        #set state
        self.current_demand = demand_stochastic(time_arr,seed_no)
        self.state = (self.current_capacity, self.current_demand)
        return np.array(self.state)
    
    

    
def agent_policy(rand_generator):
    """
    Given random number generator and state, returns an action according to the agent's policy
    Returns:
        chosen action [int]
    """
    
    # set chosen_action as 0 or 1 with equal probability
    # state is unnecessary for this agent policy
    chosen_action = rand_generator.choice([0,1,2,3])
    
    #TODO greedy action??
    
    return chosen_action 
