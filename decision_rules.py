"""
Decision rule functions for parking garage case study

N Saduagkan, Feb 2023
@nishasdk
"""

import config
import numpy as np


def design_variable_check(t:int) -> bool:
    
    #conditions between certain times where expansion is allowed, then return True if it can expand
    if 1<t<4 and config.y1_4expand:
        return True
    elif 9<t<12 and config.y9_12expand:
        return True
    elif 17<t<20 and config.y17_20expand:
        return True
    elif 5<t<9 or 13<t<17:
        return True
    else:
        return False

def check_capacity_limit(capacity: int, floor_expansion: int) -> bool:
    """ Check that maximum capacity constraint has not been reached - another floor can be built

    Args:
        capacity (np.array): capacity array
        t (int): time where the condition is checked (year)

    Returns:
        bool: True = can expand
    """ 
    
    return (capacity + floor_expansion * config.space_per_floor) <= config.floor_max * config.space_per_floor


def check_expansion_criterion(capacity: np.array, demand: np.array, t: int, year_threshold: int, capacity_threshold: float) -> bool:
    """Check that the demand in the last n years has been met. Controlled by year_threshold (design variable)

    Args:
        capacity (np.array): array of capacity over lifespan
        demand (np.array): array of demand over lifespan 
        t (int): time where condition is being checked (year)

    Returns:
        bool: True = can expand if within x% of the capacity - change capacity_threshold. 
    """
    total_demand = np.sum(demand[(t - year_threshold):t]) #demand in the last n years
    total_capacity = np.sum(capacity[(t - year_threshold):t]) #capacity in the last n years
    return total_demand >= capacity_threshold * total_capacity

def expansion_cost(floor_expansion: int, capacity: int) -> float:
    """calculates the cost of expanding (past 2 floors)

    Args:
        floor_expansion (int): how many floors to expand by
        capacity (int): current capacity 
                        (when nested in decision_rules.capacity_update, will take a value from capacity array)

    Returns:
        float: cost to expand the specified amount of floors
    """
    return config.cost_construction * config.space_per_floor *(((((1 + config.growth_factor) ** floor_expansion) -1) / config.growth_factor) * ((1 + config.growth_factor) ** ((capacity/config.space_per_floor) -1)))

def capacity_update(capacity:np.array, cost_expansion: np.array, demand: np.array, floor_expansion: int, year_threshold: int, capacity_threshold: float) -> np.array:
    """Runs through decision rules for expansion at each year and returns updated arrays. TEST WITH: capacity_update(capacity,cost_expansion,demand,1,2,1)

    Args:
        capacity (np.array): capacity array, gets updated every iteration
        cost_expansion (np.array): cost array, also updates
        demand (np.array): demand array - constant
        floor_expansion (int): how many floors to expand by
        year_threshold (int): n years to check demand
        capacity_threshold (float): usually 0 < x < 1, determines which % of capacity has to be filled for the past n years to expand

    Returns:
        np.array: outputs final capcity with expansion and the cost of expansion
    """
    
    for t in range(1,config.time_lifespan): 
        #check that it hasn't reached max expansion
        if check_capacity_limit(capacity[t-1], floor_expansion) and check_expansion_criterion(capacity,demand,t,year_threshold,capacity_threshold) and design_variable_check(t) and t > year_threshold:
            #check that in the past n year(s), demand has been sufficient as to trigger expansion
            capacity[t] = capacity[t-1] + floor_expansion * config.space_per_floor
            cost_expansion[t] = expansion_cost(floor_expansion,capacity[t])
            
            
        elif check_expansion_criterion(capacity,demand,t,year_threshold,capacity_threshold) and not(check_capacity_limit(capacity[t-1], floor_expansion)) and design_variable_check(t):
            #expand to max floors when floor_expansion is not a multiple of the max
            capacity[t] = config.floor_max * config.space_per_floor
            expansion_amount = (capacity[t] - capacity[t-1])/config.space_per_floor
            cost_expansion[t] = expansion_cost(expansion_amount,capacity[t])
            
        else:
            #if demand is not sufficient then no expansion, capacity remains the same
            capacity[t] = capacity[t-1]
    
    capacity[-1] = capacity[-2]
    return capacity, cost_expansion
