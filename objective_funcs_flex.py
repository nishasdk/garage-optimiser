"""
Flexible Objective functions

N Saduagkan, Mar 2023
@nishasdk
"""

import objective_funcs
from pandas import array
import config
import numpy as np
from numpy_financial import npv
from decision_rules import capacity_update
import typing


def cashflow_array_flex_det(floor_initial: float,y1_4expand:bool, y9_12expand:bool, y17_20expand:bool,floor_expansion: int, year_threshold: int, capacity_threshold: float) -> np.array:

    # initialise the cashflow array
    cashflow = np.full((config.time_lifespan+1), -(objective_funcs.cost_construction_initial(floor_initial) + config.cost_land), dtype='float64')
    # initialise capacity array
    capacity = np.full((config.time_lifespan+1),floor_initial * config.space_per_floor)
    #initialise expansion cost
    cost_expansion = np.zeros(config.time_lifespan+1)
    
    demand = objective_funcs.demand_deterministic(config.time_arr)
    
    capacity, cost_expansion = capacity_update(capacity,cost_expansion,demand,y1_4expand,y9_12expand,y17_20expand,floor_expansion,year_threshold,capacity_threshold)
    
    for t in range(1, config.time_lifespan):
        cashflow[t] = min(capacity[t], demand[t])*config.price - capacity[t]*config.cost_ops - config.cost_land - cost_expansion[t]
    cashflow[-1] = min(capacity[-1], demand[-1])*config.price - capacity[-1]*config.cost_ops
    return cashflow


def enpv_flex_det(floor_initial: float, y1_4expand:bool, y9_12expand:bool, y17_20expand:bool,floor_expansion: int, year_threshold: int, capacity_threshold: float) -> np.array:

    cashflow_det = cashflow_array_flex_det(floor_initial,y1_4expand,y9_12expand,y17_20expand,floor_expansion,year_threshold,capacity_threshold)
    npv_det= npv(config.rate_discount,cashflow_det)

    enpv_det = np.mean(npv_det)
    from millify import millify
    print('ENPV £' + str(millify(enpv_det,precision=2)))

    return enpv_det , npv_det

def enpv_flex_det_opti(floor_initial: float, y1_4expand:bool, y9_12expand:bool, y17_20expand:bool,floor_expansion: int, year_threshold: int, capacity_threshold: float) -> np.array:

    cashflow_det = cashflow_array_flex_det(floor_initial,y1_4expand,y9_12expand,y17_20expand,floor_expansion,year_threshold,capacity_threshold)
    npv_det= npv(config.rate_discount,cashflow_det)
    
    return -np.mean(npv_det)

def cashflow_array_flex(floor_initial: float,y1_4expand:bool, y9_12expand:bool, y17_20expand:bool,floor_expansion: int, year_threshold: int, capacity_threshold: float,seed_number: int) -> np.array:

    # initialise the cashflow array
    cashflow = np.full((config.time_lifespan+1), -(objective_funcs.cost_construction_initial(floor_initial) + config.cost_land), dtype='float64')
    # initialise capacity array
    capacity = np.full((config.time_lifespan+1),floor_initial * config.space_per_floor)
    # initialise demand scenarios
    demand = objective_funcs.demand_stochastic(config.time_arr,seed_number)
    #initialise expansion cost
    cost_expansion = np.zeros(config.time_lifespan+1)
    
    capacity, cost_expansion = capacity_update(capacity,cost_expansion,demand,y1_4expand,y9_12expand,y17_20expand,floor_expansion,year_threshold,capacity_threshold)
    
    for t in range(1, config.time_lifespan):
        cashflow[t] = min(capacity[t], demand[t])*config.price - capacity[t]*config.cost_ops - config.cost_land - cost_expansion[t]
    cashflow[-1] = min(capacity[-1], demand[-1])*config.price - capacity[-1]*config.cost_ops
    return cashflow

def enpv_flex(floor_initial: float, y1_4expand:bool, y9_12expand:bool, y17_20expand:bool,floor_expansion: int, year_threshold: int, capacity_threshold: float) -> np.array:

    #initialise the arrays
    cashflow_stoc = np.zeros(config.time_lifespan+1,dtype='float64')
    npv_stoc = np.zeros(config.sims,dtype='float64')
    
    
    for instance in range(config.sims):
            cashflow_stoc = cashflow_array_flex(floor_initial,y1_4expand,y9_12expand,y17_20expand,floor_expansion,year_threshold,capacity_threshold,seed_number=config.scenarios[instance])
            npv_stoc[instance] = npv(config.rate_discount,cashflow_stoc)

    enpv_stoc = np.mean(npv_stoc)
    from millify import millify
    print('ENPV £' + str(millify(enpv_stoc,precision=2)))

    return enpv_stoc, npv_stoc

def enpv_flex_opti(floor_initial: float, y1_4expand:bool, y9_12expand:bool, y17_20expand:bool,floor_expansion: int, year_threshold: int, capacity_threshold: float) -> np.array:

    cashflow_stoc = np.zeros(config.time_lifespan+1,dtype='float64')
    npv_stoc = np.zeros(config.sims,dtype='float64')
    for instance in range(config.sims):
            cashflow_stoc = cashflow_array_flex(floor_initial,y1_4expand,y9_12expand,y17_20expand,floor_expansion,year_threshold,capacity_threshold,seed_number=config.scenarios[instance])
            npv_stoc[instance] = npv(config.rate_discount,cashflow_stoc)

    return -np.mean(npv_stoc)