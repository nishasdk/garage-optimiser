"""
Calculates demand and cost for parking garage case study

N Saduagkan, Feb 2023
@nishasdk
"""

import config
import numpy as np
from numpy_financial import npv
import typing

def demand_deterministic(time_arr: np.array) -> np.array:
    """Function to calculate demand projection

    Args:
        time_arr (np.array): array starting at 0, ending at time_lifespan

    Returns:
        np.array: deterministic demand
    """
    # Parameter for demand model showing difference between initial and final demand values
    alpha = config.demand_10 + config.demand_20
    # Parameter for demand model showing growth speed of demand curve
    beta = -np.log(config.demand_20 / alpha) / (config.time_lifespan / 2 - 1)
    demand = config.demand_initial + config.demand_10 + config.demand_20 - alpha * np.exp(-beta * (time_arr - 1))
    return demand


def demand_stochastic(time_arr: np.array, seed_number: int) -> np.array:
    """function for calculating the stochastic demand (edited from @cesa_, change explanations commented)

    Args:
        time_arr (np.array): array starting at 0, ending at time_lifespan
        seed_number (int): random seed number for new simulation

    Returns:
        np.array: stochastic demand
    """
    # set constant seed for simulations to for standardized comparison
    np.random.seed(seed_number) #demand scenario with this seed will always be the same
    
    rD0 = round((1 - config.off_D0) * config.demand_initial +np.random.rand() * 2 * config.off_D0 * config.demand_initial)  # Realised demand in year 0

    rD10 = round((1 - config.off_D10) * config.demand_10 + np.random.rand() * 2 * config.off_D10 * config.demand_10)  # Realised additional demand years 0-10

    rDf = round((1 - config.off_Dfinal) * config.demand_20 + np.random.rand() * 2 * config.off_Dfinal * config.demand_20)  # Realised additional demand years 10-20

    # Parameter for demand model showing difference between initial and final demand values
    alpha_stoc = rD10 + rDf

    # Parameter for demand model showing growth speed of demand curve
    beta_stoc = -np.log(rDf / alpha_stoc) / (config.time_lifespan / 2 - 1)

    D_stoc = np.zeros(config.time_lifespan+1) #initialise
    
    D_stoc[1:config.time_lifespan+1] = (rD0 + rD10 + rDf - alpha_stoc *np.exp(-beta_stoc * (time_arr[1:config.time_lifespan+1] - 1)))  # projected demand vector

    for i in np.arange(2,config.time_lifespan+1):
        # projected demand vector shifted by one period to right
        D_g_proj = D_stoc[i] / D_stoc[i-1] - 1
        R_g = D_g_proj - config.volatility + np.random.rand() * 2 * config.volatility
        D_stoc[i] = D_stoc[i-1] * (1 + R_g)

    return D_stoc


def cost_construction_initial(floor_initial: float) -> float:
    """initial cost of the garage @ time = 0
    cost remains at this value for the rigid design, use exp_cost for flexible design

    Args:
        floor_initial (float): floors @ time 0

    Returns:
        float: cost of infrastructure
    """
    if floor_initial > 2:
        return config.cost_construction * config.space_per_floor * ((((1 + config.growth_factor)**(floor_initial - 1) - (1 + config.growth_factor)) / config.growth_factor)) + (2 * config.space_per_floor * config.cost_construction)
    else:
        return floor_initial * config.space_per_floor * config.cost_construction


def cashflow_array_rigid(floor_initial: float, demand_det: bool, seed_number = None) -> np.array:
    """Generates an array containing the annual cashflows across project lifespan

    Args:
        floor_initial (float): initial number of floors
        demand_det (bool): is the demand deterministic? if not, stochastic demand is used
        *args (int): seed number if demand is stochastic

    Returns:
        np.array: cashflow throughout project lifespan
    """
    
    # initialise the cashflow array
    cashflow = np.full((config.time_lifespan+1), -(cost_construction_initial(floor_initial) + config.cost_land),dtype='float64')
    # initialise capacity array
    capacity = np.full((config.time_lifespan+1),floor_initial * config.space_per_floor)
    # initialise demand scenarios
    if demand_det:
        demand = demand_deterministic(config.time_arr)
    else:
        demand = demand_stochastic(config.time_arr,seed_number)
    for i in range(1, config.time_lifespan):
        cashflow[i] = min(capacity[i], demand[i])*config.price - capacity[i]*config.cost_ops - config.cost_land
    cashflow[-1] = min(capacity[-1], demand[-1])*config.price - capacity[-1]*config.cost_ops
    
    return cashflow


def npv_det(floor_initial: float):
    """ function for printing NPV in main script

    Args:
        floor_initial (float): initial number of floors

    Returns:
        npv_det (float): the deterministic NPV
    """
    npv_garage = npv(config.rate_discount,cashflow_array_rigid(floor_initial,demand_det = True))
    
    from millify import millify
    print('Floors = '+ str(floor_initial), '| NPV £' + str(millify(npv_garage,precision=2)))
    
    return npv_det

def npv_det_opti(floor_initial: float):
    """Function used for the scipy_optimize module

    Args:
        floor_initial (float): initial number of floors

    Returns:
        npv (float): the NPV, negative due to nature of optimizing for minimum
    """
    return -npv(config.rate_discount,cashflow_array_rigid(floor_initial,demand_det = True))



def expected_npv(floor_initial: float) -> np.array:
    """ function for printing ENPV in main script

    Args:
        floor_initial (float): initial number of floors

    Returns:
        npv_det (float): the ENPV for n stochastic demand conditions
    """
    cashflow_stoc = np.zeros(config.time_lifespan+1,dtype='float64')
    npv_stoc = np.zeros(config.sims,dtype='float64')
    for instance in range(config.sims):
        cashflow_stoc = cashflow_array_rigid(floor_initial,demand_det=False,seed_number=config.scenarios[instance])
        npv_stoc[instance] = npv(config.rate_discount,cashflow_stoc)

    enpv_stoc = np.mean(npv_stoc)
    from millify import millify
    print('Floor = '+ str(floor_initial) + ' | ENPV £' + str(millify(enpv_stoc,precision=2)))

    return enpv_stoc, npv_stoc

def expected_npv_opti(floor_initial: float) -> np.array:
    """ function for scipy optimize module

    Args:
        floor_initial (float): initial number of floors

    Returns:
        npv_det (float): the ENPV, negative for minimize optimization
    """
    cashflow_stoc = np.zeros(config.time_lifespan+1,dtype='float64')
    npv_stoc = np.zeros(config.sims,dtype='float64')
    for instance in range(config.sims):
        cashflow_stoc = cashflow_array_rigid(floor_initial,demand_det=False,seed_number=config.scenarios[instance])
        npv_stoc[instance] = npv(config.rate_discount,cashflow_stoc)

    return -np.mean(npv_stoc)
