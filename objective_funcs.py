"""
Calculates demand and cost for parking garage case study

N Saduagkan, Feb 2023
@nishasdk
"""

from pandas import array
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
    alpha = config.demand_10 + config.demand_final
    # Parameter for demand model showing growth speed of demand curve
    beta = -np.log(config.demand_final / alpha) / (config.time_lifespan / 2 - 1)
    demand = config.demand_1 + config.demand_10 + config.demand_final - alpha * np.exp(-beta * (time_arr - 1))
    return demand


def demand_stochastic(time_arr: np.array, seed_number: int) -> np.array:
    """function for calculating the stochastic demand (lifted straight from @cesa_, variable explanations commented)

    Args:
        time_arr (np.array): array starting at 0, ending at time_lifespan
        seed_number (int): random seed number for new simulation

    Returns:
        np.array: stochastic demand
    """
    # set constant seed for simulations to for standardized comparison
    np.random.seed(seed_number) #demand scenario with this seed will always be the same
    rD0 = round(
        (1 - config.off_D0) * config.demand_1 +
        np.random.rand() * 2 * config.off_D0 * config.demand_1)  # Realised demand in year 0

    rD10 = round((1 - config.off_D10) * config.demand_10 + np.random.rand() * 2 * config.off_D10 * config.demand_10)  # Realised additional demand by year 10

    rDf = round(
        (1 - config.off_Dfinal) * config.demand_final + np.random.rand() * 2 * config.off_Dfinal * config.demand_final)  # Realised additional demand after year 10

    # Parameter for demand model showing difference between initial and final demand values
    alpha_stoc = rD10 + rDf

    # Parameter for demand model showing growth speed of demand curve
    beta_stoc = -np.log(rDf / alpha_stoc) / (config.time_lifespan / 2 - 1)

    D_stoc1 = (rD0 + rD10 + rDf - alpha_stoc *np.exp(-beta_stoc * (time_arr - 1)))  # projected demand vector

    # projected demand vector shifted by one period to right
    D_stoc2 = rD0 + rD10 + rDf - alpha_stoc * np.exp(-beta_stoc * (time_arr - 2))
    D_g_proj = D_stoc1 / D_stoc2 - 1
    R_g = D_g_proj - config.volatility + np.random.rand(len(time_arr)) * 2 * config.volatility

    return np.multiply(D_stoc2, (1 + R_g))


def cost_construction_initial(floor_initial: int) -> float:
    """initial cost of the garage @ time = 0
    cost remains at this value for the rigid design, use exp_cost for flexible design

    Args:
        floor_initial (int): floors @ time 0

    Returns:
        float: cost of infrastructure
    """
    return (config.cost_construction * config.space_per_floor *
            ((((1 + config.growth_factor)**(floor_initial - 1) - (1 + config.growth_factor)) / config.growth_factor)) +
            (2 * config.space_per_floor * config.cost_construction) if floor_initial > 2 else
            floor_initial * config.space_per_floor * config.cost_construction)

def occupancy(demand: int, capacity: int) -> int:
    return min(demand, capacity)


def expansion_cost(floor_expansion: int, capacity: int) -> float:
    """calculates the cost of expanding (past 2 floors)

    Args:
        floor_expansion (int): how many floors to expand by
        capacity (int): current capacity (when nested in capacity_update, will take a value from capacity array)

    Returns:
        float: cost to expand the specified amount of floors
    """
    return config.cost_construction * config.space_per_floor *(
        (
            (
                (1 + config.growth_factor) ** floor_expansion) / config.growth_factor - 1) * (
                    (1 + config.growth_factor) ** (capacity/config.space_per_floor -1) 
                )
            )

def cashflow_array(floor_initial: int, demand_det: bool, seed_number = None) -> np.array:
    """Generates an array containing the annual cashflows across project lifespan

    Args:
        floor_initial (int): initial number of floors
        demand_det (bool): is the demand deterministic? if not, stochastic demand is used
        *args (int): seed number if demand is stochastic

    Returns:
        np.array: cashflow throughout project lifespan
    """
    
    # initialise the cashflow array
    cashflow = np.full((config.time_lifespan+1), -(cost_construction_initial(floor_initial) + config.cost_land))
    # initialise capacity array
    capacity = np.full((config.time_lifespan+1),floor_initial * config.space_per_floor, dtype=int)
    # initialise demand scenarios
    if demand_det:
        demand = demand_deterministic(config.time_arr)
    else:
        demand = demand_stochastic(config.time_arr,seed_number)
    for i in range(1, config.time_lifespan):
        cashflow[i] = min(capacity[i], demand[i])*config.price - capacity[i]*config.cost_ops - config.cost_land
    cashflow[-1] = min(capacity[i], demand[i])*config.price - capacity[i]*config.cost_ops
    
    return cashflow

def net_present_value(floor_initial: int, demand_det: bool, seed_number = None):
    npv_garage = npv(config.rate_discount,cashflow_array(floor_initial,demand_det,seed_number))
    
    from millify import millify
    print('NPV £' + str(millify(npv_garage,precision=2)))
    
    return net_present_value


def expected_npv(sims: int, scenarios: np.array) -> np.array:
    """Calculates expected NPV for x demand scenarios and n simulations

    Args:
        sims (int): number of simulations
        scenarios (np.array): demand scenarios array

    Returns:
        np.array: the ENPV and the stochastic NPV array for plotting
    """
    cashflow_stoc = np.zeros(config.time_lifespan+1)
    npv_stoc = np.zeros(sims)
    for instance in range(sims):
        cashflow_stoc = cashflow_array(floor_initial=5,demand_det=False,seed_number=scenarios[instance])
        npv_stoc[instance] = npv(config.rate_discount,cashflow_stoc)

    enpv_stoc = np.mean(npv_stoc)
    from millify import millify
    print('ENPV £' + str(millify(enpv_stoc,precision=2)))
    
    return enpv_stoc, npv_stoc