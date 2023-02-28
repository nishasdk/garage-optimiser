"""
Calculates demand for parking garage case study

N Saduagkan, Feb 2023
@nishasdk
"""

import config
import numpy as np
from numpy_financial import npv
import pandas as pd
import matplotlib.pyplot as plt
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
    np.random.seed(seed_number)
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
