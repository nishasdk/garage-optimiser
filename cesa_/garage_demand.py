"""
Defining demand for multi-story parking garage

N Saduagkan, Feb 2023
@nishasdk
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing

## SEED ##
seed_number = 1

# Parameters
time_lifespan = 20  # years
time_arr = np.array(range(time_lifespan + 1))  # time array

years = list(map(str, range(time_lifespan)))  # Array of strings for np header
cost_construction = 16000  # Construction cost per parking space
cost_land = 3600000  # Annual leasing land cost
# p = 0.00# Percentage of construction cost to acquire flexibility (i.e. fatter columns). Note that this is set to 0 to determine how much the flexibility is worth. The real value can be found by subracting the acquiistion cost from the NPV for a particular design.
cost_ops = 2000  # Operating cost per parking space

growth_factor = 0.10  # Growth in construction cost per floor above two floors
space_initial = 200  # Initial number of parking space per floor
price = 10000  # Price per parking space
rate_discount = 0.12  # Discount rate
floor_min = 2  # Minimum number of floors built
floor_max = 9  # Maximum number of floors built

# Demand Variables

demand_1 = 750  # Projected year 1 demand
demand_10 = 750  # additional demand by year 10
demand_final = 250  # additional demand after year 10

# Stochastic demand variables

off_D0 = 0.5  # Realised demand in yr 1 within "x" perccentage of demand projection
off_D10 = 0.5  # Additional demand by year 10 within "x" percentage of demand projection
# Additional demand after year 10 within "x" percentage of demand projection
off_Dfinal = 0.5
# Annual volatility of demand growth within "x" percentage of growth projection
volatility = 0.15


def cost_initial(floor_initial: int) -> float:
    """initial cost of the garage @ time = 0
    cost remains at this value for the rigid design, use exp_cost for flexible design

    Args:
        floor_initial (int): floors @ time 0

    Returns:
        float: cost of infrastructure
    """
    return (
        cost_construction
        * space_initial
        * (
            (
                ((1 + growth_factor) ** (floor_initial - 1) - (1 + growth_factor))
                / growth_factor
            )
        )
        + (2 * space_initial * cost_construction)
        if floor_initial > 2
        else floor_initial * space_initial * cost_construction
    )


def demand_deterministic(time_arr: np.array) -> np.array:
    """Function to calculate demand projection

    Args:
        time_arr (np.array): array starting at 0, ending at time_lifespan

    Returns:
        np.array: deterministic demand
    """
    # Parameter for demand model showing difference between initial and final demand values
    alpha = demand_10 + demand_final
    # Parameter for demand model showing growth speed of demand curve
    beta = -np.log(demand_final / alpha) / (time_lifespan / 2 - 1)
    return demand_1 + demand_10 + demand_final - alpha * np.exp(-beta * (time_arr - 1))


def demand_stochastic(time_arr: np.array, seed_number: int) -> np.array:
    """function for calculating the stochastic demand

    Args:
        time_arr (np.array): array starting at 0, ending at time_lifespan
        seed_number (int): random seed number for new simulation

    Returns:
        np.array: stochastic demand
    """
    # set constant seed for simulations to for standardized comparison
    np.random.seed(seed_number)
    rD0 = round(
        (1 - off_D0) * demand_1 + np.random.rand() * 2 * off_D0 * demand_1
    )  # Realised demand in year 0

    rD10 = round(
        (1 - off_D10) * demand_10 + np.random.rand() * 2 * off_D10 * demand_10
    )  # Realised additional demand by year 10

    rDf = round(
        (1 - off_Dfinal) * demand_final
        + np.random.rand() * 2 * off_Dfinal * demand_final
    )  # Realised additional demand after year 10

    # Parameter for demand model showing difference between initial and final demand values
    alpha_stoc = rD10 + rDf

    # Parameter for demand model showing growth speed of demand curve
    beta_stoc = -np.log(rDf / alpha_stoc) / (time_lifespan / 2 - 1)

    D_stoc1 = (
        rD0 + rD10 + rDf - alpha_stoc * np.exp(-beta_stoc * (time_arr - 1))
    )  # projected demand vector

    # projected demand vector shifted by one period to right
    D_stoc2 = rD0 + rD10 + rDf - alpha_stoc * np.exp(-beta_stoc * (time_arr - 2))
    D_g_proj = D_stoc1 / D_stoc2 - 1
    R_g = D_g_proj - volatility + np.random.rand(len(time_arr)) * 2 * volatility

    return np.multiply(D_stoc2, (1 + R_g))


plt.figure()
plt.plot(
    time_arr[1:],
    demand_deterministic(time_arr[1:]),
    marker="",
    linestyle="-",
    color="b",
    label="Deterministic Demand",
)
plt.plot(
    time_arr[1:],
    demand_stochastic(time_arr[1:], 1),
    marker=".",
    linestyle="",
    color="r",
    label="Stochastic Demand 1",
)
plt.plot(
    time_arr[1:],
    demand_stochastic(time_arr[1:], 69),
    marker="x",
    linestyle="",
    color="g",
    label="Stochastic Demand 2",
)
plt.xlabel("Year")
plt.ylabel("Demand")
plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], time_arr)
plt.title("Deterministic and Stochastic Demand over time")
plt.legend()
plt.show()
