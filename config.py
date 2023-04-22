"""
Configuration file for garage scenario case study
Lists all variables to use in other modules

N Saduagkan, Feb 2023
@nishasdk
"""

import numpy as np

'''______________select simulation variables__________________'''

seed_no = 123 # means script always selects the same N scenarios. N is defined by sims
np.random.seed(seed_no)
sims = 2000 # number of simulations

'''___________________________________________________________'''

''' to find ENPV for a certain number of scenarios'''
scenarios = np.random.choice(sims,size=sims,replace=False) 


'''________________select design variables___________________'''
floor_initial = 3
y1_4expand = 0
y9_12expand = 1
y17_20expand = 0
floor_expansion = 1
year_threshold = 1
capacity_threshold = 0.8
'''___________________________________________________________'''

# Parameters
time_lifespan = 20  # years
time_arr = np.arange(time_lifespan + 1)  # time array

years = list(map(str, range(time_lifespan)))  # Array of strings for np header
cost_construction = 16000  # Construction cost per parking space
cost_land = 3600000  # Annual leasing land cost
# p = 0.00# Percentage of construction cost to acquire flexibility (i.e. fatter columns). Note that this is set to 0 to determine how much the flexibility is worth.
# The real value can be found by subracting the acquisition cost from the NPV for a particular design.
cost_ops = 2000  # Operating cost per parking space

growth_factor = 0.10  # Growth in construction cost per floor above two floors
space_per_floor = 200  # Initial number of parking space per floor
price = 10000  # Price per parking space
rate_discount = 0.12  # Discount rate
floor_min = 2  # Minimum number of floors built
floor_max = 9 + 1  # Maximum number of floors built (+1 is for python to pick 9 floors inclusive upperbound)

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