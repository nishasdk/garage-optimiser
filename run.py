# Parking Garage model calculator - main code (run.py)
#
# Created on Mon Feb 06 2023
#
# Copyright (c) 2023 N Saduagkan
#

import numpy as np
import config
import objective_funcs
import plotting
from numpy_financial import npv



'''______________select simulation variables__________________'''

seed_no = 69 # means script always selects the same N scenarios. N is defined by sims
np.random.seed(seed_no)
sims = 2000 # number of simulations

'''___________________________________________________________'''


''' NPV function for initial design vector - rigid and flex'''
npv_det = objective_funcs.net_present_value(floor_initial=5,demand_det=True)

''' plot the deterministic and stochastic demands'''

scenarios = np.random.choice(sims,size=sims,replace=False) 

#plot first demands
# plotting.demand_plotter(scenarios[1:8])

''' ENPV for a certain number of scenarios'''
cashflow_stoc = np.zeros(config.time_lifespan+1)
npv_stoc = np.zeros(sims)
for instance in range(sims):
    cashflow_stoc = objective_funcs.cashflow_array(floor_initial=5,demand_det=False,seed_number=scenarios[instance])
    npv_stoc[instance] = npv(config.rate_discount,cashflow_stoc)

enpv_stoc = np.mean(npv_stoc)
from millify import millify
print('ENPV £' + str(millify(enpv_stoc,precision=2)))

plotting.cdf_plotter(npv_stoc/1e6)

''' TODO: implement decision rules'''


''' TODO: optimise the rigid & flex design vectors under deterministic demand'''
''' TODO: optimise the rigid & flex design vectors under stochastic demand'''
# at this point it proves that optimal values will not be the same under a stochastic demand

''' TODO: montecarlo simulation for n cases - generate n different demand scenarios'''
# show uncertainty across scenarios

''' TODO: set up ENPV function - rigid AND flex'''

''' TODO: optimise against ENPV to find optimal design variable'''
# prove that flexibility can 

''' TODO: plot the results, histogram, cdf distribution'''


''' TODO: iterative history? sensitivity analysis? multiobjective? '''