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
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar


''' NPV function for initial design vector - rigid and flex'''
''' TODO: change functions to to take variables for optimisation'''

print('____________________________')
print('Rigid Deterministic:')
print('____________________________')
#fun = lambda x: objective_funcs.npv_det_opti(x)
#Optimise number of floors for rigid design under deterministic conditions
bnds = (config.floor_min, config.floor_max)
rigid_det_optimised = minimize_scalar(objective_funcs.npv_det_opti, method='Brent', bounds=bnds)
print(rigid_det_optimised)

for instance in range(config.floor_min,config.floor_max):
    objective_funcs.npv_det(instance)


#Rigid design
print('____________________________')
print('Rigid Stochastic:')
print('____________________________')

#Rigid design 
rigid_stoc_optimised = minimize_scalar(objective_funcs.expected_npv_opti, method='Brent', bounds=bnds)
print(rigid_stoc_optimised)

for instance in range(config.floor_min,config.floor_max):
    objective_funcs.expected_npv(instance)

#flex
print('____________________________')
print('Flex Deterministic:')
print('____________________________')
enpv_stoc_flex, npv_stoc_flex = objective_funcs.expected_npv_flex_det(config.floor_initial)


print('____________________________')
print('Flex Stochastic:')
print('____________________________')
enpv_stoc_flex, npv_stoc_flex = objective_funcs.expected_npv_flex(config.floor_initial)

''' TODO: optimise the flex design vectors under deterministic and stochastic demand'''
# at this point it proves that optimal values will not be the same under a stochastic demand
''' TODO: optimise against ENPV to find optimal design variable'''
# prove that flexibility can 


plot = False
if plot:
    #plot first demands
    plt.style.use(style='fast')
    ax1 = plt.figure(plotting.demand_plotter(config.scenarios[1:8]))
    enpv_stoc, npv_stoc = objective_funcs.expected_npv(config.sims,config.scenarios)
    #rigid stochastic demand histogram
    ax2 = plt.figure(plotting.histogram_plotter(npv_stoc/1e6))
    #rigid stochastic demand cdf
    ax3 = plt.figure(plotting.cdf_plotter(npv_stoc/1e6,enpv_stoc/1e6))
    #flex stochastic demand histogram
    ax4 = plt.figure(plotting.histogram_plotter(npv_stoc_flex/1e6))
    #rigid stochastic demand cdf
    ax5 = plt.figure(plotting.cdf_plotter(npv_stoc_flex/1e6,enpv_stoc_flex/1e6))
    '''TODO add args into plotting.cdf that includes legend titles/plot titles etc.'''
    plt.show()

''' TODO: iterative history? sensitivity analysis? multiobjective? '''