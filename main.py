# Parking Garage model calculator - main code (run.py)
#
# Created on Mon Feb 06 2023
#
# Copyright (c) 2023 N Saduagkan
#

import numpy as np
import config
import objective_funcs
import objective_funcs_flex
import plotting
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar, brute, fmin
from millify import millify


#NPV function for initial design vector - rigid and flex
#TODO: change functions to to take variables for optimisation

print('____________________________')
print('Rigid Deterministic:')
print('____________________________')
#fun = lambda x: objective_funcs.npv_det_opti(x)
#Optimise number of floors for rigid design under deterministic conditions
bnds = (config.floor_min, config.floor_max)
rigid_det_optimised = minimize_scalar(objective_funcs.npv_det_opti, bounds=bnds)
print(rigid_det_optimised)

for instance in range(config.floor_min,config.floor_max):
    objective_funcs.npv_det(instance)


#Rigid design
print('____________________________')
print('Rigid Stochastic:')
print('____________________________')

#Rigid design 
rigid_stoc_optimised = minimize_scalar(objective_funcs.expected_npv_opti, bounds=bnds)
print(rigid_stoc_optimised)

for instance in range(config.floor_min,config.floor_max):
    objective_funcs.expected_npv(instance)

#flex

print('____________________________')
print('Flex Deterministic:') 
print('____________________________')
enpv_det_flex, npv_det_flex, capacity = objective_funcs_flex.enpv_flex_det(config.floor_initial,config.y1_4expand,config.y9_12expand,config.y17_20expand,config.floor_expansion,config.year_threshold,config.capacity_threshold)

def fun(variables):
    floor_initial,y1_4expand,y9_12expand,y17_20expand,floor_expansion,year_threshold,capacity_threshold = variables
    return objective_funcs_flex.enpv_flex_det_opti(floor_initial,y1_4expand,y9_12expand,y17_20expand,floor_expansion,year_threshold,capacity_threshold)

rrange = (slice(config.floor_min, config.floor_max,1),slice(0, 2,1),slice(0, 2,1),slice(0, 2,1),slice(1,4,1),slice(1,4,1),slice(0.9,1.3,0.1))
flex_det_optimised = brute(fun, rrange, full_output=True, finish=fmin)
print(np.around(flex_det_optimised[0],decimals=2),flex_det_optimised[1])
from millify import millify
print('Decision Rules = '+ str(np.rint(flex_det_optimised[0])) + ' | ENPV £' + str(millify(-flex_det_optimised[1],precision=2)))
print(objective_funcs_flex.enpv_flex_det(4,1,0,0,1,1,1.0))

print('____________________________')
print('Flex Stochastic:')
print('____________________________')

print(objective_funcs_flex.enpv_flex(3,1,1,0,1,2,1.0))
print(objective_funcs_flex.enpv_flex(5,0,1,0,1,1,1.0))
print(objective_funcs_flex.enpv_flex(5,1,1,1,1,2,1.0))

design_variables = (config.floor_initial,config.y1_4expand,config.y9_12expand,config.y17_20expand,config.floor_expansion,config.year_threshold,config.capacity_threshold)
bnds_arr = ((config.floor_min, config.floor_max), (0, 1),(0, 1),(0, 1),(1,3),(1,3),(0.8,1.0))
fun2 = lambda design_variables: objective_funcs_flex.enpv_flex_opti(design_variables[0],design_variables[1],design_variables[2],design_variables[3],design_variables[4],design_variables[5],design_variables[6])
flex_optimised = minimize(fun2, design_variables, method='Nelder-Mead', bounds=bnds_arr)
print(flex_optimised)
#print('Decision Rules = '+ str(np.rint(flex_optimised[0])) + ' | ENPV £' + str(millify(-flex_optimised[1],precision=2)))


#TODO: plot the graphs properly

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
    #TODO add args into plotting.cdf that includes legend titles/plot titles etc.
    plt.show()

#TODO: iterative history? sensitivity analysis? multiobjective?