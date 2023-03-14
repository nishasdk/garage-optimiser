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



'''______________select simulation variables__________________'''

seed_no = 123 # means script always selects the same N scenarios. N is defined by sims
np.random.seed(seed_no)
sims = 2000 # number of simulations

'''___________________________________________________________'''

''' NPV function for initial design vector - rigid and flex'''
npv_det = objective_funcs.net_present_value(config.floor_initial,demand_det=True)

''' plot the deterministic and stochastic demands'''

scenarios = np.random.choice(sims,size=sims,replace=False) 

#plot first demands
plt.style.use(style='fast')

ax1 = plt.figure(plotting.demand_plotter(scenarios[1:8]))

''' ENPV for a certain number of scenarios'''
#rigid
print('Rigid:')
enpv_stoc, npv_stoc = objective_funcs.expected_npv(config.floor_initial,sims,scenarios,flex=False)

ax2 = plt.figure(plotting.histogram_plotter(npv_stoc/1e6))
ax3 = plt.figure(plotting.cdf_plotter(npv_stoc/1e6,enpv_stoc/1e6))

#flex
print('Flex:')
enpv_stoc_flex, npv_stoc_flex = objective_funcs.expected_npv(config.floor_initial,sims,scenarios,flex=True)
ax4 = plt.figure(plotting.histogram_plotter(npv_stoc_flex/1e6))
ax5 = plt.figure(plotting.cdf_plotter(npv_stoc_flex/1e6,enpv_stoc_flex/1e6))
plt.show()

''' TODO: optimise the rigid design under deterministic and stochastic demand'''
''' TODO: optimise the flex design vectors under deterministic and stochastic demand'''
# at this point it proves that optimal values will not be the same under a stochastic demand
''' TODO: optimise against ENPV to find optimal design variable'''
# prove that flexibility can 


''' TODO: iterative history? sensitivity analysis? multiobjective? '''