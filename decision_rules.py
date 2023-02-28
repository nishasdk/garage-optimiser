"""
Decision rule functions for parking garage case study

N Saduagkan, Feb 2023
@nishasdk
"""


# def capacity_update(capacity: np.array, year_threshold: int = None, y1_4expand: bool = None, y9_12expand: bool = None, y17_20expand: bool = None, floor_expansion: int = None, floor_initial: int = None) -> np.array:

#     if y9_12expand is None:
#         y9_12expand = {True}
#     if y17_20expand is None:
#         y17_20expand = [False]
#     if floor_expansion is None:
#         floor_expansion = {1}
#     if floor_initial is None:
#         floor_initial = {2}
#     if year_threshold is None:
#         year_threshold = {1}
#     if y1_4expand is None:
#         y1_4expand = {False}
#     demand = demand_deterministic(time_arr)

#     for t in range(1,time_lifespan-1):
#         #WAIT FOR 1 YEAR BEFORE EXPANDING
#         if year_threshold == 1:
#             if min(capacity[t],demand[t]) == capacity[t-1] and capacity[t] + floor_expansion * space_per_floor <= floor_max * space_per_floor:

#                 if (y1_4expand and t < 5) or (y9_12expand and t>8 and t<13) or (y17_20expand and t>16 and t<21):
#                     print('in the range')
#                     capacity[t] = capacity[t-1] + floor_expansion * space_per_floor
#                     cost_expansion = expansion_cost(floor_expansion,capacity[t])
                
#                 #NOTE: uncomment if they can expand during years 5-9 and 12-17
#                 #elif (t>5 and t<9) or (t > 12 and t < 17):
#                     #print('non-design')
#                     #capacity[t] = capacity[t-1] + floor_expansion * space_per_floor
#                     #cost_expansion = expansion_cost(floor_expansion,capacity[t])

#                 else:
#                     capacity[t] = floor_initial * space_per_floor
#                     cost_expansion = 0
                    
#         #WAIT FOR 2 YEARS BEFORE EXPANDING
#         if year_threshold == 2:
#             if min(capacity[t],demand[t]) + min(capacity[t-1],demand[t-1]) == capacity[t-1] and capacity[t] + floor_expansion * space_per_floor <= floor_max * space_per_floor:

#                 if (y1_4expand and t < 5) or (y9_12expand and t>8 and t<13) or (y17_20expand and t>16 and t<21):
#                     print('in the range')
#                     capacity[t] = capacity[t-1] + floor_expansion * space_per_floor
#                     cost_expansion = expansion_cost(floor_expansion,capacity[t])
                
#                 #NOTE: uncomment if they can expand during years 5-9 and 12-17
#                 #elif (t>5 and t<9) or (t > 12 and t < 17):
#                     #print('non-design')
#                     #capacity[t] = capacity[t-1] + floor_expansion * space_per_floor
#                     #cost_expansion = expansion_cost(floor_expansion,capacity[t])

#                 else:
#                     capacity[t] = floor_initial * space_per_floor
#                     cost_expansion = 0