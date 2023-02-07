# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:39:31 2020

@author: cesa_

Updated Feb 20223 @nishasdk
"""

# Parameters
T = 20  # years
cost_construction = 16000  # Construction cost per parking space
cost_land = 3600000  # Annual leasing land cost
# Percentage of construction cost to acquire flexibility (i.e. fatter columns).
cost_percentage = 0.00
# Note that this is set to 0 to determine how much the flexibility is worth.
# The real value can be found by subracting the acquiistion cost from the NPV for a particular design.
cost_ops = 2000  # Operating cost per parking space
growth_factor = 0.10  # Growth in construction cost per floor above two floors
space_initial = 200  # Initial number of parking space per floor
price = 10000  # Price per parking space
rate_discount = 0.12  # Discount rate
floor_min = 2  # Minimum number of floors built
floor_max = 8  # Maximum number of floors built
capacity_max = floor_max * space_initial


def cost_exp(capacity: int, floor_exp: int) -> float:
    """_summary_

    Args:
        capacity (int): no. of floors * space per floor
        floor_exp (int): how many floors to expand by

    Returns:
        float: updated cost after expansion occurs
    """
    return space_initial * cost_construction * ((((1 + growth_factor) ** (floor_exp)) - 1) / (growth_factor)) * ((1 + growth_factor) ** ((capacity / space_initial) - 1))


def opex(capacity: int) -> float:
    """Function to find opex

    Args:
        capacity (int): no. of floors * space per floor

    Returns:
        float: operational cost
    """
    return cost_ops * capacity
