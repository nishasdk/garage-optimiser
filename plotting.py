
import numpy as np
import matplotlib.pyplot as plt
import config
import objective_funcs
import matplotlib

def demand_plotter(scenarios: np.array):
    """Plots different stochastic scenarios of demand against deterministic case

    Args:
        scenarios (np.array): scenario number, represented by seed number
    """
    plt.figure()
    
    #Plot deterministic demand
    plt.plot(
        config.time_arr[1:],
        objective_funcs.demand_deterministic(config.time_arr[1:]),
        marker="",
        linestyle="-",
        color="b",
        label="Deterministic Demand",
    )

    #Plot stochastic demand for n scenarios
    for item in scenarios:
        plt.plot(
            config.time_arr[1:],
            objective_funcs.demand_stochastic(config.time_arr[1:], item),
            marker="",
            linestyle="--",
            label="Stochastic Demand, scenario " + str(item),
        )

    plt.xlabel("Year")
    plt.ylabel("Demand")
    plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    plt.title("Deterministic and Stochastic Demand over time")
    plt.legend(fontsize="xx-small")
    plt.show()

def histogram_plotter(npv:np.array):
    """ plots pdf histogran

    Args:
        npv (np.array): npv array for n scenarios
    """
    plt.figure()
    
    plt.hist(
        npv,
        bins = 49,
        density=True
    )

    plt.xlabel("NPV ($, Millions)")
    plt.ylabel("Probability (%)")
    plt.title("Histogram")
    plt.show()

def cdf_plotter(npv:np.array):
    """ plots cdf

    Args:
        npv (np.array): npv array for n scenarios
    """
    plt.figure()
    
    values, base = np.histogram(npv, bins = 49)
    
    plt.plot( base[:-1], np.cumsum(values))


    plt.xlabel("NPV ($, Millions)")
    plt.ylabel("Probability (%)")
    plt.title("CDF graph")
    plt.show()