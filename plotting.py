
from cProfile import label
from matplotlib.backend_bases import LocationEvent
import numpy as np
import matplotlib.pyplot as plt
import config
import objective_funcs
import matplotlib.ticker as ticker


def demand_plotter(scenarios: np.array):
    """Plots different stochastic scenarios of demand against deterministic case

    Args:
        scenarios (np.array): scenario number, represented by seed number
    """
    
    fig, ax = plt.subplots()
    
    #Plot deterministic demand
    ax.plot(
        config.time_arr[1:],
        objective_funcs.demand_deterministic(config.time_arr[1:]),
        marker="o",
        markersize="1",
        linestyle="-",
        #color="b",
        #label="Deterministic Demand"
    )

    #Plot stochastic demand for n scenarios
    for item in scenarios:
        plt.plot(
            config.time_arr[1:],
            objective_funcs.demand_stochastic(config.time_arr[1:], item),
            linestyle="--",
            linewidth="1.5",
            #color="r",
            #label="Stochastic Demand, scenario " + str(item),
        )

    # Add the grid
    
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.grid(which='major', axis='both', linestyle='--')
    plt.xlim([0, 20])

    plt.xlabel("Year")
    plt.ylabel("Demand")
    plt.title("Deterministic and Stochastic Demand over time")
    plt.legend(("Deterministic Demand","Stochastic Demand Scenarios"),fontsize="xx-small",loc='lower right')
    leg = ax.get_legend()
    leg.legendHandles[1].set_color('black')
    #("Deterministic Demand","Stochastic Demand Scenarios"),

def histogram_plotter(npv:np.array):
    """ plots pdf histogran

    Args:
        npv (np.array): npv array for n scenarios
    """
    
    plt.hist(
        npv,
        bins = 49,
        density=True,
        edgecolor='black',
        #color="b"
    )

    plt.grid('on', linestyle='--')
    plt.xlim([np.amin(npv), np.amax(npv)])
    
    plt.xlabel("NPV ($, Millions)")
    plt.ylabel("Probability (%)")
    plt.title("Histogram")

def cdf_plotter(npv:np.array, enpv:float):
    """ plots cdf

    Args:
        npv (np.array): npv array for n scenarios
    """
    
    values, base = np.histogram(npv, bins = 49)
    
    
    plt.plot( 
        base[:-1], 
        np.cumsum(values),
        #color="b",
        label="NPV rigid",
        linewidth="2"
    )
    
    plt.axvline(
        x=enpv,
        linestyle='--',
        color = 'r',
        label="ENPV rigid",
        linewidth="2"
    )
    
    plt.grid('on', linestyle='--')
    plt.xlim([np.amin(npv), np.amax(npv)])
    plt.ylim(-50,2000)
    plt.xlabel("NPV ($, Millions)")
    plt.ylabel("Probability (%)")
    plt.title("CDF graph")
    plt.legend(fontsize="x-small",loc='lower right')
