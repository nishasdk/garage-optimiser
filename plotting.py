
import numpy as np
import matplotlib.pyplot as plt
import config
import demand

def demand_plotter(scenarios: np.array):
    plt.figure()
    
    plt.plot(
        config.time_arr[1:],
        demand.demand_deterministic(config.time_arr[1:]),
        marker="",
        linestyle="-",
        color="b",
        label="Deterministic Demand",
    )

    for item in scenarios:
        plt.plot(
            config.time_arr[1:],
            demand.demand_stochastic(config.time_arr[1:], item),
            marker="",
            linestyle="--",
            label="Stochastic Demand, scenario " + str(item.index),
        )

    plt.xlabel("Year")
    plt.ylabel("Demand")
    plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    plt.title("Deterministic and Stochastic Demand over time")
    plt.legend(fontsize="xx-small")
    plt.show()
