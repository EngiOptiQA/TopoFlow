from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import numpy as np

class Optimizer(ABC):
    @abstractmethod
    def optimize(self):
        pass

    def plot_history(self):
       
        optimization_steps = np.arange(1, len(self.objective_function_list)+1)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(optimization_steps, self.objective_function_list, label="Objective function", marker='s', color='k')
        ax1.set_xlabel('Optimization Steps')
        ax1.set_ylabel('Objective Function')
        ax2.plot(optimization_steps, self.volume_fraction_list,label='Volume fraction', marker='x', color='gray')
        ax2.set_ylabel('Volume Fraction')
        fig.legend()
        fig.show()
