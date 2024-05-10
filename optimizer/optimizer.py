from abc import ABC, abstractmethod
from matplotlib import pyplot as plt

class Optimizer(ABC):
    @abstractmethod
    def optimize(self):
        pass

    def plot_history(self):
       
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.objective_function_list,label="Objective function",marker='.')
        ax2.plot(self.volume_fraction_list,label='Volume fraction',marker='.',color='r')
        fig.legend()
        fig.show()
