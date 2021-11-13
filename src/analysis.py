import numpy as np
import matplotlib.pyplot as plt
from numba import jit

class DataAnalyzer():
    def __init__(self, ResultsPath = ''):
        try:
            self.PosHistory = np.load(ResultsPath + '/PosHistory.npy')
            self.Mass = np.load(ResultsPath + '/Masses.npy')
            #load yaml with configurations
        except Exception as e:
            print(e)

        self.NBodies = self.Mass.shape[0]
        self.dt = 1

    def CalcVelFromPos(self):
        self.Vel = (self.PosHistory[self.NBodies:] - self.PosHistory[:-self.NBodies])/self.dt
        pass

    def PlotTotalKineticEnergy(self):
        self.TotalKineticEnergy = 0.5 * self.Mass * np.split(np.sum(self.Vel ** 2, axis = -1), self.Vel.shape[0]/self.NBodies) 
        plt.plot(np.ediff1d(self.TotalKineticEnergy))
        plt.show()
