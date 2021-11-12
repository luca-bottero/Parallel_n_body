from mpi4py import MPI
import numpy as np
import time
from numba import jit

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

HighMass = 10        #max generated mass
LowMass = 0.1       #min generated mass
HighVel = 0.5       #max generated Vel
LowVel = -HighVel   #min generated Vel
HighPos = 300        #max generated Pos
LowPos = -HighPos   #min generated Pos

LCube = 500
XWidth = LCube    #Plot Axis range (-XWidth,XWidth)
YWidth = LCube
ZWidth = LCube

filepath = None

NBodies = 128    #number of bodies to simulate
G = 1           #np.float32(6.67430e-11) Gravitational Costant
dt = 1        #timestep
SimTime = 100    #total simulation time
AnimDuration = 20    #total animation time in seconds

UseJit = True


NumThreads = comm.Get_size()    



def ShowSimulationLog(StartTime, EndTime):      #shows basics informations about the simulation
    TotTime = EndTime - StartTime
    print("Number of bodies: " + str(NBodies))
    print("Total iteration: " + str(SimTime/dt))
    print("Total time: " + str(TotTime))
    print("Mean time per body: " + str(TotTime/NBodies))
    print("Mean time per iteration: " + str(dt*TotTime/SimTime))
    print("Mean time per body per iteration: " + str(dt*TotTime/SimTime/NBodies))

class NBodySim():
    def __init__(self, ConfigFilepath = None):
        self.rank = comm.Get_rank()

        if ConfigFilepath is not None:
            #load initial config from file
            #TODO
            pass
        
        else:
            #compute default random initial condition
            if rank == 0:            
                self.PosHistory = np.zeros([NBodies,3])
                self.Pos = np.zeros([NBodies,3], dtype = np.float32)     #root blank initialization
                self.Vel = np.zeros([NBodies,3], dtype = np.float32)
                self.Acc = np.zeros([NBodies,3], dtype = np.float32)

                for i in range(NBodies):        #Random generation of positions and velocities
                    for j in range(3):
                        self.Pos[i][j] = np.random.uniform(low = LowPos, high = HighPos)    
                for i in range(NBodies):
                    for j in range(3):
                        self.Vel[i][j] = np.random.uniform(low = LowVel, high = HighVel)  
                        
            else:
                self.Pos = None          #thread's vectors initialization 
                self.CommPos = None
                self.CommVel = None
                self.CommAcc = None
                self.LocalPos = None
                self.LocalVel = None
                self.LocalAcc = None
                
        MassGen = np.random.uniform(low = LowMass, high = HighMass, size = NBodies)     #Masses random generation, in kg
        self.Mass = comm.bcast(MassGen, root = 0)

    @staticmethod
    @jit
    def _CompForce(Mass, Pos, LocalPos, LocalVel, LocalAcc):
        for i in range(len(LocalAcc)):      #calculate accelerations
            LocalAcc[i] = 0.
            for j in range(NBodies):
                if np.array_equal(LocalPos[i],Pos[j]) == False:
                    r = LocalPos[i] - Pos[j]                    #displacement vector
                    DCube = (r.dot(r) + 1e-3)**1.5    #compute distance
                    LocalAcc[i] += -r*G*Mass[j]/DCube           #compute acceleration
            LocalVel[i] += LocalAcc[i]*dt                       #update velocity
            LocalPos[i] += 0.5*LocalAcc[i]*dt*dt + LocalVel[i]*dt  #update local positions
        return LocalVel, LocalPos

    def ComputeForce(self):
        if UseJit:
            self.LocalVel, self.LocalPos = self._CompForce(self.Mass, self.Pos, self.LocalPos, self.LocalVel, self.LocalAcc)
        else:
            for i in range(len(self.LocalAcc)):      #calculate accelerations
                self.LocalAcc[i] = 0.
                for j in range(NBodies):
                    if np.array_equal(self.LocalPos[i],self.Pos[j]) == False:
                        r = self.LocalPos[i] - self.Pos[j]                    #displacement vector
                        DCube = (np.inner(r,r) + self.SafetyValue)**1.5    #compute distance
                        self.LocalAcc[i] += -r*G*self.Mass[j]/DCube           #compute acceleration
                self.LocalVel[i] += self.LocalAcc[i]*dt                       #update velocity
                self.LocalPos[i] += 0.5*self.LocalAcc[i]*dt*dt + self.LocalVel[i]*dt  #update local positions

    def run(self):

        #SIMULATION

        if self.rank == 0:
            self.StartTime = time.time()

        for t in np.arange(dt, SimTime, dt, dtype = np.float32):
            if rank == 0:                                   #Split array that will be sent to each node
                self.CommPos = np.array_split(self.Pos, NumThreads)
                self.CommVel = np.array_split(self.Vel, NumThreads)    
                self.CommAcc = np.array_split(self.Acc, NumThreads)

            self.Pos = comm.bcast(self.Pos, root = 0)                 #broadcast positions of all the bodies
            self.LocalPos = comm.scatter(self.CommPos, root = 0)      #positions that will be updated

            if t == dt:
                self.LocalVel = comm.scatter(self.CommVel, root = 0)  #sends vel and acc to every node for the first time
                self.LocalAcc = comm.scatter(self.CommAcc, root = 0)    
                self.SafetyValue = 1e-3       #needed to not divide by 0

            self.ComputeForce()
                
            NewPos = comm.gather(self.LocalPos, root = 0)    #each node sends LocalPos to root
            if rank == 0:
                self.Pos = np.concatenate(NewPos)            #Pos is update, needed for broadcasting
                self.PosHistory = np.concatenate((self.PosHistory,self.Pos))
        
        if self.rank == 0:
            EndTime = time.time()
            ShowSimulationLog(self.StartTime, EndTime)   #basic infos
            PosHistory = self.PosHistory[NBodies:]       #eliminates the dummy 0's at the beginning
            #ShowPlot(PosHistory)
            
            '''with open("run_out.npy", "wb") as f:    #saves data to file
                np.save(f, PosHistory)'''
            
            '''
            with open("run_out.npy", "rb") as f:    #loads data from file
                a = np.load(f, allow_pickle = True)
            '''
            
        #PosHistory[NBodies:].reshape(np.asscalar(np.array(SimTime/dt-1).astype(int)),NBodies,3)
    
instance = NBodySim()
instance.run()