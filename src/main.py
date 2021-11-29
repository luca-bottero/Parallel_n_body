from mpi4py import MPI
import numpy as np
import time
from numba import jit
import yaml

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

HighMass  = 10        #max generated mass
LowMass   = 0.1       #min generated mass
HighVel   = 0.5       #max generated Vel
LowVel    = -HighVel   #min generated Vel
HighPos   = 300        #max generated Pos
LowPos    = -HighPos   #min generated Pos

LCube   = 500
XWidth  = LCube    #Plot Axis range (-XWidth,XWidth)
YWidth  = LCube
ZWidth  = LCube

filepath = None

'''NBodies = 128    #number of bodies to simulate
G = 1           #np.float32(6.67430e-11) Gravitational Costant
dt = 0.01        #timestep
SimTime = 10    #total simulation time
AnimDuration = 20    #total animation time in seconds

UseJit = True
SaveRes = True'''

yaml.load('../config/TEST_run.yaml')

NumThreads = comm.Get_size()    





class NBodySim():
    def __init__(self, ConfigFilepath = None):
        self.rank = comm.Get_rank()
        self.ConfigFilepath = ConfigFilepath
        self._LoadConfig()
       
        #compute default random initial condition
        if rank == 0:            
            self.PosHistory = np.zeros([self.NBodies,3])
            self.Pos = np.zeros([self.NBodies,3], dtype = np.float32)     #root blank initialization
            self.Vel = np.zeros([self.NBodies,3], dtype = np.float32)
            self.Acc = np.zeros([self.NBodies,3], dtype = np.float32)

            for i in range(self.NBodies):        #Random generation of positions and velocities
                for j in range(3):
                    self.Pos[i][j] = np.random.uniform(low = LowPos, high = HighPos)    
            for i in range(self.NBodies):
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
            
        MassGen = np.random.uniform(low = LowMass, high = HighMass, size = self.NBodies)     #Masses random generation, in kg
        self.Mass = comm.bcast(MassGen, root = 0)

    def _LoadConfig(self):
        with open(self.ConfigFilepath, "r") as stream:
            try:
                self.Config = yaml.full_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        
        self.NBodies = self.Config['NBodies']
        self.G = self.Config['G']
        self.dt = self.Config['dt']
        self.SimTime = self.Config['SimTime']
        self.UseJit = self.Config['UseJit']
        self.SaveRes = self.Config['SaveRes']


    @staticmethod
    @jit(nopython = True)
    def _CompForce(NBodies, Mass, Pos, LocalPos, LocalVel, LocalAcc, dt, G):
        for i in range(len(LocalAcc)):      #calculate accelerations
            LocalAcc[i] = 0.
            for j in range(NBodies):
                if np.array_equal(LocalPos[i],Pos[j]) == False:
                    r = LocalPos[i] - Pos[j]                    #displacement vector
                    DCube = np.power((r.dot(r) + 1e-3), 1.5)    #compute distance
                    LocalAcc[i] += -r*G*Mass[j]/DCube           #compute acceleration
            LocalVel[i] += LocalAcc[i]*dt                       #update velocity
            #LocalPos[i] += 0.5*LocalAcc[i]*dt*dt + LocalVel[i]*dt  #update local positions
            LocalPos[i] += LocalVel[i]*dt
        return LocalVel, LocalPos

    def ComputeForce(self):
        if self.UseJit:
            self.LocalVel, self.LocalPos = self._CompForce(self.NBodies, self.Mass, self.Pos, self.LocalPos, 
                                                            self.LocalVel, self.LocalAcc, self.dt, self.G)
        else:
            for i in range(len(self.LocalAcc)):      #calculate accelerations
                self.LocalAcc[i] = 0.
                for j in range(self.NBodies):
                    if np.array_equal(self.LocalPos[i],self.Pos[j]) == False:
                        r = self.LocalPos[i] - self.Pos[j]                    #displacement vector
                        DCube = (r.dot(r) + self.SafetyValue)**1.5    #compute distance
                        self.LocalAcc[i] += -r*self.G*self.Mass[j]/DCube           #compute acceleration
                self.LocalVel[i] += self.LocalAcc[i]*self.dt                       #update velocity
                self.LocalPos[i] += 0.5*self.LocalAcc[i]*self.dt*self.dt + self.LocalVel[i]*self.dt  #update local positions

    def SaveResults(self, verbose = False, append = False):
        # used to save the results. Do not print anything if saving during simulation
        if verbose: print('Saving trajectories')
        with open("PosHistory.npy", "wb") as f:    
            np.save(f, self.PosHistory)

        if verbose: print('Saving mass values')
        with open('Masses.npy', 'wb') as f:
            np.save(f, self.Mass)

    def ShowSimulationLog(self, StartTime, EndTime):      #shows basics informations about the simulation
        TotTime = EndTime - StartTime
        print("Number of bodies: " + str(self.NBodies))
        print("Total iteration: " + str(self.SimTime/self.dt))
        print("Total time: " + str(TotTime))
        print("Mean time per body: " + str(TotTime/self.NBodies))
        print("Mean time per iteration: " + str(self.dt*TotTime/self.SimTime))
        print("Mean time per body per iteration: " + str(self.dt*TotTime/self.SimTime/self.NBodies))


    def run(self):

        if self.rank == 0:
            self.StartTime = time.time()

        for t in np.arange(self.dt, self.SimTime, self.dt, dtype = np.float32):    
            
            #MAIN LOOP
            
            if rank == 0:                                   #Split array that will be sent to each node
                self.CommPos = np.array_split(self.Pos, NumThreads)
                self.CommVel = np.array_split(self.Vel, NumThreads)    
                self.CommAcc = np.array_split(self.Acc, NumThreads)

            self.Pos = comm.bcast(self.Pos, root = 0)                 #broadcast positions of all the bodies
            self.LocalPos = comm.scatter(self.CommPos, root = 0)      #positions that will be updated

            if np.round(t) == np.round(self.dt):
                self.LocalVel = comm.scatter(self.CommVel, root = 0)  #sends vel and acc to every node for the first time
                self.LocalAcc = comm.scatter(self.CommAcc, root = 0)    
                self.SafetyValue = 1e-3       #needed to not divide by 0

            self.ComputeForce()
                
            NewPos = comm.gather(self.LocalPos, root = 0)    #each node sends LocalPos to root
            if rank == 0:
                self.Pos = np.concatenate(NewPos)            #Pos is update, needed for broadcasting
                self.PosHistory = np.concatenate((self.PosHistory,self.Pos))
        
        #SIMULATION ENDS
        if self.rank == 0:
            EndTime = time.time()
            self.ShowSimulationLog(self.StartTime, EndTime)   #basic infos
            self.PosHistory = self.PosHistory[self.NBodies:]       #eliminates the dummy 0's at the beginning
            #ShowPlot(PosHistory)
            
            if self.SaveRes:
                self.SaveResults(verbose = True)
                
            
            '''
            with open("run_out.npy", "rb") as f:    #loads data from file
                a = np.load(f, allow_pickle = True)
            '''
            
        #PosHistory[self.NBodies:].reshape(np.asscalar(np.array(SimTime/dt-1).astype(int)),self.NBodies,3)
    
instance = NBodySim('../config/TEST_run.yaml')
instance.run()