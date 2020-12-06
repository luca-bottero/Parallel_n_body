from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

HighMass = 10
LowMass = 1

NBodies = 10
G = 1 #np.float(6.67430e-11)
dt = 0.1
SimTime = 10

NumThreads = comm.Get_size()    
    
def run():
    if rank == 0:
        Pos = np.zeros([NBodies,3], dtype = np.float64)     #root initialization
        Vel = np.zeros([NBodies,3], dtype = np.float64)
        Acc = np.zeros([NBodies,3], dtype = np.float64)
        for i in range(NBodies):
            for j in range(3):
                Pos[i][j] = np.random.uniform(low = -10, high = 10)    
    else:
        Pos = None          #threads initialization
        CommPos = None
        CommVel = None
        CommAcc = None
        LocalPos = None
        LocalVel = None
        LocalAcc = None
        
    MassGen = np.random.uniform(low = LowMass, high = HighMass, size = NBodies)     #Masses, in kg
    Mass = comm.bcast(MassGen, root = 0)

    '''
    https://info.gwdg.de/wiki/doku.php?id=wiki:hpc:mpi4py#gather
    '''

    for t in np.arange(dt, SimTime, dt, dtype = np.float):
        if rank == 0:                                   #Split array that will be sent to each node
            CommPos = np.array_split(Pos, NumThreads)
            CommVel = np.array_split(Vel, NumThreads)    
            CommAcc = np.array_split(Acc, NumThreads)

        Pos = comm.bcast(Pos, root = 0)                 #broadcast positions of all the bodies
        LocalPos = comm.scatter(CommPos, root = 0)      #positions that will be updated

        if t == dt:
            LocalVel = comm.scatter(CommVel, root = 0)  #sends vel and acc to every node for the first time
            LocalAcc = comm.scatter(CommAcc, root = 0)    
            SafetyVec = np.array([1.,1.,1.])*1e-3        

        for i in range(len(LocalAcc)):      #calculate accelerations
            for j in range(NBodies):
                if np.array_equal(LocalPos[i],Pos[j]) == False:
                    r = LocalPos[i] - Pos[j]
                    DCube = (np.inner(r,r) + SafetyVec)**1.5
                    LocalAcc[i] += r*G*Mass[j]/DCube
            LocalVel[i] += LocalAcc[i]*dt                       #update velocity
            LocalPos[i] += 0.5*LocalAcc[i]*dt*dt + LocalVel[i]  #update local positions
            
        NewPos = comm.gather(LocalPos, root = 0)    #each node sends LocalPos to root
        if rank == 0:
            Pos = np.concatenate(NewPos)            #Pos is update, needed for broadcasting
            print(Pos[0])
        


run()   

    






