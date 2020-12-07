from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

HighMass = 1   #maxi generated mass
LowMass = 0.1    #mini generated mass
HighVel = 0.1   #maxi generated Vel
LowVel = -0.1   #mini generated Vel
HighPos = 50   #maxi generated Pos
LowPos = -50    #mini generated Pos

XWidth = 100    #Plot Axis range (-XWidth,XWidth)
YWidth = 100
ZWidth = 100

NBodies = 10
G = 1           #np.float(6.67430e-11) Gravitational Costant
dt = 0.01        #timestep
SimTime = 30    #total simulation time

NumThreads = comm.Get_size()    


def update_lines(PosHistory) :
    for line, data in zip(lines, dataLines) :
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2,:num])
    return lines

def ShowPlot(PosHistory):
    t = np.linspace(dt,dt,SimTime)
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    data = PosHistory
    
    data = []
    timerange = np.asscalar(np.array(SimTime/dt-1).astype(int))

    for i in range(NBodies):
        dat = []
        for x in range(3):
            dat.append([PosHistory[NBodies*t + i][x] for t in range(timerange)])     #Pos of i-th body in time
        data.append(dat)
    
    data = np.asarray(data)

    trajectories = [ax.plot(dat[0, 0:timerange], dat[1, 0:timerange], dat[2, 0:timerange])[0] for dat in data] #[ax.plot(dat[0, -100:100],dat[1, -100:100],dat[2, -100:100])[0] for dat in data]

    # Setting the axes properties
    ax.set_xlim3d([-XWidth, XWidth])
    ax.set_xlabel('X')
    ax.set_ylim3d([-YWidth, YWidth])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-ZWidth, ZWidth])
    ax.set_zlabel('Z')

    ax.set_title('Trajectories')

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, fargs=(data, trajectories),
                                interval=500, blit=False)

   
def run():
    if rank == 0:
        PosHistory = np.zeros([NBodies,3])
        Pos = np.zeros([NBodies,3], dtype = np.float64)     #root initialization
        Vel = np.zeros([NBodies,3], dtype = np.float64)
        Acc = np.zeros([NBodies,3], dtype = np.float64)
        for i in range(NBodies):
            for j in range(3):
                Pos[i][j] = np.random.uniform(low = LowPos, high = HighPos)    
        for i in range(NBodies):
            for j in range(3):
                Vel[i][j] = np.random.uniform(low = LowVel, high = HighVel)  

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
            SafetyVec = np.array([1.,1.,1.])*1e-3       #needed to not divide by 0

        for i in range(len(LocalAcc)):      #calculate accelerations
            for j in range(NBodies):
                if np.array_equal(LocalPos[i],Pos[j]) == False:
                    r = LocalPos[i] - Pos[j]
                    DCube = (np.inner(r,r) + SafetyVec)**1.5
                    LocalAcc[i] += -r*G*Mass[j]/DCube
            LocalVel[i] += LocalAcc[i]*dt                       #update velocity
            LocalPos[i] += 0.5*LocalAcc[i]*dt*dt + LocalVel[i]  #update local positions
            
        NewPos = comm.gather(LocalPos, root = 0)    #each node sends LocalPos to root
        if rank == 0:
            Pos = np.concatenate(NewPos)            #Pos is update, needed for broadcasting
            PosHistory = np.concatenate((PosHistory,Pos))
            #print(Pos)
            #print("\n")
    
    if rank == 0:
        PosHistory = PosHistory[NBodies:]       #eliminates the dummy 0's at the beginning
        ShowPlot(PosHistory)
        #PosHistory[NBodies:].reshape(np.asscalar(np.array(SimTime/dt-1).astype(int)),NBodies,3)
    
run()
plt.show()