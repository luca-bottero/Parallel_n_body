from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import pathlib

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

HighMass = 1        #max generated mass
LowMass = 0.1       #min generated mass
HighVel = 0.       #max generated Vel
LowVel = -HighVel   #min generated Vel
HighPos = 50        #max generated Pos
LowPos = -HighPos   #min generated Pos

XWidth = 100    #Plot Axis range (-XWidth,XWidth)
YWidth = 100
ZWidth = 100

NBodies = 20  
G = 1           #np.float(6.67430e-11) Gravitational Costant
dt = 0.1        #timestep
SimTime = 30    #total simulation time
AnimDur = 1    #total animation time in seconds

NumThreads = comm.Get_size()    

def update_lines(num, dataLines, lines) :
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
        #line.set_marker("o")
    #for i in range(NBodies):
     #   line.set_data()
    return lines

def ShowPlot(PosHistory):
    t = np.linspace(dt,dt,SimTime)
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    data = PosHistory
    
    data = []
    TimeRange = np.asscalar(np.array(SimTime/dt-1).astype(int))

    for i in range(NBodies):
        dat = []
        for x in range(3):
            dat.append([PosHistory[NBodies*t + i][x] for t in range(TimeRange)])     #Pos of i-th body in time: 
        data.append(dat)
    
    data = np.asarray(data)

    trajectories = [ax.plot(dat[0, 0:TimeRange], dat[1, 0:TimeRange], dat[2, 0:TimeRange])[0] for dat in data] #[ax.plot(dat[0, -100:100],dat[1, -100:100],dat[2, -100:100])[0] for dat in data]

    # Setting the axes properties
    ax.set_xlim3d([-XWidth, XWidth])
    ax.set_xlabel('X')
    ax.set_ylim3d([-YWidth, YWidth])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-ZWidth, ZWidth])
    ax.set_zlabel('Z') 

    ax.set_title('Trajectories')

    # Creating the Animation object
    IntvTime = AnimDur/(SimTime/dt)
    line_ani = animation.FuncAnimation(fig, update_lines, fargs=(data, trajectories), interval = IntvTime, repeat = True)
    #line_ani.save("out.mp4", bitrate=-1)
    plt.show()

   
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
        
        with open("run_out.npy", "wb") as f:    #saves data to file
            np.save(f, PosHistory)
        
        '''
        with open("run_out.npy", "rb") as f:    #loads data from file
            a = np.load(f, allow_pickle = True)
        '''
        
        #PosHistory[NBodies:].reshape(np.asscalar(np.array(SimTime/dt-1).astype(int)),NBodies,3)
    
run()
plt.show()