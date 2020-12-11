from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import time

'''
TO DO LIST:
Evaluate use of ALLGATHER instead of going through root each time
Fix inefficeincies in simulation code 

Fix the mp4 movie output

Run on multiple computers

Clean the code
    separate different part in different files

Initial condition from file

More accurate benchmark:
    create simple benchmark tool for either single and multiple computers

Check for collision and implement collision mechanism
'''


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

HighMass = 10        #max generated mass
LowMass = 0.1       #min generated mass
HighVel = 0.       #max generated Vel
LowVel = -HighVel   #min generated Vel
HighPos = 100        #max generated Pos
LowPos = -HighPos   #min generated Pos

LCube = 200
XWidth = LCube    #Plot Axis range (-XWidth,XWidth)
YWidth = LCube
ZWidth = LCube

NBodies = 64    #number of bodies to simulate
G = 1           #np.float(6.67430e-11) Gravitational Costant
dt = 0.1        #timestep
SimTime = 300    #total simulation time
AnimDuration = 20    #total animation time in seconds

NumThreads = comm.Get_size()    

def update_lines(num, dataLines, lines) :       #needed to animate trajectories
    for line, data in zip(lines, dataLines):
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
        #line.set_marker("o")
    return lines

def ShowPlot(PosHistory):               #Plots an animation of the trajectories in time and saves it to a mp4 movie
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
    
    data = np.asarray(data)         #bodies positions
    trajectories = [ax.plot(dat[0, 0:TimeRange], dat[1, 0:TimeRange], dat[2, 0:TimeRange])[0] for dat in data]  #trajectories line

    # Setting the axes properties
    ax.set_xlim3d([-XWidth, XWidth])
    ax.set_xlabel('X')
    ax.set_ylim3d([-YWidth, YWidth])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-ZWidth, ZWidth])
    ax.set_zlabel('Z') 
    ax.set_title('Trajectories')

    # Creating the Animation object
    IntvTime = AnimDuration/(SimTime/dt)     #used to make an animation that has a duration of AnimDuration
    line_ani = animation.FuncAnimation(fig, update_lines, fargs=(data, trajectories), interval = IntvTime, repeat = True)
    line_ani.save("out.mp4",fps = 10, dpi = 200)    #saving animation as a mp4 movie
    plt.show()

def ShowSimulationLog(StartTime, EndTime):      #shows basics informations about the simulation
    TotTime = EndTime - StartTime
    print("Number of bodies: " + str(NBodies))
    print("Total iteration: " + str(SimTime/dt))
    print("Total time: " + str(TotTime))
    print("Mean time for body: " + str(TotTime/NBodies))
    print("Mean time for iteration: " + str(dt*TotTime/SimTime))
    print("Mean time for body for iteration: " + str(dt*TotTime/SimTime/NBodies))


   
def run():
    if rank == 0:
        PosHistory = np.zeros([NBodies,3])
        Pos = np.zeros([NBodies,3], dtype = np.float64)     #root blank initialization
        Vel = np.zeros([NBodies,3], dtype = np.float64)
        Acc = np.zeros([NBodies,3], dtype = np.float64)

        for i in range(NBodies):        #Random generation of positions and velocities
            for j in range(3):
                Pos[i][j] = np.random.uniform(low = LowPos, high = HighPos)    
        for i in range(NBodies):
            for j in range(3):
                Vel[i][j] = np.random.uniform(low = LowVel, high = HighVel)  
    else:
        Pos = None          #thread's vectors initialization initialization
        CommPos = None
        CommVel = None
        CommAcc = None
        LocalPos = None
        LocalVel = None
        LocalAcc = None
        
    MassGen = np.random.uniform(low = LowMass, high = HighMass, size = NBodies)     #Masses random generation, in kg
    Mass = comm.bcast(MassGen, root = 0)

    #SIMULATION
    if rank == 0:
        StartTime = time.time()

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
            SafetyValue = 1e-3       #needed to not divide by 0

        for i in range(len(LocalAcc)):      #calculate accelerations
            LocalAcc[i] = 0.
            for j in range(NBodies):
                if np.array_equal(LocalPos[i],Pos[j]) == False:
                    r = LocalPos[i] - Pos[j]                    #displacement vector
                    DCube = (np.inner(r,r) + SafetyValue)**1.5    #compute distance
                    LocalAcc[i] += -r*G*Mass[j]/DCube           #compute acceleration
            LocalVel[i] += LocalAcc[i]*dt                       #update velocity
            LocalPos[i] += 0.5*LocalAcc[i]*dt*dt + LocalVel[i]  #update local positions
            
        NewPos = comm.gather(LocalPos, root = 0)    #each node sends LocalPos to root
        if rank == 0:
            Pos = np.concatenate(NewPos)            #Pos is update, needed for broadcasting
            PosHistory = np.concatenate((PosHistory,Pos))
    
    if rank == 0:
        EndTime = time.time()
        ShowSimulationLog(StartTime, EndTime)   #basic infos
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