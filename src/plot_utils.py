import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

class PlotUtils():
    def __init__(self, ResultsPath = ''):
        try:
            self.PosHistory = np.load(ResultsPath + '/PosHistory.npy')
            self.Mass = np.load(ResultsPath + '/Masses.npy')
            #load yaml with configurations
        except Exception as e:
            print('An exception occurred:')
            print(e)

        self.NBodies = self.Mass.shape[0]
        self.dt = 0.1
        self.SimTime = 10000 #np.round(self.PosHistory.shape[0]/self.NBodies)

    def Animation3D(self):
        fig = plt.figure()
        ax = p3.Axes3D(fig)

        #data = PosHistory
        
        data = []
        TimeRange = np.asscalar(np.array(self.SimTime/self.dt).astype(int))

        print(TimeRange)

        for i in range(self.NBodies):
            dat = []
            for x in range(3):
                dat.append([self.PosHistory[self.NBodies*t + i][x] for t in range(TimeRange - 1)])     #Pos of i-th body in time: 
            data.append(dat)
        
        data = np.asarray(data)         #bodies positions
        trajectories = [ax.plot(dat[0, 0:TimeRange], dat[1, 0:TimeRange], dat[2, 0:TimeRange])[0] for dat in data]  #trajectories line

        # Setting the axes properties
        '''ax.set_xlim3d([-XWidth, XWidth])
        ax.set_xlabel('X')
        ax.set_ylim3d([-YWidth, YWidth])
        ax.set_ylabel('Y')
        ax.set_zlim3d([-ZWidth, ZWidth])
        ax.set_zlabel('Z') 
        ax.set_title('Trajectories')'''

        def init():
            ln1, = plt.plot([], [], '-r')
            ln2, = plt.plot([], [], '-b')
            ln3, = plt.plot([], [], '-g')
            return ln1,ln2,ln3,
        
        def update_lines(num, dataLines, lines) :       #needed to animate trajectories
            for line, data in zip(lines, dataLines):
                #line.set_data(data[0:2, :num])         #for lines
                #line.set_3d_properties(data[2, :num])
                line.set_data(data[0:2, num-1])          #for dots
                line.set_3d_properties(data[2, num-1])
                line.set_marker("o")
            return lines

        # Creating the Animation object
        #IntvTime = AnimDuration/(self.SimTime/self.dt)     #used to make an animation that has a duration of AnimDuration
        line_ani = animation.FuncAnimation(fig, update_lines, fargs=(data, trajectories), init_func=init, repeat = True, blit = True)
        #line_ani.save("out.mp4",fps = 10, dpi = 200)    #saving animation as a mp4 movie, needs fix
        
        return line_ani
