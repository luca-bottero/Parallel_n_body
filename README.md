# Parallel n-body solver for python

A gravitational n-body simulator in Python with parallel support using mpi4py.

## How to use
Run using 4 cores
```
cd src
mpirun -n 4 python3 main.py
```
Profile with cProfiler and save the result
```
cd src
mpirun -n 4 python3 -m cProfile -s cumtime main.py > cprofile.txt
```

It does support multi-machine parallel calculations. If there are N bodies and M nodes, each node compute the force on N/M bodies and updates their positions.

The output is a file which contains the position of every body at every timestep. Also, a 3D animation can be produced. 


