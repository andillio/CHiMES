import numpy as np
try:
    import cupy as cp
except ImportError:
    pass
import time
import datetime
import utils as u
import os



class SimObj(object):

    def __init__(self):

        # simulation parameters
        self.N = 5 # number of modes in simulation

        self.dt = 1.0e-3 # time step

        self.omega0 = 1.

        self.C = 2. # k dependent potential constant

        self.Lambda0 = .1  # k independent potential constant

        self.kord = np.arange(self.N)
        self.kx = None 
        self.x = None 
        self.dx = None 
        self.L = None
        self.dk = None

        self.T = 0 # total elapsed simulation time

        # meta parameters
        self.frames = 100 # total data drops
        self.framesteps = 10 # steps in between data drops

        self.time0 = time.time()

        self.ofile = "test"

        self.solvers = []

        self.hbar = 1. 
        self.mpart = 1.


    def MakeMetaFile(self, **kwargs):
        try:
            os.mkdir("../Data/" + self.ofile + "/")
        except OSError:
            pass
        f = open("../Data/" + self.ofile + "/" + self.ofile + "Meta.txt", 'w+')
        f.write("sim start: " + str(datetime.datetime.now()) + '\n\n')

        for key, value in kwargs.items():
            f.write(str(key) + ": " + str(value) + '\n')
        f.write('\n')

        f.close()


    # runs the simulation
    def Run(self, verbose = True):
        
        # loops through data drops
        for i in range(self.frames):
            
            # update the simulation time
            dt_ = self.dt
            self.T += dt_ * self.framesteps

            # loop through each solver
            for solver_ in self.solvers: 
                if solver_.working: # check if solver has broken

                    # ------------ step 3d ---------------- #
                    # update dynamical variables
                    solver_.Update(dt_, self)
                    # ------------------------------------- #
                
                if solver_.working:
                    # ------------ step 3e ---------------- #
                    # output state of dynamical variables
                    solver_.DataDrop(i, self.ofile)
                    # ------------------------------------- #

            # info drop on terminal
            if verbose:
                u.repeat_print(('%i hrs, %i mins, %i s remaining.' %u.remaining(i + 1, self.frames, self.time0)))


    def ValidIndex(self, i):
        return (i < self.N) and (i >= 0)

    def GetC(self, i, j):
        C_ij = 0.
        
        if (i != j):
            C_ij = self.C / (self.kord[i] - self.kord[j])**2 + self.Lambda0
        else:
            C_ij = self.Lambda0

        return C_ij

    def GetLam(self, ind1, ind2, ind3, ind4):
        C_ij = self.Lambda0
        
        if ind1 != ind3 and np.abs(self.C) > 0:
            C_ij += .5*self.C / (self.kord[ind1] - self.kord[ind3])**2
        if ind1 != ind4 and np.abs(self.C) > 0:
            C_ij += .5*self.C / (self.kord[ind1] - self.kord[ind4])**2

        return C_ij

    def EndSim(self, text = True):
        for solver_ in self.solvers:
            # output state of dynamical variables
            solver_.DataDrop(self.frames, self.ofile)
        if text:
            print('\nsim completed in %i hrs, %i mins, %i s' %u.hms(time.time()-self.time0))
            print("output: ", self.ofile)
