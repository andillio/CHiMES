import SimObj as S 
#import DiffObj_RK as E 
import FME_solverObjOdd as E 
import time 
import numpy as np
import utils as u
import di_FME_analysis
import scipy.fftpack as sp

r = 1
#ofile = "M4test_DOE_r"+str(r) # name of directory to be created
ofile = "Odd_M4_r" + str(r) # name of directory to be created
#ofile = "testRepl_r" + str(r) # name of directory to be created
#ofile = "testExpFast" + str(r) # name of directory to be created
gauss = False

dt = 1e-4 / np.sqrt(r) / 4.

frames = 300
framesteps = int(256 * np.sqrt(r)) * 4

#IC = np.asarray([0,24,32,0,0])
#IC = np.asarray([12,8,10,8,6])
IC = np.asarray([0,2,2,1]) * r
#IC = np.asarray([0,2,2,1,0]) * r
N = len(IC)

omega0 = 1. 
lambda0 = 0.#-omega0/10. 
C = -.1 / r

dIs = [di_FME_analysis] # data interpreters

def initSim():
    print("Initializing simultion object")
    s = S.SimObj()

    s.N = N
    s.dt = dt 
    s.omega0 = omega0
    s.C = C
    s.Lambda0 = lambda0 
    s.frames = frames
    s.framesteps = framesteps
    s.ofile = ofile

    s.kord = np.arange(N) - N/2
    s.kx = sp.fftshift(s.kord)
    kmax = np.max(np.abs(s.kord))
    s.L = np.pi*s.N/kmax
    s.dx = s.L/s.N 
    s.x = s.dx*(.5+np.arange(-s.N/2, s.N/2))
    s.dk = 1./s.dx

    s.Norm = np.sqrt(np.sum(IC))

    s.MakeMetaFile(N = N, dt = dt, frames = frames, framesteps = framesteps, IC = IC,
        omega0 = omega0, Lamda0 = lambda0)

    return s


def initExp(s, aa_on = False, ba_on = False, tag = ""):
    # get dynamical variables object
    d = E.DynVars(N)

    d.kord = np.arange(N) - N/2

    d.is_dispersion_quadratic = True

    d.tag = tag

    d.a = np.sqrt( IC ) + 0j
    #d.a *= np.exp(np.random.uniform(0,2*np.pi,N)*1j)
    np.random.seed(1)
    phi = np.random.uniform(0, 2 * np.pi, s.N)
    d.a *= np.exp(phi*1j)

    d.aa_on = aa_on
    d.ba_on = ba_on

    d.gauss = gauss

    d.SetHigherMomentsFromField()

    d.kord = np.arange(N) - N/2

    d.ReadyDir(ofile)

    return d


def main():
    time0 = time.time()

    s = initSim() # initialize the simulation object

    e_min = initExp(s, False, False) # classical model
    #e_mid = initExp(s, True, False, "mid") # classical model
    e_max = initExp(s, True, True, "max") # second order approx

    print 'initialization completed in %i hrs, %i mins, %i s' %u.hms(time.time()-time0)
    print "begining sim", ofile

    s.time0 = time.time()
    s.solvers = [e_min, e_max]
    #s.solvers = [e_min]
    s.Run()
    s.EndSim()
    
    time0 = time.time()

    tags_ = ['',  'max']
    #tags_ = ['']
    if len(dIs) >= 1:
        print "begining data interpretation"

        for j in range(len(dIs)):
            for i in range(len(tags_)):
                dIs[j].main(ofile, tags_[i])
        print 'analysis completed in %i hrs, %i mins, %i s' %u.hms(time.time()-time0)
    u.ding()


if __name__ == '__main__':
    main()