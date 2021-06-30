# imports
import SimObj as S
import ClObject as Cl
import DiffObj_RK as E 
import time 
import numpy as np 
import utils as u 
import scipy.fftpack as sp
import matplotlib.pyplot as plt
import di_FME_analysis

r = 1 # logrithmically scales the total number of particles
h = 1 # scales hbar
ofile = "Sin_h" + str(h) + "_r" + str(r) # data directory
frames = 600 # number of data drops
framesteps = 20 # number of time steps between data drops

CUPY = False # if true it will try run on cuda
CL = True # should a classical particles simulation be run
if CUPY:
    import DiffObjCP as E
    import DiffObj_RKcp as E 
    import cupy as np 
    import cupy.fft as sp
    import ClObjectCP as Cl


N = 256 # number of modes/grid cells
factor = 2**r
n = int(1024*factor) # number of particles
L = 1. # box size
dx = L/N # cell size
dk = 1./dx # inverse cell size
dt = 1e-4 # time step

Mtot = 1. # total mass
G = .1/np.pi/4. # Gravitational constant
mpart = Mtot / n # particle mass
hbar = 2.5e-4*mpart 
hbar_ = hbar / mpart 
Npart = (Mtot/mpart) # total number of particles 
Norm = np.sqrt(Mtot/mpart) # norm of classical field

x = dx*(.5+np.arange(-N/2, N/2)) # grid in natural order
kx = 2*np.pi*sp.fftfreq(N, d = L/N) # fourier grid in fourier order
kx_ord = sp.fftshift(kx) # fourier grid in natural order
y = sp.fftfreq(len(kx_ord), d=dk) *N # grid in fourier order

omega0 = hbar/mpart * h # kinetic constant
lambda0 = 0 # contact interaction constant
C = -4*np.pi*G*Mtot/omega0/Norm**2 # long range interaction constant

dIs = [di_FME_analysis] # data interpreters

# creates the simulation object
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

    s.kord = kx_ord
    s.kx = kx 
    s.x = x 
    s.dx = dx
    s.L = L 
    s.dk = dk 

    s.Norm = Norm

    s.T = 0 

    s.MakeMetaFile(N = N, dt = dt, frames = frames, framesteps = framesteps, Mtot = Mtot,
        omega0 = omega0, Lamda0 = lambda0, C= C, mpart = mpart, L = L, Norm = Norm, n = n)

    return s

# creates the field moment expansion solver
def initExp(s, aa_on = False, ba_on = False, tag = ""):
    # get dynamical variables object
    d = E.DynVars(N)

    d.kord = kx_ord

    d.is_dispersion_quadratic = True

    d.tag = tag

    # make the field op
    rho = 1. + .1*np.cos(y*2*np.pi)
    psi_y = np.sqrt(rho)

    R = sp.fft(psi_y, axis = 0)
    R = sp.fftshift(R, axes = (0))
    a = R / np.sqrt(N) + 0j
    d.a = a
    d.a /= np.sqrt((np.abs(d.a)**2).sum())
    d.a *= Norm

    d.aa_on = aa_on
    d.ba_on = ba_on

    d.gauss = False

    d.SetHigherMomentsFromField()

    d.ReadyDir(ofile)

    psi = sp.fftshift(sp.ifft(sp.ifftshift(d.a)) * np.sqrt(N))
    rho = np.abs(psi)**2

    return d, rho


# creates the classical particle solver
def initCl(s,rho):
    cl = Cl.DynVars(n)

    cl.mpart = Mtot / n

    cl.N = N 

    cl.C = 4*np.pi*G

    cl.SetPositionsFromDensity(x, rho, s) 

    cl.ReadyDir(ofile)

    return cl



def main():
    time0 = time.time() # timing

    s = initSim() # initialize the simulation object

    e_min, rho = initExp(s, False, False) # classical model
    e_max, rho = initExp(s, True, True, "max") # second order approx 

    if CL:
        cl = initCl(s,rho) # classical particles solver

    print('initialization completed in %i hrs, %i mins, %i s' %u.hms(time.time()-time0))
    print("begining sim", ofile)

    s.time0 = time.time()

    # set solvers
    if CL:
        s.solvers = [cl,e_min, e_max]
    else:
        s.solvers = [e_min, e_max]

    # ruun sim
    s.Run()
    s.EndSim()

    # data analysis
    time0 = time.time()
    if len(dIs) >= 1:
        print("begining data interpretation")
        for j in range(len(dIs)):
            dIs[j].main(ofile)
        print('analysis completed in %i hrs, %i mins, %i s' %u.hms(time.time()-time0))
    u.ding()

if __name__ == "__main__":
    main()