import numpy as np 
import matplotlib.pyplot as plt 
import husimi_sp as hu
import scipy.fftpack as sp 
import utils as u_


N = 256
L = 1. 
dx = L/N 
x = dx*(.5+np.arange(-N/2, N/2))
kx = 2*np.pi*sp.fftfreq(N, d = L/N)
u = np.asarray(sp.fftshift(2*np.pi*sp.fftfreq(N, d = dx)))
hbar = 1.0e-2
u = hbar*np.asarray(sp.fftshift(2*np.pi*sp.fftfreq(N, d = dx)))

dt = 1e-4
Tfinal = 5.

sig_x = dx*5
K = hu.K_H_(x, x.copy(), u, hbar, sig_x, L)

A = 100. 
B = 73.
C = B*1e-3
mu = L/4.

V = A*x**2 + C*x**4
def F(x):
    return -2*A*x- 4*B*C**3
V_ = V + B*x**4
def F_(x):
    return -2*A*x - 4*B*x**3


def GetPQ(psi):
    q = np.sum(x*np.abs(psi)**2)*dx
    q2 = np.sum(x*x*np.abs(psi)**2)*dx
    psi_k = sp.fft(psi)
    psi_k /= np.sqrt(np.sum(np.abs(psi_k)**2))
    p = np.sum(kx*np.abs(psi_k)**2)
    p2 = np.sum(kx*kx*np.abs(psi_k)**2)
    Varq = q2 - q**2
    Varp = hbar**2*(p2 - p**2)
    a2 = .5*(p+q*1j)
    return q, p*hbar, Varq / mu**2 


def makeFig(psi, psiT, psiT_ , psiT2, psiT2_,  r,v,rT,vT, rT2, vT2  ,rT_,vT_, rT2_, vT2_):
    fig, axs = plt.subplots(2,3,figsize = (16,12))

    H = hu.f_H(psi, K, dx)
    HT = hu.f_H(psiT, K, dx)
    HT_ = hu.f_H(psiT_, K, dx)
    HT2 = hu.f_H(psiT2, K, dx)
    HT2_ = hu.f_H(psiT2_, K, dx)

    cmin = 0.
    cmax = np.max(H)*.7

    im = axs[0][0].imshow(H, interpolation = 'none', extent=[-L/2.01, L/2.01, np.min(u),np.max(u)], aspect = 'auto', origin = 'lower',cmap = 'binary')
    im.set_clim(vmin=cmin, vmax=cmax)	
    axs[0][0].text(.8,.9,r'$T = 0$'  , ha='center', va='center', transform= axs[0][0].transAxes, bbox = {'facecolor': 'white', 'pad': 5})
    axs[0][0].plot(r,v, 'ro')
    p,q, Q = GetPQ(psi)
    axs[0][0].plot(p,q, 'c^', linewidth = 5)
    axs[0][0].text(.2,.9,r'$Q_q = %.2f$' %(Q) , ha='center', va='center', transform= axs[0][0].transAxes, bbox = {'facecolor': 'white', 'pad': 5})
    

    im = axs[1][0].imshow(H, interpolation = 'none', extent=[-L/2.01, L/2.01, np.min(u),np.max(u)], aspect = 'auto', origin = 'lower',cmap = 'binary')
    im.set_clim(vmin=cmin, vmax=cmax)	
    axs[1][0].plot(r,v, 'ro')
    p,q, Q = GetPQ(psi)
    axs[1][0].plot(p,q, 'c^', linewidth = 5)
    axs[1][0].text(.2,.9,r'$Q_q = %.2f$' %(Q) , ha='center', va='center', transform= axs[1][0].transAxes, bbox = {'facecolor': 'white', 'pad': 5})


    im = axs[0][1].imshow(HT, interpolation = 'none', extent=[-L/2.01, L/2.01, np.min(u),np.max(u)], aspect = 'auto', origin = 'lower',cmap = 'binary')
    im.set_clim(vmin=cmin, vmax=cmax)	
    axs[0][1].text(.8,.9,r'$T = 5$'  , ha='center', va='center', transform= axs[0][1].transAxes, bbox = {'facecolor': 'white', 'pad': 5})
    axs[0][1].plot(rT,vT, 'ro')
    p,q, Q = GetPQ(psiT)
    axs[0][1].plot(p,q, 'c^', linewidth = 5)
    axs[0][1].text(.2,.9,r'$Q_q = %.2f$' %(Q) , ha='center', va='center', transform= axs[0][1].transAxes, bbox = {'facecolor': 'white', 'pad': 5})


    im = axs[1][1].imshow(HT_, interpolation = 'none', extent=[-L/2.01, L/2.01, np.min(u),np.max(u)], aspect = 'auto', origin = 'lower',cmap = 'binary')
    im.set_clim(vmin=cmin, vmax=cmax)	
    axs[1][1].plot(rT_,vT_, 'ro')
    p,q, Q = GetPQ(psiT_)
    axs[1][1].plot(p,q, 'c^', linewidth = 5)
    axs[1][1].text(.2,.9,r'$Q_q = %.2f$' %(Q) , ha='center', va='center', transform= axs[1][1].transAxes, bbox = {'facecolor': 'white', 'pad': 5})

    im = axs[0][2].imshow(HT2, interpolation = 'none', extent=[-L/2.01, L/2.01, np.min(u),np.max(u)], aspect = 'auto', origin = 'lower',cmap = 'binary')
    im.set_clim(vmin=cmin, vmax=cmax)	
    axs[0][2].text(.8,.9,r'$T = 10$'  , ha='center', va='center', transform= axs[0][2].transAxes, bbox = {'facecolor': 'white', 'pad': 5})
    axs[0][2].plot(rT2,vT2, 'ro')
    p,q, Q = GetPQ(psiT2)
    axs[0][2].plot(p,q, 'c^', linewidth = 5)
    axs[0][2].text(.2,.9,r'$Q_q = %.2f$' %(Q) , ha='center', va='center', transform= axs[0][2].transAxes, bbox = {'facecolor': 'white', 'pad': 5})

    im = axs[1][2].imshow(HT2_, interpolation = 'none', extent=[-L/2.01, L/2.01, np.min(u),np.max(u)], aspect = 'auto', origin = 'lower',cmap = 'binary')
    im.set_clim(vmin=cmin, vmax=cmax)	
    axs[1][2].plot(rT2_,vT2_, 'ro')
    p,q, Q = GetPQ(psiT2_)
    axs[1][2].plot(p,q, 'c^', linewidth = 5)
    axs[1][2].text(.2,.9,r'$Q_q = %.2f$' %(Q) , ha='center', va='center', transform= axs[1][2].transAxes, bbox = {'facecolor': 'white', 'pad': 5})

    axs[0][0].set_xticklabels([])
    axs[0][1].set_yticklabels([])
    axs[0][1].set_xticklabels([])
    axs[1][1].set_yticklabels([])
    axs[0][2].set_yticklabels([])
    axs[0][2].set_xticklabels([])
    axs[1][2].set_yticklabels([])

    axs[1][0].set_xlabel(r'$q \, [L]$')
    axs[1][1].set_xlabel(r'$q \, [L]$')
    axs[1][2].set_xlabel(r'$q \, [L]$')

    axs[0][0].set_ylabel(r'$p \, [\hbar / L]$')
    axs[1][0].set_ylabel(r'$p \, [\hbar / L]$')

    x0 = axs[0][0].get_position().x0
    y0 = axs[0][0].get_position().y1
    h = .015
    w = axs[0][2].get_position().x1 - x0
    cbar_ax = fig.add_axes([x0,y0, w, h])
    cb = fig.colorbar(im, cax=cbar_ax, orientation = "horizontal")
    cbar_ax.xaxis.set_ticks_position("top")

    plt.subplots_adjust(wspace=0, hspace=0)    

    return fig



def EvolveWaveFuncs(psi_, Phi, r_, v_, a):
    T = 0
    psi = psi_.copy()

    r = r_
    v = v_
    while(T < Tfinal):

        r += v*dt/2. 
        v += a(r) * dt
        r += v*dt/2. 

        # update position half-step
        psi_k = sp.fft(psi)
        psi_k *= np.exp(-1j*dt*hbar*(kx**2)/(4.))
        psi = sp.ifft(psi_k)

        # update momentum full-step
        psi *= np.exp(-1j*dt*Phi/hbar)

        # update position half-step
        psi_k = sp.fft(psi)
        psi_k *= np.exp(-1j*dt*hbar*(kx**2)/(4.))
        psi = sp.ifft(psi_k)

        T += dt

    return psi, r, v 


def GetICs():
    sig = L/20.

    arg = (x-mu)**2 / sig**2
    psi = np.exp(-arg)
    psi /= np.sqrt(np.sum(np.abs(psi)**2)*dx )

    return psi, mu, 0.

if __name__ == "__main__":
    u_.orientPlot()    

    psi, r, v = GetICs()
    psiT, rT, vT = EvolveWaveFuncs(psi, V, r, v, F)
    psiT_, rT_, vT_ = EvolveWaveFuncs(psi, V_, r, v, F_)
    psiT2, rT2, vT2 = EvolveWaveFuncs(psiT, V, rT, vT, F)
    psiT2_, rT2_, vT2_ = EvolveWaveFuncs(psiT_, V_, rT_, vT_, F_)
    
    fig = makeFig(psi, psiT, psiT_ , psiT2, psiT2_,  r,v,rT,vT, rT2, vT2 ,  rT_,vT_, rT2_, vT2_)
    fig.savefig("../Figs/" + "QuantumErrors.pdf",bbox_inches = 'tight')