import numpy as np
try:
    import cupy as cp
except ImportError:
    pass
import time
import datetime
import utils as u
import os
import scipy.fftpack as sp

# class for central moment expansion dynamic variables
class DynVars(object):

    # takes N, grid size of type int
    def __init__(self, N):

        print "Diff Obj"

        self.a = np.ones(N) + 0j

        self.dada = np.zeros((N,N)) + 0j

        self.dbda = np.zeros((N,N)) + 0j

        self.aa_on = False
        self.ba_on = False

        self.gauss = False

        self.tag = ""

        self.working = True

        self.is_dispersion_quadratic = False
        self.kord = np.arange(N) # momentum modes


    def SetHigherMomentsFromField(self):
        N = len(self.a)
        self.dada = np.zeros((N,N)) + 0j

        self.dbda = np.zeros((N,N)) + 0j


    def ReadyDir(self,ofile):
        try:
            os.mkdir("../Data/" + ofile + "/a" + self.tag)
        except OSError:
            pass
        
        if self.aa_on:
            try:
                os.mkdir("../Data/" + ofile + "/aa" + self.tag)
            except OSError:
                pass
        
        if self.ba_on:
            try:
                os.mkdir("../Data/" + ofile + "/ba" + self.tag)
            except OSError:
                pass

    def DataDrop(self,i,ofile_):
        np.save("../Data/" + ofile_ + "/a" + self.tag + "/" + "drop" + str(i) + ".npy", self.a)
        if self.aa_on:
            np.save("../Data/" + ofile_ + "/aa" + self.tag + "/" + "drop" + str(i) + ".npy", self.dada + np.outer(self.a, self.a))
        if self.ba_on:
            np.save("../Data/" + ofile_ + "/ba" + self.tag + "/" + "drop" + str(i) + ".npy", self.dbda + np.outer(np.conj(self.a), self.a))

    def Update(self, dt_, s):

        for i in range(s.framesteps):

            a0 = self.a.copy()
            b0 = np.conj(a0).copy()
            dada0 = self.dada.copy()
            dbda0 = self.dbda.copy()

            k_a1 = self.Update_a(dt_,a0, b0, dada0, dbda0, s)
            k_dada1 = self.Update_aa(dt_, a0, b0, dada0, dbda0, s)
            k_dbda1 = self.Update_ba(dt_, a0, b0, dada0, dbda0, s)

            a1 = a0 + k_a1*.5
            b1 = np.conj(a1)
            dada1 = dada0 + k_dada1*.5
            dbda1 = dbda0 + k_dbda1*.5

            k_a2 = self.Update_a(dt_,a1, b1, dada1, dbda1, s)
            k_dada2 = self.Update_aa(dt_, a1, b1, dada1, dbda1, s)
            k_dbda2 = self.Update_ba(dt_, a1, b1, dada1, dbda1, s)

            a2 = a0 + k_a2*.5
            b2 = np.conj(a2)
            dada2 = dada0 + k_dada2*.5
            dbda2 = dbda0 + k_dbda2*.5

            k_a3 = self.Update_a(dt_,a2, b2, dada2, dbda2, s)
            k_dada3 = self.Update_aa(dt_, a2, b2, dada2, dbda2, s)
            k_dbda3 = self.Update_ba(dt_, a2, b2, dada2, dbda2, s)

            a3 = a0 + k_a3
            b3 = np.conj(a3)
            dada3 = dada0 + k_dada3
            dbda3 = dbda0 + k_dbda3

            k_a4 = self.Update_a(dt_,a3, b3, dada3, dbda3, s)
            k_dada4 = self.Update_aa(dt_, a3, b3, dada3, dbda3, s)
            k_dbda4 = self.Update_ba(dt_, a3, b3, dada3, dbda3, s)

            # update a
            self.a += (1./6)*(k_a1 + 2*k_a2 + 2*k_a3 + k_a4) 
            #print (self.a - a).sum()
            #assert(0)
            if self.aa_on:
                # update aa
                self.dada += (1./6)*(k_dada1 + 2*k_dada2 + 2*k_dada3 + k_dada4)
            if self.ba_on:
                # update ab
                self.dbda += (1./6)*(k_dbda1 + 2*k_dbda2 + 2*k_dbda3 + k_dbda4)


        if True in np.isnan(self.a):
            self.working = False
        if True in np.isnan(self.dada):
            self.working = False
        if True in np.isnan(self.dbda):
            self.working = False
        if self.getQ(s) > .5:
            print "exceeded Q threshold, " + self.tag
            self.working = False

    def Update_a(self, dt_, a, b, dada, dbda, s):
        a_ = np.zeros(len(a)) + 0j

        One = np.ones(s.N)

        zeros = np.zeros(s.N/2)
        zeros2j = np.zeros((s.N,s.N/2))
        zeros2iL = np.zeros((s.N/2,2*s.N))

        a_pad = np.concatenate((zeros, a, zeros))

        pad_psi = sp.ifft(sp.ifftshift(a_pad)) * np.sqrt(2*s.N) #* s.dk
        pad_psi_ = np.conj(pad_psi)

        kx_pad = 2*np.pi*sp.fftfreq(2*s.N, d = s.L/s.N)
        kern_alt_pad = 1./kx_pad**2
        kern_alt_pad[kx_pad == 0] = 0

        if self.is_dispersion_quadratic:
            a_ += -1j * dt_ * .5 * self.kord**2 * s.omega0*a
        else:
            a_ += -1j * dt_ * self.kord * s.omega0*a

        # C_ij * b[i] * a[j] * a[p+i-j]
        term = pad_psi*pad_psi_ * dt_ * s.C 
        term = sp.fft(term)
        term = term * kern_alt_pad
        term = sp.ifft(term)
        term = term*pad_psi
        term = self.psi2a(term,s)[s.N/2:3*s.N/2]
        da_ = term * np.sqrt(s.N)**2 / np.sqrt(8.) #* np.sqrt(s.L)**6
        a_ += -1j * da_

        if self.ba_on:
            #  dbda[i][j] * a[p+i-j]
            term = np.concatenate((zeros2j,dbda,zeros2j), axis=1) 
            term = np.concatenate((zeros2iL,term,zeros2iL), axis=0)
            term = self.b2psi_(term,s, axis = 0)
            term = self.a2psi(term,s, axis = 1)
            term = np.diag(term)
            term = sp.fft(term)
            term = term * kern_alt_pad
            term = sp.ifft(term)
            term = term * pad_psi * dt_ * s.C 
            term = self.psi2a(term, s)[s.N/2:3*s.N/2]
            da_ = term / np.sqrt(8.) # / np.sqrt(s.N)**2  * np.sqrt(s.L)**4
            a_ += -1j* da_ 

            # dbda[i][p+i-j] * a[j]
            term = np.concatenate((zeros2j,dbda,zeros2j), axis=1)
            term = np.concatenate((zeros2iL,term,zeros2iL), axis=0)
            term = self.b2psi_(term,s, axis = 0) * dt_ * s.C 
            term = np.einsum("ij,i->ij", term, pad_psi)
            term = sp.fft(term, axis = 0)
            term = np.einsum("ij,i->ij", term, kern_alt_pad)
            term = sp.ifft(term, axis = 0)
            term = self.a2psi(term,s, axis = 1)
            term = np.diag(term)
            term = self.psi2a(term,s)[s.N/2:3*s.N/2]
            da_ = term / np.sqrt(8.)# / np.sqrt(s.N)**2  * np.sqrt(s.L)**4    
            a_ += -1j* da_
        
        if self.aa_on:
            # dada[p+i-j][j] * b[i]
            term = np.concatenate((zeros2j,dada,zeros2j), axis=1)
            term = np.concatenate((zeros2iL,term,zeros2iL), axis=0)
            term = self.a2psi(term,s, axis = 1) * dt_ * s.C 
            term = np.einsum("ij,j->ij", term, pad_psi_)
            term = sp.fft(term, axis = 1)
            term = np.einsum("ij,j->ij", term, kern_alt_pad)
            term = sp.ifft(term, axis = 1)
            term = self.a2psi(term,s, axis = 0)
            term = np.diag(term)
            term = self.psi2a(term,s)[s.N/2:3*s.N/2]
            da_ = term / np.sqrt(2.) * np.sqrt(s.N)**2 # * np.sqrt(s.L)**6 / np.sqrt(s.N)**2
            a_ += -1j* da_
        return a_

    def psi2a(self,A, s, axis=0): # defined on k_ord
        R = sp.fft(A, axis = axis)
        R = sp.fftshift(R, axes = (axis))
        return R / np.sqrt(s.N)#**3)

    def a2psi(self,A, s, axis=0): # defined on y
        return sp.ifft(sp.ifftshift(A, axes = (axis)), axis = axis) * np.sqrt(s.N) #* s.dk

    def b2psi_(self,A, s, axis=0): # defined on y
        return sp.fft(sp.ifftshift(A, axes = (axis)), axis = axis) * np.sqrt(s.N)

    def psi_2b(self, A, s, axis=0): # defined on y
        return sp.fftshift(sp.ifft(A, axis = axis), axes = (axis)) / np.sqrt(s.N)

    def Update_aa(self, dt_, a, b, dada, dbda, s):
        aa = self.dada + np.outer(self.a, self.a) + 0j
        aa_ = np.zeros((len(a), len(a))) + 0j
        zeros = np.zeros(s.N/2)
        zeros2j = np.zeros((s.N,s.N/2))
        zeros2i = np.zeros((s.N/2,s.N))
        zeros2iL = np.zeros((s.N/2,2*s.N))

        One = np.ones(s.N)

        a_pad = np.concatenate((zeros, a, zeros))

        pad_psi = sp.ifft(sp.ifftshift(a_pad)) * np.sqrt(2*s.N)# * s.dk
        pad_psi_ = np.conj(pad_psi)

        One = np.ones(s.N)
        deltak = np.einsum("i,j->ij", s.kord, One) - np.einsum("i,j->ji", s.kord, One)
        kern = 1./deltak**2
        kern[deltak == 0] = 0
        kern_alt = 1./s.kx**2
        kern_alt[s.kx == 0] = 0
        kern_alt_ord = 1./s.kord**2
        kern_alt_ord[s.kord==0]=0
        kx_pad = 2*np.pi*sp.fftfreq(2*s.N, d = s.L/s.N)
        kern_alt_pad = 1./kx_pad**2
        kern_alt_pad[kx_pad == 0] = 0

        kx_pad_ord = sp.fftshift(kx_pad)
        kern_alt_pad_ord = 1./kx_pad_ord**2
        kern_alt_pad_ord[kx_pad_ord == 0] = 0

        if self.is_dispersion_quadratic:
            aa_ += -1j * dt_ * .5 * (np.einsum("i,j->ij", s.kord, One)**2 + np.einsum("i,j->ji", s.kord, One)**2)*s.omega0*dada
        else:
            aa_ += -1j * dt_ * (np.einsum("i,j->ij", s.kord, One) + np.einsum("i,j->ji", s.kord, One))*s.omega0*dada

        
        # C_ij*aa[l_,j_]*b[k_]*a[p]  
        term = np.concatenate((zeros2i,dada,zeros2i), axis = 0)
        term = self.a2psi(term,s, axis = 0) * dt_ * s.C 
        term = np.einsum("ij,i->ij", term, pad_psi_)
        term = sp.fft(term, axis = 0)
        term = np.einsum("ij,i->ij", term, kern_alt_pad)
        term = sp.ifft(term, axis = 0)
        term = np.einsum("ij,i->ij", term, pad_psi)
        term = self.psi2a(term,s, axis = 0)[s.N/2:3*s.N/2]
        daa_ = term * np.sqrt(s.N)**2 / np.sqrt(4.)# * np.sqrt(s.L)**6
        aa_ += -1j  * (daa_ + daa_.T)
    

        # C_ij*aa[j_,p]*b[k_]*a[l_]
        aa_pad = np.concatenate((zeros2j, dada, zeros2j), axis = 1)
        aa_pad = self.a2psi(aa_pad,s, axis = 1) * dt_ * s.C
        term = pad_psi*pad_psi_
        term = sp.fft(term)
        term = term*kern_alt_pad
        term = sp.ifft(term)
        term = np.einsum("ij,j->ij", aa_pad, term)
        term = self.psi2a(term,s, axis = 1)
        term = term[:,s.N/2:3*s.N/2]
        term = np.einsum("ij->ji", term)
        daa_ =  term * np.sqrt(s.N)**2 / np.sqrt(4.) #* np.sqrt(s.L)**6
        aa_ += -1j  * (daa_ + daa_.T) 

        # C_ij*ba[k_,j_]*a[l_]*a[p]
        term = np.concatenate( (zeros2i, dbda, zeros2i), axis = 0 )
        term = self.b2psi_(term,s, axis = 0)  * dt_ * s.C
        term = np.einsum("ij,i->ij", term, pad_psi)
        term = sp.fft(term, axis = 0)
        term = np.einsum("ij,i->ij", term, kern_alt_pad)
        term = sp.ifft(term, axis = 0) # get V_y on axis 0
        term = np.einsum("ij,i->ij", term, pad_psi) 
        daa_ = self.psi2a(term,s, axis = 0)[s.N/2:3*s.N/2, :] / np.sqrt(16)
        aa_ += -1j * (daa_ + daa_.T)

        # C_ni * aa[n][l]
        aa_pad = np.concatenate((zeros2j, aa, zeros2j), axis = 1)
        aa_pad = np.concatenate((zeros2iL, aa_pad, zeros2iL), axis =0)
        K = np.diag(kern_alt_pad_ord)
        K = self.a2psi(K,s, axis = 0)
        K = self.a2psi(K,s, axis = 1)
        aa_x = self.a2psi(aa_pad,s, axis = 0)
        aa_x = self.b2psi_(aa_x,s, axis = 1) * dt_ * s.C
        term = K*aa_x 
        term = self.psi2a(term,s, axis = 0)
        term = self.psi_2b(term,s, axis = 1)  * np.sqrt(s.N)**2 
        daa_ = term[s.N/2:3*s.N/2,s.N/2:3*s.N/2]
        aa_ += -1j * daa_

        return aa_

    def Update_ba(self, dt_, a, b, dada, dbda, s):
        ba_ = np.zeros((len(a), len(a)))+0j
        zeros = np.zeros(s.N/2)
        zeros2i = np.zeros((s.N/2,s.N))

        One = np.ones(s.N)


        kx_pad = 2*np.pi*sp.fftfreq(2*s.N, d = s.L/s.N)
        kern_alt_pad = 1./kx_pad**2
        kern_alt_pad[kx_pad == 0] = 0
         
        a_pad = np.concatenate((zeros, a, zeros))
        pad_psi = sp.ifft(sp.ifftshift(a_pad)) * np.sqrt(2*s.N) #* s.dk
        pad_psi_ = np.conj(pad_psi)

        if self.is_dispersion_quadratic:
            ba_ += 1j * dt_ * .5 * (np.einsum("i,j->ij", s.kord, One)**2 - np.einsum("i,j->ji", s.kord, One)**2) * s.omega0*dbda
        else:
            ba_ += 1j * dt_ * (np.einsum("i,j->ij", s.kord, One) - np.einsum("i,j->ji", s.kord, One)) *s.omega0*dbda

        # dada[j][k] * b[i+k-l] * b[l]
        term = np.concatenate((zeros2i,dada,zeros2i), axis = 0)
        term = self.a2psi(term, s, axis = 0) * dt_ * s.C
        term = np.einsum("ij,i->ij", term, pad_psi_)
        term = sp.fft(term, axis = 0)
        term = np.einsum("ij,i->ij", term, kern_alt_pad)
        term = sp.ifft(term, axis = 0)
        term = np.einsum("ij,i->ij", term, pad_psi_)
        term = self.psi_2b(term, s, axis = 0)[s.N/2:3*s.N/2]
        daa_ = 1j  * term * np.sqrt(s.N)**4#2 * np.sqrt(s.L)**6
        ba_ += daa_ + np.conj(daa_.T)

        # dbda[i+k-l][j] * b[l] * a[k]
        term = pad_psi*pad_psi_ * dt_ * s.C 
        term = sp.fft(term)
        term = term * kern_alt_pad
        term = sp.ifft(term)
        ba_pad = np.concatenate((zeros2i,dbda,zeros2i), axis = 0)
        ba_pad = self.b2psi_(ba_pad,s, axis = 0)
        term = np.einsum("ij,i->ij", ba_pad, term)
        term = self.psi_2b(term,s, axis = 0)
        daa_ = 1j * term[s.N/2:3*s.N/2] * np.sqrt(s.N)**2 / 2.# * np.sqrt(s.L)**4
        ba_ += daa_ + np.conj(daa_.T)

        #  ba[l][j] * b[p] * a[k]
        term = np.concatenate( (zeros2i, dbda, zeros2i), axis = 0 )
        term = self.b2psi_(term,s, axis = 0) * dt_ * s.C
        term = np.einsum("ij,i->ij", term, pad_psi)
        term = sp.fft(term, axis = 0)
        term = np.einsum("ij,i->ij", term, kern_alt_pad)
        term = sp.ifft(term, axis = 0)
        term = np.einsum("ij,i->ij", term, pad_psi_)
        term = self.psi_2b(term,s, axis = 0)[s.N/2:3*s.N/2]
        daa_ = 1j  * term * np.sqrt(s.N)**2 / 2. #* np.sqrt(s.L)**4
        ba_ += daa_ + np.conj(daa_.T)

        return ba_

    def getQ(self,s):
        ntot = np.abs(s.Norm)**2
        Q = np.trace(np.abs(self.dbda))/ntot
        return Q