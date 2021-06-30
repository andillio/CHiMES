#import numpy as np 
#import cupy as cp
try:
    import cupy as cp
except ImportError:
    import numpy as cp
import numpy as np
import pylab as pyl 
import scipy.fftpack as sp
import time
import sys
import matplotlib.pyplot as plt
import os
import pyfftw
import shutil

# computes the potential from the wavefunction
def compute_phi(rho_r, C_, dx, kx = None, fac = 500):
	rho_k = sp.fft(rho_r)
	phi_k = None
	if kx is None:
		kx = 2*np.pi*sp.fftfreq(len(rho_r), d = dx)
	with np.errstate(divide ='ignore', invalid='ignore'):
		# np.errstate(...) suppresses the divide-by-zero warning for k = 0
		phi_k = -(C_*rho_k)/(kx**2)
	phi_k[0] = 0.0 # set k=0 component to 0
	phi_r = np.real(sp.ifft(phi_k))
	return phi_r

# computes the potential from the wavefunction
def compute_phi_cp(rho_r, C_, dx, kx = None, fac = 500):
	rho_k = cp.fft.fft(rho_r)
	phi_k = None
	if kx is None:
		kx = 2*np.pi*cp.fft.fftfreq(len(rho_r), d = dx)
	with np.errstate(divide ='ignore', invalid='ignore'):
		# np.errstate(...) suppresses the divide-by-zero warning for k = 0
		phi_k = -(C_*rho_k)/(kx**2)
	phi_k[0] = 0.0 # set k=0 component to 0
	phi_r = cp.real(cp.fft.ifft(phi_k))
	return phi_r

def compute_phi2D(rho_r, C_, dx, kx, ky):
    # TODO: we know rho_r is real, so using rfft would be better 
    rho_k = sp.fft2(rho_r)
    with np.errstate(divide='ignore', invalid='ignore'):
        # np.errstate(...) suppresses the divide-by-zero warning for k=0
        phi_k = -(C_*rho_k)/(kx**2+ky**2) 
    phi_k[0,0] = 0.0 # zero the k=0 component 
    phi_r = np.real(sp.ifft2(phi_k))
    return phi_r


def getKineticE(psi,kx = None, hbar_ = None):
	if kx is None:
		kx = 2*np.pi*sp.fftfreq(len(psi))
	if hbar_ is None:
		hbar_ = 1.
	rho_k = np.abs(sp.fft(psi))**2
	return np.mean(np.abs(((kx**2 / 2.)*rho_k*(hbar_**2))))


def getE_kin(psi,kx,hbar_):
	psi_k = sp.fft(psi)
	psi_ = sp.ifft(kx**2 * psi_k)
	return hbar_**2 / 2. * np.mean(np.real(np.conj(psi)*psi_))

# returns quantum potential
def VQ(rho, D2):
	A = np.sqrt(rho)
	A_ = np.matmul(D2, A)
	return A_/A

def VQ_est(psi,hbar_,D2):
	A = np.abs(psi)
	return hbar_**2 / 2. * A * np.matmul(D2,A)

def normalize(y,dx):
	return y/(np.sum(y)*dx)

def normalize_cp(y,dx):
	return y/(cp.sum(y)*dx)

def makeOverDensity(y):
	return y - np.mean(y)

def array_make_periodic(x,w):
    w[x>=1] += 1
    x[x>=1.] -= 1.

    w[x<0] -= 1 
    x[x<0.]  +=1.

def fast_CIC_deposit(x,mi,Ngrid,periodic=1):
    """cloud in cell density estimator
    """
    if ((np.size(mi)) < (np.size(x))):
        m=x.copy()
        m[:]=mi
    else:
        m=mi

    dx = 1./Ngrid
    rho = np.zeros(Ngrid)
 
    left = x-0.5*dx
    right = left+dx
    xi = np.int32(left/dx)
    frac = (1.+xi-left/dx)
    ind = pyl.where(left<0.)
    frac[ind] = (-(left[ind]/dx))
    xi[ind] = Ngrid-1
    xir = xi.copy()+1
    xir[xir==Ngrid] = 0
    rho  = pyl.bincount(xi,  weights=frac*m, minlength=Ngrid)
    rho2 = pyl.bincount(xir, weights=(1.-frac)*m, minlength=Ngrid)

    rho += rho2
    
    return rho*Ngrid


def fast_CIC_deposit_cp(x,mi,Ngrid,periodic=1):
    """cloud in cell density estimator
    """
    m=mi

    dx = 1./Ngrid
    rho = cp.zeros(Ngrid)
 
    left = x-0.5*dx
    right = left+dx
    xi = (left/dx).astype(cp.int32)
    frac = (1.+xi-left/dx)
    ind = cp.where(left<0.)
    frac[ind] = (-(left[ind]/dx))
    xi[ind] = Ngrid-1
    xir = xi.copy()+1
    xir[xir==Ngrid] = 0
    rho  = cp.bincount(xi,  weights=frac*m, minlength=Ngrid)
    rho2 = cp.bincount(xir, weights=(1.-frac)*m, minlength=Ngrid)

    rho += rho2
    
    return rho*Ngrid



def CIC_2d(x,y,mpart,grid_size):

	BoxSize = 1.
	pos = np.zeros((len(x), 2))
	pos[:,0] = x
	pos[:,1] = y

	index_u=np.zeros((len(x),2),dtype=np.int32)
	index_d=np.zeros((len(x),2),dtype = np.int32)

	u=np.ones((len(x),2))*1.
	d=np.ones((len(x),2))*1.

	inv_cell_size = grid_size/BoxSize

	density = np.zeros(grid_size*grid_size)

	dist_track = np.zeros((len(x),2),dtype=np.int32)

	dist = pos*inv_cell_size

	u       = dist - np.int32(dist)

	d       = 1.0 - u

	index_d = (np.int32(dist))%grid_size

	index_u = index_d + 1

	index_u = index_u%grid_size

	density += reshapeAndAdd(index_d[:,0], index_d[:,1], mpart*d[:,0]*d[:,1], grid_size)
	density += reshapeAndAdd(index_d[:,0],index_u[:,1], mpart*d[:,0]*u[:,1], grid_size)
	density += reshapeAndAdd(index_u[:,0],index_d[:,1], mpart*u[:,0]*d[:,1], grid_size)
	density += reshapeAndAdd(index_u[:,0],index_u[:,1], mpart*u[:,0]*u[:,1], grid_size)

	return np.reshape(density, (grid_size, grid_size))*grid_size*grid_size


def reshapeAndAdd(ind1, ind2, frac, N):
	ind = N*ind1 + ind2
	return pyl.bincount(ind,  weights=frac, minlength=N*N)	


def CIC_acc_cp(x,m,Ngrid,C,dx,kx):
    dx = 1./Ngrid
    xg = (0.5+cp.arange(Ngrid))/Ngrid
    rho = fast_CIC_deposit_cp(x,m,Ngrid) # size Ngrid
    Phi = compute_phi_cp(rho, C, dx, kx) # size Ngrid
    a = a_from_Phi_cp(Phi) # size Ngrid
    left = x-0.50000*dx # size n
    xi = (left/dx).astype(cp.int32) # size n
    frac = (1.+xi-left/dx) 
    ap = (frac)*(cp.roll(a,0))[xi] + (1.-frac) * (cp.roll(a,-1))[xi]
    return ap


def CIC_acc(x,m,Ngrid,C,dx,kx):
    dx = 1./Ngrid
    xg = (0.5+np.arange(Ngrid))/Ngrid
    rho = fast_CIC_deposit(x,m,Ngrid) # size Ngrid
    Phi = compute_phi(rho, C, dx, kx) # size Ngrid
    a = a_from_Phi(Phi) # size Ngrid
    left = x-0.50000*dx # size n
    xi = np.int64(left/dx) # size n
    frac = (1.+xi-left/dx) 
    ap = (frac)*(np.roll(a,0))[xi] + (1.-frac) * (np.roll(a,-1))[xi]
    return ap


def CIC_acc2D(x,y,m,Ngrid,C,dx,kx,ky):
    dx = 1./Ngrid
    xg = (0.5+np.arange(Ngrid))/Ngrid # (N)
    rho = CIC_2d(x,y,m,Ngrid) # (N,N)
    Phi = compute_phi2D(rho, C, dx, kx,ky) # (N,N)
    ax, ay = a_from_Phi2D(Phi) # (N, N), (N,N)
    left_x = x-0.50000*dx # (n)
    left_y = y-0.50000*dx # (n)
    xi = np.int64(left_x/dx) # (n)
    yi = np.int64(left_y/dx) # (n)
    frac_x = (1.+xi-left_x/dx) # ()
    frac_y = (1.+yi-left_y/dx)
    ax = (frac_x)*(np.roll(ax,0,axis=0))[xi,yi] + (1.-frac_x) * (np.roll(ax,-1,axis=0))[xi,yi]
    ay = (frac_y)*(np.roll(ay,0,axis=1))[xi,yi] + (1.-frac_y) * (np.roll(ay,-1,axis=1))[xi,yi]
    return ax, ay

def a_from_Phi_cp(Phi):
    """Calculate  - grad Phi  from Phi assuming a periodic domain
    domain the is 0..1 and dx=1./len(Phi)
    """
    N = len(Phi)
    dx = 1./N
    ax = - central_difference_cp(Phi,0)/dx
    return ax


def a_from_Phi(Phi):
    """Calculate  - grad Phi  from Phi assuming a periodic domain
    domain the is 0..1 and dx=1./len(Phi)
    """
    N = len(Phi)
    dx = 1./N
    ax = - central_difference(Phi,0)/dx
    return ax

#TODO
def a_from_Phi2D(Phi):
    """Calculate  - grad Phi  from Phi assuming a periodic domain
    domain the is 0..1 and dx=1./len(Phi)
    """
    N = len(Phi)
    dx = 1./N
    ax = - central_difference(Phi,0)/dx
    ay = - central_difference(Phi,1)/dx
    return ax, ay


def central_difference_cp(y, axis_ = 0):
    """ Central difference:  (y[i+1]-y[i-1])/2 
    """
    return (cp.roll(y,-1, axis = axis_)-cp.roll(y,1, axis = axis_))/2


def central_difference(y, axis_ = 0):
    """ Central difference:  (y[i+1]-y[i-1])/2 
    """
    return (np.roll(y,-1, axis = axis_)-np.roll(y,1, axis = axis_))/2


def remaining(done, total, start):
	Dt = time.time() - start
	return hms((Dt*total)/float(done) - Dt)

# given a time T in s
# returns (hours, mins, secs) remaining
def hms(T):
	r = T
	hrs = int(r)/(60*60)
	mins = int(r%(60*60))/(60)
	s = int(r%60)
	return (hrs, mins, s)

def repeat_print(string):
    sys.stdout.write('\r' +string)
    sys.stdout.flush()

def Tag2Array(tag_):
    tags_ = tag_.replace('[','')
    tags_ = tags_.replace(']','')
    tags_ = tags_.split(' ')

    tags = []

    for i in range(len(tags_)):
        if (len(tags_[i]) > 0): 
            tags.append(int(tags_[i]))

    return np.array(tags) 

def getMetaKno(name, **kwargs):
	dir_ = ''
	if 'dir' in kwargs.keys():
		dir_ = kwargs['dir']
	f = open("../" + dir_ + name + "/" + name + "Meta.txt")

	print("reading meta info...")
	
	metaParams = {}

	for key_ in kwargs.keys():
		metaParams[key_] = None

	for line in f.readlines():
		for key_ in kwargs.keys():
			if key_ + ":" in line:
				print(line)
				number = line.split(":")[1]
				#metaParams[key_] = re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]
				if key_ == "N" or key_ == "frames" or key_ == "n" or key_ == "drops":
					metaParams[key_] = int(number)
				elif key_ == "IC":
					metaParams[key_] = Tag2Array(number)
				elif key_ == 'dir':
					pass
				else:
					metaParams[key_] = float(number)

	f.close()
	return metaParams

# gets the names of files in the given directory organized by time stamp (ascending)
def getNames(name):
	files = ["../" + name + "/" + file for file in os.listdir("../" + name) if (file.lower().endswith('.npy'))]
	files.sort(key=os.path.getmtime)
	return files

def getNamesInds(name):
	files = ["../" + name + "/" + file for file in os.listdir("../" + name) if (file.lower().endswith('.npy'))]
	
	inds = []

	for i in range(len(files)):
		file_ = files[i]
		file_ = file_.replace("drop",'')
		file_ = file_.replace('.npy','')
		ind_ = int(file_.split("/")[-1])
		inds.append(ind_)
	
	sortInds = np.array(inds).argsort()

	files = np.array(files)

	files = files[sortInds]

	return files


def orientPlot(fontSize = 22):
	plt.rc("font", size=fontSize)

	plt.figure(figsize=(6,6))

	fig,ax = plt.subplots(figsize=(6,6))

	plt.rc("text", usetex=True)

	plt.rcParams["axes.linewidth"]  = 1.5

	plt.rcParams["xtick.major.size"]  = 8

	plt.rcParams["xtick.minor.size"]  = 3

	plt.rcParams["ytick.major.size"]  = 8

	plt.rcParams["ytick.minor.size"]  = 3

	plt.rcParams["xtick.direction"]  = "in"

	plt.rcParams["ytick.direction"]  = "in"

	plt.rcParams["legend.frameon"] = 'False'


def GetIndexes(T,N,times):
	inds = []
	for i in range(len(times)):
		t = times[i]
		ind_ = int(t*N/T)
		ind_ = np.min([ind_, N-1])
		ind_ = np.max([0,ind_])
		inds.append(ind_)
	return inds


def readyDir(ofile, tag):
	try: # attempt to make the directory
		os.mkdir("../" + ofile + "/" + tag)
	except OSError:
		try: # assuming directory already exists, delete it and try again
			print("removing and recreating an existing directory")
			shutil.rmtree("../" + ofile + "/" + tag)
			readyDir(ofile, tag)
		except OSError:
			pass


def ding():
	dur1 = .15
	dur2 = .15
	freq1 = 600
	freq2 = 700
	#os.system('play  --no-show-progress --null --channels 1 synth %s sine %f' % (dur1, freq1))
	#time.sleep(.04)
	#os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (dur2, freq2))


class figObj(object):

	def __init__(self):
		self.ax = None
		self.im = None

		self.axs = []
		self.ims = []

		self.axSP = None
		self.imSP = None

		self.axMS = None
		self.imMS = None

		self.axVN = None
		self.imVN = None

		self.axCL = None
		self.imCL = None

		self.time_text = None

		self.frames = None

		self.Tfinal = None
		self.t = []
		self.E = []
		self.T = []
		self.V = []

		self.decimate = None
		self.times = None
		self.SP = None
		self.shift = None
		self.colorscale = None
		self.CL_res_shift = None
		self.tags = None
		self.include = None
		self.inds = None
		self.axis = None

		self.N_images = None
		self.fileNames_r = None
		self.fileNames_v = None
		self.fileNames_rx = None
		self.fileNames_vx = None
		self.fileNames_ry = None
		self.fileNames_vy = None
		self.fileNames_psi = None
		self.fileNames_Psi = None
		self.fileNames_rho = None

		self.x = None
		self.dx = None
		self.L = None
		self.N = None
		self.C = None

		self.K = None

		self.meta = None


def setFileNames(fo, name, tags = ["SP", "VN", "MS", "CL"]):
	if "CL" in tags:
		fo.fileNames_r = getNames(name + "/" + "r")
		fo.fileNames_v = getNames(name + "/" + "v")
		fo.N_images = len(fo.fileNames_r)
	if "MS" in tags:
		fo.fileNames_Psi = getNames(name + "/" + "Psi")
		fo.N_images = len(fo.fileNames_Psi)
	if "SP" in tags:
		fo.fileNames_psi = getNames(name + "/" + "psi")
		fo.N_images = len(fo.fileNames_psi)
	if "VN" in tags:
		fo.fileNames_rho = getNames(name + "/" + "rho")
		fo.N_images = len(fo.fileNames_rho)
	if "CL2D" in tags:
		fo.fileNames_rx = getNames(name + "/" + "rx")
		fo.fileNames_vx = getNames(name + "/" + "vx")
		fo.fileNames_ry = getNames(name + "/" + "ry")
		fo.fileNames_vy = getNames(name + "/" + "vy")
		fo.N_images = len(fo.fileNames_rx)

def PrintTimeUpdate(done, total, time0):
    repeat_print(('%i hrs, %i mins, %i s remaining.' %remaining(done, total, time0)))

def PrintCompletedTime(time0):
    print('\ncompleted in %i hrs, %i mins, %i s' %hms(time.time()-time0))

