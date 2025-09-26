'''
This is my working library of useful functions and constants of nature

Andrew MacRae. Last Updated Feb 1, 2021

TODO:
- Animatin helper functions
- Add plotstyle script

'''

# ---  constants ---
amu = 1.66053906660e-27
c = 299792458   # Speed of light
rBohr = 5.2917721090329e-11 # Bohr radius
muBohr = 9.2740100657e-24
e = 1.60217662e-19 # elementary charge
hbar = 1.0545718e-34 # hbar, obv
epsilon0 = 8.85418782e-12  # Permittivity of free space
mu0 = 1.25663706e-6 # Permeability of free space
kB = 1.380649e-23; #Boltzmann Constant
Rgas = 8.31446261815324 # ideal gas constant in J/mol*K
Nav = 6.02214076e23 # Avagadro's number
PI = 3.14159265358979 # ... I'm not sure why I did this
me = 9.10938356e-31 # Mass of the electron

light_year = 9460730472580800
astro_unit = 149597870700

# Properties of Rubidium
dRb = 2.8e-29 # dipole moment of Rb *** varies by +/- 0.5e-29 for different transisitions
GammaRb = 2*PI*5.746e6
lambdaRb87_D1 = 794.978851156e-9
lambdaRb87_D2 = 780.241209686e-9

# --- Function Definitions ---

# Numerical Differentiation/Integration (slightly redundant wrt numpy)
def nintegrate(fn,dx,xL,xR):
    """ Simple Numerical Integration of scalar function fn between (xL,xR), with spacing dx.
        Further improvements would take on vector functions as well as employ adaptive step sizing.
    """
    N = int((xR-xL)/dx) # Total number of points
    sm = 0  # Running sum
    for k in range(0,N):
        dF = fn(xL + k*dx) + 4*fn(xL + (k+0.5)*dx) + fn(xL + (k+1)*dx)
        sm = sm + dF
        return (xR-xL)*sm/(6*N)

def mean(x):
    """ Return the first moment """
    sm,sz = 0,x.size
    if sz<=0:
        return 0
    for xI in x:
        sm = sm+xI
    return sm/sz

def var(x):
    """ Return the first moment """
    sm = 0
    mn = mean(x)
    sz = x.size
    if sz<=0:
        return 0
    for xI in x:
        sm = sm+(xI-mn)**2
    return sm/sz
    
def std(x):
    import numpy
    return numpy.sqrt(var(x))

def skew(x):
    """ Returns third moment """
    sm,sz = 0,x.size
    mn,st = mean(x),std(x)
    if st==0 or sz == 0:
        return 0
    
    for xI in x:
        sm = sm+((xI-mn)/st)**3
    return sm/sz

def rCorr(x,y):
    """ Calculates the Pearson r coefficient of correlation """
    sm,sz = 0,x.size
    if x.size != y.size:
        return None
    mnX,mnY = mean(x),mean(y)
    stdX,stdY = std(x),std(y)
    
    for k in range(sz):
        sm = sm+(x[k]-mnX)*(y[k]-mnY)
    return sm/(sz*stdX*stdY)

def diff(fn,x0,h=1e-5):
    """ Symmetric difference method of a single valued function """
    return (fn(x0+h) - fn(x0-h))/(2*h)

def ndiff(x,dx):
    """ Numerical differentiation of a vector
        Uses symmetric differentiation on middle points, and 3-point method at endpoints
    """
    v = x*0
    sz = x.size
    if(sz < 2):
        return x
    for k in range(2,sz-1):
        v[k] = (x[k+1] - x[k-1])/(2*dx)
    v[0] = (4*x[1]-3*x[0]-x[2])/(2*dx)
    v[sz-1] = -(4*x[sz-2]-3*x[sz-1]-x[sz-3])/(2*dx);
    return v

# AMO functions
def getRb_VapourPressure(T,units = 'Torr',iso = 'Rb85'): # To do: Add option for isotope
    import numpy as np
    """Use Claussius-Clapyron relation to return atomic density for a  given temperature T [Kelvin]"""

    # Allow for either scalar or vector input
    scl_flg = False 
    if np.isscalar(T):
        scl_flg = True
        T = np.array([T])

# Melting and boiling points are essentially the same for the two isotopes
    Tk = 273.15 # 0C in K
    Torr2Pa = 133.3223684211 # Conversion of Torr to Pascal
    Tmelt = 39.31 + Tk 
    Tboil = 688 + Tk        

    # Mask that selects various phases of temperature array
    Tz = T<0
    Ts = (T<=Tmelt)*(T>0)
    Tl = (T<=Tboil)*(T>Tmelt)
    Tv = (T>=Tboil)
    
    Npts = len(T)
    
    if iso == 'Rb85':
        logPv = np.ones(Npts)*Tz*1e-9
        logPv += (2.881 + 4.857 - 4215/T)*Ts
        logPv += (2.881 + 4.312 - 4040/T)*Tl
        logPv += (2.881 + 4.312 - 4040/Tboil)*Tv
    else:
        logPv = np.ones(Npts)*Tz*1e-9
        logPv += (-94.04826 - 1961.258/T - 0.03771687*T + 42.57526*np.log10(T))*Ts
        logPv += (15.88253 - 4529.635/T + 0.00058663 *T - 2.99138* np.log10(T))*Tl
        logPv += (15.88253 - 4529.635/T - 0.00058663 *T + 2.99138* np.log10(T))*Tv

    if scl_flg:
        P = 10**logPv[0]
    else:
        P = 10**logPv
    if units == 'Pa':
        P *= Torr2Pa
    return P

def getNRb(T,iso = 'Rb85'):
    isotope = iso
    P = getRb_VapourPressure(T,units='Pa',iso = isotope)
    return P/(kB*T)

def getDopplerWidth(T,iso = 'Rb85',m=87*amu,lmd = lambdaRb87_D2,in_Hz = True):
    """ Return standard deviation of Doppler profile in Hz
        Isotopes can be Rb85 or Rb87. If using other element, specify mass(m) and wavelength(lmd)
    """
    import numpy as np
    if iso == 'Rb85':
        m0,lmd0 = 85*amu, lambdaRb87_D2
    elif iso == 'Rb87':
        m0,lmd0 = 87*amu, lambdaRb87_D1
    else:
        m0,lmd0 = m,lmd

    if in_Hz:
        return (np.sqrt(kB*T/m0)/lmd0)
    else:
        return (np.sqrt(kB*T/m0)/(2*np.pi*lmd0))

def chi2lev(freqs,linewidth,rabifreq = 0):
    """ Calculates the susceptibility of a single 2-level atom
        The default value of zero (negligible) Rabi frequency corresponds to purely linear response.
        To incorporate an atomic ensemble, multiply by the atomic density
    """
    chi1 = (dRb**2/(hbar*epsilon0))*1/(1j*GammaRb/2 + freqs)
    chi3 = (2*dRb**4/(epsilon0*hbar**3))*(1j*GammaRb/2 - freqs)/(.25*GammaRb**2 + freqs**2)**2
    return chi1 + chi3*(rabifreq**2)
    
def chi3lev(freqs,Dc,Gs,Gc,Oc,gbc):
    """ Calculates the EIT susceptibility of a 3 level atom in a lambda configuration as seen by a weak probe field
        It is assumed that the probe field being weak means that it is much less than the coupling field
            freqs: frequency vector
            Dc: coupling field detuning
            Gs(c): linewidth of the signal(coupling) field transitions
            Oc: Coupling field Rabi frequency
            gbc: ground-state dephasing
    """
    return (dRb**2/(hbar*epsilon0)) * (freqs - Dc + 1j*gbc)/(Oc**2 - (freqs - Dc + 1j*gbc)*(freqs+1j*(Gs+Gc)/2))

def chi4lev(freqs,Dc,Dm,Gs,Gc,Gm,Oc,Om,gbc):
    """ Calculates the susceptibility of a 4 level atom in N-Type configuration as seen by a weak probe field
        It is assumed that the probe field being weak means that it is much less than the coupling field
            freqs: frequency vector
            Dc: coupling field detuning
            Gs(c)[m]: linewidth of the signal(coupling)[modulation] field transitions
            Oc(m): Coupling(modulation) field Rabi frequency
            gbc: ground-state dephasing
    """
    gEIT = (gbc-1j*(freqs-Dc))
    g3 = (Gc-1j*freqs)
    g4 = (Gm-1j*(Dm-freqs+Dm))
    return 1j*(dRb**2/(hbar*epsilon0))*(gEIT + (Om**2)/g4)/(Oc**2 + g3*gEIT + (Om**2)*g3/g4)
    

def load_from_gax(fname,colname):
    import numpy
    import csv
    
    ret = numpy.array([])
    
    with open(fname, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:                
            ret = numpy.append(ret,float(row[colname]))
        
    return ret
    
# Fitting Routines
def wlls(x,y,stdy):
    '''Weighted linear least squares: given precise values x and measured values y, with uncertainty stdy, 
    estimate slope, intercept, and the standard deviatations of these estimates'''
    
    import numpy as np
    w = 1/(stdy**2)
    W = sum(w)
    xW = np.dot(w,x)/W
    yW = np.dot(w,y)/W
    xyW = np.dot(x*y,w)/W
    xxW = np.dot(x**2,w)/W
    
    vr = (xxW - xW**2)
    
    m = (xyW - xW*yW)/vr
    b = yW - m*xW
    
    sm = np.sqrt(1/(W*vr))
    sb = np.sqrt(xxW*sm)
    
    return m,b,sm,sb

def hann_avg(x,N):
    import numpy as np
    '''Run a Hanning Window to smooth out data
        x: a numpy array
        N: order of the filter (must be odd for symetric window)
    '''
    y = x*0.0
    if(N%2==0):        
        print('Warning: hann_avg accepts odd order. Order rounded from '+ str(N) + ' to '+str(N+1))
        N = N+1
    wnd = np.hanning(N)/np.sum(np.hanning(N))
    mid = int((N-1)/2)
    
    for k in np.arange(mid+1):
        y[k] = x[k]
        y[np.size(x)-k-1] = x[np.size(x)-k-1]

    for k in np.arange(mid,np.size(x)-mid-1):
        y[k] = np.sum(x[k-mid:k+mid+1]*wnd)
    return y  

def get_channel_idx(fName,chan):
    import csv
    with open(fName, newline='') as scope_data:
        reader = csv.reader(scope_data)
        for row in reader:
            for k in range(len(row)):
                if row[k] == chan:
                    return k + 3
    return -1

def get_scope_data(fName,chan,avg=1):
    '''Scrape data from Tektronix TDS 2012C Oscilloscope
        fName: full path to csv file
        chan: Must be one of {'CH1','CH2', or 'MATH'}
    '''
    import numpy as np
    import csv
    trc = np.array([])
    tm = np.array([])
        
    chan_idx = get_channel_idx(fName,chan)
    time_idx = chan_idx - 1
    # Throw error here 

    with open(fName, newline='') as scope_data:
        reader = csv.reader(scope_data)
        for row in reader:
            tm = np.append(tm,float(row[time_idx]))
            trc = np.append(trc,float(row[chan_idx]))
    ret = trc
    if avg <= 1:
        ret = hann_avg(trc,11)
    return ret,tm
    
#OK ... now construct multisine
def mSine(tTot,Fs,fMfN=1):
    ''' 
        - tTot is the total time of the trace
        - Fs is the sample frequency (1/sample-time)
        - fMfN: max frequency to include as a fraction of the Nyquist frequency
    '''
    import numpy as np
# Create a multisine forcing function with constant spectral power up to fMax
    #First create indices, given fMax, total trace time and sampling freq
    fMax = Fs * fMfN
    if fMax>Fs/2:
        fMax = Fs/2
    nMax = np.int(tTot*fMax);
    nTot = np.int(tTot*Fs);
    # Next, construct first half of spectrum amplitude ...
    Y = np.zeros(nTot)    
    Y[0:np.int(nMax)]=1
    # ... and apply Schroeder phase to minimize crest factor
    phs = np.arange(nTot)
    Y2 = Y*np.exp(1j*phs*(phs+1)*(phs+2))
    # next make second half of spectra the cpx conj of first for a real-valued time series.
    Y2[nTot:nTot-nMax:-1] = np.conj(Y2[1:nMax])
    return np.real(np.fft.ifft(Y2))

# ---- Fresnel Equations -----
# fields
def fresnel_rP(theta_i,ni,nt,cpx = False):
    import numpy as np
    theta_c = np.arcsin(min(1,nt/ni))
    msk = theta_i<theta_c
    tht_i= theta_i*msk + (1-msk)*theta_c
    Ci = np.cos(tht_i) 
    Ct = np.sqrt(1 - (ni*np.sin(tht_i)/nt)**2)
    if not cpx:
        return (ni*Ct - nt*Ci)/(ni*Ct + nt*Ci)
    else:
        ret_vec = (ni*Ct - nt*Ci)/(ni*Ct + nt*Ci) + 0j
        phs_num = -2*ni*nt**2*np.cos(theta_i)*np.sqrt(np.abs(ni**2 * np.sin(theta_i)**2 - nt**2))
        phs_den = ni**4*np.cos(theta_i)**2 - ni**2 * np.sin(theta_i)**2 + (ni*nt)**4
        ret_vec *= np.exp(1j*(1-msk)*np.arctan2(phs_num,phs_den))
        return ret_vec

def fresnel_tP(theta_i,ni,nt):
    import numpy as np
    theta_c = np.arcsin(min(nt/ni,1))
    msk = theta_i<theta_c
    tht_i= theta_i*msk + (1-msk)*theta_c

    Ci = np.cos(tht_i) 
    Ct = np.sqrt(1 - (ni*np.sin(tht_i)/nt)**2)
    return 2*ni*Ci/(ni*Ct + nt*Ci)

def fresnel_rS(theta_i,ni,nt,cpx = False):
    import numpy as np
    theta_c = np.arcsin(min(1,nt/ni))
    msk = theta_i<theta_c
    tht_i= theta_i*msk + (1-msk)*theta_c
    Ci = np.cos(tht_i) 
    Ct = np.sqrt(1 - (ni*np.sin(tht_i)/nt)**2)
    return (ni*Ci - nt*Ct)/(ni*Ci + nt*Ct)

def fresnel_tS(theta_i,ni,nt):
    import numpy as np
    theta_c = np.arcsin(min(1,nt/ni))
    msk = theta_i<theta_c
    tht_i= theta_i*msk + (1-msk)*theta_c
    Ci = np.cos(tht_i) 
    Ct = np.sqrt(1 - (ni*np.sin(tht_i)/nt)**2)
    return 2*ni*Ci/(ni*Ci + nt*Ct)

# Powers
def fresnel_RP(theta_i,ni,nt):
    return fresnel_rP(theta_i,ni,nt)**2

def fresnel_TP(theta_i,ni,nt):
    import numpy as np
    theta_c = np.arcsin(min(1,nt/ni))
    msk = theta_i<theta_c
    tht_i= theta_i*msk + (1-msk)*theta_c    
    Ci = np.cos(tht_i) 
    Ct = np.sqrt(1 - (ni*np.sin(tht_i)/nt)**2)
    return fresnel_tP(tht_i,ni,nt)**2 *nt*Ct/(ni*Ci)


def fresnel_RS(theta_i,ni,nt):
    return fresnel_rS(theta_i,ni,nt)**2
def fresnel_TS(theta_i,ni,nt):
    import numpy as np
    theta_c = np.arcsin(min(1,nt/ni))
    msk = theta_i<theta_c
    tht_i= theta_i*msk + (1-msk)*theta_c    
    Ci = np.cos(tht_i) 
    Ct = np.sqrt(1 - (ni*np.sin(tht_i)/nt)**2)
    return fresnel_tS(tht_i,ni,nt)**2 *nt*Ct/(ni*Ci)


# ------- Animation stuff

def find_all_png(file_base):
    import glob
    pngs = glob.glob(file_base+"*.png")
    pngs.sort()
    buf = []
    for png in pngs:
        buf.append(png)
    return buf

def cr_mp4(image_list,mp4_name,mfps = 60):
    import imageio
    writer = imageio.get_writer(mp4_name, fps=mfps)
    for im in image_list:
        writer.append_data(imageio.imread(im))
    writer.close()

def cr_gif(image_list, gif_name, durr = 0.05,optimize_gif = False):
    import imageio,os
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
        # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF', duration = durr,loop = 0)

    if optimize_gif:
        from pygifsicle import optimize
        print('Attempting to reduce filesize.')
        print(f'Current filesize: {round(os.path.getsize(gif_name)/1e6,2)} MB')
        optimize(gif_name)
        print(f'New filesize: {round(os.path.getsize(gif_name)/1e6,2)} MB')
    

def del_all_png(file_base):
    import os, glob
    for f in glob.glob(file_base+'*.png'):
        os.remove(f)

def make_frames(plotfunk,frames,dpi_res,f_base='./tiempo'):
    from matplotlib import pyplot as plt
    for k in range(len(frames)):
        fig = plotfunk(k)
        plt.savefig(f_base+str(1000+k)+'.png', format='png', dpi=dpi_res, bbox_inches='tight')
        plt.close(fig)

def make_anim(f_name,plot_funk,frames,optgif = True,f_base = './tiempo',dpi=100, duration = 0.05, fps = 60, file_type = 'gif'):
    import time
    tic = time.perf_counter()
    print('Creating frames:')
    make_frames(plot_funk,frames,f_base = f_base,dpi_res = dpi)
    buf = find_all_png(f_base)
    if file_type == 'gif':
        print('Creating gif: '+f_name)
        cr_gif(buf,f_name,optimize_gif=optgif,durr = duration)
    else:
        print('Creating mp4: '+f_name)
        cr_mp4(buf,f_name,mfps = fps)

    del_all_png(f_base)
    toc = time.perf_counter()
    print(f"Time to create animation: {toc - tic:0.4f} seconds")

def gen_sin2_ramp(ti,tf,Npts):
    print('Warning! This is an older vesion! use cos_ramp() or cos_window() instead!!!')
    import numpy as np
    th = np.linspace(0,np.pi/2,Npts)
    return np.sin(th)**2*(tf-ti) + ti

def cos_ramp(t,t1,dt):
    import numpy as np
    N = len(t)
    x = np.zeros(N)
    
    if t1 < t[0]:
        return x+1
    if t1-dt > t[-1]:
        return x
    
    idxL = 0
    idxR = len(t)-1
    
    iL = np.argwhere(t>=t1-dt)
    if len(iL)>0:
        idxL = iL[0][0]
    
    iR = np.argwhere(t>=t1)
    if len(iR)>0:
        idxR = iR[0][0]
    x[idxL:idxR+1] = np.cos(.5*np.pi*(t[idxL:idxR+1]-t1)/dt)**2
    x[idxR+1:] = 1
    return x

def cos_window(t,t1,t2,dt):
    return cos_ramp(t,t1,dt) - cos_ramp(t,t2+dt,dt)