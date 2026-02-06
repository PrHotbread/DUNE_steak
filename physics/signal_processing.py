import numpy as np
from numba import njit
from scipy import integrate
from scipy import interpolate
from scipy import signal

from tqdm import tqdm
from .interpolation import linear_interp_field


        
"""
The module allows to calculate the signal from the SR-theorem and make the convolution by the transfert function
"""
    
def signal_process(current,elec):
    deltaQ_collect,t_sampling,Sconv,t_convolve=conv(current[2],elec)
    deltaQ_induc1,t_sampling,Sconv1,t_convolve=conv(current[0],elec)
    deltaQ_induc2,t_sampling,Sconv2,t_convolve=conv(current[1],elec)
    return deltaQ_collect, deltaQ_induc1, deltaQ_induc2, t_sampling, Sconv, Sconv1, Sconv2, t_convolve



def conv(Sraw,response):
    """ Electronic response reading"""
    if response == "top":
        electronics = np.loadtxt("/Users/pinchault/Documents/work/research/DUNE_steak/steak/electronics/RepElectroTop.txt")
        reltime = electronics[:,0]
        ampl = electronics[:,1]
        sampling = sampling_top # sampling time for the top electronics
    elif response == "bot":
        reltime,ampl = np.loadtxt("/Users/pinchault/Documents/work/research/DUNE_steak/steak/electronics/RepElectroBot.txt")
        sampling = sampling_bot # sampling time for the bot electronics
   
    """Electronic response normalize and sampling"""
    reltime_interp = np.arange(0, np.max(reltime), params['te'], dtype = np.float32)
    ampl_interp = interpolation(reltime_interp, reltime, ampl) # re-sampling the transfert function to make the convolution with the signal
    response_norm = ampl_interp / integrate.trapz( ampl_interp, reltime_interp) #Normalize the response for integrate conservation --> charge conservation
    
    t_convolve = np.linspace(0, params['acquisition_time'] + reltime_interp[-1], len(params['drift_time']) + len(reltime_interp)-1) # Time axis after the full convolution (time response + time signal)
    
    S_conv = params['te'] * signal.fftconvolve(Sraw, response_norm, "full") # Convolution normalize by the sampling time

    """ ADC convertor """
    sample_rate = np.arange(0, max(t_convolve), sampling) #Electronics sampling time axis
    S_sampling = interpolation(sample_rate, t_convolve, S_conv)
    Q = integrate.cumulative_trapezoid(S_sampling, sample_rate)
    deltaQ = (Q[1:] - Q[:-1])#*ADC/1e-15
    
    return deltaQ, sample_rate[:-2], S_conv, t_convolve





#@njit
def ramo(vol: float, x: float, y: float, z: float, vx: float, vy: float, vz: float, Ew0: float, Ew1: float, Ew2: float, step_x: float, n_step: int):
    """ electric charge """
    q = - 1 #np.float32(- 1.602e-19) #elemental electric charge

    """ empty array for current storage"""
    ind1 = np.zeros(n_step, dtype = np.float32)
    ind2 = np.zeros(n_step, dtype = np.float32)
    coll = np.zeros(n_step, dtype = np.float32)
    
    for idx in range(n_step):

        i = np.int32(np.round(x[idx] / step_x))
        j = np.int32(np.round(y[idx] / step_x))
        k = np.int32(np.round(z[idx] / step_x))
        
        """ Trilinear field interpolation """
        
        """ Weighting field view0 """
    
        Exw0, Eyw0, Ezw0 = linear_interp_field(x[idx], y[idx], z[idx], vol, i, j, k, step_x, step_x, Ew0[0], Ew0[1], Ew0[2])
        
        """ Weighting field view1 """
        
        Exw1, Eyw1, Ezw1 = linear_interp_field(x[idx], y[idx], z[idx], vol, i, j, k, step_x, step_x, Ew1[0], Ew1[1], Ew1[2])
        
        """ Weighting field view2 """
        
        Exw2, Eyw2, Ezw2 = linear_interp_field(x[idx], y[idx], z[idx], vol, i, j, k, step_x, step_x, Ew2[0], Ew2[1], Ew2[2])

        """ Instantaneous current calculation by Shockley-Ramo theorem """
        ind1[idx] = + q * (vx[idx] * Exw0 + vy[idx] * Eyw0+ vz[idx] * Ezw0)
        ind2[idx] = + q * (vx[idx] * Exw1 + vy[idx] * Eyw1 + vz[idx] * Ezw1)
        coll[idx] = + q * (vx[idx] * Exw2 + vy[idx] * Eyw2 + vz[idx] * Ezw2)
  
    return ind1, ind2, coll
    
def interpolation(te,t,f):
    f=interpolate.interp1d(t,f,fill_value="extrapolate")
    return f(te)

""" Electronic params """
#Te=1/1e3 #100 000 points / microsconde
#tmax=20 #20 [micro sec]
#sampling_time=np.arange(0,tmax,Te) #numerical sample rate
ADC=256 #ADC value by fC
sampling_top=0.4 #Top electronic sampling [micro s]
sampling_bot=0.512 #Bot electronic sampling [micro s]
