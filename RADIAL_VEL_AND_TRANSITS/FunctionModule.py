import numpy as np
import scipy.optimize as opt

########################
### GLOBAL VARIABLES ###
########################

#this is to get the required global variables such as wavelengths, template spectra and period.
a = np.load('JSAstroLab2023_rv_data_20332692.npz')
wavelengths, velocities, template = a['wavelength'], a['velocity'], a['spec_template']

b = np.load('JSAstroLab2023_transit_data_20332692.npz')
P = b['P']


#########################
### GENERAL FUNCTIONS ###
#########################

def residuals(params, f, x, y,s=1):
    """
    Takes in the parameters, function, x and y values
    Function returns the residuals from a given function (sine or transit in this lab)
    """
    
    res = (y - f(params,x) )/s
    return res

def chi2(params, f, x, y):
    """
    This function was made for the optimize.fmin non linear fit, however it is now only used for comparing the chi^2 of the inital vs final parameters from curve_fit 
    Takes in parameters, a function , x and y coordinates
    """
    
    chi2 = (np.sum( (y - f(params,x))**2 ))
    return chi2

def gaussian(x, A, B, C, D):
    """
    Gaussian function
    """
    
    y = A*np.exp(-1*( ((x-B)**2) / (2*C**2) ) ) + D
    #A = height of the peak
    #B = x value of the peak 
    #C = standard deviation (C^2 will be variance), FWHM = ( 2sqrt(2 ln(2)) )*C   - use for error vs std?
    #D is the y offset
    return y

def sine(params, x):
    """
    This sine function takes the parameters as an array, making it simpler for plotting, residuals, chi^2 etc
    """
    y = params[0]*np.sin(params[1]*x +params[2]) + params[3]
    return y

def sine_curve_fit(x, A, B, C, D):
    """
    This sine function takes the parameters individually and is used for the curve_fit function
    """
    y = A*np.sin(B*x +C) + D
    return y 


#################################
### RADIAL VELOCITY FUNCTIONS ###
#################################

def keylistr(file):
    """
    Gives out the keys for the spectra and the times so they can be used to create arrays
    This function assumes the first 3 keys in the file are not to be used and that the keys alternate between a spectrum and timestamp
    """
    
    remove = 3
    
    working = list(file.keys())
    keylist = working[remove:] #remove the first three (wvlen, vel, template)
    
    #this list alternates between a spectrum and a timestamp, spectra on the even indexes and time on the uneven
    spectra_keys = [keylist[i] for i in range(len(keylist)) if i%2 == 0] #modulo lets us seperate evens and odds
    time_keys = [keylist[i] for i in range(len(keylist)) if i%2 != 0]
        #lists of the keys corresponding to the spectra and times respectively 
        
    return spectra_keys, time_keys

def correlate(spectrum):#,template):
    """
    Takes in a spectrum and correlates it with the template spectrum from the .npz file
    """
    
    global template #the template spectra from outside the fn (from .npz file)
    global wavelengths
    global velocities #arrays of vs and wvlens from outside the fn
    #using global variables means I only have to define the template etc once and the function has less inputs 
    
    corr = np.correlate(template - np.median(template), spectrum - np.median(spectrum),'same') #same makes the array output the same size as the overlap of the 2 inputs
    #for each value the medians are taken away, numpy does this without having to have a for loop or list comprehension
    return corr

def velocity_findr(spectrum):
    """
    Takes a spectrum and correlates it, then fits a gaussian to the correlation and returns the radial velocity and error
    """
    
    global velocities
    
    corr = correlate(spectrum)
    
    popt, pcov = opt.curve_fit(gaussian, velocities, corr) #linear regression fitting a gaussian to correlation
    A,B,C,D = popt #these are the parameters of the gaussian (see gaussian fn for explanation)

    errs = np.sqrt(np.diag(pcov)) #will also take this error and compare it to the C value
    
    return B,C, errs[1]


#########################
### TRANSIT FUNCTIONS ###
#########################

def transit(params, z, mode):
    """
    This is the transit function from the lab script, it will take in the parameters and z values(normalized distance)
    It returns the flux 
    Originally it had 2 modes, single and multi, multi allowing an array to be inputted and sinlge for one value
    This was then removed as curve_fit was used to find the best fit 
    """
    
    rho = params[2]
    f_oot = params[4]
    
    if mode == 'single':
        #this is to take each input indiviudally
        
        if z > 1 + rho:
            return 1*f_oot
        
        elif z <= 1-rho:
            return (1- rho**2)*f_oot
        
        elif z > 1 - rho or z <= 1 + rho:
            k_0 = np.arccos( (rho**2 +z**2 -1)/(2*rho*z) )
            k_1 = np.arccos( (1 - rho**2 + z**2)/(2*z) )
            
            f = 1 - (1/np.pi)*( (rho**2)*k_0 + k_1 - np.sqrt( (4*z**2 - (1+z**2-rho**2)**2 )/4 ) )
            return f*f_oot

def transit_curvefit(z, T_0, a_over_R, rho, i, f_oot):
    """
    Again this function was defined to be used with curve fit as it requires the parameters to be inputted individually
    As above it returns flux from an input of z and parameters
    """
    
    if z > 1 + rho:
        return 1*f_oot
        
    elif z <= 1-rho:
        return (1- rho**2)*f_oot
        
    elif z > 1 - rho or z <= 1 + rho:
        k_0 = np.arccos( (rho**2 +z**2 -1)/(2*rho*z) )
        k_1 = np.arccos( (1 - rho**2 + z**2)/(2*z) )
            
        f = 1 - (1/np.pi)*( (rho**2)*k_0 + k_1 - np.sqrt( (4*z**2 - (1+z**2-rho**2)**2 )/4 ) )
        return f*f_oot
    

#the functions below take parameters and return various values used in the transit curve fitting
def phi_find(params, t):
    """
    This function finds phi (orbital phase) from the period and transit time
    """
    global P #the P from the .npz file 
    
    phi = (2*np.pi*(t - params[0]))/P
    return phi

def z_find(params, phi):
    """
    This function finds z (normalized distance) from the inclination, the semi major axis, and the orbital phase
    """
    
    #params[1] = a_over_R,  [3] = i
    z = (params[1])*np.sqrt( (np.sin(phi))**2 + (np.cos(params[3])*np.cos(phi))**2  )
    return z

def flux_findr(params, time):
    """
    Uses phi and z finding functions and the transit function to return the flux of the transit curve
    """
    
    flux = np.ones(time.size)
    
    for t in range(len(time)): #goes through the entire time list
        phi = phi_find(params,time[t]) #finds phi
        z = z_find(params, phi) #finds z
        f = transit(params, z, 'single') #finds f
        flux[t] = f
        
    return flux

def flux_findr_curvefit(time, T_0, a_over_R, rho, i, f_oot):
    """
    This function is again created for curve_fit, with individual parameters inputted
    It is the same as the flux_findr() function and returns the flux of the transit curve 
    """
    
    flux = np.ones(time.size)
    params = [T_0, a_over_R, rho, i, f_oot] #puts parameters into a vector to be used by flux_findr()
    
    for t in range(len(time)):
        phi = phi_find(params,time[t]) #finds phi
        z = z_find(params, phi) #finds z
        f = transit(params, z, 'single') #finds f
        flux[t] = f
        
    return flux

def flux_plottr(params, time):
    """
    This function takes times individually, not as an array, so that the transit curve can be plotted easily
    """
    
    phi = phi_find(params,time) #finds phi
    z = z_find(params, phi) #finds z
    f = transit(params, z, 'single') #finds flux

    return f    
    