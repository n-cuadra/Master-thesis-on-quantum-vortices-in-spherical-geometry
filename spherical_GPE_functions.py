#This file contains all the functions I use for the numerical treatment of the Gross Pitaevskii equation in spherical coordinates

import numpy as np
import pyshtools as pysh
import spherical_GPE_params as params
import cmocean
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, gmres, lgmres, gcrotmk, bicgstab
import scienceplots

#set some parameters for plotting

plt.style.use('science')
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'xtick.labelsize': 10})
plt.rcParams.update({'ytick.labelsize': 10})


lstart = params.lmax - 20 #sh degree above which filtering will start (initially). This is defind here, because it needs to be adjusted globally. See the function timestep_grid for its use

#cotangent

def cot(x):
    return np.tan(np.pi/2 - x)

#transformation from spherical to cartesian coordinates

def sph2cart(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

#transformation from cartesian to spherical coordinates

def cart2sph(x, y, z):
    r = np.sqrt(x^2 + y^2 + z^2)
    theta = np.arccos(z/r)
    phi = np.arctan2(x,y)
    return theta, phi


####################### INITIAL CONDITION #############################################################

#initial phase angle of a vortex antivortex dipole at the positions (theta_plus, phi_plus) and (theta_minus, phi_minus) on the grid.
#The origin of this function is the stereographic projection of the phase of a vortex antivortex dipole in lfat 2D

def phase(theta, phi, theta_plus, phi_plus, theta_minus, phi_minus):
    denom1 = cot(theta/2) * np.sin(phi) - cot(theta_plus/2) * np.sin(phi_plus)
    num1 = cot(theta/2) * np.cos(phi) - cot(theta_plus/2) * np.cos(phi_plus)
    denom2= cot(theta/2) * np.sin(phi) - cot(theta_minus/2) * np.sin(phi_minus)
    num2 = cot(theta/2) * np.cos(phi) - cot(theta_minus/2) * np.cos(phi_minus)
    
    phase = np.arctan2(num1, denom1) - np.arctan2(num2, denom2)
    return phase


#model initial magnitude of the wavefunction (a function that goes from 0 to 1 over the length scale of the healing length xi at the position of the vortices)
#It's mathematical form is basically 1 - e^(l/xi), where l is the arc length separating a point on the sphere from the vortex core, xi is the healing length
#This is not at all close to the actual density profile. It's only purpose is to accelerate the production of the vortex core via imaginary time evolution
#depending on use case, it is sufficient to imprint the phase above

def initial_magnitude(theta, phi, theta_plus, phi_plus, theta_minus, phi_minus, xi):
    
    x = np.sin(theta) * np.cos(phi)
    x_plus = np.sin(theta_plus) * np.cos(phi_plus)
    x_minus = np.sin(theta_minus) * np.cos(phi_minus)
    y = np.sin(theta) * np.sin(phi)
    y_plus = np.sin(theta_plus) * np.sin(phi_plus)
    y_minus = np.sin(theta_minus) * np.sin(phi_minus)
    z = np.cos(theta)
    z_plus = np.cos(theta_plus)
    z_minus = np.cos(theta_minus)
    
    d_plus = np.sqrt((x - x_plus)**2 + (y - y_plus)**2 + (z - z_plus)**2)
    d_minus = np.sqrt((x - x_minus)**2 + (y - y_minus)**2 + (z - z_minus)**2)
    
    arc_length_plus = 2 * np.arcsin(d_plus / 2)
    arc_length_minus = 2 * np.arcsin(d_minus / 2)
    
    return 1. - np.exp(- arc_length_plus / xi) - np.exp(- arc_length_minus / xi)

#same model initial magnitude, but for only one vortex

def one_vortex_magnitude(theta, phi, theta_v, phi_v, xi):
    x, y, z = sph2cart(theta, phi)
    
    x_v, y_v, z_v = sph2cart(theta_v, phi_v)
    
    d = np.sqrt((x - x_v)**2 + (y - y_v)**2 + (z - z_v)**2)
    
    arc_length = 2 * np.arcsin(d / 2)
    
    return 1. - np.exp(- arc_length / xi)


#function that generates gridded data for a vortex dipole using the magnitude and phase above
#xi: healing length
#bg_dens: density far from vortices

def IC_vortex_dipole(theta_plus, phi_plus, theta_minus, phi_minus, xi, bg_dens):
    phaseangle = phase(params.THETA, params.PHI, theta_plus, phi_plus, theta_minus, phi_minus)
    magplus = one_vortex_magnitude(params.THETA, params.PHI, theta_plus, phi_plus, xi)
    magminus = one_vortex_magnitude(params.THETA, params.PHI, theta_minus, phi_minus, xi)
    psi = np.sqrt(bg_dens) * magplus * magminus * np.exp(1j * phaseangle)
    
    return psi

#################### DERIVATIVES ###############################

#derivative with respect to azimuthal angle phi of a complex function psi
#This uses the fact that spherical harmonics are eigenfunctions of this derivative with eigenvalues i*m

#first, initialize array that has the entries i*m to be multiplied with the coefficients
def deriv_phi_multiplier(i, l, m):
    return 1.0j * m * (-1.)**i

deriv_phi_multiplier_array = np.fromfunction(deriv_phi_multiplier, shape = (2, params.lmax + 1, params.lmax + 1), dtype = np.complex128) #create array from the multiplier function. The sh coeffs have to be multiplied with i*m to get the derivative wrt phi

def deriv_phi(psi):
    sh_coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #expand sh coefficients of psi
    sh_coeffs = sh_coeffs * deriv_phi_multiplier_array #modify coeffs with the above array
    psi = pysh.expand.MakeGridDHC(sh_coeffs, norm = 4, sampling = 2, extend = False) #back to real space, with the gridded data of the modified coefficients 
    return psi

#angular laplacian of a complex function psi
#This uses the fact that spherical harmonics are eigenfunctions of the angular laplacian with eigenvalues -l(l+1)

#first, initialize array that has the entries -l(l+1) to be multiplied with the coefficients
def laplacian_multiplier(i, l, m):
    return - l * (l + 1)

laplacian_multiplier_array = np.fromfunction(laplacian_multiplier, shape = (2, params.lmax + 1, params.lmax + 1), dtype = np.float64) #create array from the multiplier function. The sh coeffs have to be multiplied with -l(l+1) to get the laplacian


def Laplacian(psi):
    sh_coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #expand sh coefficients of psi
    sh_coeffs = sh_coeffs * laplacian_multiplier_array #modify coeffs with the above array
    psi = pysh.expand.MakeGridDHC(sh_coeffs, norm = 4, sampling = 2, extend = False) #back to real space, with the gridded data of the modified coefficients 
    return psi

#angular laplacian for a real valued function f
#same as above, except the function is expanded in real spherical harmonics

def Laplacianr(f):
    coeffs = pysh.expand.SHExpandDH(f, norm = 4, sampling = 2) 
    coeffs = coeffs * laplacian_multiplier_array
    f = pysh.expand.MakeGridDH(coeffs, norm = 4, sampling = 2, extend = False)
    return f


############################## CONSERVED QUANTITIES #####################################################

#norm function, calculates particle number of the condensate with wave function psi
#This first function calculates the particle number with spectral methods, using spherical harmonics
#In some cases, a represantion in spherical harmonics might not be numerically accurate. Refer to get_norm2 then

def get_norm(psi):
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #sh coefficients of wavefunction
    norm = np.sum(np.abs(coeffs)**2) #sum over the absolute value squared of all coefficients
    return norm

#This function calculates the particle number as well, but in real space without spectral methods
#It's a bit longer, but it ensures the correct computation no matter the grid size

def get_norm2(psi):
    N = np.shape(psi)[0] #grid length
    dangle = np.pi/N #grid point separation
    theta, phi = np.linspace(0,  np.pi, N, endpoint = False), np.linspace(0, 2 * np.pi, 2 * N, endpoint = False) #create angular arrays
    THETA, PHI = np.meshgrid(theta, phi, indexing = 'ij') #creat grid
    norm = np.sum(dangle**2 * np.sin(THETA) * np.abs(psi)**2) #compute norm which formally is an integral over the whole sphere but transformed into a sum on the finitely sized grid
    return norm

#calculate energy of condensate with wave function psi with spectral methods
#g: contact interaction strength in units of hbar^2/m
#omega: frequency of external rotation in units of hbar/(mR^2)
#G: gravitational constant in units of hbar^2/(m^2R^3). I've never used this, but implemented it nonetheless. So it is possible to calculate gravitational energy but currently I don't know to what end

#functions to create multplier arrays
def kinetic_multiplier(i, l, m):
    return 0.5 * l * (l + 1) 
def angmom_multiplier(i, l, m):
    return m * (-1.)**i
 
#create multiplier arrays 
kinetic_multiplier_array = np.fromfunction(kinetic_multiplier, shape = (2, params.lmax + 1, params.lmax + 1), dtype = np.float64)
angmom_multiplier_array = np.fromfunction(angmom_multiplier, shape = (2, params.lmax + 1, params.lmax + 1), dtype = np.float64)


def get_energy(psi, g, omega, G = 0):
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #sh coeffs of psi
    coeffs2 = pysh.expand.SHExpandDH(np.abs(psi)**2, norm = 4, sampling = 2) #sh coeffs of |psi|^2 to calculate interaction energy
    
    #the respective energies are calculated as sums over the coefficients modified by the multipliers, this is a consequence of parseval's theorem for spherical harmonics
    ekin = np.sum(kinetic_multiplier_array * np.abs(coeffs)**2)
    erot = omega * np.sum(angmom_multiplier_array * np.abs(coeffs)**2)
    eint = np.sum(0.5 * g * coeffs2**2)
    
    #if G is nonzero, gravitational energy is also calculated and returned
    if G:
        coeffs_grav = pysh.expand.SHExpandDHC(psi * (np.cos(params.theta_grid) + 1), norm = 4, sampling = 2)
        eg = np.real(np.sum(G * coeffs * np.conj(coeffs_grav)))
        return ekin, eint, erot, eg
    
    return ekin, eint, erot

#calculate angular momentum of condensate with wave function in z direction

def get_ang_momentum(psi):
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #sh coeffs of psi
    
    #again, angular momentum is calculated as a sum of modified coefficients
    mom = np.sum(angmom_multiplier_array * np.abs(coeffs)**2)
    
    return mom

############################## TIME EVOLUTION ##############################################################

#The numerical method for time evolution is split stepping. There are three different functions for it.
#The way that I do it is creating an array that is multiplied with the array storing the sh coeffs and thereby performing the kinetic and rotational timestep

def timestep_multiplier(dt, omega, gamma = 0):
    def step(i, l, m):
        return np.exp(- (1.0j + gamma) * 0.5 * l * (l + 1) * params.dt ) * np.exp(- (1.0j + gamma) * m * omega * params.dt * (-1.)**i)

    multiplier_array = np.fromfunction(step, shape = (2, params.lmax + 1, params.lmax + 1), dtype = np.complex128) #create array of same shape as coeffs with entries from step
    return multiplier_array

def imaginary_timestep_multiplier(dt, omega):
    def step(i, l, m):#this function will be mutiplied entry wise with coeffs with entry indices i, l, m
        return np.exp(- 0.5 * l * (l + 1) * dt ) * np.exp(- m * omega * dt * (-1.)**i)
    
    multiplier_array = np.fromfunction(step, shape = (2, params.lmax + 1, params.lmax + 1), dtype = np.float64)
    return multiplier_array

#This function performs one timestep and takes as input sh coefficients and returns the sh coefficients of the new wave function after the timestep.
#It was rarely used and is therefore the most barebone
#coeffs: array of sh coefficients
#dt: size of timestep in units of mR^2/hbar
#g: contact interaction strength in units of hbar^2/m
#omega: frequency of external rotation in units of hbar/(mR^2)


def timestep_coeffs(coeffs, dt, g, multiplier_array):
    coeffs = coeffs * multiplier_array #modify coeffs
    psi = pysh.expand.MakeGridDHC(coeffs, norm = 4, sampling = 2, extend = False) #create gridded data in (N, 2*N) array from coeffs
    psi = psi * np.exp(-1.0j * g * dt * np.abs(psi)**2) #timestep of nonlinear term
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #calculate expansion coefficients from the gridded data
    return coeffs

#This function performs one timestep as well, but takes as input the wave function psi defined on the grid directly.
#It has a bunch of additional features.
#psi: Nx2N array containing the gridded data
#dt: size of timestep in units of mR^2/hbar
#g: contact interaction strength in units of hbar^2/m
#omega: frequency of external rotation in units of hbar/(mR^2)
#G: gravitational constant in units of hbar^2/(m^2R^3). If you want to include gravity in the time-evolution, set this to be non-zero. Default is zero
#mu: chemical potential in units of hbar^2/(m R^2). May or may not be included. Default is not inclued (mu = 0)
#gamma: Strength of dissipation. Dissipation is included in this function. Default is no dissipation (gamma = 0)
#filtering: Boolean to turn on/off filtering during time evolution. Default is on (filtering = True). To combat aliasing, filtering may be included during the time evolution.
#Filtering modifies the coefficients according to the exponential filter function exp(-alpha (l - lstart)), for all coefficients with l > lstart (see top of file). It is dynamically adjusted if the spectrum grows at the edges

def timestep_grid(psi, dt, g, multiplier_array, G = 0, mu = 0, gamma = 0, filtering = True):
    psi = psi * np.exp(-(1.0j + gamma) * g * 0.5 * dt * np.abs(psi)**2) #half a timestep of nonlinear term
    if G:
        psi = psi * np.exp(-1.0j * G * (np.cos(params.theta_grid) + 1) * dt)
        
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2)
    
    coeffs = coeffs * multiplier_array
    
    if filtering:
        spectrum = pysh.spectralanalysis.spectrum(coeffs, normalization = 'ortho') #calculate spectrum of coefficients 
        global lstart
        #if spectrum grows at the edges, modify lstart in this fashion
        if (spectrum[lstart] > spectrum[lstart - 10]):
            lstart = lstart - 5
            if (lstart < 2 * params.lmax // 3):
                lstart = 2 * params.lmax // 3
        #create function for exponential filter
        def exp_filter(i, l, m): 
            alpha = 0.01 
            return np.exp(- alpha * (l - lstart))
        
        filter_multiplier = np.fromfunction(exp_filter, shape = np.shape(coeffs), dtype = np.float64) #create array from the filter with same shape as coeffs
        coeffs[:, lstart:, :] = coeffs[:, lstart:, :] * filter_multiplier[:, lstart:, :] #apply filter from lstart onwards
        
    
    psi = pysh.expand.MakeGridDHC(coeffs, norm = 4, sampling = 2, extend = False) #create gridded data in (N, 2*N) array from coeffs
    
    if mu:
        psi = psi * np.exp((1.0j + gamma) * dt * mu)
    psi = psi * np.exp(-(1.0j + gamma) * g * 0.5 * dt * np.abs(psi)**2) #half a timestep of nonlinear term
    return psi

#This function provides one timestep in imaginary time.
#psi: Nx2N array containing the gridded data
#dt: size of timestep in units of mR^2/hbar
#g: contact interaction strength in units of hbar^2/m
#omega: frequency of external rotation in units of hbar/(mR^2)
#particle_number: This must be the particle number of psi. I do not included the chemical potential in my imaginary time evolution. Instead, the particle number is kept fixed
#keep_phase: In some scenarios, you may want to keep the phase of psi constant and only subject the density to imaginary time evolution. If so, set keep_phase to True. Default is True.

def imaginary_timestep_grid(psi, dt, g, particle_number, multiplier_array, keep_phase = True):
    phase = np.angle(psi)
    psi = psi * np.exp(- g * dt * np.abs(psi)**2) #half timestep of nonlinear term
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2)
    
    coeffs = coeffs * multiplier_array
    
    psi = pysh.expand.MakeGridDHC(coeffs, norm = 4, sampling = 2, extend = False) #create gridded data in (N, 2*N) array from coeffs
    norm = get_norm2(psi)
    if keep_phase:
        psi = np.sqrt(particle_number) * np.abs(psi) * np.exp(1.0j * phase) / np.sqrt(norm)
    else:
        psi = np.sqrt(particle_number) * psi / np.sqrt(norm)
    return psi

#This is a seperate function that applies the exponential filter to the wave function psi, to be used outside of time evolution
#psi: Nx2N array containing the gridded data
#lstart: sh degree above which to apply the filter
#alpha: strength of filter
#k: exponent of filter

def filtering(psi, lstart, alpha, k):
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2)
    
    def exp_filter(i, l, m): #exponential filter
        return np.exp(- alpha * (l - lstart)**k)
    
    filter_multiplier = np.fromfunction(exp_filter, shape = np.shape(coeffs), dtype = np.float64) #create array from the filter with same shape as coeffs
    coeffs[:, lstart:, :] = coeffs[:, lstart:, :] * filter_multiplier[:, lstart:, :] #apply filter from lstart onwards
    
    psi = pysh.expand.MakeGridDHC(coeffs, norm = 4, sampling = 2, extend = False) #create gridded data in (N, 2*N) array from coeffs
    
    return psi
    
    
################### VORTEX TRACKING ############################

#This function calculates the position of a vortex of the wave function psi near guessed position (theta_guess, phi_guess) using the Newton-Raphson method
#counter: keeps track of how many iterations have been performed

def vortex_tracker(psi, theta_guess, phi_guess, counter = 0):
    N = np.shape(psi)[0] #calculate grid size
    lmax = N//2 - 1 #calculate maximum sh degree
    
    #expand sh coefficients of wave function, its real part and imaginary part
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) 
    coeffs_real = pysh.expand.SHExpandDH(np.real(psi), norm = 1, sampling = 2) 
    coeffs_imag = pysh.expand.SHExpandDH(np.imag(psi), norm = 1, sampling = 2) 
    
    #calculate gradient of real and imaginary part
    psi_theta_real, psi_phi_real = pysh.expand.MakeGradientDH(coeffs_real, sampling = 2) 
    psi_theta_imag, psi_phi_imag = pysh.expand.MakeGradientDH(coeffs_imag, sampling = 2)
    
    #expand sh coefficients for both components of the gradient for real and imaginary part
    coeffs_theta_real = pysh.expand.SHExpandDH(psi_theta_real, norm = 4, sampling = 2)
    coeffs_theta_imag = pysh.expand.SHExpandDH(psi_theta_imag, norm = 4, sampling = 2)
    coeffs_phi_real = pysh.expand.SHExpandDH(psi_phi_real, norm = 4, sampling = 2)
    coeffs_phi_imag = pysh.expand.SHExpandDH(psi_phi_imag, norm = 4, sampling = 2)
    
    Jacobian = np.zeros((2,2), dtype = np.float64) # initialize jacobian
    
    #calculate the wave function and the Jacobian at position (theta_guess, phi_guess) in sh representation
    psi_guess = np.sum(coeffs * pysh.expand.spharm(lmax, theta_guess, phi_guess, normalization = 'ortho', kind = 'complex', degrees = False))
    Jacobian[0, 0] = np.sum(coeffs_theta_real * pysh.expand.spharm(lmax, theta_guess, phi_guess, normalization = 'ortho', kind = 'real', degrees = False))
    Jacobian[0, 1] = np.sum(coeffs_phi_real * pysh.expand.spharm(lmax, theta_guess, phi_guess, normalization = 'ortho', kind = 'real', degrees = False))
    Jacobian[1, 0] = np.sum(coeffs_theta_imag * pysh.expand.spharm(lmax, theta_guess, phi_guess, normalization = 'ortho', kind = 'real', degrees = False))
    Jacobian[1, 1] = np.sum(coeffs_phi_imag * pysh.expand.spharm(lmax, theta_guess, phi_guess, normalization = 'ortho', kind = 'real', degrees = False))
    
    if (np.abs(psi_guess)**2 > 0.1 * np.max(np.abs(psi)**2)): #if density at guessed position is larger than 10% of maximum, possibly cannot guarantee convergence, so the tracking must be aborted
        print('Guessed position too far away from vortex core. Try again!')
        print('Density at guessed position: ' + str(np.abs(psi_guess)**2))
        print('It took ' + str(counter) + ' iterations to arrive here')
        return 0, 0
    
    if (np.abs(psi_guess)**2 < 1e-8 * np.max(np.abs(psi)**2)): #if the density at the guessed position is smaller than 1e-7 assume a reasonably converged solution and return the guess
        print('Number of iterations to convergence: ' + str(counter))
        return theta_guess, phi_guess
    
    inverse_jacobian = np.linalg.inv(Jacobian)
    
    #new coordinates
    theta_new = theta_guess - inverse_jacobian[0, 0] * np.real(psi_guess) - inverse_jacobian[0, 1] * np.imag(psi_guess)
    phi_new = phi_guess - inverse_jacobian[1, 0] * np.real(psi_guess) - inverse_jacobian[1, 1] * np.imag(psi_guess)    
    
    return vortex_tracker(psi, theta_new, phi_new, counter + 1) #recur the function with the new coordinates as the new guesses


############## PLOTTING ROUTINES ############

#function to plot 2d spectrum of a given function with sh coefficients stored in the array coeffs
#path: string, optional parameter to save the spectrum to path
#title: string, optional parameter include a title in the plot

def plot_2dspectrum(coeffs, path = '', title = ''):
    grid = np.concatenate((np.flip(coeffs[1,:,1:], axis = 1), coeffs[0, :, :]), axis = 1) #form a 2D array that has l as the first index and m going from -lmax to lmax in the second index ([0] = -lmax, [-1] = lmax)                   
    spectrum = np.abs(grid)**2 #calculate absolute value squared to plot
    spectrum[spectrum == 0.0] = np.nan #coefficients that don't exist are zero in the array, so set these to NaN

    #create coordinate grid for pcolormesh
    lmax = len(coeffs[0,:,0]) - 1
    ls = np.arange(0, lmax + 1, 1)
    ms = np.arange(- lmax, lmax + 1, 1)

    #plot log of spectrum as pcolormesh
    fig, ax = plt.subplots(1, 1, figsize = (6,4))
    if title:
        plt.suptitle(title, fontsize = 12)
    mappable = ax.pcolormesh(ms, ls, np.log(spectrum), cmap = cmocean.cm.haline, vmin = np.nanmin(np.log(spectrum)),  vmax = np.nanmax(np.log(spectrum)))
    ax.invert_yaxis()
    fig.colorbar(mappable, label = 'Power per coefficient', ax = ax)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_xlabel(r'Spherical harmonic order $m$', fontsize = 15)
    ax.set_ylabel(r'Spherical harmonic degree $l$', fontsize = 15)
    if path:
        fig.savefig(path, dpi = 300, bbox_inches = 'tight')
    return None


#psi: wavefunction to plot
#wf_title: string of title of density + phase plot
#spectrum_title: string of title of spectrum plot
#spectrum2d_title: string of title of 2d spectrum plot
#wf_path: string of path where to save density + phase plot
#spectrum_path: string of path where to save spectrum plot
#spectrum_path: string of path where to save 2d spectrum plot
#dpi: dpi of saved plots
#ftype: filetype of saved plots
def plot(psi, wf_title = '', spectrum_title = '', spectrum2d_title = '', wf_subtitle = '', wf_path = '', spectrum_path = '', spectrum2d_path = '', dpi = 600, ftype = 'jpg'):
    
    dens = np.abs(psi)**2 #calculate condensate density
    phase_angle = np.angle(psi) #calculate phase of condensate
    
    N = np.shape(psi)[0]
    lmax = N//2 - 1    
    
    #plot wave function
    
    fig, ax = plt.subplots(2, 1, figsize = (9, 8))
    plt.subplots_adjust(hspace=0.2)
    
    mappable1 = ax[0].pcolormesh(dens, cmap = cmocean.cm.thermal, vmin = 0, vmax = np.max(dens))
    ax[0].invert_yaxis()
    ax[0].set_ylabel(r'Latitude', fontsize = 16)
    ax[0].set_yticks(ticks = (0, N//4, N//2, 3 * N // 4, N), labels = ('90°', '45°', '0°', '-45°', '-90°'))
    ax[0].set_xticks(ticks = (0, N//2, N, 3 * N / 2, 2 * N), labels=('0°', '90°', '180°', '270°', '360°'))
    ax[0].tick_params(axis='both', which='major', labelsize=14)
    cbardens = fig.colorbar(mappable1, cmap = cmocean.cm.thermal, ax = ax[0], location = 'right')
    cbardens.ax.set_ylabel(r'$n$ $\left[1/R^2\right]$', fontsize=16)
    cbardens.ax.tick_params(labelsize=14)
    
    
    mappable2 = ax[1].pcolormesh(phase_angle, cmap = cmocean.cm.balance, vmin = -np.pi, vmax = np.pi)
    ax[1].invert_yaxis()
    ax[1].set_xlabel(r'Longitude', fontsize = 16)
    ax[1].set_ylabel(r'Latitude', fontsize = 16)
    ax[1].set_yticks(ticks = (0, N//4, N//2, 3 * N // 4, N), labels = ('90°', '45°', '0°', '-45°', '-90°'))
    ax[1].set_xticks(ticks = (0, N//2, N, 3 * N / 2, 2 * N), labels=('0°', '90°', '180°', '270°', '360°'))
    ax[1].tick_params(axis='both', which='major', labelsize=14)
    cbarphase = fig.colorbar(mappable2, cmap = cmocean.cm.balance, ax = ax[1], location = 'right')
    cbarphase.ax.set_ylabel(r'Phase', fontsize=16)
    cbarphase.ax.set_yticks(ticks = [-np.pi, 0, np.pi], labels = [r'$-\pi$', 0, r'$+\pi$'])
    cbarphase.ax.tick_params(labelsize=14)
    
    
    if wf_title:
        fig.suptitle(wf_title, x = 0.46, y = 0.93, fontsize = 18)
        
    if wf_subtitle:
        fig.text(x = 0.42, y = 0, s = wf_subtitle, fontsize = 14, ha = 'left')
    
    if wf_path:
        fig.savefig(wf_path, dpi = dpi, bbox_inches = 'tight', format = ftype)
    
    #plot spectrum
    
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #get sh coefficients for the wave function
    spectrum = pysh.spectralanalysis.spectrum(coeffs, normalization = 'ortho') # creat array that contains the spectrum
    fig, ax = plt.subplots(1, 1)
    ls = np.arange(0, lmax + 1, 1)
    ax.plot(ls, spectrum, label = 'Power per degree')
    ax.grid()
    ax.legend(fontsize = 10)
    ax.set_yscale('log')
    ax.set_xlim(0, lmax)
    ax.set_xlabel(r'Spherical harmonic degree $l$')
    ax.set_ylabel('Power')
    
    if spectrum_title:
        fig.suptitle(spectrum_title, fontsize = 12)
    
    if spectrum_path:
        fig.savefig(spectrum_path, dpi = dpi, bbox_inches = 'tight', format = ftype)
        
    
    #plot 2d spectrum 
    
    plot_2dspectrum(coeffs, spectrum2d_path, spectrum2d_title)
    
    plt.show()
    return None


################################### NEWTON RAPHSON ######################################################################

#solve the linear system A * x = b
#the scipy linalg methods require that all vectors are in 1D form, therefore v is a 1D array of length 4N^2, 2 times all the points in the grid 
#the real and imaginary part of delta psi are arrays of length 2N^2
#and I can reshape them onto the N x 2N grid to perform the effects of the linear operators in the Jacobian
#NR2D is for the NR method where the chemical potential is kept constant


#sGPE functional
#this will be the used to calculate the residual of the NR method
def Functional(psi, mu, g, omega):  
    F = - 0.5 * Laplacian(psi) + g * np.abs(psi)**2 * psi - 1.0j * omega * deriv_phi(psi) - mu * psi
    return F


#function to plot sGPE functional |F|^2 / particle_number, its logarithm and the phase of F
#path: string of path where to save the plot

def Fplot(F, particle_number, path = ''):
    fig, ax = plt.subplot_mosaic(
        [['A'],['B'],['C']],
        figsize = (7, 9)
    )
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    func =  np.abs(F)**2 / particle_number
    
    mappable1 = ax['A'].pcolormesh(func, cmap = cmocean.cm.thermal, vmin = np.min(func), vmax = np.max(func))
    ax['A'].invert_yaxis()
    ax['A'].set_xlabel(r'$\phi$')
    ax['A'].set_ylabel(r'$\theta$')
    ax['A'].set_yticks(ticks = (0, params.N//2, params.N), labels = ('0°', '90°', '180°'))
    ax['A'].set_xticks(ticks = (0, params.N//2, params.N, 3 * params.N / 2, 2 * params.N), labels=('0°', '90°', '180°', '270°', '360°'))
    fig.colorbar(mappable1, cmap = cmocean.cm.thermal, label = r'$|F|^2 / N$', ax = ax['A'], location = 'right')
    
    mappable2 = ax['B'].pcolormesh(np.log(func), cmap = cmocean.cm.thermal, vmin = np.min(np.log(func )), vmax = np.max(np.log(func)))
    ax['B'].invert_yaxis()
    ax['B'].set_xlabel(r'$\phi$')
    ax['B'].set_ylabel(r'$\theta$')
    ax['B'].set_yticks(ticks = (0, params.N//2, params.N), labels = ('0°', '90°', '180°'))
    ax['B'].set_xticks(ticks = (0, params.N//2, params.N, 3 * params.N / 2, 2 * params.N), labels=('0°', '90°', '180°', '270°', '360°'))
    fig.colorbar(mappable2, cmap = cmocean.cm.thermal, label = r'$\log \left ( |F|^2 / N \right ) $', ax = ax['B'], location = 'right')
    
    mappable3 = ax['C'].pcolormesh(np.angle(F), cmap = cmocean.cm.balance, vmin = -np.pi, vmax = np.pi)
    ax['C'].invert_yaxis()
    ax['C'].set_xlabel(r'$\phi$')
    ax['C'].set_ylabel(r'$\theta$')
    ax['C'].set_yticks(ticks = (0, params.N//2, params.N), labels = ('0°', '90°', '180°'))
    ax['C'].set_xticks(ticks = (0, params.N//2, params.N, 3 * params.N / 2, 2 * params.N), labels=('0°', '90°', '180°', '270°', '360°'))
    cb = fig.colorbar(mappable3, cmap = cmocean.cm.balance, label = r'Phase of $F$', ax = ax['C'], location = 'right')
    cb.ax.set_yticks(ticks = [-np.pi, 0, np.pi], labels = [r'$-\pi$', 0, r'$+\pi$'])
    
    if path:
        fig.savefig(path, dpi = 300, format = 'jpg', bbox_inches = 'tight')
    return None


#function to plot the residuals as a function of the iteration number 
#currently unused and untested

def resplot(residuals, counter, omega):
    plt.plot(np.arange(0, counter + 1, 1), residuals[0: counter + 1], linestyle = 'None', marker = '.')
    plt.xlabel('Iterations')
    plt.ylabel('NR residual')
    plt.yscale('log')
    plt.savefig('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/residuals_' + str(omega) + '.pdf', dpi = 300, bbox_inches = 'tight')
    return None


#matvec function that contains the information of the whole Jacobian to construct the linear operator, v is now a 1D array of length 4N^2 
#psig: initial guess for NR method, must have shape (N, 2N)
#mu: dimensionless chemical potential
#g: dimensionless interaction parameter
#omega: dimensionless rotation frequency


def matvec_NR2D(v, psig, mu, g, omega):
    deltapsir = v[:2 * params.N**2] #first 2N^2 entries correspond to real part of delta psi
    deltapsii = v[2 * params.N**2:] #second 2N^2 entries correspond to imaginary part of delta psi
    
    #reshape to the grid shape
    deltapsir_grid = np.reshape(deltapsir, newshape = (params.N, 2 * params.N), order = 'C')
    deltapsii_grid = np.reshape(deltapsii, newshape = (params.N, 2 * params.N), order = 'C')
    
    #create sh coeffs
    coeffs_deltapsir = pysh.expand.SHExpandDHC(deltapsir_grid, norm = 4, sampling = 2)
    coeffs_deltapsii = pysh.expand.SHExpandDHC(deltapsii_grid, norm = 4, sampling = 2)
    
    #calculate Laplacian of deltapsir and deltapsii
    laplacian_deltapsir = pysh.expand.MakeGridDHC(coeffs_deltapsir * laplacian_multiplier_array, norm = 4, sampling = 2, extend = False)
    laplacian_deltapsii = pysh.expand.MakeGridDHC(coeffs_deltapsii * laplacian_multiplier_array, norm = 4, sampling = 2, extend = False)
    
    #calculate derivate wrt phi of deltapsir and deltapsii    
    deriv_deltapsir = pysh.expand.MakeGridDHC(coeffs_deltapsir * deriv_phi_multiplier_array, norm = 4, sampling = 2, extend = False)
    deriv_deltapsii = pysh.expand.MakeGridDHC(coeffs_deltapsii * deriv_phi_multiplier_array, norm = 4, sampling = 2, extend = False)
    
    #calculate the entries of A * v on the grid and then flatten them
    A11_deltapsir = - 0.5 * np.real(laplacian_deltapsir) + g * (3 * np.real(psig)**2 + np.imag(psig)**2) * deltapsir_grid - mu * deltapsir_grid
    A12_deltapsii = 2 * g * np.real(psig) * np.imag(psig) * deltapsii_grid + omega * np.real(deriv_deltapsii)
    entry1 = np.ravel(A11_deltapsir + A12_deltapsii, order = 'C')
    
    A21_deltapsir = 2 * g * np.real(psig) * np.imag(psig) * deltapsir_grid - omega * np.real(deriv_deltapsir)
    A22_deltapsii = - 0.5 * np.real(laplacian_deltapsii) + g * (3 * np.imag(psig)**2 + np.real(psig)**2) * deltapsii_grid - mu * deltapsii_grid
    entry2 = np.ravel(A21_deltapsir + A22_deltapsii, order = 'C')
    
    result = np.concatenate((entry1, entry2))

    return result

#full implementation of NR method
#psig: initial guess for NR method, must have shape (N, 2N)
#mu: dimensionless chemical potential
#g: dimensionless interaction parameter
#omega: dimensionless rotation frequency
#epsilon: tolerance of NR method. Once residual falls below epsilon, the method has converged
#counter: integer to count the number of iterations
#maxcounter: maximum number of iterations after which the procedure will be aborted

def NR2D(psig, mu, g, omega, epsilon, Fpath = '', counter = 0, maxcounter = 20):

    F = Functional(psig, mu, g, omega) #compute Functional of psig
    F_coeffs = pysh.expand.SHExpandDHC(F, norm = 4, sampling = 2) #compute SH coeffs of functional
    norm = np.sqrt(np.sum(np.abs(F_coeffs)**2) / get_norm(psig)) #compute norm of functional
    Fpath = 'E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/F_' + str(np.round(mu, 0)) + '_' + str(np.round(omega, 4)) + '_' + str(counter) + '.jpg' #set path where to save the plot of sGPE functional
    Fplot(F, get_norm(psig), path = Fpath) #plot the sGPE functional

    print(counter)
    print(norm)

    
    if (norm < epsilon): #if norm is smaller than epsilon, convergence is achieved and psinew is returned
        print('Iterations to convergence: ', counter)
        return psig
    
    #if maxcounter of iterations is reached, abort
    if (counter == maxcounter):
        print('Iterations to convergence: ', counter)
        return psig
        
    
    def mv(v):
        return matvec_NR2D(v, psig, mu, g, omega)

    A = LinearOperator(shape = (4 * params.N**2, 4 * params.N**2), 
                                 matvec = mv, 
                                 dtype = np.float64)
    
    Fr_flat = np.ravel(np.real(F), order = 'C') #1D array of real part of functional
    Fi_flat = np.ravel(np.imag(F), order = 'C') #1D array of imaginary part of functional
    b = np.concatenate((Fr_flat, Fi_flat)) #right hand side of linearised problem as a 1D array  
    
    def callback(xk):
        residual = np.linalg.norm(b - A * xk) / np.linalg.norm(b)
        print(residual)  
        
    #result, info = gmres(A, b, rtol = .1, callback = callbackgmres, callback_type= 'pr_norm') #perform algorithm to solve linear equation
    result, info = lgmres(A, b, rtol = .09, callback = callback, maxiter = 10000) #perform algorithm to solve linear equation

    if (info != 0):
        print('Linear solver did not converge. Info: ' + str(info) + '. Counter: ' + str(counter + 1))
        return np.zeros(shape = (params.N, 2 * params.N), dtype = np.complex128)
    
    #reshape result of linear solver psi into real part and imaginary part of delta psi on the grid
    deltapsir = np.reshape(result[:2 * params.N**2], newshape = (params.N, 2 * params.N), order = 'C')
    deltapsii = np.reshape(result[2 * params.N**2:], newshape = (params.N, 2 * params.N), order = 'C')
    
    deltapsi = deltapsir + 1.0j * deltapsii #compute deltapsi
    psinew = psig - deltapsi
    
    #below I included  a plotting routine for deltapsi, but currently unused
    '''
    #plot deltapsi
    
    wf_title= r'$\Delta \Psi$ for Iteration: ' + str(counter + 1) 
    spectrum2d_title = '2D Spectrum for Iteration: ' + str(counter + 1)
    spectrum_title = 'Spectrum for Iteration: ' + str(counter + 1)
    wf_path = 'E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/wf_deltapsi' + str(omega) + '_' + str(counter + 1) + '.pdf'
    spectrum_path = 'E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/spectrum_deltapsi' + str(omega) + '_' + str(counter + 1) + '.pdf'
    spectrum2d_path = 'E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/spectrum2d_deltapsi' + str(omega) + '_' + str(counter + 1) + '.jpg'
    plot(deltapsi, wf_title=wf_title, spectrum_title=spectrum_title, spectrum2d_title=spectrum2d_title, wf_path=wf_path, spectrum_path=spectrum_path, spectrum2d_path=spectrum2d_path)
    '''
    #recur NR method with psinew as next guess
    return NR2D(psinew, mu, g, omega, epsilon, counter = counter + 1)    





    