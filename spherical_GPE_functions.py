import numpy as np
import pyshtools as pysh
import spherical_GPE_params as params
import cmocean
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, gmres, lgmres, gcrotmk, bicgstab

import scienceplots

#set some parameters for plotting

plt.style.use('science')
plt.rcParams.update({'font.size': 7})
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')


lstart = params.lmax - 20 #sh degree above which filtering will start (initially)

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

#initial phase angle of a vortex antivortex pair at the positions (theta_plus, phi_plus) and (theta_minus, phi_minus) on the grid

def phase(theta, phi, theta_plus, phi_plus, theta_minus, phi_minus):
    denom1 = cot(theta/2) * np.sin(phi) - cot(theta_plus/2) * np.sin(phi_plus)
    num1 = cot(theta/2) * np.cos(phi) - cot(theta_plus/2) * np.cos(phi_plus)
    denom2= cot(theta/2) * np.sin(phi) - cot(theta_minus/2) * np.sin(phi_minus)
    num2 = cot(theta/2) * np.cos(phi) - cot(theta_minus/2) * np.cos(phi_minus)
    
    phase = np.arctan2(num1, denom1) - np.arctan2(num2, denom2)
    return phase


#model initial magnitude of the wavefunction (a function that goes from 0 to 1 over the length scale of the healing length xi at the position of the vortices)

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

def IC_vortex_dipole(theta_plus, phi_plus, theta_minus, phi_minus, xi, bg_dens):
    phaseangle = phase(params.THETA, params.PHI, theta_plus, phi_plus, theta_minus, phi_minus)
    magplus = one_vortex_magnitude(params.THETA, params.PHI, theta_plus, phi_plus, xi)
    magminus = one_vortex_magnitude(params.THETA, params.PHI, theta_minus, phi_minus, xi)
    psi = np.sqrt(bg_dens) * magplus * magminus * np.exp(1j * phaseangle)
    
    return psi

#################### DERIVATIVES ###############################

#derivative with respect to azimuthal angle phi of a function psi

def deriv_phi(psi):
    sh_coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #expand sh coefficients of psi
    
    def multiplier(i, l, m):
        return 1.0j * m * (-1.)**i
    
    multiplier_array = np.fromfunction(multiplier, shape = np.shape(sh_coeffs), dtype = np.complex128) #create array from the multiplier function. The sh coeffs have to be multiplied with i*m to get the derivative wrt phi
    sh_coeffs = sh_coeffs * multiplier_array #modify coeffs with the above array
    psi = pysh.expand.MakeGridDHC(sh_coeffs, norm = 4, sampling = 2, extend = False) #back to real space, with the gridded data of the modified coefficients 
    return psi

#angular laplacian of a function psi

def Laplacian(psi):
    sh_coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #expand sh coefficients of psi
    
    def multiplier(i, l, m):
        return - l * (l + 1)
    
    multiplier_array = np.fromfunction(multiplier, shape = np.shape(sh_coeffs), dtype = np.float64) #create array from the multiplier function. The sh coeffs have to be multiplied with -l(l+1) to get the laplacian
    sh_coeffs = sh_coeffs * multiplier_array #modify coeffs with the above array
    psi = pysh.expand.MakeGridDHC(sh_coeffs, norm = 4, sampling = 2, extend = False) #back to real space, with the gridded data of the modified coefficients 
    return psi

#angular laplacian for a real valued function f

def Laplacianr(f):
    coeffs = pysh.expand.SHExpandDH(f, norm = 4, sampling = 2) 
    
    def multiplier(i, l, m):
        return - l * (l + 1)
    multiplier_array = np.fromfunction(multiplier, shape = np.shape(coeffs), dtype = np.float64)
    coeffs = coeffs * multiplier_array
    f = pysh.expand.MakeGridDH(coeffs, norm = 4, sampling = 2, extend = False)
    return f


############################## CONSERVED QUANTITIES #####################################################

#norm function, calculates particle number of the condensate

def get_norm(psi):
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #sh coefficients of wavefunction
    norm = np.sum(np.abs(coeffs)**2) #sum over the absolute value squared of all coefficients
    return norm

#calculate energy of condensate

def get_energy(psi, g, omega, G = 0):
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2)
    coeffs2 = pysh.expand.SHExpandDH(np.abs(psi)**2, norm = 4, sampling = 2)
    
    def kinetic(i, l, m):
        return 0.5 * l * (l + 1) 
    def rotation(i, l, m):
        return omega * m * (-1.)**i
    
    kinetic_multiplier = np.fromfunction(kinetic, shape = (2, params.lmax + 1, params.lmax + 1), dtype = np.float64)
    rotation_multiplier = np.fromfunction(rotation, shape = (2, params.lmax + 1, params.lmax + 1), dtype = np.float64)
    
    ekin = np.sum(kinetic_multiplier * np.abs(coeffs)**2)
    erot = np.sum(rotation_multiplier * np.abs(coeffs)**2)
    eint = np.sum(0.5 * g * coeffs2**2)
    
    if G:
        coeffs_grav = pysh.expand.SHExpandDHC(psi * (np.cos(params.theta_grid) + 1), norm = 4, sampling = 2)
        eg = np.real(np.sum(G * coeffs * np.conj(coeffs_grav)))
        return ekin, eint, erot, eg
    
    return ekin, eint, erot

#calculate angular momentum of condensate in z direction

def get_ang_momentum(psi):
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2)
    
    def angmom(i, l, m):
        return m * (-1)**i
    
    mom_multiplier = np.fromfunction(angmom, shape = np.shape(coeffs), dtype = np.float64)
    mom = np.sum(mom_multiplier * np.abs(coeffs)**2)
    
    return mom

############################## TIME EVOLUTION ##############################################################

#timestep with split stepping method where the sh coefficients are the input and output

def timestep_coeffs(coeffs, dt, g, omega):
    def step(i, l, m): #this function will be mutiplied entry wise with coeffs with entry indices i, l, m
        return np.exp(- 1.0j * 0.5 * l * (l + 1) * params.dt ) * np.exp(- 1.0j * m * params.omega * params.dt * (-1.)**i)
    
    step_multiplier = np.fromfunction(step, shape = np.shape(coeffs), dtype = np.complex128) #create array of same shape as coeffs with entries from step
    coeffs = coeffs * step_multiplier
    
    psi = pysh.expand.MakeGridDHC(coeffs, norm = 4, sampling = 2, extend = False) #create gridded data in (N, 2*N) array from coeffs
    psi = psi * np.exp(-1.0j * g * dt * np.abs(psi)**2) #timestep of nonlinear term
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #calculate expansion coefficients from the gridded data
    return coeffs

#same timestep as above, except the input and output is the gridded data (includes a filter for dealiasing)

def timestep_grid(psi, dt, g, omega, G = 0, mu = 0, filtering = True):
    psi = psi * np.exp(-1.0j * g * 0.5 * dt * np.abs(psi)**2)
    if G:
        psi = psi * np.exp(-1.0j * G * (np.cos(params.theta_grid) + 1) * dt)
        
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2)
    
    def step(i, l, m): #this function will be mutiplied entry wise with coeffs with entry indices i, l, m
        return np.exp(- 1.0j * 0.5 * l * (l + 1) * params.dt ) * np.exp(- 1.0j * m * omega * params.dt * (-1.)**i)
    
    step_multiplier = np.fromfunction(step, shape = np.shape(coeffs), dtype = np.complex128) #create array of same shape as coeffs with entries from step
    coeffs = coeffs * step_multiplier
    
    spectrum = pysh.spectralanalysis.spectrum(coeffs, normalization = 'ortho')
    
    if filtering:
        global lstart
        if (spectrum[lstart] > spectrum[lstart - 10]):
            lstart = lstart - 5
            if (lstart < 2 * params.lmax // 3):
                lstart = 2 * params.lmax // 3
        
        def exp_filter(i, l, m): #exponential filter
            alpha = 0.01 
            return np.exp(- alpha * (l - lstart))
        
        filter_multiplier = np.fromfunction(exp_filter, shape = np.shape(coeffs), dtype = np.float64) #create array from the filter with same shape as coeffs
        coeffs[:, lstart:, :] = coeffs[:, lstart:, :] * filter_multiplier[:, lstart:, :] #apply filter from lstart onwards
        
    
    psi = pysh.expand.MakeGridDHC(coeffs, norm = 4, sampling = 2, extend = False) #create gridded data in (N, 2*N) array from coeffs
    
    if mu:
        psi = psi * np.exp(1.0j * dt * mu)
    psi = psi * np.exp(-1.0j * g * 0.5 * dt * np.abs(psi)**2)
    return psi

#the same timestep, but for imaginary time on the grid

def imaginary_timestep_grid(psi, dt, g, omega, particle_number, keep_phase = True):
    phase = np.angle(psi)
    psi = psi * np.exp(- 0.5 * g * dt * np.abs(psi)**2) #half timestep of nonlinear term
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2)
    
    def step(i, l, m):#this function will be mutiplied entry wise with coeffs with entry indices i, l, m
        return np.exp(- 0.5 * l * (l + 1) * dt ) * np.exp(- m * omega * dt * (-1.)**i)
    
    step_multiplier = np.fromfunction(step, shape = np.shape(coeffs), dtype = np.float64)
    coeffs = coeffs * step_multiplier
    
    psi = pysh.expand.MakeGridDHC(coeffs, norm = 4, sampling = 2, extend = False) #create gridded data in (N, 2*N) array from coeffs
    psi = psi * np.exp(- 0.5 * g * dt * np.abs(psi)**2) #half timestep of nonlinear term
    norm = get_norm(psi)
    if keep_phase:
        psi = np.sqrt(particle_number) * np.abs(psi) * np.exp(1.0j * phase) / np.sqrt(norm)
    else:
        psi = np.sqrt(particle_number) * psi / np.sqrt(norm)
    return psi

#filtering method to use outside of the timesteps

def filtering(psi, lstart, alpha, k):
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2)
    
    def exp_filter(i, l, m): #exponential filter
        return np.exp(- alpha * (l - lstart)**k)
    
    filter_multiplier = np.fromfunction(exp_filter, shape = np.shape(coeffs), dtype = np.float64) #create array from the filter with same shape as coeffs
    coeffs[:, lstart:, :] = coeffs[:, lstart:, :] * filter_multiplier[:, lstart:, :] #apply filter from lstart onwards
    
    psi = pysh.expand.MakeGridDHC(coeffs, norm = 4, sampling = 2, extend = False) #create gridded data in (N, 2*N) array from coeffs
    
    return psi
    
    
################### VORTEX TRACKING ############################

def vortex_tracker(psi, theta_guess, phi_guess, counter = 0):
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
    psi_guess = np.sum(coeffs * pysh.expand.spharm(params.lmax, theta_guess, phi_guess, normalization = 'ortho', kind = 'complex', degrees = False))
    Jacobian[0, 0] = np.sum(coeffs_theta_real * pysh.expand.spharm(params.lmax, theta_guess, phi_guess, normalization = 'ortho', kind = 'real', degrees = False))
    Jacobian[0, 1] = np.sum(coeffs_phi_real * pysh.expand.spharm(params.lmax, theta_guess, phi_guess, normalization = 'ortho', kind = 'real', degrees = False))
    Jacobian[1, 0] = np.sum(coeffs_theta_imag * pysh.expand.spharm(params.lmax, theta_guess, phi_guess, normalization = 'ortho', kind = 'real', degrees = False))
    Jacobian[1, 1] = np.sum(coeffs_phi_imag * pysh.expand.spharm(params.lmax, theta_guess, phi_guess, normalization = 'ortho', kind = 'real', degrees = False))
    
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



############## PLOTTING ROUTINES (plots density and phase of wavefunction, plots spectrum of wavefunction) ############

def plot_2dspectrum(coeffs, path, title):
    grid = np.concatenate((np.flip(coeffs[1,:,1:], axis = 1), coeffs[0, :, :]), axis = 1) #form a 2D array that has l as the first index and m going from -lmax to lmax in the second index ([0] = -lmax, [-1] = lmax)                   
    spectrum = np.abs(grid)**2 #calculate absolute value squared to plot
    spectrum[spectrum == 0.0] = np.nan #coefficients that don't exist are zero in the array, so set these to NaN

    #create coordinate grid for pcolormesh
    lmax = len(coeffs[0,:,0]) - 1
    ls = np.arange(0, lmax + 1, 1)
    ms = np.arange(- lmax, lmax + 1, 1)

    #plot log of spectrum as pcolormesh
    fig, ax = plt.subplots(1, 1, figsize = (7,5))
    if title:
        plt.suptitle(title, fontsize = 12)
    mappable = ax.pcolormesh(ms, ls, np.log(spectrum), cmap = cmocean.cm.haline, vmin = np.nanmin(np.log(spectrum)),  vmax = np.nanmax(np.log(spectrum)))
    ax.invert_yaxis()
    fig.colorbar(mappable, label = 'Power per coefficient', ax = ax)
    ax.set_xlabel(r'Spherical harmonic order $m$')
    ax.set_ylabel(r'Spherical harmonic degree $l$')
    if path:
        fig.savefig(path, dpi = 300, bbox_inches = 'tight')
    return None


#psi: wavefunction to plot
#wf_title: string of title of density + phase plot
#spectrum_title: string of title of spectrum plot
#wf_path: string of path where to save density + phase plot
#spectrum_path: string of path where to save spectrum plot
#dpi: dpi of saved plots
#ftype: filetype of saved plots
def plot(psi, wf_title = '', spectrum_title = '', spectrum2d_title = '', wf_path = '', spectrum_path = '', spectrum2d_path = '', dpi = 600, ftype = 'jpg'):
    
    dens = np.abs(psi)**2 #calculate condensate density
    phase_angle = np.angle(psi) #calculate phase of condensate
    
    N = np.shape(psi)[0]
    lmax = N//2 - 1    
    
    #plot wave function
    
    fig, ax = plt.subplots(2, 1, figsize = (9, 8))
    plt.subplots_adjust(hspace=0.2)


    mappable1 = ax[0].pcolormesh(dens, cmap = cmocean.cm.thermal, vmin = np.min(dens), vmax = np.max(dens))
    ax[0].invert_yaxis()
    ax[0].set_ylabel(r'Latitude')
    ax[0].set_yticks(ticks = (0, N//4, N//2, 3 * N // 4, N), labels = ('90°', '45°', '0°', '-45°', '-90°'))
    ax[0].set_xticks(ticks = (0, N//2, N, 3 * N / 2, 2 * N), labels=('0°', '90°', '180°', '270°', '360°'))
    fig.colorbar(mappable1, cmap = cmocean.cm.thermal, label = r'$nR^2$', ax = ax[0], location = 'right')

    mappable2 = ax[1].pcolormesh(phase_angle, cmap = cmocean.cm.balance, vmin = -np.pi, vmax = np.pi)
    ax[1].invert_yaxis()
    ax[1].set_xlabel(r'Longitude')
    ax[1].set_ylabel(r'Latitude')
    ax[1].set_yticks(ticks = (0, N//4, N//2, 3 * N // 4, N), labels = ('90°', '45°', '0°', '-45°', '-90°'))
    ax[1].set_xticks(ticks = (0, N//2, N, 3 * N / 2, 2 * N), labels=('0°', '90°', '180°', '270°', '360°'))
    cb = fig.colorbar(mappable2, cmap = cmocean.cm.balance, label = r'Phase', ax = ax[1], location = 'right')
    cb.ax.set_yticks(ticks = [-np.pi, 0, np.pi], labels = [r'$-\pi$', 0, r'$+\pi$'])
    
    
    if wf_title:
        fig.suptitle(wf_title, x = 0.46, y = 0.93, fontsize = 16)
    
    if wf_path:
        fig.savefig(wf_path, dpi = dpi, bbox_inches = 'tight', format = ftype)
    
    #plot spectrum
    
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #get sh coefficients for the wave function
    clm = pysh.SHCoeffs.from_array(coeffs, normalization='ortho', lmax = params.lmax) #create a SHCoeffs instance for the wavefunction to plot spectrum (at the bottom)
    fig, ax = plt.subplots(1, 1)
    clm.plot_spectrum(unit = 'per_l', show = False, ax = ax)
    ax.set_xlim(0, lmax)
    
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
#the scipy linalg methods require that all vectors are in 1D form, therefore v is a 1D array of length 4N^2 + 1, 2 times all the points in the grid + term for particle conservation
#the real and imaginary part of delta psi are arrays of length 2N^2
#and I can reshape them onto the N x 2N grid to perform the effects of the linear operators in the Jacobian
#NR2D is for the NR method where the chemical potential is kept constant
#NR3D includes the chemical potential as parameter to be varied, keeping the particle number constant

#sGPE functional
def Functional(psi, mu, g, omega):  
    F = - 0.5 * Laplacian(psi) + g * np.abs(psi)**2 * psi - 1.0j * omega * deriv_phi(psi) - mu * psi
    return F


#function to plot sGPE functional
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

def resplot(residuals, counter, omega):
    plt.plot(np.arange(0, counter + 1, 1), residuals[0: counter + 1], linestyle = 'None', marker = '.')
    plt.xlabel('Iterations')
    plt.ylabel('NR residual')
    plt.yscale('log')
    plt.savefig('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/residuals_' + str(omega) + '.pdf', dpi = 300, bbox_inches = 'tight')
    return None


#matvec function that contains the information of the whole Jacobian to construct the linear operator, v is now a 1D array of length 4N^2 

def matvecsimple(v, psig, mu, g, omega):
    deltapsir = v[:2 * params.N**2] #first 2N^2 entries correspond to real part of delta psi
    deltapsii = v[2 * params.N**2:] #second 2N^2 entries correspond to imaginary part of delta psi
    
    
    deltapsi = deltapsir + 1j * deltapsii
    deltapsi_grid = np.reshape(deltapsi, newshape = (params.N, 2 * params.N), order = 'C')
    
    
    yy = -0.5 * Laplacian(deltapsi_grid) + g * (2 * np.abs(psig)**2 + psig**2) * deltapsi_grid - mu * deltapsi_grid - 1j * omega * deriv_phi(deltapsi_grid)
    
    yy = np.ravel(yy, order = 'C')

    return np.concatenate((np.real(yy), np.imag(yy)))


def matvec_NR2D(v, psig, mu, g, omega):
    deltapsir = v[:2 * params.N**2] #first 2N^2 entries correspond to real part of delta psi
    deltapsii = v[2 * params.N**2:] #second 2N^2 entries correspond to imaginary part of delta psi
    
    deltapsir_grid = np.reshape(deltapsir, newshape = (params.N, 2 * params.N), order = 'C')
    deltapsii_grid = np.reshape(deltapsii, newshape = (params.N, 2 * params.N), order = 'C')
    
    #calculate the entries of A * v on the grid and then flatten them
    A11_deltapsir = - 0.5 * Laplacianr(deltapsir_grid) + g * (3 * np.real(psig)**2 + np.imag(psig)**2) * deltapsir_grid - mu * deltapsir_grid
    A12_deltapsii = 2 * g * np.real(psig) * np.imag(psig) * deltapsii_grid + omega * np.real(deriv_phi(deltapsii_grid))
    entry1 = np.ravel(A11_deltapsir + A12_deltapsii, order = 'C')
    
    A21_deltapsir = 2 * g * np.real(psig) * np.imag(psig) * deltapsir_grid - omega * np.real(deriv_phi(deltapsir_grid))
    A22_deltapsii = - 0.5 * Laplacianr(deltapsii_grid) + g * (3 * np.imag(psig)**2 + np.real(psig)**2) * deltapsii_grid - mu * deltapsii_grid
    entry2 = np.ravel(A21_deltapsir + A22_deltapsii, order = 'C')
    
    result = np.concatenate((entry1, entry2))

    return result

#full implementation of NR method

def NR2D(psig, mu, g, omega, epsilon,  counter = 0, maxcounter = 20):

    F = Functional(psig, mu, g, omega) #compute Functional of psig
    F_coeffs = pysh.expand.SHExpandDHC(F, norm = 4, sampling = 2) #compute SH coeffs of functional
    norm = np.sqrt(np.sum(np.abs(F_coeffs)**2) / get_norm(psig)) #compute norm of functional
    Fpath = 'E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/F_' + str(omega) + '_' + str(counter) + '.jpg'
    Fplot(F, get_norm(psig), path = Fpath)

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
    
    def callbackgmres(pr_norm): 
        print(pr_norm)    
        
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
    
    #plot deltapsi
    wf_title= r'$\Delta \Psi$ for Iteration: ' + str(counter + 1) 
    spectrum2d_title = '2D Spectrum for Iteration: ' + str(counter + 1)
    spectrum_title = 'Spectrum for Iteration: ' + str(counter + 1)
    wf_path = 'E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/wf_deltapsi' + str(omega) + '_' + str(counter + 1) + '.pdf'
    spectrum_path = 'E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/spectrum_deltapsi' + str(omega) + '_' + str(counter + 1) + '.pdf'
    spectrum2d_path = 'E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/spectrum2d_deltapsi' + str(omega) + '_' + str(counter + 1) + '.jpg'
    plot(deltapsi, wf_title=wf_title, spectrum_title=spectrum_title, spectrum2d_title=spectrum2d_title, wf_path=wf_path, spectrum_path=spectrum_path, spectrum2d_path=spectrum2d_path)
    
    #recur NR method with psinew as next guess
    return NR2D(psinew, mu, g, omega, epsilon, counter = counter + 1)    

#second try

def A(v, psig, mug, g, omega):
    #extract deltapsi, deltapsi*, deltamu from input vector v
    deltapsi = v[:2 * params.N**2]
    deltapsiconj = v[2 * params.N**2:-1]
    deltamu = v[-1]
    
    #reshape deltapsi, deltapsi* onto the N x 2N grid
    deltapsi_grid = np.reshape(deltapsi, newshape = (params.N, 2 * params.N), order = 'C')
    deltapsiconj_grid = np.reshape(deltapsiconj, newshape = (params.N, 2 * params.N), order = 'C')
    
    #calculate the entries of A * v on the grid and then flatten them
    A11_deltapsi = - 0.5 * Laplacian(deltapsi_grid) + 2 * g * np.abs(psig)**2 * deltapsi_grid - 1.0j * omega * deriv_phi(deltapsi_grid) - mug * deltapsi_grid
    A12_deltapsiconj = g * psig**2 * deltapsiconj_grid
    A13_deltamu = psig * deltamu
    entry1 = np.ravel(A11_deltapsi + A12_deltapsiconj + A13_deltamu, order = 'C')
    
    A21_deltapsi = g * np.conj(psig)**2 * deltapsi_grid
    A22_deltapsiconj = - 0.5 * Laplacian(deltapsiconj_grid) + 2 * g * np.abs(psig)**2 * deltapsiconj_grid + 1.0j * omega * deriv_phi(deltapsiconj_grid) - mug * deltapsiconj_grid
    A23_deltamu = np.conj(psig) * deltamu
    entry2 = np.ravel(A21_deltapsi + A22_deltapsiconj + A23_deltamu)
    
    coeffsg = pysh.expand.SHExpandDHC(psig, norm = 4, sampling = 2)
    coeffs = pysh.expand.SHExpandDHC(deltapsi_grid, norm = 4, sampling = 2)
    A31_deltapsi = - np.sum(np.conj(coeffsg) * coeffs)
    A32_deltapsiconj = - np.sum(coeffsg * np.conj(coeffs))
    entry3 = np.array([A31_deltapsi + A32_deltapsiconj])
    
    result = np.concatenate((entry1, entry2, entry3))
    
    return result

def NR3D(psig, mug, g, omega, particle_number, epsilon, counter = 0):
    F = Functional(psig, mug, g, omega) # compute F Functional
    G = particle_number - get_norm(psig) #compute G functional
    F_flat = np.ravel(F, order = 'C') #flatten F functional
    b = np.concatenate((F_flat, np.conj(F_flat), np.array([G]))) #generate rhs of A * x = b
    
    F_coeffs = pysh.expand.SHExpandDHC(F, norm = 4, sampling = 2) #compute CH coeffs of functional
    norm = np.sqrt(np.sum(np.abs(F_coeffs)**2)) #compute norm of functional    
    
    print(counter)
    print(norm)
    print(G)
    
    
    if (norm < epsilon): #if norm is smaller than epsilon, convergence is achieved and psig is returned
        print('Iterations to convergence: ', counter)
        return psig, mug
    
    def mv(v):
        return A(v, psig, mug, g, omega)
    
    NR_operator = LinearOperator(shape = (4 * params.N**2 + 1, 4 * params.N**2 + 1), 
                                 matvec = mv, 
                                 dtype = np.complex128)
    
    #create starting guess for bicgstab algorithm
    psig_flat = np.ravel(psig, order = 'C') 
    psi0 = np.concatenate((psig_flat, np.conj(psig_flat), np.array([mug])))

    result, info = bicgstab(NR_operator, b, x0 = psi0, rtol = 0.1) #perform algorithm to solve linear equation
    
    if (info != 0):
        print('Linear solver did not converge. Info: ' + str(info) + '. Counter: ' + str(counter + 1))
        return np.zeros(shape = (params.N, 2 * params.N), dtype = np.complex128), 0
    
    #reshape result of bicgstab psi into real part and imaginary part of delta psi on the grid
    deltapsi = np.reshape(result[:2 * params.N**2], newshape = (params.N, 2 * params.N), order = 'C')
    deltamu = result[-1]
    
    psinew = psig - deltapsi 
    munew = mug - np.real(deltamu)
    
    #recur NR method with psinew, munew as next guess
    return NR3D(psinew, munew, g, omega, particle_number, epsilon, counter + 1)



    