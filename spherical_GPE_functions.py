import numpy as np
import pyshtools as pysh
import spherical_GPE_params as params
import cmocean
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres, bicg, lgmres, tfqmr

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

#define initial phase angle of a vortex antivortex pair at the positions (theta_plus, phi_plus) and (theta_minus, phi_minus)


def phase(theta, phi, theta_plus, phi_plus, theta_minus, phi_minus):
    denom1 = cot(theta/2) * np.sin(phi) - cot(theta_plus/2) * np.sin(phi_plus)
    num1 = cot(theta/2) * np.cos(phi) - cot(theta_plus/2) * np.cos(phi_plus)
    denom2= cot(theta/2) * np.sin(phi) - cot(theta_minus/2) * np.sin(phi_minus)
    num2 = cot(theta/2) * np.cos(phi) - cot(theta_minus/2) * np.cos(phi_minus)
    
    phase = np.arctan2(num1, denom1) - np.arctan2(num2, denom2)
    return phase


#define a model initial magnitude of the wavefunction (a function that goes from 0 to 1 over the length scale of the healing length xi at the position of the vortices)


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


def one_vortex_magnitude(theta, phi, theta_v, phi_v, xi):
    x, y, z = sph2cart(theta, phi)
    
    x_v, y_v, z_v = sph2cart(theta_v, phi_v)
    
    d = np.sqrt((x - x_v)**2 + (y - y_v)**2 + (z - z_v)**2)
    
    arc_length = 2 * np.arcsin(d / 2)
    
    return 1. - np.exp(- arc_length / xi)

#define a function that generates gridded data for the wave function


def generate_gridded_wavefunction(theta_plus, phi_plus, theta_minus, phi_minus, xi, bg_dens):
    psi = np.zeros(shape = (params.N, 2*params.N), dtype = np.complex128)
    
    for i in range(params.N//2):
        for j in range(2*params.N):
            psi[i,j] = np.sqrt(bg_dens) * one_vortex_magnitude(params.theta[i], params.phi[j], theta_plus, phi_plus, xi) * np.exp(1.0j * phase(params.theta[i], params.phi[j], theta_plus, phi_plus, theta_minus, phi_minus))       
            
    for i in range(params.N//2, params.N):
        for j in range(2 * params.N):
            psi[i,j] = np.sqrt(params.bg_dens) * one_vortex_magnitude(params.theta[i], params.phi[j], theta_minus, phi_minus, xi) * np.exp(1.0j * phase(params.theta[i], params.phi[j], theta_plus, phi_plus, theta_minus, phi_minus)) 
    return psi

    
#define norm function, this calculates particle number of the condensate


def get_norm(psi):
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #sh coefficients of wavefunction
    norm = np.sum(np.abs(coeffs)**2) #sum over the absolute value squared of all coefficients
    return norm


#define derivative wrt azimuthal coordinate


def deriv_phi(psi):
    sh_coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #expand sh coefficients of psi
    
    def multiplier(i, l, m):
        return 1.0j * m * (-1.)**i
    
    multiplier_array = np.fromfunction(multiplier, shape = np.shape(sh_coeffs), dtype = np.complex128) #create array from the multiplier function. The sh coeffs have to be multiplied with i*m to get the derivative wrt phi
    sh_coeffs = sh_coeffs * multiplier_array #modify coeffs with the above array
    psi = pysh.expand.MakeGridDHC(sh_coeffs, norm = 4, sampling = 2, extend = False) #back to real space, with the gridded data of the modified coefficients 
    return psi

#define angular Laplacian

def Laplacian(psi):
    sh_coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #expand sh coefficients of psi
    
    def multiplier(i, l, m):
        return - l * (l + 1)
    
    multiplier_array = np.fromfunction(multiplier, shape = np.shape(sh_coeffs), dtype = np.float64) #create array from the multiplier function. The sh coeffs have to be multiplied with -l(l+1) to get the laplacian
    sh_coeffs = sh_coeffs * multiplier_array #modify coeffs with the above array
    psi = pysh.expand.MakeGridDHC(sh_coeffs, norm = 4, sampling = 2, extend = False) #back to real space, with the gridded data of the modified coefficients 
    return psi

#angular Laplacian for real valued function

def Laplacianr(f):
    coeffs = pysh.expand.SHExpandDH(f, norm = 4, sampling = 2) 
    
    def multiplier(i, l, m):
        return - l * (l + 1)
    multiplier_array = np.fromfunction(multiplier, shape = np.shape(coeffs), dtype = np.float64)
    coeffs = coeffs * multiplier_array
    f = pysh.expand.MakeGridDH(coeffs, norm = 4, sampling = 2, extend = False)
    return f

#calculate energy of condensate (conserved quantitiy)


def get_energy(psi, g, omega, G = 0):
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2)
    coeffs2 = pysh.expand.SHExpandDH(np.abs(psi)**2, norm = 4, sampling = 2)
    coeffs_grav = pysh.expand.SHExpandDHC(psi * (np.cos(params.theta_grid) + 1), norm = 4, sampling = 2)
    
    def kinetic(i, l, m):
        return 0.5 * l * (l + 1) 
    def rotation(i, l, m):
        return omega * m * (-1.)**i
    
    kinetic_multiplier = np.fromfunction(kinetic, shape = (2, params.lmax + 1, params.lmax + 1), dtype = np.float64)
    rotation_multiplier = np.fromfunction(rotation, shape = (2, params.lmax + 1, params.lmax + 1), dtype = np.float64)
    #gravity = G * params.dangle**2 * np.sin(params.theta_grid) * np.abs(psi)**2 * (np.cos(params.theta_grid) + 1)
    
    ekin = np.sum(kinetic_multiplier * np.abs(coeffs)**2)
    erot = np.sum(rotation_multiplier * np.abs(coeffs)**2)
    eint = np.sum(0.5 * g * coeffs2**2)
    eg = np.real(np.sum(G * coeffs * np.conj(coeffs_grav)))

    
    return ekin, eint, erot, eg


#calculate angular momentum of condensate in z direction (another conserved quantity).

def get_ang_momentum(psi):
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2)
    
    def angmom(i, l, m):
        return m * (-1)**i
    
    mom_multiplier = np.fromfunction(angmom, shape = np.shape(coeffs), dtype = np.float64)
    mom = np.sum(mom_multiplier * np.abs(coeffs)**2)
    
    return mom



#one timestep with split stepping method where the sh coefficients are the input and output


def timestep_coeffs(coeffs, dt, g, omega):
    def step(i, l, m): #this function will be mutiplied entry wise with coeffs with entry indices i, l, m
        return np.exp(- 1.0j * 0.5 * l * (l + 1) * params.dt ) * np.exp(- 1.0j * m * params.omega * params.dt * (-1.)**i)
    
    step_multiplier = np.fromfunction(step, shape = np.shape(coeffs), dtype = np.complex128) #create array of same shape as coeffs with entries from step
    coeffs = coeffs * step_multiplier
    
    psi = pysh.expand.MakeGridDHC(coeffs, norm = 4, sampling = 2, extend = False) #create gridded data in (N, 2*N) array from coeffs
    psi = psi * np.exp(-1.0j * g * dt * np.abs(psi)**2) #timestep of nonlinear term
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #calculate expansion coefficients from the gridded data
    return coeffs

#the same timestep as above, except the input and output is the gridded data, i.e. the wavefunction

def timestep_grid(psi, dt, g, omega, G = 0, include_gravity = False):
    psi = psi * np.exp(-1.0j * g * 0.5 * dt * np.abs(psi)**2)
    if include_gravity:
        psi = psi * np.exp(-1.0j * G * (np.cos(params.theta_grid) + 1) * dt)
        
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2)
    
    def step(i, l, m): #this function will be mutiplied entry wise with coeffs with entry indices i, l, m
        return np.exp(- 1.0j * 0.5 * l * (l + 1) * params.dt ) * np.exp(- 1.0j * m * params.omega * params.dt * (-1.)**i)
    
    step_multiplier = np.fromfunction(step, shape = np.shape(coeffs), dtype = np.complex128) #create array of same shape as coeffs with entries from step
    coeffs = coeffs * step_multiplier
    
    spectrum = pysh.spectralanalysis.spectrum(coeffs, normalization = 'ortho')
    
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
    psi = psi * np.exp(-1.0j * g * 0.5 * dt * np.abs(psi)**2)
    return psi

#the same timestep, but for imaginary time

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

def imaginary_timestep_grid2(psi, dt, g, omega, particle_number):
    coeffs = pysh.expand.SHExpandDHC(griddh = psi, norm = 4, sampling = 2)
    len_l = np.size(coeffs, axis = 1)  #size of sh_coeffs array in l and m indices(degree and order)
    for l in range(len_l):
        for m in range(len_l):
            for i in range(2):
                coeffs[i,l,m] *= np.exp(- 0.5 * l * (l + 1) * dt ) * np.exp(- m * omega * dt * (-1.)**i)  #timestep of kinetic and rotating term
    psi = pysh.expand.MakeGridDHC(coeffs, norm = 4, sampling = 2, extend = False) #create gridded data in (N, 2*N) array from coeffs
    psi *= np.exp(- g * dt * np.abs(psi)**2) #timestep of nonlinear term
    norm = get_norm(psi)
    psi = np.sqrt(particle_number) * psi / np.sqrt(norm)
    return psi

#filtering

def filtering(psi, lstart, alpha, k):
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2)
    spectrum = pysh.spectralanalysis.spectrum(coeffs, normalization = 'ortho')
    
    if (spectrum[lstart - 1] > spectrum[lstart - 10]):
        lstart = lstart - 5
        if (lstart < 2 * params.lmax // 3):
            lstart = 2 * params.lmax // 3
    
    def exp_filter(i, l, m): #exponential filter
        return np.exp(- alpha * (l - lstart)**k)
    
    filter_multiplier = np.fromfunction(exp_filter, shape = np.shape(coeffs), dtype = np.float64) #create array from the filter with same shape as coeffs
    coeffs[:, lstart:, :] = coeffs[:, lstart:, :] * filter_multiplier[:, lstart:, :] #apply filter from lstart onwards
    
    psi = pysh.expand.MakeGridDHC(coeffs, norm = 4, sampling = 2, extend = False) #create gridded data in (N, 2*N) array from coeffs
    
    return psi
    
    
    

#vortex tracker

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



#plotting routine (plots density and phase of wavefunction, plots spectrum of wavefunction)
#psi: wavefunction to plot
#include_title: True to include a title on the plots
#wf_title: string of title of density + phase plot
#spectrum_title: string of title of spectrum plot
#save: True to save both plots
#wf_path: string of path where to save density + phase plot
#spectrum_path: string of path where to save spectrum plot
#dpi: dpi of saved plots
#ftype: filetype of saved plots
def plot(psi, include_title = False, wf_title = '', spectrum_title = '', save = False, wf_path = '', spectrum_path = '', dpi = 300, ftype = 'pdf'):
    
    dens = np.abs(psi)**2 #calculate condensate density
    phase_angle = np.angle(psi) #calculate phase of condensate
    
    
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #get sh coefficients for the wave function     
    dens_coeffs = pysh.expand.SHExpandDH(dens, norm = 4, sampling = 2) #get sh coefficients for the density
    phase_coeffs = pysh.expand.SHExpandDH(phase_angle, norm = 4, sampling = 2) #get sh coefficients for the phase
    
    clm = pysh.SHCoeffs.from_array(coeffs, normalization='ortho', lmax = params.lmax) #create a SHCoeffs instance for the wavefunction to plot spectrum (at the bottom)
    dens_clm = pysh.SHCoeffs.from_array(dens_coeffs, normalization='ortho', lmax = params.lmax) #create a SHCoeffs instance from the coefficient array for the density 
    phase_clm = pysh.SHCoeffs.from_array(phase_coeffs, normalization='ortho', lmax = params.lmax) #create a SHCoeffs instance from the coefficient array for the phase

    dens_grid = dens_clm.expand() #create a SHGrid instance for the density 
    phase_grid = phase_clm.expand() #create a SHGrid instance for the phase
    
    #plot
    
    gridspec_kw = dict(height_ratios = (1, 1), hspace = 0.5)

    fig, axes = plt.subplots(2, 1, gridspec_kw = gridspec_kw, figsize = (10, 6))
    
    if include_title:
        plt.suptitle(wf_title, fontsize = 12)

    #subplot for denstiy

    dens_grid.plot(cmap = cmocean.cm.thermal, 
                   colorbar = 'right', 
                   cb_label = 'Density', 
                   xlabel = '', 
                   tick_interval = [90,45], 
                   tick_labelsize = 6, 
                   axes_labelsize = 7,
                   ax = axes[0],  
                   show = False)
    
    cb2 = axes[0].images[-1].colorbar
    cb2.mappable.set_clim(np.min(dens), np.max(dens))
    
    
    #subplot for phase

    phase_grid.plot(cmap = cmocean.cm.balance, 
                    colorbar = 'right',
                    cb_label = 'Phase',
                    tick_interval = [90,45], 
                    cb_tick_interval = np.pi,
                    tick_labelsize = 6, 
                    axes_labelsize = 7, 
                    ax = axes[1],  
                    show = False)
    
   
    cb2 = axes[1].images[-1].colorbar
    cb2.mappable.set_clim(-np.pi, np.pi)
    cb2.ax.set_yticks(ticks = [-np.pi, 0, np.pi], labels = [r'$-\pi$', 0, r'$+\pi$'])
    
    if save:
        plt.savefig(wf_path, dpi = dpi, bbox_inches = 'tight', format = ftype)
    
    #plot spectrum
    
    clm.plot_spectrum(unit = 'per_l', show = False)
    
    if include_title:
        plt.title(spectrum_title, fontsize = 12)
    
    if save:
        plt.savefig(spectrum_path, dpi = dpi, bbox_inches = 'tight', format = ftype)
    plt.show()
    return None


################################### NEWTON RAPHSON ######################################################################

#solve the linear system A * x = b
#the scipy linalg methods require that all vectors are in 1D form, therefore v is a 1D array of length 4N^2 + 1, 2 times all the points in the grid + term for particle conservation
#the real and imaginary part of delta psi are arrays of length 2N^2
#and I can reshape them onto the N x 2N grid to perform the effects of the linear operators in the Jacobian

#sGPE functional
def Functional(psi, mu, g, omega):  
    F = - 0.5 * Laplacian(psi) + g * np.abs(psi)**2 * psi - 1.0j * omega * deriv_phi(psi) - mu * psi
    return F

#matvec function that contains the information of the whole Jacobian to construct the linear operator, v is now a 1D array of length 4N^2 
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

def NR2D(psig, mu, g, omega, epsilon, counter = 0):
    F = Functional(psig, mu, g, omega) #compute Functional of psig
    F_coeffs = pysh.expand.SHExpandDHC(F, norm = 4, sampling = 2) #compute SH coeffs of functional
    norm = np.sqrt(np.sum(np.abs(F_coeffs)**2) / get_norm(psig)) #compute norm of functional
    
    
    print(counter)
    print(norm)
    print(get_norm(psig))
    
    if (norm < epsilon): #if norm is smaller than epsilon * energy, convergence is achieved and psig is returned
        print('Iterations to convergence: ', counter)
        return psig
    
    def mv(v):
        return matvec_NR2D(v, psig, mu, g, omega)

    NR_operator = LinearOperator(shape = (4 * params.N**2, 4 * params.N**2), 
                                 matvec = mv, 
                                 dtype = np.float64)
    
    Fr_flat = np.ravel(np.real(F), order = 'C') #1D array of real part of functional
    Fi_flat = np.ravel(np.imag(F), order = 'C') #1D array of imaginary part of functional
    b = np.concatenate((Fr_flat, Fi_flat)) #right hand side of linearised problem as a 1D array
    
    #create starting guess for bicgstab algorithm
    psigr_flat = np.ravel(np.real(psig), order = 'C') 
    psigi_flat = np.ravel(np.imag(psig), order = 'C')
    psi0 = np.concatenate((psigr_flat, psigi_flat))

    result, info = gmres(NR_operator, b, x0 = psi0, rtol = .1, restart = 10) #perform algorithm to solve linear equation

    if (info != 0):
        print('Linear solver did not converge. Info: ' + str(info) + '. Counter: ' + str(counter + 1))
        return np.zeros(shape = (params.N, 2 * params.N), dtype = np.complex128)
    
    #reshape result of bicgstab psi into real part and imaginary part of delta psi on the grid
    deltapsir = np.reshape(result[:2 * params.N**2], newshape = (params.N, 2 * params.N), order = 'C')
    deltapsii = np.reshape(result[2 * params.N**2:], newshape = (params.N, 2 * params.N), order = 'C')
    
    deltapsi = deltapsir + 1.0j * deltapsii #compute deltapsi
    
    psinew = psig - deltapsi 
    
    #recur NR method with psinew, munew as next guess
    return NR2D(psinew, mu, g, omega, epsilon, counter + 1)    

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



    