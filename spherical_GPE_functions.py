import numpy as np
import pyshtools as pysh
import matplotlib.colors as col
import hsluv
import spherical_GPE_params as params



#cotangent

def cot(x):
    return np.tan(np.pi/2 - x)

#transformation from spherical to cartesian coordinates


def sph2cart(theta, phi, r):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
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


def initial_magnitude_1(theta, phi, theta_plus, phi_plus, theta_minus, phi_minus, xi):
    x = np.sin(theta) * np.cos(phi)
    x_plus = np.sin(theta_plus) * np.cos(phi_plus)
    x_minus = np.sin(theta_minus) * np.cos(phi_minus)
    y = np.sin(theta) * np.sin(phi)
    y_plus = np.sin(theta_plus) * np.sin(phi_plus)
    y_minus = np.sin(theta_minus) * np.sin(phi_minus)
    z = np.cos(theta)
    z_plus = np.cos(theta_plus)
    z_minus = np.cos(theta_minus)
    
    exp1 = np.exp(- ((x - x_plus)**2 + (y - y_plus)**2 + (z - z_plus)**2) / xi**2)
    exp2 = np.exp(- ((x - x_minus)**2 + (y - y_minus)**2 + (z - z_minus)**2) / xi**2)
    
    return 1. - exp1 - exp2


def initial_magnitude_2(theta, phi, theta_plus, phi_plus, theta_minus, phi_minus, xi):
    
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
    x, y, z = sph2cart(theta, phi, 1)
    
    x_v, y_v, z_v = sph2cart(theta_v, phi_v, 1)
    
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
    len_m = np.size(sh_coeffs, axis = 2) 
    for m in range(len_m):
        for i in range(2):
            sh_coeffs[i,:,m] *= 1.0j * m * (-1.)**i  #spherical harmonics are eigenfunctions of del_phi, so need only to multiply coefficients with i*m to perform derivative
    psi = pysh.expand.MakeGridDHC(sh_coeffs, norm = 4, sampling = 2, extend = False) #back to real space, with the gridded data of the modified coefficients 
    return psi

#define angular Laplacian

def Laplacian(psi):
    sh_coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #expand sh coefficients of psi
    len_l = np.size(sh_coeffs, axis = 1) 
    for l in range(len_l):
        sh_coeffs[:,l,:] *= - l * (l + 1) #spherical harmonics are eigenfunctions of angular Laplacian, so need only to multiply coefficients with -l(l+1) to perform Laplacian
    psi = pysh.expand.MakeGridDHC(sh_coeffs, norm = 4, sampling = 2, extend = False) #back to real space, with the gridded data of the modified coefficients 
    return psi

#calculate energy of condensate (conserved quantitiy)


def get_energy(psi, g, omega):
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
    
    return ekin, eint, erot


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
    len_l = np.size(coeffs, axis = 1)  #size of sh_coeffs array in l and m indices(degree and order)
    for l in range(len_l):
        for m in range(len_l):
            for i in range(2):
                coeffs[i,l,m] *= np.exp(- 1.0j * 0.5 * l * (l + 1) * dt ) * np.exp(- 1.0j * m * omega * dt * (-1.)**i)  #timestep of kinetic and rotating term
    psi = pysh.expand.MakeGridDHC(coeffs, norm = 4, sampling = 2, extend = False) #create gridded data in (N, 2*N) array from coeffs
    psi *= np.exp(-1.0j * g * dt * np.abs(psi)**2) #timestep of nonlinear term
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #calculate expansion coefficients from the gridded data
    return coeffs

#the same timestep as above, except the input and output is the gridded data, i.e. the wavefunction

def timestep_grid(psi, dt, g, omega):
    psi = psi * np.exp(-1.0j * g * 0.5 * dt * np.abs(psi)**2)
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2)
    
    def step(i, l, m): #this function will be mutiplied entry wise with coeffs with entry indices i, l, m
        return np.exp(- 1.0j * 0.5 * l * (l + 1) * params.dt ) * np.exp(- 1.0j * m * params.omega * params.dt * (-1.)**i)
    
    step_multiplier = np.fromfunction(step, shape = np.shape(coeffs), dtype = np.complex128) #create array of same shape as coeffs with entries from step
    coeffs = coeffs * step_multiplier
    
    psi = pysh.expand.MakeGridDHC(coeffs, norm = 4, sampling = 2, extend = False) #create gridded data in (N, 2*N) array from coeffs
    psi = psi * np.exp(-1.0j * g * 0.5 * dt * np.abs(psi)**2)
    return psi

#the same timestep, but for imaginary time

def imaginary_timestep_grid(psi, dt, g, omega, particle_number):
    #phase = np.angle(psi)
    psi = psi * np.exp(- 0.5 * g * dt * np.abs(psi)**2) #half timestep of nonlinear term
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2)
    
    def step(i, l, m):#this function will be mutiplied entry wise with coeffs with entry indices i, l, m
        return np.exp(- 0.5 * l * (l + 1) * dt ) * np.exp(- m * omega * dt * (-1.)**i)
    
    step_multiplier = np.fromfunction(step, shape = np.shape(coeffs), dtype = np.float64)
    coeffs = coeffs * step_multiplier
    
    psi = pysh.expand.MakeGridDHC(coeffs, norm = 4, sampling = 2, extend = False) #create gridded data in (N, 2*N) array from coeffs
    psi = psi * np.exp(- 0.5 * g * dt * np.abs(psi)**2) #half timestep of nonlinear term
    norm = get_norm(psi)
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


#vortex tracker

def vortex_tracker(psi, theta_guess, phi_guess, counter = 0):
    #expand sh coefficients of wave function, its real part and imaginary part
    coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) 
    coeffs_real = pysh.expand.SHExpandDH(np.real(psi), norm = 4, sampling = 2) 
    coeffs_imag = pysh.expand.SHExpandDH(np.imag(psi), norm = 4, sampling = 2) 
    
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
    
    if (np.abs(psi_guess)**2 > 0.05 * np.max(np.abs(psi)**2)): #if density at guessed position is larger than 5% of maximum, possibly cannot guarantee convergence, so the tracking must be aborted
        print('Guessed position too far away from vortex core. Try again!')
        print('It took ' + str(counter) + ' iterations to arrive here')
        return 0, 0
    
    if (np.abs(psi_guess)**2 < 1e-7 * np.max(np.abs(psi)**2)): #if the density at the guessed position is smaller than 1e-10 assume a reasonably converged solution and return the guess
        print('Number of iterations to convergence: ' + str(counter))
        return theta_guess, phi_guess
    
    inverse_jacobian = np.linalg.inv(Jacobian)
    
    #new coordinates
    theta_new = theta_guess - inverse_jacobian[0, 0] * np.real(psi_guess) - inverse_jacobian[0, 1] * np.imag(psi_guess)
    phi_new = phi_guess - inverse_jacobian[1, 0] * np.real(psi_guess) - inverse_jacobian[1, 1] * np.imag(psi_guess)
    
    
    return vortex_tracker(psi, theta_new, phi_new, counter + 1) #recur the function with the new coordinates as the new guesses

#custom cyclic colormap to visualize the phase of the condensate

def phasemap(N = 256, use_hpl = True):
    h = np.ones(N) # hue
    h[:N//2] = 11.6 # red 
    h[N//2:] = 258.6 # blue
    s = 100 # saturation
    l = np.linspace(0, 100, N//2) # luminosity
    l = np.hstack( (l,l[::-1] ) )

    colorlist = np.zeros((N,3))
    for ii in range(N):
        if use_hpl:
            colorlist[ii,:] = hsluv.hpluv_to_rgb((h[ii], s, l[ii]))
        else:
            colorlist[ii,:] = hsluv.hsluv_to_rgb((h[ii], s, l[ii]))
    colorlist[colorlist > 1] = 1 # correct numeric errors
    colorlist[colorlist < 0] = 0 
    return col.ListedColormap(colorlist)


    
    
    