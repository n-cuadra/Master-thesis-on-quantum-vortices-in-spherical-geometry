import numpy as np
import pyshtools as pysh
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
    num1 = cot(theta/2) * np.sin(phi) - cot(theta_plus/2) * np.sin(phi_plus)
    denom1 = cot(theta/2) * np.cos(phi) - cot(theta_plus/2) * np.cos(phi_plus)
    num2 = cot(theta/2) * np.sin(phi) - cot(theta_minus/2) * np.sin(phi_minus)
    denom2 = cot(theta/2) * np.cos(phi) - cot(theta_minus/2) * np.cos(phi_minus)
    
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
    
    


#define norm function, this calculates particle number of the condensate

def get_norm(psi):
    norm = 0
    for i in range(params.N):
        for j in range(2*params.N):
            norm += np.sin(params.theta[i]) * params.dangle**2 * np.abs(psi[i,j])**2
    return norm


#define derivative wrt azimuthal coordinate

def deriv_phi(psi):
    sh_coeffs = pysh.expand.SHExpandDHC(griddh = psi, norm = 4, sampling = 2) #expand sh coefficients of psi
    len_m = np.size(sh_coeffs, axis = 2) 
    for m in range(len_m):
        for i in range(2):
            sh_coeffs[i,:,m] *= 1.0j * m * (-1.)**i  #spherical harmonics are eigenfunctions of del_phi, so need only to multiply coefficients with i*m to perform derivative
    psi = pysh.expand.MakeGridDHC(sh_coeffs, norm = 4, sampling = 2, extend = False) #back to real space, with the gridded data of the modified coefficients 
    return psi

#define angular Laplacian

def Laplacian(psi):
    sh_coeffs = pysh.expand.SHExpandDHC(griddh = psi, norm = 4, sampling = 2) #expand sh coefficients of psi
    len_l = np.size(sh_coeffs, axis = 1) 
    for l in range(len_l):
        sh_coeffs[:,l,:] *= - l * (l + 1) #spherical harmonics are eigenfunctions of angular Laplacian, so need only to multiply coefficients with -l(l+1) to perform Laplacian
    psi = pysh.expand.MakeGridDHC(sh_coeffs, norm = 4, sampling = 2, extend = False) #back to real space, with the gridded data of the modified coefficients 
    return psi

#calculate energy of condensate (conserved quantitiy)

def get_energy(psi, g, omega):
    energy = 0
    Laplace_psi = Laplacian(psi) #array of Laplacian of wavefunction
    deriv_phi_psi = deriv_phi(psi) #array of derivative wrt azimuthal angle of wavefunction
    conj_psi = np.conjugate(psi) #array of complex conjugate of wavefunction
    for i in range(params.N):
        for j in range(2*params.N):
            energy += params.dangle**2 * np.sin(params.theta[i]) * (  - 0.5 * conj_psi[i,j] * Laplace_psi[i,j] + g * np.abs(psi[i,j])**4 - 1.0j * omega * conj_psi[i,j] * deriv_phi_psi[i,j]  ) #compute the hamiltonian
    energy = np.real(energy)
    return energy
    
#calculate angular momentum of condensate in z direction (another conserved quantity).

def get_ang_momentum(psi):
    mom = 0
    deriv_phi_psi = deriv_phi(psi) #array of derivative wrt azimuthal angle of wavefunction
    conj_psi = np.conjugate(psi) #array of complex conjugate of wavefunction
    for i in range(params.N):
        for j in range(2*params.N):
            mom += - 1.0j * params.dangle**2 * np.sin(params.theta[i]) * conj_psi[i,j] * deriv_phi_psi[i,j] #compute the angular momentum integral
    mom = np.real(mom)
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
    coeffs = pysh.expand.SHExpandDHC(griddh = psi, norm = 4, sampling = 2) #calculate expansion coefficients from the gridded data
    return coeffs

#the same timestep as above, except the input and output is the gridded data, i.e. the wavefunction

def timestep_grid(psi, dt, g, omega):
    coeffs = pysh.expand.SHExpandDHC(griddh = psi, norm = 4, sampling = 2)
    len_l = np.size(coeffs, axis = 1)  #size of sh_coeffs array in l and m indices(degree and order)
    for l in range(len_l):
        for m in range(len_l):
            for i in range(2):
                coeffs[i,l,m] *= np.exp(- 1.0j * 0.5 * l * (l + 1) * dt ) * np.exp(- 1.0j * m * omega * dt * (-1.)**i)  #timestep of kinetic and rotating term
    psi = pysh.expand.MakeGridDHC(coeffs, norm = 4, sampling = 2, extend = False) #create gridded data in (N, 2*N) array from coeffs
    psi *= np.exp(-1.0j * g * dt * np.abs(psi)**2) #timestep of nonlinear term
    return psi

#vortex tracker

def vortex_tracker(psi, theta_guess, phi_guess):
    
    Jacobian = np.zeros((2,2), dtype = np.float64) # initialize jacobian
    
    coeffs = pysh.expand.SHExpandDHC(griddh = psi, norm = 4, sampling = 2) #expand sh coefficients from wave function psi
    
    psi_guess = 0 #calculate the wave function psi at position (theta_guess, phi_guess)
    for l in range(params.lmax + 1):
        for m in range(l + 1):
            psi_guess += coeffs[0, l, m] * pysh.expand.spharm_lm(l = l, m = m, theta = theta_guess, phi = phi_guess, normalization = 'ortho', kind = 'complex', degrees = False)
            psi_guess += coeffs[1, l, m] * pysh.expand.spharm_lm(l = l, m = -m, theta = theta_guess, phi = phi_guess, normalization = 'ortho', kind = 'complex', degrees = False)
    
    print('density = ', np.abs(psi_guess)**2)                
    
    if (np.abs(psi_guess)**2 > 0.01 * np.max(np.abs(psi)**2)): #if density at guessed position is larger than 1% of maximum, possibly cannot guarantee convergence, so the tracking must be aborted
        print('Guessed position too far away from vortex core. Try again!')
        return 0, 0
    
    machine_epsilon = np.finfo(np.float64).eps #machine epsilon of numpy float 64
    if (np.abs(psi_guess)**2 < machine_epsilon * np.max(np.abs(psi)**2)): #if the density at the guessed position is smaller than the machine precision fraction of the maximum density assume a reasonably converged solution and return the guess
        return theta_guess, phi_guess
    
    psi_phi = 0 #calculate the derivative of the wave function wrt phi at position (theta_guess, phi_guess)
    for l in range(params.lmax + 1):
        for m in range(l + 1):
            psi_phi += 1.0j * m * coeffs[0, l, m] * pysh.expand.spharm_lm(l = l, m = m, theta = theta_guess, phi = phi_guess, normalization = 'ortho', kind = 'complex', degrees = False)   
            psi_phi += - 1.0j * m * coeffs[1, l, m] * pysh.expand.spharm_lm(l = l, m = -m, theta = theta_guess, phi = phi_guess, normalization = 'ortho', kind = 'complex', degrees = False)
            
    
    print('psi_phi=', psi_phi)
    Jacobian[0,1] = np.real(psi_phi)/np.sin(theta_guess)
    Jacobian[1,1] = np.imag(psi_phi)/np.sin(theta_guess)
    
    psi_theta = 0 #calculate the derivative of the wave function wrt theta at position (theta_guess, phi_guess)
    for l in range(params.lmax + 1):
        for m in range(l + 1):
            if (m == l):#for m = l, the spherical harmonics in the second term of the derivative don't exist (which is not a problem in the equation, because the prefactor is 0, but here we need to take care)
                psi_theta += coeffs[0, l, l] * l * cot(theta_guess) * pysh.expand.spharm_lm(l = l, m = l, theta = theta_guess, phi = phi_guess, normalization = 'ortho', kind = 'complex', degrees = False)
                psi_theta += coeffs[1, l, l] * l * cot(theta_guess) * pysh.expand.spharm_lm(l = l, m = -l, theta = theta_guess, phi = phi_guess, normalization = 'ortho', kind = 'complex', degrees = False)
            else:
                psi_theta += coeffs[0, l, m] * (m * cot(theta_guess) * pysh.expand.spharm_lm(l = l, m = m, theta = theta_guess, phi = phi_guess, normalization = 'ortho', kind = 'complex', degrees = False) + np.sqrt((l - m) * (l + m + 1)) * np.exp(-1.0j * phi_guess) * pysh.expand.spharm_lm(l = l, m = m + 1, theta = theta_guess, phi = phi_guess, normalization = 'ortho', kind = 'complex', degrees = False))
                psi_theta += coeffs[1, l, m] * (m * cot(theta_guess) * pysh.expand.spharm_lm(l = l, m = -m, theta = theta_guess, phi = phi_guess, normalization = 'ortho', kind = 'complex', degrees = False) - np.sqrt((l - m) * (l + m + 1)) * np.exp(1.0j * phi_guess) * pysh.expand.spharm_lm(l = l, m = - m - 1, theta = theta_guess, phi = phi_guess, normalization = 'ortho', kind = 'complex', degrees = False))
    
    
    print('psi_theta=', psi_theta)
    Jacobian[0,0] = np.real(psi_theta)
    Jacobian[1,0] = np.imag(psi_theta)
    inverse_jacobian = np.linalg.inv(Jacobian)
    
    theta_new = theta_guess - inverse_jacobian[0,0] * np.real(psi_guess) - inverse_jacobian[0,1] * np.imag(psi_guess)
    phi_new = phi_guess - inverse_jacobian[1,0] * np.real(psi_guess) - inverse_jacobian[1,1] * np.imag(psi_guess)
    print(theta_new, phi_new)
    
    return vortex_tracker(psi, theta_new, phi_new) #recur the function with the new coordinates as the new guesses