import numpy as np

#simulation parameters (adjustable)
N = 512 #grid points
end = 40000 #number of steps in simulation
scattering_length = 1e-4 #scattering length in units of the radius
omega = 2 #rotating frequency in Hz
theta_plus, phi_plus = np.deg2rad(45), np.pi   #position of vortex 
theta_minus, phi_minus = np.pi - theta_plus, np.pi #position of antivortex
dt = 1e-3 #time step
mu = 400 #chemical potential
G = 1

#these are calculated from the values above, don't change these!
lmax = N//2 - 1 #maximum degree of spherical harmonics
g = - 2 * np.pi / np.log(0.5 * np.sqrt(255 * (255 + 1)) * scattering_length * np.exp(np.euler_gamma)) #unitless interaction strength
bg_dens = mu / g #condensate density far from vortices
xi = 1/np.sqrt(2 * g * bg_dens) #healing length

#coordinate system
theta, phi = np.linspace(0,  np.pi, N, endpoint = False), np.linspace(0, 2 * np.pi, 2 * N, endpoint = False)  #rectangular grid with polar angle theta and azimuthal angle phi
dangle = theta[1] - theta[0]  #grid spacing
THETA, PHI = np.meshgrid(theta, phi, indexing = 'ij') #the theta and phi values expanded over the whole grid as a N x 2N array
    
