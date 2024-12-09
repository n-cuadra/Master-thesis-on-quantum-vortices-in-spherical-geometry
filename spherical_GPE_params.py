import numpy as np

#constants
bohr_radius = 5.2918e-5 #bohr radius in micrometers
hbar = 1.055e-22 #hbar in units of kg * µm^2 / s


#simulation parameters (adjustable)
N = 512 #grid points
end = 100000 #number of steps in simulation
mass = 1.443e-25 #mass of the atoms in kg
R = 50.0 #radius of sphere in µm
scattering_length = 1e-4 #scattering length in units of the radius
omega_units = 20 #rotating frequency in Hz
theta_plus, phi_plus = np.deg2rad(45), np.pi   #position of vortex 
theta_minus, phi_minus = np.pi - theta_plus, np.pi #position of antivortex
dt = 2e-5  #time step
bg_dens = 1000 #condensate density far from vortices
G = 1



#these are calculated from the values above, don't change these!

lmax = N//2 - 1 #maximum degree of spherical harmonics
g = - 2 * np.pi / np.log(0.5 * np.sqrt(lmax * (lmax + 1)) * scattering_length * np.exp(np.euler_gamma)) #unitless interaction strength
real_dt = 1000 * dt *  R**2 * mass / hbar #one timestep in real time in ms
omega = omega_units * R**2 * mass / hbar #unitless rotating frequency
xi = 1/np.sqrt(2 * g * bg_dens) #healing length 
mu = g * bg_dens + 1

#coordinate system

theta, phi = np.linspace(0,  np.pi, N, endpoint = False), np.linspace(0, 2 * np.pi, 2 * N, endpoint = False)  #rectangular grid with polar angle theta and azimuthal angle phi
dangle = theta[1] - theta[0]  #grid spacing
THETA, PHI = np.meshgrid(theta, phi, indexing = 'ij') #the theta and phi values expanded over the whole grid as a N x 2N array
    

