import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh

from matplotlib import cm


#constants
bohr_radius = 5.2918e-5 #bohr radius in micrometers
hbar = 1.055e-22 #hbar in units of kg * µm^2 / s


#simulation parameters (adjustable)
N = 512 #grid points
R = 50.0 #radius of sphere in µm
mass = 1.443e-25 #mass of the atoms in kg
scattering_length = 100.0 #scattering length in units of the Bohr radius
omega_units = 5.0 #rotating frequency in Hz
theta_plus = np.pi/5. #position of vortex on the upper hemisphere
dt = 1.0e-5  #time step
epsilon = 1.0e-4 #strength of bogoliubov excitations

#these are calculated from the values above, don't change these!


lmax = N//2 - 1 #maximum degree of spherical harmonics
g = - np.pi / np.log(np.sqrt(lmax * (lmax + 1)) * scattering_length * bohr_radius * np.exp(np.euler_gamma) / (2 * R) ) #unitless interaction strength
real_dt = 1000 * dt *  R**2 * mass / hbar #one timestep in real time in ms
omega = omega_units * R**2 * mass / hbar #unitless rotating frequency



#coordinate system

theta, phi = np.linspace(0,  np.pi, N, endpoint = False), np.linspace(0, 2 * np.pi, 2 * N, endpoint = False)  #rectangular grid with polar angle theta and azimuthal angle phi
dangle = np.pi / N  #grid spacing

#define the analytic expression for the bogoliubov dispersion (only the positive branch)

def bogoliubov_dispersion(l, m, omega, g):
    disp = m * omega + np.sqrt(0.5 * l * (l + 1) * ( 0.5 * l * (l + 1)  + 2 * g  ))
    return disp


#one timestep with split stepping method where the sh coefficients are the input and output

def timestep_coeffs(coeffs, dt):
    len_l = np.size(coeffs, axis = 1)  #size of sh_coeffs array in l and m indices(degree and order)
    for l in range(len_l):
        for m in range(len_l):
            for i in range(2):
                coeffs[i,l,m] *= np.exp(- 1.0j * 0.5 * l * (l + 1) * dt ) * np.exp(- 1.0j * m * omega * dt * (-1.)**i)   #timestep of kinetic and rotating term
    psi = pysh.expand.MakeGridDHC(coeffs, norm = 4, sampling = 2, extend = False) #create gridded data in (N, 2*N) array from coeffs
    psi *= np.exp(-1.0j * g * dt * np.abs(psi)**2) #timestep of nonlinear term
    coeffs = pysh.expand.SHExpandDHC(griddh = psi, norm = 4, sampling = 2) #calculate expansion coefficients from the gridded data
    return coeffs

#initialize spherical harmonic coefficients
clm = pysh.SHCoeffs.from_zeros(lmax = lmax, normalization = 'ortho', kind = 'complex') #initialize SHCoeffs instance with only zeros

clm.set_coeffs(ls = 0, ms = 0, values = 1.) #set the l=0 m=0 coefficient to 1
clm.set_coeffs(ls = 1, ms = (1, 0, -1), values = epsilon) # set all three l = 1 coefficients to epsilon


coeffs = clm.to_array(normalization = 'ortho', lmax = lmax) #create array of the coefficients

coeffs_t = np.zeros((lmax, 2, lmax + 1, lmax + 1), dtype = np.complex128) #initialize array to store the time evolution of the sh coefficients


frequencies = 2 * np.pi * np.fft.fftfreq(lmax, d = 10 * dt)
frequencies_ord = np.fft.fftshift(frequencies)

steps = 10 * lmax

#run simulation

for q in range(steps):
    if (q % 10 == 0): #every ten steps append the current array of all sh coefficients
        index = q // 10
        coeffs_t[index, :, :, :] = coeffs
    coeffs = timestep_coeffs(coeffs, dt)

#%%

plt.rc('font', family='sans-serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rcParams['mathtext.fontset'] = 'cm'

coeffs_omega = np.fft.fft(coeffs_t, axis = 0)
coeffs_omega = np.fft.fftshift(coeffs_omega, axes = 0)

data = np.log(np.abs(coeffs_omega[:,0,:,0]))

mappable = plt.imshow(data, cmap = cm.inferno, vmin = np.min(data), vmax = np.max(data), extent=[0, lmax, frequencies_ord.min(), frequencies_ord.max()], aspect = 'auto')

plt.colorbar(mappable, label = r'$\log \ |\psi^{l}_0(\omega)|$')

plt.xlabel(r'$m$')
plt.ylabel(r'$\omega_{l,0}$')


plt.savefig('J:/Uni - Physik/Master/6. Semester/Masterarbeit/Media/spherical_bogoliubov_dispersion_1.pdf', dpi = 300)
plt.show()


#%%

l = np.linspace(0, lmax, lmax)
omega_analytic_l = bogoliubov_dispersion(l, 0, omega, g)


data = np.log(np.abs(coeffs_omega[:,0,:,0]))


plt.plot(l, omega_analytic_l, 'w--', lw = 0.5, label = 'analytic dispersion relation')

mappable = plt.imshow(data, cmap = cm.inferno, vmin = np.min(data), vmax = np.max(data), extent=[0, lmax, frequencies_ord.min(), frequencies_ord.max()], aspect = 'auto')

plt.colorbar(mappable, label = r'$\log \ |\psi^{l}_0(\omega)|$')

plt.xlabel(r'$l$')
plt.ylabel(r'$\omega_{l,0}$')

plt.legend(fontsize = 5)
plt.savefig('J:/Uni - Physik/Master/6. Semester/Masterarbeit/Media/spherical_bogoliubov_dispersion_2.pdf', dpi = 300)
plt.show()


#%%

m = np.linspace(0, lmax, lmax)
omega_analytic_m = bogoliubov_dispersion(lmax + 1, m, omega, g)

data = np.log(np.abs(coeffs_omega[:,0,-1,:]))

#plt.plot(m, omega_analytic_m, lw = 1, label = 'analytic dispersion relation')

mappable = plt.imshow(data, cmap = cm.inferno, vmin = np.min(data), vmax = np.max(data), extent=[0, lmax, frequencies_ord.min(), frequencies_ord.max()], aspect = 'auto')

plt.colorbar(mappable, label = r'$\log \ |\psi^{255}_m(\omega)|$')

plt.xlabel(r'$m$')
plt.ylabel(r'$\omega_{255,m}$')

#plt.legend(fontsize = 5)
plt.savefig('J:/Uni - Physik/Master/6. Semester/Masterarbeit/Media/spherical_bogoliubov_dispersion_3.pdf', dpi = 300)
plt.show()

