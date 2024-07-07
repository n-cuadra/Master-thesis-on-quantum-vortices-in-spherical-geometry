import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
import spherical_GPE_functions as sgpe

from matplotlib import cm


#constants
bohr_radius = 5.2918e-5 #bohr radius in micrometers
hbar = 1.055e-22 #hbar in units of kg * µm^2 / s


#simulation parameters (adjustable)
N = 512 #grid points
R = 50.0 #radius of sphere in µm
mass = 1.443e-25 #mass of the atoms in kg
scattering_length = 100.0 #scattering length in units of the Bohr radius
omega_units = 10.0 #rotating frequency in Hz
theta_plus = np.pi/5. #position of vortex on the upper hemisphere
dt = 1.0e-5  #time step
epsilon = 1.0e-4 #strength of bogoliubov excitations
bg_dens = 1000. #condensate density

#these are calculated from the values above, don't change these!


lmax = N//2 - 1 #maximum degree of spherical harmonics
g = - np.pi / np.log(np.sqrt(lmax * (lmax + 1)) * scattering_length * bohr_radius * np.exp(np.euler_gamma) / (2 * R) ) #unitless interaction strength
real_dt = 1000 * dt *  R**2 * mass / hbar #one timestep in real time in ms
omega = omega_units * R**2 * mass / hbar #unitless rotating frequency



#coordinate system

theta, phi = np.linspace(0,  np.pi, N, endpoint = False), np.linspace(0, 2 * np.pi, 2 * N, endpoint = False)  #rectangular grid with polar angle theta and azimuthal angle phi
dangle = np.pi / N  #grid spacing

#define the analytic expression for the bogoliubov dispersion (only the positive branch)

def bogoliubov_dispersion(l, m, omega, g, bg_dens, positive_sign):
    if positive_sign:
        return m * omega + np.sqrt(0.5 * l * (l + 1) * ( 0.5 * l * (l + 1)  + 2 * g * bg_dens ))
    else:
        return m * omega - np.sqrt(0.5 * l * (l + 1) * ( 0.5 * l * (l + 1)  + 2 * g * bg_dens ))


#initialize spherical harmonic coefficients
clm = pysh.SHCoeffs.from_zeros(lmax = lmax, normalization = 'ortho', kind = 'complex') #initialize SHCoeffs instance with only zeros

clm.set_coeffs(ls = 0, ms = 0, values = 2.0 * np.sqrt(np.pi * bg_dens)) #set the l=0 m=0 coefficient to 2 sqrt(pi) sqrt(condensate density)

for l in range(1, lmax + 1):
    for m in range(l + 1):
        clm.set_coeffs(ls = l, ms = m, values = epsilon) # set all other coefficients to epsilon
        clm.set_coeffs(ls = l, ms = -m, values = epsilon)

clm.plot_spectrum(unit = 'per_l', show = False)

coeffs = clm.to_array(normalization = 'ortho', lmax = lmax) #create array of the coefficients

print(coeffs[0,0,0], coeffs[0,1,1], coeffs[0,200,50], coeffs[1,100,10])

coeffs_t = np.zeros((lmax, 2, lmax + 1, lmax + 1), dtype = np.complex128) #initialize array to store the time evolution of the sh coefficients


frequencies = 2 * np.pi * np.fft.fftfreq(lmax, d = 10 * dt)
frequencies_ord = np.fft.fftshift(frequencies)

steps = 10 * lmax

#run simulation

for q in range(steps):
    if (q % 10 == 0): #every ten steps append the current array of all sh coefficients
        index = q // 10
        coeffs_t[index, :, :, :] = coeffs
    coeffs = sgpe.timestep_coeffs(coeffs, dt)

#%%

plt.rc('font', family='sans-serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rcParams['mathtext.fontset'] = 'cm'


#coeffs_t = np.fft.ifftshift(coeffs_t, axes = 0)
coeffs_omega = np.fft.fft(coeffs_t, axis = 0)
coeffs_omega = np.fft.fftshift(coeffs_omega, axes = 0)


data = np.log(np.abs(coeffs_omega[:,0,:,0]))

mappable = plt.imshow(data, cmap = cm.inferno, vmin = np.min(data), vmax = np.max(data), extent=[0, lmax, frequencies_ord.min(), frequencies_ord.max()], aspect = 'auto')

plt.colorbar(mappable, label = r'$\log \ |\psi^{l}_0(\omega)|$')

plt.xlabel(r'$l$')
plt.ylabel(r'$\omega_{l,0}$')


plt.savefig('J:/Uni - Physik/Master/Masterarbeit/Media/spherical_bogoliubov_dispersion_1.pdf', dpi = 300)
plt.show()


#%%

l = np.linspace(0, lmax, lmax)
omega_analytic_l_p = bogoliubov_dispersion(l, 0, omega, g, bg_dens, True)
omega_analytic_l_m = bogoliubov_dispersion(l, 0, omega, g, bg_dens, False)



data = np.log(np.abs(coeffs_omega[:,0,:,0]))


plt.plot(l, omega_analytic_l_p, 'w--', lw = 0.5, label = 'analytic dispersion relation')
plt.plot(l, omega_analytic_l_m, 'w--', lw = 0.5)

mappable = plt.imshow(data, cmap = cm.inferno, vmin = np.min(data), vmax = np.max(data), extent=[0, lmax, frequencies_ord.min(), frequencies_ord.max()], aspect = 'auto')

plt.colorbar(mappable, label = r'$\log \ |\psi^{l}_0(\omega)|$')

plt.xlabel(r'$l$')
plt.ylabel(r'$\omega_{l,0}$')

plt.legend(fontsize = 5)
plt.savefig('J:/Uni - Physik/Master/Masterarbeit/Media/spherical_bogoliubov_dispersion_2.pdf', dpi = 300)
plt.show()


#%%
ltest = 200
m = np.linspace(0, lmax, lmax)
omega_analytic_m_p = bogoliubov_dispersion(ltest, m, omega, g, bg_dens, True)
omega_analytic_m_m = bogoliubov_dispersion(ltest, m, omega, g, bg_dens, False)

data = np.log(np.abs(coeffs_omega[:,0,ltest,:ltest]))

plt.plot(m, omega_analytic_m_p, 'w--', lw = 1, label = 'analytic dispersion relation')
plt.plot(m, omega_analytic_m_m, 'w--', lw = 1)

mappable = plt.imshow(data, cmap = cm.inferno, vmin = np.min(data), vmax = np.max(data), extent=[0, ltest, frequencies_ord.min(), frequencies_ord.max()], aspect = 'auto')

plt.colorbar(mappable, label = r'$\log \ |\psi^{200}_m(\omega)|$')

plt.xlabel(r'$m$')
plt.ylabel(r'$\omega_{200,m}$')

plt.legend(fontsize = 5)
plt.savefig('J:/Uni - Physik/Master/Masterarbeit/Media/spherical_bogoliubov_dispersion_5.pdf', dpi = 300)
plt.show()


#%%

clmt = pysh.SHCoeffs.from_array(coeffs_t[-1,:,:,:], normalization='ortho', lmax = lmax)
clmt.plot_spectrum(unit = 'per_l', show = False)
