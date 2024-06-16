import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh

from matplotlib import cm


#parameters
N = 512 #grid points
R = 1.0 #radius of sphere
theta_plus = np.pi/5 #position of vortex on the upper hemisphere
dt = 1.e-4  #time step
omega = 2 * np.pi * 5 #frequency in Hz
real_dt = dt * 1000 / omega #length of one timestep in real time in ms
alpha = 5. / omega # linear coefficient
g = 1. / omega   #nonlinear coefficient
bg_dens = 1. #condensate density far from vortices
lmax = N//2 - 1 #maximum degree of spherical harmonics
epsilon = 1e-4 #strength of bogoliubov excitations
sigma = 1./24. #width of gaussian


#coordinate system

theta, phi = np.linspace(0,  np.pi, N, endpoint = False), np.linspace(0, 2 * np.pi, 2 * N, endpoint = False)  #rectangular grid with polar angle theta and azimuthal angle phi
dangle = np.pi / N  #grid spacing


#one timestep with split stepping method where the sh coefficients are the input and output

def timestep_coeffs(coeffs, dt):
    len_l = np.size(coeffs, axis = 1)  #size of sh_coeffs array in l and m indices(degree and order)
    for l in range(len_l):
        for m in range(len_l):
            for i in range(2):
                coeffs[i,l,m] *= np.exp(- 1.0j * alpha * l * (l + 1) * dt / 2) * np.exp(- 1.0j * m * dt * (-1.)**i)  #timestep of kinetic and rotating term
    psi = pysh.expand.MakeGridDHC(coeffs, norm = 4, sampling = 2, extend = False) #create gridded data in (N, 2*N) array from coeffs
    psi *= np.exp(1.0j * g * dt * np.abs(psi)**2) #timestep of nonlinear term
    coeffs = pysh.expand.SHExpandDHC(griddh = psi, norm = 4, sampling = 2) #calculate expansion coefficients from the gridded data
    return coeffs

#initialize spherical harmonic coefficients
clm = pysh.SHCoeffs.from_zeros(lmax = lmax, normalization = 'ortho', kind = 'complex') #initialize SHCoeffs instance with only zeros

clm.set_coeffs(ls = 0, ms = 0, values = 1.) #set the l=0 m=0 coefficient to 1
clm.set_coeffs(ls = 1, ms = (1, 0, -1), values = epsilon) # set all three l = 1 coefficients to epsilon


coeffs = clm.to_array(normalization = 'ortho', lmax = lmax) #create array of the coefficients

steps = 10 * lmax

coeffs_t = np.zeros((lmax, 2, lmax + 1, lmax + 1), dtype = np.complex128) #initialize array to store the time evolution of the sh coefficients

#run simulation

for q in range(steps):
    if (q % 10 == 0): #every ten steps append the current array of all sh coefficients
        index = q // 10
        coeffs_t[index, :, :, :] = coeffs
    coeffs = timestep_coeffs(coeffs, dt)

#%%
def bogoliubov_dispersion(l, m, alpha, g):
    disp = m +np.sqrt(alpha * l * (l + 1) / 2 * ( alpha * l * (l + 1) / 2 + 2 * g  ))
    return disp

l = np.linspace(0, lmax, lmax)
omega_analytic_l = bogoliubov_dispersion(l, 0, alpha, g)

m = np.linspace(0, lmax, lmax)
omega_analytic_m = bogoliubov_dispersion(lmax, m, alpha, g)

omega = 2 * np.pi * np.fft.fftfreq(lmax, d = 10 * dt)
omega_ord = np.fft.fftshift(omega)

coeffs_omega = np.fft.fft(coeffs_t, axis = 0)
coeffs_omega = np.fft.fftshift(coeffs_omega, axes = 0)

data = np.log(np.abs(coeffs_omega[:,0, -1,:]) + 1e-50)

#plt.plot(m, omega_analytic_m, lw = 1, label = 'analytic dispersion relation')

mappable = plt.imshow(data, cmap = cm.inferno, vmin = np.min(data), vmax = np.max(data), extent=[0, lmax, omega_ord.min(), omega_ord.max()], aspect = 'auto')

plt.colorbar(mappable, label = r'$\log |\psi^{255}_0(\omega)|$')

plt.xlabel('m')
plt.ylabel(r'$\omega_{255,m}$')


plt.savefig('J:/Uni - Physik/Master/6. Semester/Masterarbeit/Media/spherical_bogoliubov_dispersion_3', dpi = 300)
plt.show()


