import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
import spherical_GPE_functions as sgpe
import spherical_GPE_params as params

omega = np.linspace(0.1, 1.0, 10)
theta_plus = np.linspace(0.1, 7*np.pi/18, 100)
theta_minus = np.pi - theta_plus

energy = np.zeros(shape = (100, 10), dtype = np.float64)

#initialize wavefunction

psi = np.zeros(shape = (params.N, 2*params.N), dtype = np.complex128)

for k in range(100):
    for i in range(params.N//2):
        for j in range(2*params.N):
            psi[i,j] = np.sqrt(params.bg_dens) * sgpe.one_vortex_magnitude(params.theta[i], params.phi[j], theta_plus[k], params.phi_plus, params.xi) * np.exp(1.0j * sgpe.phase(params.theta[i], params.phi[j], theta_plus[k], params.phi_plus, theta_minus[k], params.phi_minus))
    for i in range(params.N//2):
        for j in range(2*params.N):
            psi[i,j] = np.sqrt(params.bg_dens) * sgpe.one_vortex_magnitude(params.theta[i], params.phi[j], theta_minus[k], params.phi_minus, params.xi) * np.exp(1.0j * sgpe.phase(params.theta[i], params.phi[j], theta_plus[k], params.phi_plus, theta_minus[k], params.phi_minus))   
    for q in range(10):
        energy[k, q] = sgpe.get_energy(psi, params.g, omega[q])
                

#%%

plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'cm'


#plt.plot(theta_plus, energy[:,0], label = r'$\~\omega = $ 0.1', linewidth = 0.8)

#plt.plot(theta_plus, energy[:,1], label = r'$\~\omega = $ 0.2', linewidth = 0.8)

#plt.plot(theta_plus, energy[:,2], label = r'$\~\omega = $ 0.3', linewidth = 0.8)

#plt.plot(theta_plus, energy[:,3], label = r'$\~\omega = $ 0.4', linewidth = 0.8)
#plt.plot(theta_plus, energy[:,4], label = r'$\~\omega = $ 0.5', linewidth = 0.8)
plt.plot(theta_plus, energy[:,5], label = r'$\~\omega = 0.6$', linewidth = 0.8)
plt.plot(theta_plus, energy[:,6], label = r'$\~\omega = 0.7$', linewidth = 0.8)
plt.plot(theta_plus, energy[:,7], label = r'$\~\omega =  0.8$', linewidth = 0.8)
plt.plot(theta_plus, energy[:,8], label = r'$\~\omega =  0.9$', linewidth = 0.8)
#plt.plot(theta_plus, energy[:,9], label = r'$\~\omega =  1.0', linewidth = 0.8)

plt.xlabel(r'$\theta_+$')
plt.ylabel(r'$E$ $\left[ \frac{\hbar^2}{m R^2}  \right]$')
plt.legend(fontsize = 'x-small')
plt.savefig('J:/Uni - Physik/Master/Masterarbeit/Media/energy as a function of omega.pdf', format = 'pdf', dpi = 300)