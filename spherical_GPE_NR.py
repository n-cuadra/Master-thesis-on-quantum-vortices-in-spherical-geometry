import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
import spherical_GPE_functions as sgpe
import spherical_GPE_params as params
import cmocean
import scienceplots

#set some parameters for plotting

plt.style.use('science')
plt.rcParams.update({'font.size': 7})
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')


#%%
#psi = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/psi_-0.708.txt', delimiter= ',', dtype = np.complex128)

psi = sgpe.IC_vortex_dipole(params.theta_plus, params.phi_plus, params.theta_minus, params.phi_minus, params.xi, params.bg_dens)
particle_number = sgpe.get_norm(psi)

for _ in range(300):
    psi = sgpe.imaginary_timestep_grid(psi, params.dt, params.g, 0.0, particle_number, keep_phase = False)
    
    

omega = np.arange(0.708, 0.808, 0.01)

angmom = np.zeros(len(omega), dtype = np.float64)
energykin = np.zeros(len(omega), dtype = np.float64)
energyint = np.zeros(len(omega), dtype = np.float64)
energyrot = np.zeros(len(omega), dtype = np.float64)

for i in range(len(omega)):
    psi = sgpe.NR2D(psi, mu = params.mu, g = params.g, omega = -omega[i], epsilon = 3e-4)
    #np.savetxt('J:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/psi_' + str(-omega[i]) + '.txt', psi, delimiter = ',', header = 'NR solution for omega = ' + str(-omega[i]))
    ekin, eint, erot = sgpe.get_energy(psi, params.g, -omega[i])
    angmom[i] = sgpe.get_ang_momentum(psi)
    energykin[i] = ekin 
    energyint[i] = eint 
    energyrot[i] = erot
    #wf_path = 'J:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/wf_' + str(-omega[i]) + '.pdf'
    #spectrum_path = 'J:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/spectrum_' + str(-omega[i]) + '.pdf'
    wf_title = r'Wave function for $\tilde\omega =$ ' + str(-omega[i])
    spectrum_title = r'Spectrum for $\tilde\omega =$ ' + str(-omega[i])
    sgpe.plot(psi, spectrum_title = spectrum_title, wf_title = wf_title)


#np.savetxt('J:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/ekin.txt', energykin, delimiter = ',')
#np.savetxt('J:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/eint.txt', energyint, delimiter = ',')
#np.savetxt('J:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/erot.txt', energyrot, delimiter = ',')
#np.savetxt('J:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/angular_momentum.txt', angmom, delimiter = ',')


#%%

data = np.loadtxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Measuring the angular velocities of vortices/omegacomplete.txt', delimiter = ',')

omega = -data[0, 20:-2]
thetaplus = data[1, 20:-2]

energyrot = np.zeros(len(omega), dtype = np.float64)
energyint = np.zeros(len(omega), dtype = np.float64)
energykin = np.zeros(len(omega), dtype = np.float64)
angmom = np.zeros(len(omega), dtype = np.float64)


for i in range(len(omega)):
    psi = np.sqrt(params.bg_dens) * sgpe.initial_magnitude(params.theta_grid, params.phi_grid, thetaplus[i], np.pi, np.pi - thetaplus[i], np.pi, params.xi) * np.exp(1.0j * sgpe.phase(params.theta_grid, params.phi_grid, thetaplus[i], np.pi, np.pi - thetaplus[i], np.pi))
    particle_number = sgpe.get_norm(psi)


    for _ in range(300):
        psi = sgpe.imaginary_timestep_grid(psi, params.dt, params.g, 0.0, particle_number, keep_phase = False)
        
    psi = sgpe.NR2D(psi, params.mu, params.g, omega[i], epsilon = 5e-4)
    #sgpe.plot(psi)
    
    ekin, eint, erot = sgpe.get_energy(psi, params.g, omega[i])
    angmom[i] = sgpe.get_ang_momentum(psi)
    energykin[i] = ekin
    energyint[i] = eint
    energyrot[i] = erot
    
np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/ekin.txt', energykin, delimiter = ',')
np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/eint.txt', energyint, delimiter = ',')
np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/erot.txt', energyrot, delimiter = ',')
np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/angular_momentum.txt', angmom, delimiter = ',')


    