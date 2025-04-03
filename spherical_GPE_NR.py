import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
import spherical_GPE_functions as sgpe
import spherical_GPE_params as params
import cmocean
import scienceplots


#set some parameters for plotting

plt.style.use('science')
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'xtick.labelsize': 10})
plt.rcParams.update({'ytick.labelsize': 10})




data = np.loadtxt('C:/Users/cuadr/OneDrive/Desktop/Masterarbeit/Measurements/Simulations of two vortices/Measuring the angular velocities of vortices/omegacomplete.txt', delimiter = ',')
thetaplus = data[1, 24]
omega = np.array([np.round(data[0, 24], 3)])


imaginary_timestep_multiplier_array = sgpe.imaginary_timestep_multiplier(dt = params.dt, omega = omega[0])

psi = sgpe.IC_vortex_dipole(thetaplus, np.pi, np.pi - thetaplus, np.pi, params.xi, params.bg_dens)
particle_number = sgpe.get_norm(psi)
sgpe.plot(psi)
#for _ in range(500):
    #psi = sgpe.imaginary_timestep_grid(psi, params.dt, params.g, 0.0, particle_number, keep_phase = True)


#psi = np.loadtxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/mu = 150/psi_150_-5.9.txt', delimiter = ',', dtype = np.complex128)
#psi2 = np.loadtxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/mu = 50/psi_50_-8.0.txt', delimiter = ',', dtype = np.complex128)
#psi = np.sqrt(params.bg_dens) * psi1 * psi2 / np.max(np.abs(psi1)) / np.max(np.abs(psi2))

#sgpe.plot(psi)

#omega = np.arange(8, 9, 1)



angmom = np.zeros(len(omega), dtype = np.float64)
energykin = np.zeros(len(omega), dtype = np.float64)
energyint = np.zeros(len(omega), dtype = np.float64)
e0 = np.zeros(len(omega), dtype = np.float64)

for i in range(len(omega)):
    psi = sgpe.NR2D(psi, mu = params.mu, g = params.g, omega = -omega[i], epsilon = 9e-5)
    np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/psi_' + str(np.round(params.mu, 0)) + '_' + str(-omega[i]) + '.txt', psi, delimiter = ',', header = 'NR solution for omega = ' + str(-omega[i]))
    ekin, eint, erot = sgpe.get_energy(psi, params.g, -omega[i])
    angmom[i] = sgpe.get_ang_momentum(psi)
    energykin[i] = ekin 
    energyint[i] = eint 
    e0[i] = params.g * sgpe.get_norm(psi)**2 / (8 * np.pi)
    wf_path = 'E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/wf_' + str(np.round(params.mu, 0)) + '_' + str(-omega[i]) + '.png'
    spectrum_path = 'E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/spectrum_' + str(np.round(params.mu, 0)) + '_'  + str(-omega[i]) + '.png'
    spectrum2d_path = 'E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/spectrum2d_' + str(np.round(params.mu, 0)) + '_'  + str(-omega[i]) + '.png'
    wf_title = r'Wave function for $\tilde\omega =$ ' + str(-omega[i])
    spectrum_title = r'Spectrum for $\tilde\omega =$ ' + str(-omega[i])
    spectrum2d_title = r'2D Spectrum for $\tilde\omega =$ ' + str(-omega[i])
    sgpe.plot(psi, spectrum_title = spectrum_title, wf_title = wf_title, spectrum2d_title=spectrum2d_title, wf_path=wf_path, spectrum_path=spectrum_path, spectrum2d_path=spectrum2d_path)
omegastart = omega[0]
omegaend = omega[-1]
np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/e0_' + str(np.round(params.mu, 0)) + '_' + str(omegastart) + '-'+ str(omegaend)  + '.txt', e0, delimiter = ',')
np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/ekin_' + str(np.round(params.mu, 0)) + '_' + str(omegastart) + '-'+ str(omegaend)  + '.txt', energykin, delimiter = ',')
np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/eint_' + str(np.round(params.mu, 0)) + '_' + str(omegastart) + '-'+ str(omegaend)  + '.txt', energyint, delimiter = ',')
np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/angular_momentum_' + str(np.round(params.mu, 0)) + '_' + str(omegastart) + '-'+ str(omegaend)  + '.txt', angmom, delimiter = ',')

