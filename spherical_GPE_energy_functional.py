import numpy as np
import matplotlib.pyplot as plt
import spherical_GPE_functions as sgpe
import spherical_GPE_params as params
import pyshtools as pysh

num = 50

theta_plus = np.linspace(0.01, 20*np.pi/180, num)


ekin = np.zeros(shape = num, dtype = np.float64)
erot = np.zeros(shape = num, dtype = np.float64)
eint = np.zeros(shape = num, dtype = np.float64)

#initialize wavefunction

for k in range(num):
    psi = sgpe.generate_gridded_wavefunction(theta_plus[k], np.pi, np.pi - theta_plus[k], np.pi, params.xi, params.bg_dens)
    particle_number = sgpe.get_norm(psi)
    for _ in range(300):
        psi = sgpe.imaginary_timestep_grid(psi, params.dt, params.g, 0.0, particle_number)
        
    kin, inter, rot = sgpe.get_energy(psi, params.g, 1.0)
    ekin[k] = kin
    eint[k] = inter
    erot[k] = rot
    
    coeffs = pysh.expand.SHExpandDH(griddh = np.abs(psi)**2, norm = 4, sampling = 2)
    coeffs_phase = pysh.expand.SHExpandDH(griddh = np.angle(psi), norm = 4, sampling = 2)
    clm_phase = pysh.SHCoeffs.from_array(coeffs_phase, normalization='ortho', lmax = params.lmax)
    clm = pysh.SHCoeffs.from_array(coeffs, normalization='ortho', lmax = params.lmax)
    grid = clm.expand()
    grid_phase = clm_phase.expand()
    grid.plot(colorbar = 'right', cb_label = 'Density', show = False)
    grid_phase.plot(colorbar = 'right', cb_label = 'Phase', show = False)
    plt.show()
    print(k)           


np.savetxt('J:/Uni - Physik/Master/Masterarbeit/Data/Recreating the results of Padavic et al/ekin20.txt', ekin, delimiter = ',')
np.savetxt('J:/Uni - Physik/Master/Masterarbeit/Data/Recreating the results of Padavic et al/erot20.txt', erot, delimiter = ',')
np.savetxt('J:/Uni - Physik/Master/Masterarbeit/Data/Recreating the results of Padavic et al/eint20.txt', eint, delimiter = ',')


#%%

plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'cm'

theta_plus_degrees = 180 * theta_plus / np.pi


plt.plot(theta_plus_degrees, ekin + eint + erot * 0.4, label = r'$\~\omega = $ 0.49', linewidth = 0.8)
plt.plot(theta_plus_degrees, ekin + eint + erot * 0.6, label = r'$\~\omega = $ 0.5', linewidth = 0.8)
plt.plot(theta_plus_degrees, ekin + eint + erot * 0.8, label = r'$\~\omega = $ 0.51', linewidth = 0.8)




#plt.xticks(ticks = (0, 10, 20, 30, 40, 50, 60, 70, 80), labels = ('0°', '10°', '20°', '30°', '40°', '50°', '60°', '70°', '80°'))


plt.xlabel(r'$\theta_+$')
plt.ylabel(r'$E_{\text{tot}}$ $\left[ \frac{\hbar^2}{m R^2}  \right]$')
#plt.legend(fontsize = 'x-small')
#plt.savefig('J:/Uni - Physik/Master/Masterarbeit/Media/total energy as a function of theta and omega5.pdf', format = 'pdf', dpi = 300) 

#%%

plt.plot(theta_plus_degrees, ekin, linewidth = 0.8)
plt.xticks(ticks = (0, 10, 20, 30, 40, 50, 60, 70), labels = ('0°', '10°', '20°', '30°', '40°', '50°', '60°', '70°'))

plt.xlabel(r'$\theta_+$')
plt.ylabel(r'$E_{\text{kin}}$ $\left[ \frac{\hbar^2}{m R^2}  \right]$')
#plt.savefig('J:/Uni - Physik/Master/Masterarbeit/Media/kinetic energy as a function of theta and omega.pdf', format = 'pdf', dpi = 300)

#%%

plt.plot(theta_plus_degrees, erot[:,4], label = r'$\~\omega = $ 0.5', linewidth = 0.8)
plt.plot(theta_plus_degrees, erot[:,5], label = r'$\~\omega = $ 0.6', linewidth = 0.8)
plt.plot(theta_plus_degrees, erot[:,6], label = r'$\~\omega = $ 0.7', linewidth = 0.8)
plt.plot(theta_plus_degrees, erot[:,7], label = r'$\~\omega = $ 0.8', linewidth = 0.8)
plt.plot(theta_plus_degrees, erot[:,8], label = r'$\~\omega = $ 0.9', linewidth = 0.8)
plt.xticks(ticks = (0, 10, 20, 30, 40, 50, 60, 70), labels = ('0°', '10°', '20°', '30°', '40°', '50°', '60°', '70°'))


plt.xlabel(r'$\theta_+$')
plt.ylabel(r'$E_{\text{rot}}$ $\left[ \frac{\hbar^2}{m R^2}  \right]$')
plt.legend(fontsize = 'x-small')
#plt.savefig('J:/Uni - Physik/Master/Masterarbeit/Media/rotation energy as a function of theta and omega.pdf', format = 'pdf', dpi = 300)

#%%

plt.plot(theta_plus_degrees, eint, linewidth = 0.8)
plt.xticks(ticks = (0, 10, 20, 30, 40, 50, 60, 70), labels = ('0°', '10°', '20°', '30°', '40°', '50°', '60°', '70°'))


plt.xlabel(r'$\theta_+$')
plt.ylabel(r'$E_{\text{int}}$ $\left[ \frac{\hbar^2}{m R^2}  \right]$')
#plt.savefig('J:/Uni - Physik/Master/Masterarbeit/Media/interaction energy as a function of theta and omega.pdf', format = 'pdf', dpi = 300)

