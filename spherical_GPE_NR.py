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


#initialize wavefunction. this will function as the first guess psig in the NR method
psi = sgpe.generate_gridded_wavefunction(params.theta_plus, params.phi_plus, params.theta_minus, params.phi_minus, params.xi, params.bg_dens)
particle_number = sgpe.get_norm(psi)
print(particle_number)
for _ in range(300):
    psi = sgpe.imaginary_timestep_grid(psi, params.dt, params.g, 0.0, particle_number)

print(sgpe.get_norm(psi))
#%%   

psif = sgpe.NR(psi, params.g, params.omega, params.mu)

dens = np.abs(psif)**2
phase = np.angle(psif)

dens_coeffs = pysh.expand.SHExpandDH(griddh = dens, norm = 4, sampling = 2) #get sh coefficients for the density
phase_coeffs = pysh.expand.SHExpandDH(griddh = phase, norm = 4, sampling = 2) #get sh coefficients for the phase

dens_clm = pysh.SHCoeffs.from_array(dens_coeffs, normalization='ortho', lmax = params.lmax) #create a SHCoeffs instance from the coefficient array for the density 
phase_clm = pysh.SHCoeffs.from_array(phase_coeffs, normalization='ortho', lmax = params.lmax) #create a SHCoeffs instance from the coefficient array for the phase

dens_grid = dens_clm.expand() #create a SHGrid instance for the density 
phase_grid = phase_clm.expand() #create a SHGrid instance for the phase

gridspec_kw = dict(height_ratios = (1, 1), hspace = 0.5)
density_cmap = cmocean.cm.thermal
phase_cmap = cmocean.cm.balance

fig, axes = plt.subplots(2, 1, gridspec_kw = gridspec_kw, figsize = (10, 6))

#plt.suptitle('Time evolution of two vortices at $t = $' + str(time) + r'$\frac{m R^2}{\hbar}$', fontsize = 12)

#subplot for denstiy

dens_grid.plot(cmap = density_cmap, 
               colorbar = 'right', 
               cb_label = 'Density', 
               xlabel = '', 
               tick_interval = [90,45], 
               tick_labelsize = 6, 
               axes_labelsize = 7,
               ax = axes[0],  
               show = False)

cb2 = axes[0].images[-1].colorbar
cb2.mappable.set_clim(0., np.max(dens))


#subplot for phase

phase_grid.plot(cmap = phase_cmap, 
                colorbar = 'right',
                cb_label = 'Phase',
                tick_interval = [90,45], 
                cb_tick_interval = np.pi,
                tick_labelsize = 6, 
                axes_labelsize = 7, 
                ax = axes[1],  
                show = False)


cb2 = axes[1].images[-1].colorbar
cb2.mappable.set_clim(-np.pi, np.pi)
cb2.ax.set_yticklabels([r'$-\pi$', 0, r'$+\pi$'])

print(sgpe.get_norm(psif))

