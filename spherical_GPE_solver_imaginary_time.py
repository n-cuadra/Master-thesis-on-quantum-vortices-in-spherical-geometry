import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
import spherical_GPE_functions as sgpe
import spherical_GPE_params as params
import scienceplots
import cmocean

#set some parameters for plotting


plt.style.use('science')
plt.rcParams.update({'font.size': 7})
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

#initialize wavefunction

psi = sgpe.generate_gridded_wavefunction(params.theta_plus, params.phi_plus, params.theta_minus, params.phi_minus, params.xi, params.bg_dens)
particle_number = sgpe.get_norm(psi)

for _ in range(200):
    psi = sgpe.imaginary_timestep_grid(psi, params.dt, params.g, 0.0, particle_number)


#some stuff needed for plotting

density_cmap = cmocean.cm.thermal
phase_cmap = cmocean.cm.balance
gridspec_kw = dict(height_ratios = (1, 1), hspace = 0.5)



#SIMULATION
############################################################

#after the if statement follows the plotting routine to give a plot of the density and phase of the wave function and a plot of the spectrum

plot_number = 20
conserved_number = 1000

t = np.zeros(conserved_number + 1, dtype = np.float64) #initialize array of passed time in the simulation
energy_t = np.zeros(conserved_number + 1, dtype = np.float64) #initialize array of energy as a function of time
angmom = np.zeros(conserved_number + 1, dtype = np.float64)



for q in range(params.end + 1):
    
    if (q % (params.end // plot_number) == 0):  #plot plot_number of times during simulation
        timer = round(params.dt * q, 3) #real time that has passed at this point in the simulation in ms
        
        
        dens = np.abs(psi)**2 #calculate condensate density
        phase_angle = np.angle(psi) #calculate phase of condensate
        
        #norm = sgpe.get_norm(psi) #calculate norm of condensate
        #energy = sgpe.get_energy(psi, params.g, params.omega) #calculate energy of condensate
        #mom = get_ang_momentum(psi) #calculate angular momentum of condensate
        
        coeffs = pysh.expand.SHExpandDHC(griddh = psi, norm = 4, sampling = 2) #get sh coefficients for the wave function     
        dens_coeffs = pysh.expand.SHExpandDH(griddh = dens, norm = 4, sampling = 2) #get sh coefficients for the density
        phase_coeffs = pysh.expand.SHExpandDH(griddh = phase_angle, norm = 4, sampling = 2) #get sh coefficients for the phase
        
        clm = pysh.SHCoeffs.from_array(coeffs, normalization='ortho', lmax = params.lmax) #create a SHCoeffs instance for the wavefunction to plot spectrum (at the bottom)
        dens_clm = pysh.SHCoeffs.from_array(dens_coeffs, normalization='ortho', lmax = params.lmax) #create a SHCoeffs instance from the coefficient array for the density 
        phase_clm = pysh.SHCoeffs.from_array(phase_coeffs, normalization='ortho', lmax = params.lmax) #create a SHCoeffs instance from the coefficient array for the phase

        dens_grid = dens_clm.expand() #create a SHGrid instance for the density 
        phase_grid = phase_clm.expand() #create a SHGrid instance for the phase

        #plot

        fig, axes = plt.subplots(2, 1, gridspec_kw = gridspec_kw, figsize = (10, 6))
        
        plt.suptitle(r'Imaginary time evolution of two vortices at $\tau = $' + str(timer) + r'$\frac{m R^2}{\hbar}$', fontsize = 12)

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
        
        cb1 = axes[0].images[-1].colorbar
        cb1.mappable.set_clim(0., np.max(dens))
        
        
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
        cb2.ax.set_yticks(ticks = [-np.pi, 0, np.pi], labels = [r'$-\pi$', 0, r'$+\pi$'])
        
        
        #these lines put a textbox with some simulation parameters into the density plot
        
        #textstr ='\n'.join((r'$\omega=%.1f$' % (params.omega, ) + r'$\hbar/(m R^2)$', r'$g = %.3f$' % (params.g, ) + r'$\hbar^2/m$'))
        #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        #axes[0].text(10, 80, textstr, fontsize=7, verticalalignment='top', bbox=props)
        
        filename = './wf_' + str(timer) + '.pdf'

        plt.savefig(fname = filename, dpi = 300, bbox_inches = 'tight', format = 'pdf')
        
        #plot spectrum
        
        clm.plot_spectrum(unit = 'per_l', show = False)
        
        plt.title(r'Spectrum at $\tau = $' + str(timer) + r'$\frac{m R^2}{\hbar}$')

        filename = './spectrum_' + str(timer) + '.pdf'

        plt.savefig(fname = filename, dpi = 300, bbox_inches = 'tight', format = 'pdf')
        plt.show()
        
    if (q % (params.end // conserved_number) == 0): #do this conserved_number of times
        index = q // (params.end // conserved_number)
        t[index] = params.dt * q
        ekin, eint, erot = sgpe.get_energy(psi, params.g, 0.0)
        energy_t[index] = ekin + eint + erot 
        angmom[index] = sgpe.get_ang_momentum(psi)
    psi = sgpe.imaginary_timestep_grid(psi, params.dt, params.g, 0.0, particle_number, keep_phase = False)


np.savetxt('./energy.txt', energy_t, delimiter = ',')
np.savetxt('./angular_momentum.txt', angmom, delimiter = ',')

#%%

plt.plot(angmom[:680], energy_t[:680], lw = 0.7)
plt.ylabel(r'$E_{\text{tot}}$ $\left[ \frac{\hbar^2}{m R^2}  \right]$')
plt.xlabel(r'$L_z [\hbar]$')
filename = './energy-momentum.pdf'
plt.savefig(fname = filename, dpi = 300, bbox_inches = 'tight', format = 'pdf')

#%%

plt.plot(t, energy_t, lw = 0.7)
plt.ylabel(r'$E_{\text{tot}}$ $\left[ \frac{\hbar^2}{m R^2}  \right]$')
plt.xlabel(r'$t \left[\frac{m R^2}{\hbar}\right]$')
plt.savefig('./energy.pdf', dpi = 300, bbox_inches = 'tight', format = 'pdf')

#%%

plt.plot(t, angmom, lw = 0.7)
plt.ylabel(r'$L_z [\hbar]$')
plt.xlabel(r'$t \left[\frac{m R^2}{\hbar}\right]$')
plt.savefig('./angular_momentum.pdf', dpi = 300, bbox_inches = 'tight', format = 'pdf')

