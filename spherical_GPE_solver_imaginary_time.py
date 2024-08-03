import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
import spherical_GPE_functions as sgpe
import spherical_GPE_params as params

from cartopy import crs
from matplotlib import cm

#set some parameters for plotting


plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'cm'

#initialize wavefunction

psi = sgpe.generate_gridded_wavefunction(params.theta_plus, params.phi_plus, params.theta_minus, params.phi_minus, params.xi, params.bg_dens)

particle_number = sgpe.get_norm(psi)




#some stuff needed for plotting

mycmap = cm.seismic
myprojection = crs.Mollweide(central_longitude=180.)
gridspec_kw = dict(height_ratios = (1, 1), hspace = 0.5)



#SIMULATION
############################################################

#after the if statement follows the plotting routine to give a plot of the density and phase of the wave function and a plot of the spectrum

t = np.zeros(params.end//10 + 1, dtype = np.float64) #initialize array of passed time in the simulation
energy_t = np.zeros(params.end//10 + 1, dtype = np.float64) #initialize array of energy as a function of time

plot_number = 5

for q in range(params.end + 1): 
    if (q % (params.end // plot_number) == 0):  #plot 10 times during simulation
        time = round(params.real_dt * q, 2) #real time that has passed at this point in the simulation in ms
        
        
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
        
        plt.suptitle('Imaginary time evolution of two vortices after ' + str(time) + 'ms')

        #subplot for denstiy

        dens_grid.plot(cmap = mycmap, 
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

        phase_grid.plot(cmap = mycmap, 
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
        
        
        #these lines put a textbox with some simulation parameters into the density plot
        
        textstr ='\n'.join((r'$\omega=%.1f$' % (params.omega_units, ) + 'Hz', r'$g = %.3f$' % (params.g, ) + r'$\hbar^2/m$'))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axes[0].text(10, 80, textstr, fontsize=7, verticalalignment='top', bbox=props)
        
        #put the conserved quantities below the plots
        
        #axes[1].text(-1, -150, 'Norm = ' + str(norm), fontsize = 'x-small')
        #axes[1].text(-1, -180, 'Energy = ' + str(energy), fontsize = 'x-small')
        #axes[1].text(-1, -210, 'Angular momentum = ' + str(mom), fontsize = 'x-small')
        
        filename = 'J:/Uni - Physik/Master/Masterarbeit/Media/Simulations of two vortices/' + str(time) + 'ms.pdf'

        #plt.savefig(fname = filename, dpi = 300, bbox_inches = 'tight', format = 'pdf')
        
        #plot spectrum
        
        clm.plot_spectrum(unit = 'per_l', show = False)
        
        plt.title('Spectrum after ' + str(time) + 'ms')

        filename = 'J:/Uni - Physik/Master/Masterarbeit/Media/Simulations of two vortices/spectrum_' + str(time) + 'ms.pdf'

        #plt.savefig(fname = filename, dpi = 300, bbox_inches = 'tight', format = 'pdf')
        plt.show()
    if (q % 10 == 0): #do this every 10 steps
        index = q // 10
        t[index] = params.real_dt * q
        ekin, eint, erot = sgpe.get_energy(psi, params.g, params.omega)
        energy_t[index] = ekin + eint + erot
    psi = sgpe.imaginary_timestep_grid(psi, params.dt, params.g, params.omega, particle_number)

header = 'bg_dens = ' + str(params.bg_dens) + ', theta+ = %.7f' % (params.theta_plus, ) + ', phi+ = %.7f' % (params.phi_plus, ) + ', theta- = %.7f' % (params.theta_minus, ) + ', phi- = %.7f' % (params.phi_minus, )
np.savetxt('J:/Uni - Physik/Master/Masterarbeit/Data/Initial conditions/initial condition4.txt', psi, delimiter = ',', header = header)

#%%
#plot the energy as a function of time

plt.plot(t, energy_t)
plt.ylabel(r'$E_{\text{tot}}$ $\left[ \frac{\hbar^2}{m R^2}  \right]$')
plt.xlabel(r'$t$ [ms]')
filename = 'J:/Uni - Physik/Master/Masterarbeit/Media/Simulations of two vortices/energy.pdf'
#plt.savefig(fname = filename, dpi = 300, bbox_inches = 'tight', format = 'pdf')


