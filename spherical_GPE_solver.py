import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
import spherical_GPE_functions as sgpe
import spherical_GPE_params as params
import cmocean
import time
import scienceplots

#set some parameters for plotting

plt.style.use('science')
plt.rcParams.update({'font.size': 7})
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

#initialize wavefunction

psi = sgpe.generate_gridded_wavefunction(params.theta_plus, params.phi_plus, params.theta_minus, params.phi_minus, params.xi, params.bg_dens)
particle_number = sgpe.get_norm(psi)

for _ in range(300):
    psi = sgpe.imaginary_timestep_grid(psi, params.dt, params.g, 0.0, particle_number)


#psi = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Initial conditions/initial condition NR.txt', delimiter = ',', dtype = np.complex128)


#some stuff needed for plotting

density_cmap = cmocean.cm.thermal
phase_cmap = cmocean.cm.balance
gridspec_kw = dict(height_ratios = (1, 1), hspace = 0.5)

#SIMULATION
############################################################

#after the first if statement follows the plotting routine to give a plot of the density and phase of the wave function and a plot of the spectrum

conservative_number = 10#number of times you would like to record conserved quantities
plot_number = 10 #number of times you would like to plot
tracking_number = 20 #number of times you would like to track vortices

t = np.zeros(conservative_number + 1, dtype = np.float64) #initialize array of passed time in the simulation
particle_number_t = np.zeros(conservative_number + 1, dtype = np.float64) #initialize array of particle number as a function of time
energy_t = np.zeros(conservative_number + 1, dtype = np.float64) #initialize array of energy as a function of time
angular_momentum_t = np.zeros(conservative_number + 1, dtype = np.float64) #initialize array of angular momentum as a function of time

#initialize arrays for vortex tracking
t_tracker = np.zeros(tracking_number + 1, dtype = np.float64)
vortex_tracker = np.zeros((4, tracking_number + 1), dtype = np.float64)


vortex_tracking = False #True if you want to track vortex position 
conserved_tracking = False #True if you want to record conserved quantities

timestamp = time.time() #measure the time, this will be used to create unique file names, that's all

for q in range(params.end + 1): 
    if (q % (params.end // plot_number) == 0):  #plot plot_number times during simulation, and
        print('lstart: ', sgpe.lstart)
        time = round(params.dt * q, 10) #time that has passed at this point in the simulation
        
        
        dens = np.abs(psi)**2 #calculate condensate density
        phase_angle = np.angle(psi) #calculate phase of condensate
        
        
        coeffs = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2) #get sh coefficients for the wave function     
        dens_coeffs = pysh.expand.SHExpandDH(dens, norm = 4, sampling = 2) #get sh coefficients for the density
        phase_coeffs = pysh.expand.SHExpandDH(phase_angle, norm = 4, sampling = 2) #get sh coefficients for the phase
        
        clm = pysh.SHCoeffs.from_array(coeffs, normalization='ortho', lmax = params.lmax) #create a SHCoeffs instance for the wavefunction to plot spectrum (at the bottom)
        dens_clm = pysh.SHCoeffs.from_array(dens_coeffs, normalization='ortho', lmax = params.lmax) #create a SHCoeffs instance from the coefficient array for the density 
        phase_clm = pysh.SHCoeffs.from_array(phase_coeffs, normalization='ortho', lmax = params.lmax) #create a SHCoeffs instance from the coefficient array for the phase

        dens_grid = dens_clm.expand() #create a SHGrid instance for the density 
        phase_grid = phase_clm.expand() #create a SHGrid instance for the phase
        
        #plot

        fig, axes = plt.subplots(2, 1, gridspec_kw = gridspec_kw, figsize = (10, 6))
        
        plt.suptitle('Time evolution of two vortices at $t = $' + str(time) + r'$\frac{m R^2}{\hbar}$', fontsize = 12)

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
        cb2.ax.set_yticks(ticks = [-np.pi, 0, np.pi], labels = [r'$-\pi$', 0, r'$+\pi$'])
        
        
        #the following lines put a textbox with some simulation parameters into the density plot
        
        #textstr ='\n'.join((r'$\omega=%.1f$' % (params.omega_units, ) + 'Hz', r'$g = %.3f$' % (params.g, ) + r'$\hbar^2/m$'))
        #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        #axes[0].text(10, 80, textstr, fontsize=7, verticalalignment='top', bbox=props)
        
        filename = './wf_' + str(int(round(timestamp, 0))) + '_' + str(time) + '.pdf'

        #plt.savefig(fname = filename, dpi = 300, bbox_inches = 'tight', format = 'pdf')
        
        #plot spectrum
        
        clm.plot_spectrum(unit = 'per_l', show = False)
        
        plt.title(r'Spectrum at $t = $' + str(time) + r'$\frac{m R^2}{\hbar}$')

        filename = './spectrum_' + str(int(round(timestamp, 0))) + '_' + str(time) + '.pdf'

        #plt.savefig(fname = filename, dpi = 300, bbox_inches = 'tight', format = 'pdf')
        plt.show()
        
        
        
    if conserved_tracking:
        if (q % (params.end // conservative_number) == 0): #conservative_number times record the time, particle number, energy and angular momentum in the arrays initialized above (to plot the conserved quantities as function of time below)
            index = q // (params.end // conservative_number)
            t[index] = params.dt * q
            particle_number_t[index] = sgpe.get_norm(psi)
            ekin, eint, erot = sgpe.get_energy(psi, params.g, params.omega) 
            energy_t[index] = ekin + eint + erot
            angular_momentum_t[index] = sgpe.get_ang_momentum(psi)
            
    if vortex_tracking:
        if (q % (params.end // tracking_number) == 0):
            density = np.abs(psi)**2
            #calculate the minimum density on the northern and southern hemisphere
            mindensity_plus =  np.min(density[0:params.N//2,:])
            mindensity_minus =  np.min(density[params.N//2:-1,:])
            
            #get the indices for these densities
            iplus, jplus = np.argwhere(density == mindensity_plus)[0]
            iminus, jminus = np.argwhere(density == mindensity_minus)[0]
            
            #get the coordinates for the minima
            theta_g_plus, phi_g_plus = params.theta[iplus], params.phi[jplus]
            theta_g_minus, phi_g_minus = params.theta[iminus], params.phi[jminus]
            
            #use these as an initial guess for vortex tracking
            theta_v_plus, phi_v_plus = sgpe.vortex_tracker(psi, theta_g_plus, phi_g_plus)
            theta_v_minus, phi_v_minus = sgpe.vortex_tracker(psi, theta_g_minus, phi_g_minus)
            
            #record the tracked coordinates
            index = q // (params.end // tracking_number)
            t_tracker[index] = params.dt * q
            vortex_tracker[0, index] = theta_v_plus
            vortex_tracker[1, index] = theta_v_minus
            vortex_tracker[2, index] = phi_v_plus
            vortex_tracker[3, index] = phi_v_minus
            
    
    psi = sgpe.timestep_grid(psi, params.dt, params.g, params.omega)
    
   
    

#np.savetxt('./vortex_tracker_' + str(int(round(timestamp, 0))) + '.txt', vortex_tracker, delimiter = ',')
#np.savetxt('./t_tracker_' + str(int(round(timestamp, 0))) + '.txt', t_tracker, delimiter = ',')
#np.savetxt('./t_' + str(int(round(timestamp, 0))) + '.txt', t, delimiter = ',')
#np.savetxt('./etot_' + str(int(round(timestamp, 0))) + '.txt', energy_t, delimiter = ',')
#np.savetxt('./particle_number_' + str(int(round(timestamp, 0))) + '.txt', particle_number_t, delimiter = ',')
#np.savetxt('./angular_momentum_' + str(int(round(timestamp, 0))) + '.txt', angular_momentum_t, delimiter = ',')











