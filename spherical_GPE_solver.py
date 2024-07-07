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

psi = np.zeros(shape = (params.N, 2*params.N), dtype = np.complex128)
magnitude = np.zeros(shape = (params.N, 2*params.N), dtype = np.float64)

for i in range(params.N//2):
    for j in range(2*params.N):
        psi[i,j] = np.sqrt(params.bg_dens) * sgpe.one_vortex_magnitude(params.theta[i], params.phi[j], params.theta_plus, params.phi_plus, params.xi) * np.exp(1.0j * sgpe.phase(params.theta[i], params.phi[j], params.theta_plus, params.phi_plus, params.theta_minus, params.phi_minus)) 
        magnitude[i,j] = sgpe.one_vortex_magnitude(params.theta[i], params.phi[j], params.theta_plus, params.phi_plus, params.xi)
        
for i in range(params.N//2, params.N):
    for j in range(2 * params.N):
        psi[i,j] = np.sqrt(params.bg_dens) * sgpe.one_vortex_magnitude(params.theta[i], params.phi[j], params.theta_minus, params.phi_minus, params.xi) * np.exp(1.0j * sgpe.phase(params.theta[i], params.phi[j], params.theta_plus, params.phi_plus, params.theta_minus, params.phi_minus)) 
        magnitude[i,j] = sgpe.one_vortex_magnitude(params.theta[i], params.phi[j], params.theta_minus, params.phi_minus, params.xi)


print(np.min(magnitude))
#%%
density = np.abs(psi)**2
mindensity = np.min(density[0:params.N//2,:])
i, j = np.argwhere(density == mindensity)[0]
theta_guess, phi_guess = params.theta[i], params.phi[j]


#print(theta_guess, phi_guess)
print(params.theta_plus, params.phi_plus+0.001)
theta_v, phi_v = sgpe.vortex_tracker(psi, params.theta_plus, params.phi_plus+0.001)

print(theta_v, phi_v)
#%%
#some stuff needed for plotting

mycmap = cm.seismic
myprojection = crs.Mollweide(central_longitude=180.)
gridspec_kw = dict(height_ratios = (1, 1), hspace = 0.5)

#SIMULATION
############################################################

#after the first if statement follows the plotting routine to give a plot of the density and phase of the wave function and a plot of the spectrum


t = np.zeros(params.end//10 + 1, dtype = np.float64) #initialize array of passed time in the simulation
particle_number_t = np.zeros(params.end//10 + 1, dtype = np.float64) #initialize array of particle number as a function of time
energy_t = np.zeros(params.end//10 + 1, dtype = np.float64) #initialize array of energy as a function of time
angular_momentum_t = np.zeros(params.end//10 + 1, dtype = np.float64) #initialize array of angular momentum as a function of time

plot_number = 1000 #number of times you would like to plot during the simulation

for q in range(params.end + 1): 
    if (q % (params.end // plot_number) == 0):  #plot plot_number times during simulation
        time = round(params.real_dt * q, 2) #real time that has passed at this point in the simulation in ms
        
        
        dens = np.abs(psi)**2 #calculate condensate density
        phase_angle = np.angle(psi) #calculate phase of condensate
        
        coeffs = pysh.expand.SHExpandDH(griddh = psi, norm = 4, sampling = 2) #get sh coefficients for the wave function     
        dens_coeffs = pysh.expand.SHExpandDH(griddh = dens, norm = 4, sampling = 2) #get sh coefficients for the density
        phase_coeffs = pysh.expand.SHExpandDH(griddh = phase_angle, norm = 4, sampling = 2) #get sh coefficients for the phase
        
        clm = pysh.SHCoeffs.from_array(coeffs, normalization='ortho', lmax = params.lmax) #create a SHCoeffs instance for the wavefunction to plot spectrum (at the bottom)
        dens_clm = pysh.SHCoeffs.from_array(dens_coeffs, normalization='ortho', lmax = params.lmax) #create a SHCoeffs instance from the coefficient array for the density 
        phase_clm = pysh.SHCoeffs.from_array(phase_coeffs, normalization='ortho', lmax = params.lmax) #create a SHCoeffs instance from the coefficient array for the phase

        dens_grid = dens_clm.expand() #create a SHGrid instance for the density 
        phase_grid = phase_clm.expand() #create a SHGrid instance for the phase

        #plot

        fig, axes = plt.subplots(2, 1, gridspec_kw = gridspec_kw, figsize = (10, 6))
        
        plt.suptitle('Time evolution of two vortices after ' + str(time) + 'ms')

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
        
        cb2 = axes[0].images[-1].colorbar
        cb2.mappable.set_clim(0., np.max(dens))
        
        
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
        
        
        #the following lines put a textbox with some simulation parameters into the density plot
        
        textstr ='\n'.join((r'$\omega=%.1f$' % (params.omega_units, ) + 'Hz', r'$g = %.3f$' % (params.g, ) + r'$\hbar^2/m$'))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axes[0].text(10, 80, textstr, fontsize=7, verticalalignment='top', bbox=props)
        
        filename = 'J:/Uni - Physik/Master/Masterarbeit/Media/Simulations of two vortices/Animation/' + str(time) + 'ms.jpg'

        plt.savefig(fname = filename, dpi = 1200, bbox_inches = 'tight', format = 'jpg')
        
        #plot spectrum
        
        #clm.plot_spectrum(unit = 'per_l', show = False)
        
        #plt.title('Spectrum after ' + str(time) + 'ms')

        #filename = 'J:/Uni - Physik/Master/Masterarbeit/Media/Simulations of two vortices/spectrum_' + str(time) + 'ms.pdf'

        #plt.savefig(fname = filename, dpi = 300, bbox_inches = 'tight', format = 'pdf')
        plt.show()
    #if (q % 10 == 0): #every 10 steps record the time, particle number, energy and angular momentum in the arrays initialized above (to plot the conserved quantities as function of time below)
        #index = q // 10
        #t[index] = params.real_dt * q
        #particle_number_t[index] = sgpe.get_norm(psi)
        #energy_t[index] = sgpe.get_energy(psi, params.g, params.omega) 
        #angular_momentum_t[index] = sgpe.get_ang_momentum(psi)
    psi = sgpe.timestep_grid(psi, params.dt, params.g, params.omega)

#%%
#plot the conserved quantities as a function of time

energy_t *= 1.924e-19 #energy in eV
elimhelper = np.max(energy_t) - np.min(energy_t)

plt.plot(t, energy_t)
plt.ylabel(r'$E$ [eV]')
plt.xlabel(r'$t$ [ms]')
plt.ylim(np.min(energy_t) - elimhelper/10, np.max(energy_t) + elimhelper/10)
filename = 'J:/Uni - Physik/Master/Masterarbeit/Media/Simulations of two vortices/energy.pdf'
plt.savefig(fname = filename, dpi = 300, bbox_inches = 'tight', format = 'pdf')


#%%
nlimhelper = np.max(particle_number_t) - np.min(particle_number_t)

plt.plot(t, particle_number_t)
plt.ylabel(r'$N$')
plt.xlabel(r'$t$ [ms]')
plt.ylim(np.min(particle_number_t) - nlimhelper/10, np.max(particle_number_t) + nlimhelper/10 )
filename = 'J:/Uni - Physik/Master/Masterarbeit/Media/Simulations of two vortices/particle number.pdf'
plt.savefig(fname = filename, dpi = 300, bbox_inches = 'tight', format = 'pdf')


#%%


plt.plot(t, angular_momentum_t)
plt.ylabel(r'$L_z$ [$\hbar$]')
plt.xlabel(r'$t$ [ms]')
filename = 'J:/Uni - Physik/Master/Masterarbeit/Media/Simulations of two vortices/angular momentum.pdf'
plt.savefig(fname = filename, dpi = 300, bbox_inches = 'tight', format = 'pdf')










