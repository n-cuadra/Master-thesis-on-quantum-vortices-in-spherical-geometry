import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
import spherical_GPE_functions as sgpe

from cartopy import crs
from matplotlib import cm

#set some parameters for plotting


plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'cm'

#constants
bohr_radius = 5.2918e-5 #bohr radius in micrometers
hbar = 1.055e-22 #hbar in units of kg * µm^2 / s


#simulation parameters (adjustable)
N = 512 #grid points
end = 10000 #number of steps in simulation
mass = 1.443e-25 #mass of the atoms in kg
R = 50.0 #radius of sphere in µm
scattering_length = 100.0 #scattering length in units of the Bohr radius
omega_units = 0.0 #rotating frequency in Hz
theta_plus, phi_plus = 0.001, np.pi #position of vortex 
theta_minus, phi_minus = np.pi - theta_plus, np.pi#position of antivortex
dt = 2.0e-5  #time step
bg_dens = 400. #condensate density far from vortices
xi = 0.05 #healing length 


#these are calculated from the values above, don't change these!

lmax = N//2 - 1 #maximum degree of spherical harmonics
g = - np.pi / np.log(np.sqrt(lmax * (lmax + 1)) * scattering_length * bohr_radius * np.exp(np.euler_gamma) / (2 * R) ) #unitless interaction strength
real_dt = 1000 * dt *  R**2 * mass / hbar #one timestep in real time in ms
omega = omega_units * R**2 * mass / hbar #unitless rotating frequency


#coordinate system

theta, phi = np.linspace(0,  np.pi, N, endpoint = False), np.linspace(0, 2 * np.pi, 2 * N, endpoint = False)  #rectangular grid with polar angle theta and azimuthal angle phi
dangle = np.pi / N  #grid spacing


#coordinate system

theta, phi = np.linspace(0,  np.pi, N, endpoint = False), np.linspace(0, 2 * np.pi, 2 * N, endpoint = False)  #rectangular grid with polar angle theta and azimuthal angle phi
dangle = np.pi / N  #grid spacing


#initialize wavefunction

psi = np.zeros(shape = (N, 2*N), dtype = np.complex128)
magnitude = np.zeros((N, 2*N), dtype = np.float64)

for i in range(N):
    for j in range(2*N):
        psi[i,j] = sgpe.initial_magnitude_2(theta[i], phi[j], theta_plus, phi_plus, theta_minus, phi_minus, xi) * np.exp(1.0j * sgpe.phase(theta[i], phi[j], theta_plus, phi_plus, theta_minus, phi_minus)) 
        magnitude[i,j] = sgpe.initial_magnitude_2(theta[i], phi[j], theta_plus, phi_plus, theta_minus, phi_minus, xi)

print(np.min(magnitude))
        
norm = sgpe.get_norm(psi)
psi = np.sqrt(5000) * psi / np.sqrt(norm)


#some stuff needed for plotting

mycmap = cm.seismic
myprojection = crs.Mollweide(central_longitude=180.)
gridspec_kw = dict(height_ratios = (1, 1), hspace = 0.5)



#SIMULATION
############################################################

#after the if statement follows the plotting routine to give a plot of the density and phase of the wave function and a plot of the spectrum

end = 10000 #number of steps in simulation

t = np.zeros(end//10, dtype = np.float64) #initialize array of passed time in the simulation
energy_t = np.zeros(end//10, dtype = np.float64) #initialize array of energy as a function of time

for q in range(end + 1): 
    if (q % (end / 10) == 0):  #plot 10 times during simulation
        time = round(real_dt * q, 2) #real time that has passed at this point in the simulation in ms
        
        
        dens = np.abs(psi)**2 #calculate condensate density
        phase_angle = np.angle(psi) #calculate phase of condensate
        
        norm = sgpe.get_norm(psi) #calculate norm of condensate
        energy = sgpe.get_energy(psi) #calculate energy of condensate
        #mom = get_ang_momentum(psi) #calculate angular momentum of condensate
        
        coeffs = pysh.expand.SHExpandDH(griddh = psi, norm = 4, sampling = 2) #get sh coefficients for the wave function     
        dens_coeffs = pysh.expand.SHExpandDH(griddh = dens, norm = 4, sampling = 2) #get sh coefficients for the density
        phase_coeffs = pysh.expand.SHExpandDH(griddh = phase_angle, norm = 4, sampling = 2) #get sh coefficients for the phase
        
        clm = pysh.SHCoeffs.from_array(coeffs, normalization='ortho', lmax = lmax) #create a SHCoeffs instance for the wavefunction to plot spectrum (at the bottom)
        dens_clm = pysh.SHCoeffs.from_array(dens_coeffs, normalization='ortho', lmax = lmax) #create a SHCoeffs instance from the coefficient array for the density 
        phase_clm = pysh.SHCoeffs.from_array(phase_coeffs, normalization='ortho', lmax = lmax) #create a SHCoeffs instance from the coefficient array for the phase

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
        
        
        #these lines put a textbox with some simulation parameters into the density plot
        
        textstr ='\n'.join((r'$\omega=%.1f$' % (omega_units, ) + 'Hz', r'$\Delta t =%.3f$' % (real_dt, ) + 'ms', r'$g = %.3f$' % (g, ) + r'$\hbar^2/m$'))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axes[0].text(10, 80, textstr, fontsize=7, verticalalignment='top', bbox=props)
        
        #put the conserved quantities below the plots
        
        axes[1].text(-1, -150, 'Norm = ' + str(norm), fontsize = 'x-small')
        axes[1].text(-1, -180, 'Energy = ' + str(energy), fontsize = 'x-small')
        #axes[1].text(-1, -210, 'Angular momentum = ' + str(mom), fontsize = 'x-small')
        
        filename = 'J:/Uni - Physik/Master/Masterarbeit/Media/Simulations of two vortices/' + str(time) + 'ms.pdf'

        plt.savefig(fname = filename, dpi = 300, bbox_inches = 'tight', format = 'pdf')
        
        #plot spectrum
        
        clm.plot_spectrum(unit = 'per_l', show = False)
        
        plt.title('Spectrum after ' + str(time) + 'ms')

        filename = 'J:/Uni - Physik/Master/Masterarbeit/Media/Simulations of two vortices/spectrum_' + str(time) + 'ms.pdf'

        plt.savefig(fname = filename, dpi = 300, bbox_inches = 'tight', format = 'pdf')
        plt.show()
    if (q % 10 == 0): #do this every 10 steps
        index = q // 10
        t[index] = real_dt * q
        energy_t[index] = sgpe.get_energy(psi)
    psi = sgpe.timestep_grid(psi, dt, g, omega)


#%%
#plot the energy as a function of time

energy_t *= 1.924e-19 #energy in eV
elimhelper = np.max(energy_t) - np.min(energy_t)

plt.plot(t, energy_t)
plt.ylabel(r'$E$ [eV]')
plt.xlabel(r'$t$ [ms]')
plt.ylim(np.min(energy_t) - elimhelper/10, np.max(energy_t) + elimhelper/10)
filename = 'J:/Uni - Physik/Master/Masterarbeit/Media/Simulations of two vortices/energy.pdf'
plt.savefig(fname = filename, dpi = 300, bbox_inches = 'tight', format = 'pdf')


