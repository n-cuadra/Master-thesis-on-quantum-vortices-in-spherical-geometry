import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
import spherical_GPE_functions as sgpe
import spherical_GPE_params as params
import time
import scienceplots

#set some parameters for plotting

plt.style.use('science')
plt.rcParams.update({'font.size': 7})
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

#initialize wavefunction



#alpha = 100
#m = 1
#Vp = alpha * np.sin(params.theta_grid)**m * np.cos(m * params.phi_grid)


#psi = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Initial conditions/initial condition NR.txt', delimiter = ',', dtype = np.complex128)



psi = np.sqrt(params.bg_dens) * sgpe.initial_magnitude(params.theta_grid, params.phi_grid, params.theta_plus, params.phi_plus, params.theta_minus, params.phi_minus, params.xi) * np.exp(1.0j * sgpe.phase(params.theta_grid, params.phi_grid, params.theta_plus, params.phi_plus, params.theta_minus, params.phi_minus))
particle_number = sgpe.get_norm(psi)


for _ in range(1000):
    psi = sgpe.imaginary_timestep_grid(psi, params.dt, params.g, 0.0, particle_number, keep_phase = False)


#SIMULATION
############################################################

#after the first if statement follows the plotting routine to give a plot of the density and phase of the wave function and a plot of the spectrum

conservative_number = 10#number of times you would like to record conserved quantities
plot_number = 100 #number of times you would like to plot
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
        
        
        wf_title = r'Time evolution of two vortices at $t = $ ' + str(time) + r'$\frac{m R^2}{\hbar}$'
        spectrum_title = r'Spectrum at $t = $ ' + str(time) + r'$\frac{m R^2}{\hbar}$'
        wf_path = './wf_' + str(int(round(timestamp, 0))) + '_' + str(time) + '.pdf'
        spectrum_path = './spectrum_' + str(int(round(timestamp, 0))) + '_' + str(time) + '.pdf'
        sgpe.plot(psi, wf_title = wf_title, spectrum_title = spectrum_title)
        
        
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
            
    
    psi = sgpe.timestep_grid(psi, dt = params.dt, g =  params.g, omega = 0, mu = params.mu)
    #tstop = 1e-2
    #if (params.dt * q <= tstop):
        #psi = psi * np.exp(-1.0j * Vp * (1 - params.dt * q / tstop) * params.dt)
    
  
    

#np.savetxt('./vortex_tracker_' + str(int(round(timestamp, 0))) + '.txt', vortex_tracker, delimiter = ',')
#np.savetxt('./t_tracker_' + str(int(round(timestamp, 0))) + '.txt', t_tracker, delimiter = ',')
#np.savetxt('./t_' + str(int(round(timestamp, 0))) + '.txt', t, delimiter = ',')
#np.savetxt('./etot_' + str(int(round(timestamp, 0))) + '.txt', energy_t, delimiter = ',')
#np.savetxt('./particle_number_' + str(int(round(timestamp, 0))) + '.txt', particle_number_t, delimiter = ',')
#np.savetxt('./angular_momentum_' + str(int(round(timestamp, 0))) + '.txt', angular_momentum_t, delimiter = ',')











