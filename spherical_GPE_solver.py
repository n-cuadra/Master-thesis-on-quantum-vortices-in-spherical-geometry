import numpy as np
import matplotlib.pyplot as plt
import spherical_GPE_functions as sgpe
import spherical_GPE_params as params
import time
import scienceplots
import pyshtools as pysh

#set some parameters for plotting

plt.style.use('science')
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'xtick.labelsize': 10})
plt.rcParams.update({'ytick.labelsize': 10})


#initialize wavefunction


Rand = np.zeros((params.N, 2 * params.N), dtype = np.float64)
Rand[0,:] = np.random.rand()
Rand[1:,:] = np.random.rand(params.N - 1, 2 * params.N)


#degrees = np.arange(256, dtype=float)
#degrees[0] = 1
#power = degrees**(-4)
#clm = pysh.SHCoeffs.from_random(power, seed=12345)
#clm_ortho = clm.convert(normalization='ortho')
#coeffs = clm.to_array()
#psi = pysh.expand.MakeGridDHC(coeffs, norm = 4, sampling = 2, extend = False)
#psi = psi * np.sqrt(4 * np.pi * params.mu / params.g / sgpe.get_norm(psi))




psi = np.sqrt(params.bg_dens) * np.exp(1j * 2 * np.pi * Rand)
particlenumber = sgpe.get_norm2(psi)
#RJ = psi[:, params.N//2: 3 * params.N // 2]
#psi = np.loadtxt('./psi_f.txt', delimiter=',', dtype=np.complex128)
#psi = np.concatenate((RJ, np.flip(RJ, axis = 1)), axis = 1)

omega = 8
#eta =  0.01 * np.max(np.abs(psi)) * np.exp(1j * 2 * np.pi * Rand) #noise term


#timestep_multiplier_array = sgpe.timestep_multiplier(dt = params.dt, omega = params.omega)
imaginary_timestep_multiplier_array = sgpe.imaginary_timestep_multiplier(dt = params.dt, omega = omega)


#psi = psi + eta 



#SIMULATION
############################################################

#after the first if statement follows the plotting routine to give a plot of the density and phase of the wave function and a plot of the spectrum

conservative_number = 100#number of times you would like to record conserved quantities
plot_number = 30 #number of times you would like to plot
tracking_number = 100 #number of times you would like to track vortices

t = np.zeros(conservative_number + 1, dtype = np.float64) #initialize array of passed time in the simulation
particle_number_t = np.zeros(conservative_number + 1, dtype = np.float64) #initialize array of particle number as a function of time
ekin_t = np.zeros(conservative_number + 1, dtype = np.float64) #initialize array of energy as a function of time
eint_t = np.zeros(conservative_number + 1, dtype = np.float64)
e0_t = np.zeros(conservative_number + 1, dtype = np.float64)
angular_momentum_t = np.zeros(conservative_number + 1, dtype = np.float64) #initialize array of angular momentum as a function of time

#initialize arrays for vortex tracking
t_tracker = np.zeros(tracking_number + 1, dtype = np.float64)
vortex_tracker = np.zeros((4, tracking_number + 1), dtype = np.float64)


vortex_tracking = False #True if you want to track vortex position 
conserved_tracking = False #True if you want to record conserved quantities

timestamp = time.time() #measure the time, this will be used to create unique file names, that's all

#coeffs_t = np.zeros((plot_number + 1, 2, params.lmax + 1, params.lmax + 1), dtype = np.complex128)

for q in range(params.end + 1): 
    if (q % (params.end // plot_number) == 0):  #plot plot_number times during simulation, and
        #print('lstart: ', sgpe.lstart)
        current_time = '%.1f' % round(params.dt * q, 4)  #time that has passed at this point in the simulation as a string
        ekin, eint, erot = sgpe.get_energy(psi, params.g, omega)
        print('Total energy: ', ekin + eint + erot - params.g * sgpe.get_norm(psi)**2 / (8 * np.pi))
        print('Kinetic energy: ', ekin)
        print('Interaction energy: ', eint)
        print('Angular Momentum: ', sgpe.get_ang_momentum(psi))
        print('E0: ', params.g * sgpe.get_norm(psi)**2 / (8 * np.pi), params.mu * sgpe.get_norm2(psi))
        index = q // (params.end // plot_number)
        
        wf_title = r'$\tilde\omega = $ ' + str(omega)
        wf_subtitle = r'$\tau = $ ' + current_time #+ r'$\frac{m R^2}{\hbar}$'
        spectrum_title = r'Spectrum at $t = $ ' + current_time + r'$\frac{M R^2}{\hbar}$'
        #wf_path = 'E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Imaginary Time of Vortex Pole solution/wf_rp_' + str(omega) + '_' + str(index) + '.png'
        #spectrum_path = './spectrum_' + str(int(round(timestamp, 0))) + '_' + current_time + '.pdf'
        wf_path = ''
        spectrum_path = ''
        sgpe.plot(psi, wf_subtitle=wf_subtitle, wf_title = wf_title, spectrum_title = spectrum_title, wf_path = wf_path, ftype = 'png', dpi = 1200)
        
        
        #coeffs_t[index, :, :, :] = pysh.expand.SHExpandDHC(psi, norm = 4, sampling = 2)
        
    if conserved_tracking:
        if (q % (params.end // conservative_number) == 0): #conservative_number times record the time, particle number, energy and angular momentum in the arrays initialized above (to plot the conserved quantities as function of time below)
            index = q // (params.end // conservative_number)
            t[index] = params.dt * q
            particle_number_t[index] = sgpe.get_norm(psi)
            ekin, eint, erot = sgpe.get_energy(psi, params.g, params.omega) 
            ekin_t[index] = ekin 
            eint_t[index] = eint
            e0_t[index] = params.g * sgpe.get_norm(psi)**2 / (8 * np.pi)
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
            
    
    #psi = sgpe.timestep_grid(psi, dt = params.dt, g = params.g, mu = params.mu, multiplier_array = timestep_multiplier_array, filtering = False)
    psi = sgpe.imaginary_timestep_grid(psi, dt = params.dt, g = params.g, particle_number = particlenumber, multiplier_array = imaginary_timestep_multiplier_array, keep_phase = False)


np.savetxt('./psi_f.txt', psi, delimiter = ',') 

#np.savetxt('./vortex_tracker_' + str(int(round(timestamp, 0))) + '.txt', vortex_tracker, delimiter = ',')
#np.savetxt('./t_tracker_' + str(int(round(timestamp, 0))) + '.txt', t_tracker, delimiter = ',')
#np.savetxt('./t_' + str(int(round(timestamp, 0))) + '.txt', t, delimiter = ',')
#np.savetxt('./ekin_' + str(int(round(timestamp, 0))) + '.txt', ekin_t, delimiter = ',')
#np.savetxt('./eint_' + str(int(round(timestamp, 0))) + '.txt', eint_t, delimiter = ',')
#np.savetxt('./e0_' + str(int(round(timestamp, 0))) + '.txt', e0_t, delimiter = ',')
#np.savetxt('./particle_number_' + str(int(round(timestamp, 0))) + '.txt', particle_number_t, delimiter = ',')
#np.savetxt('./angular_momentum_' + str(int(round(timestamp, 0))) + '.txt', angular_momentum_t, delimiter = ',')

