import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.ticker as mticker
import spherical_GPE_params as params


plt.style.use('science')
plt.rcParams.update({'font.size': 7})
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

etot = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/#19 everything simulation/etot_1726217149.txt', delimiter = ',', dtype = np.float64)
angmom = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/#19 everything simulation/angular_momentum_1726217149.txt', delimiter = ',', dtype = np.float64)
particle_number = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/#19 everything simulation/particle_number_1726217149.txt', delimiter = ',', dtype = np.float64)
t = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/#19 everything simulation/t_1726217149.txt', delimiter = ',', dtype = np.float64)
t_tracker = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/#19 everything simulation/t_tracker_1726217149.txt', delimiter = ',', dtype = np.float64)
vortex_tracker = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/#19 everything simulation/vortex_tracker_1726217149.txt', delimiter = ',', dtype = np.float64)


#%%

fig, ax = plt.subplot_mosaic([['a', 'b', 'c']], figsize = (7.5, 2))
plt.subplots_adjust(wspace=0.4, hspace=0.2)


ax['a'].plot(t[1:], etot[1:], lw = 0.7, color = 'purple')
ax['a'].set(xlabel = r'$t \left[\frac{m R^2}{\hbar}\right]$', ylabel = r'$E_{\text{tot}}$ $\left[ \frac{\hbar^2}{m R^2}  \right]$')

ax['b'].plot(t[1:], particle_number[1:], lw = 0.7, color = 'orange')
ax['b'].set(xlabel = r'$t \left[\frac{m R^2}{\hbar}\right]$', ylabel = r'$N$')

ax['c'].plot(t[1:], angmom[1:], lw = 0.7, color = 'green')
ax['c'].set(xlabel = r'$t \left[\frac{m R^2}{\hbar}\right]$', ylabel = r'$L_z$ [$\hbar$]')

fig.savefig('J:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/#19 everything simulation/conserved_quantities.pdf', dpi = 300, format = 'pdf')

#%%

fig, ax = plt.subplot_mosaic(
    [
        ["A", "A", "A", "A"],
        ["B", "B", "B", "B"],
    ],
    figsize = (8, 4.5)
)
plt.subplots_adjust(wspace=0.3, hspace=0.2)

t_tracker2 = t_tracker * params.hbar / (1000 * params.R**2 * params.mass)

ax['A'].plot(t_tracker2[1:], np.rad2deg(vortex_tracker[0, 1:]), lw = 0.3, color = 'olive', linestyle = '--', marker = '.', label = r'$\theta_+$', markersize = 3)
ax['A'].set_ylabel(r'$\theta$ [°]', fontsize = 7)
ax['A'].legend(loc = 'upper right')
ax['A'].invert_yaxis()
ax['A'].yaxis.set_major_formatter(mticker.ScalarFormatter())
ax['A'].yaxis.get_major_formatter().set_scientific(False)
ax['A'].yaxis.get_major_formatter().set_useOffset(False)



ax['B'].plot(t_tracker2[1:], np.rad2deg(vortex_tracker[1, 1:]), lw = 0.3, color = 'navy', linestyle = '--', marker = '.', label = r'$\theta_-$', markersize = 3)
ax['B'].set_xlabel(r'$t \left[\frac{m R^2}{\hbar}\right]$', fontsize = 7)
ax['B'].set_ylabel(r'$\theta$ [°]', fontsize = 7)
ax['B'].legend(loc = 'lower right')
ax['B'].invert_yaxis()
ax['B'].yaxis.set_major_formatter(mticker.ScalarFormatter())
ax['B'].yaxis.get_major_formatter().set_scientific(False)
ax['B'].yaxis.get_major_formatter().set_useOffset(False)

fig.savefig('J:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/#19 everything simulation/vortex_tracking.pdf', dpi = 300, format = 'pdf')
