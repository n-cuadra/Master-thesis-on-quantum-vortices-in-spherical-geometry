import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.ticker as mticker


plt.style.use('science')
plt.rcParams.update({'font.size': 7})
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

etot = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Data/#17 everything simulation/etot_1724341790.txt', delimiter = ',', dtype = np.float64)
angmom = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Data/#17 everything simulation/angular_momentum_1724341790.txt', delimiter = ',', dtype = np.float64)
particle_number = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Data/#17 everything simulation/particle_number_1724341790.txt', delimiter = ',', dtype = np.float64)
t = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Data/#17 everything simulation/t_1724341790.txt', delimiter = ',', dtype = np.float64)
t_tracker = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Data/#17 everything simulation/t_tracker_1724341790.txt', delimiter = ',', dtype = np.float64)
vortex_tracker = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Data/#17 everything simulation/vortex_tracker_1724341790.txt', delimiter = ',', dtype = np.float64)

#%%

fig, ax = plt.subplot_mosaic([['a', 'b', 'c']], figsize = (7.5, 2.5))
plt.subplots_adjust(wspace=0.4, hspace=0.2)


ax['a'].plot(t/1000, etot, lw = 0.7, color = 'purple')
ax['a'].set(xlabel = r'$t$ [s]', ylabel = r'$E_{\text{tot}}$ $\left[ \frac{\hbar^2}{m R^2}  \right]$')

ax['b'].plot(t/1000, particle_number, lw = 0.7, color = 'orange')
ax['b'].set(xlabel = r'$t$ [s]', ylabel = r'$N$')

ax['c'].plot(t/1000, angmom, lw = 0.7, color = 'green')
ax['c'].set(xlabel = r'$t$ [s]', ylabel = r'$L_z$ [$\hbar$]')

fig.savefig('J:/Uni - Physik/Master/Masterarbeit/Media/Simulations of two vortices/#17 everything simulation/conserved_quantities.pdf', dpi = 300, format = 'pdf')

#%%

fig, ax = plt.subplot_mosaic(
    [
        ["A", "A", "A", "A"],
        ["B", "B", "B", "B"],
    ],
    figsize = (8, 4.5)
)
plt.subplots_adjust(wspace=0.3, hspace=0.2)

ax['A'].plot(t_tracker/1000, np.rad2deg(vortex_tracker[0]), lw = 0.5, color = 'olive', linestyle = '--', marker = '.', label = r'$\theta_+$', mew = 0.5)
ax['A'].set_ylabel(r'$\theta$ [°]', fontsize = 7)
ax['A'].legend(loc = 'upper right')
ax['A'].invert_yaxis()
ax['A'].yaxis.set_major_formatter(mticker.ScalarFormatter())
ax['A'].yaxis.get_major_formatter().set_scientific(False)
ax['A'].yaxis.get_major_formatter().set_useOffset(False)



ax['B'].plot(t_tracker/1000, np.rad2deg(vortex_tracker[1]), lw = 0.5, color = 'navy', linestyle = '--', marker = '.', label = r'$\theta_-$', mew = 0.5)
ax['B'].set_xlabel(r'$t$ [s]', fontsize = 7)
ax['B'].set_ylabel(r'$\theta$ [°]', fontsize = 7)
ax['B'].legend(loc = 'lower right')
ax['B'].invert_yaxis()
ax['B'].yaxis.set_major_formatter(mticker.ScalarFormatter())
ax['B'].yaxis.get_major_formatter().set_scientific(False)
ax['B'].yaxis.get_major_formatter().set_useOffset(False)

fig.savefig('J:/Uni - Physik/Master/Masterarbeit/Media/Simulations of two vortices/#17 everything simulation/vortex_tracking.pdf', dpi = 300, format = 'pdf')
