import numpy as np
import matplotlib.pyplot as plt


plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'cm'

ekin = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Data/Recreating the results of Padavic et al/ekin20_2.txt', comments = '#', delimiter = ',', dtype = np.float64)
eint = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Data/Recreating the results of Padavic et al/eint20_2.txt', comments = '#', delimiter = ',', dtype = np.float64)
erot = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Data/Recreating the results of Padavic et al/erot20_2.txt', comments = '#', delimiter = ',', dtype = np.float64)

theta_plus = np.linspace(0.01, np.pi/9, 200)
theta_plus_degrees = 180 * theta_plus / np.pi



gridspec_kw = dict(height_ratios = (1, 1), hspace = 0.5)

fig, axes = plt.subplots(1, 3, figsize = (10, 2))




axes[0].plot(theta_plus_degrees, ekin + eint - erot * 0.49, label = r'$\~\omega = $ 0.49', linewidth = 0.8)
axes[0].set_ylabel(r'$E_{\text{tot}}$ $\left[ \frac{\hbar^2}{m R^2}  \right]$')
axes[0].set_xlabel(r'$\theta_+$')
axes[0].set_xticks(ticks = (0, 10, 20), labels = ('0°', '10°', '20°'))
axes[0].set_xlim(-1, 20)
axes[0].set_ylim(8.594821e6, 8.594821e6 + 10)
axes[0].set_yticks(ticks = (0 + 8.594821e6, 2 + 8.594821e6, 4 + 8.594821e6, 6 + 8.594821e6, 8 + 8.594821e6, 10 + 8.594821e6), labels = (0, 2, 4, 6, 8, 10))
axes[0].annotate('+8.594821e6', xy=(0, 1.02), xycoords = 'axes fraction', fontsize=7)
axes[0].set_title(r'$\~\omega = $ -0.49')

axes[1].plot(theta_plus_degrees, ekin + eint - erot * 0.5, label = r'$\~\omega = $ 0.5', linewidth = 0.8)
axes[1].set_xlabel(r'$\theta_+$')
axes[1].set_xticks(ticks = (0, 10, 20), labels = ('0°', '10°', '20°'))
axes[1].set_xlim(-1, 20)
axes[1].set_ylim(8.594708e6, 8.594708e6 + 10)
axes[1].set_yticks(ticks = (0 + 8.594708e6, 2 + 8.594708e6, 4 + 8.594708e6, 6 + 8.594708e6, 8 + 8.594708e6, 10 + 8.594708e6),  labels = (0, 2, 4, 6, 8, 10))
axes[1].annotate('+8.594708e6', xy=(0, 1.02), xycoords = 'axes fraction', fontsize=7)
axes[1].set_title(r'$\~\omega = $ -0.5')

axes[2].plot(theta_plus_degrees, ekin + eint - erot * 0.51, label = r'$\~\omega = $ 0.51', linewidth = 0.8)
axes[2].set_xlabel(r'$\theta_+$')
axes[2].set_xticks(ticks = (0, 10, 20), labels = ('0°', '10°', '20°'))
axes[2].set_xlim(-1, 20)#
axes[2].set_ylim(8.594595e6, 8.594595e6 + 10)
axes[2].set_yticks(ticks = (0 + 8.594595e6, 2 + 8.594595e6, 4 + 8.594595e6, 6 + 8.594595e6, 8 + 8.594595e6, 10 + 8.594595e6),  labels = (0, 2, 4, 6, 8, 10))
axes[2].annotate('+8.594595e6', xy=(0, 1.02), xycoords = 'axes fraction', fontsize=7)
axes[2].set_title(r'$\~\omega = $ -0.51')



#fig.savefig('J:/Uni - Physik/Master/Masterarbeit/Media/etot.pdf', format = 'pdf', dpi = 300, bbox_inches = 'tight')

#%%

fig, ax = plt.subplots()
ax.plot(theta_plus_degrees, eint, lw = 0.8)
ax.set_xticks(ticks = (0, 10, 20), labels = ('0°', '10°', '20°'))
ax.set_ylabel(r'$E_{\text{int}}$ $\left[ \frac{\hbar^2}{m R^2}  \right]$')
#fig.savefig('J:/Uni - Physik/Master/Masterarbeit/Media/eint.pdf', format = 'pdf', dpi = 300, bbox_inches = 'tight')



