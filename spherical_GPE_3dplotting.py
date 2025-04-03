import numpy as np
import pyshtools as pysh
import spherical_GPE_functions as sgpe
import spherical_GPE_params as params
import matplotlib.pyplot as plt
import matplotlib.cm
import cmocean
import scienceplots
from scipy.sparse.linalg import LinearOperator, bicgstab
from mpl_toolkits.mplot3d import Axes3D

#set some parameters for plotting


#set some parameters for plotting

plt.style.use('science')
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'xtick.labelsize': 10})
plt.rcParams.update({'ytick.labelsize': 10})



#psi = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Initial conditions/initial condition4.txt', delimiter = ',', dtype = np.complex128)

psi = sgpe.IC_vortex_dipole(np.pi/4, np.pi, np.pi - np.pi/4, np.pi, params.xi, params.bg_dens)
particle_number = sgpe.get_norm(psi)
print(particle_number)

for _ in range(500):
    psi = sgpe.imaginary_timestep_grid(psi, params.dt, params.g, 0.0, particle_number, keep_phase = False)

#%%
density = np.abs(psi)**2
phase = np.angle(psi)

x, y, z = sgpe.sph2cart(params.THETA, params.PHI)


# Create a figure and a 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


fcolorsdens = (density - density.min())/(density.max() - density.min())
colorsdens = cmocean.cm.thermal(fcolorsdens)


fcolorsphase = (phase - phase.min())/(phase.max() - phase.min())
colorsphase = cmocean.cm.balance(fcolorsphase)

# Plot the surface
surf = ax.plot_surface(x, y, z, facecolors=colorsdens, rstride=1, cstride=1, antialiased=True, cmap = cmocean.cm.thermal)
#surf = ax.plot_surface(x, y, z, facecolors=colorsphase, rstride=1, cstride=1, antialiased=True, cmap = cmocean.cm.balance)

# Add color bar
mappable = plt.cm.ScalarMappable(cmap=cmocean.cm.thermal)
#mappable = plt.cm.ScalarMappable(cmap=cmocean.cm.balance)
mappable.set_array(density)
#mappable.set_array(phase)
cbar = plt.colorbar(mappable, ax=ax, shrink=0.4, aspect=15, pad = 0.001, location = 'bottom', anchor = (0.5, 2.0))

cbar.ax.set_xlabel(r'$n$ $\left[1/R^2\right]$', fontsize=16)
#cbar.ax.set_xlabel(r'Phase', fontsize=16)
#cbar.mappable.set_clim(-np.pi, np.pi)
#cbar.ax.set_xticks(ticks = [-np.pi, 0, np.pi], labels = [r'$-\pi$', 0, r'$+\pi$'])
cbar.ax.tick_params(labelsize=14)


# Set labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#set ticks
ax.set_xticks(ticks = [-1, 0, 1], labels = [r'$-R$', 0, r'$R$'])
ax.set_yticks(ticks = [-1, 0, 1], labels = [r'$-R$', 0, r'$R$'])
ax.set_zticks(ticks = [-1, 0, 1], labels = [r'$-R$', 0, r'$R$'])
# Keep tick labels but remove tick marks
ax.tick_params(color = 'white', which = 'both')
#make gridlines white
ax.xaxis._axinfo['grid'].update(color='white', linewidth=0.5)
ax.yaxis._axinfo['grid'].update(color='white', linewidth=0.5)
ax.zaxis._axinfo['grid'].update(color='white', linewidth=0.5)
#make axis lines transparent
ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  
ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  
ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 

# Change the color of the grid area to light blue
ax.xaxis.pane.fill = True
ax.yaxis.pane.fill = True
ax.zaxis.pane.fill = True
color = '#9EE6E2'
ax.xaxis.pane.set_facecolor(color)  
ax.yaxis.pane.set_facecolor(color)  
ax.zaxis.pane.set_facecolor(color)  

# Adjust the viewing angle
ax.view_init(elev=15, azim=150)
ax.set_box_aspect(None, zoom=0.8)

# Show the plot
plt.show()

# Save the plot as a PDF
fig.savefig('./density3dplot_example.png', dpi=300, bbox_inches = 'tight', format = 'png', transparent = True)

