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

plt.style.use('science')
plt.rcParams.update({'font.size': 9})
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

cmap = cmocean.cm.thermal

psi = np.loadtxt('J:/Uni - Physik/Master/Masterarbeit/Initial conditions/initial condition4.txt', delimiter = ',', dtype = np.complex128)
density = np.abs(psi)**2
phase = np.angle(psi)

x, y, z = sgpe.sph2cart(params.theta_grid, params.phi_grid)


# Create a figure and a 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')



fcolors = density
fmax, fmin = fcolors.max(), fcolors.min()
fcolors = (fcolors - fmin)/(fmax - fmin)
colors = cmocean.cm.thermal(fcolors)

# Plot the surface
surf = ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, antialiased=True, cmap = cmap)

# Add color bar
mappable = plt.cm.ScalarMappable(cmap=cmap)
mappable.set_array(density)
cbar = plt.colorbar(mappable, ax=ax, shrink=0.4, aspect=15, pad = 0.001, label = 'Density', location = 'bottom', anchor = (0.5, 2.0))
#cbar.mappable.set_clim(-np.pi, np.pi)
#cbar.ax.set_xticks(ticks = [-np.pi, 0, np.pi], labels = [r'$-\pi$', 0, r'$+\pi$'])

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
fig.savefig('J:/Uni - Physik/Master/Masterarbeit/density3dplot.png', dpi=300, bbox_inches = 'tight', format = 'png', transparent = True)

