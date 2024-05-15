import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


#parameters

R = 1.0 #radius of sphere
theta_plus = np.pi/3 #position of vortex on the upper hemisphere

#cotangent
def cot(x):
    return np.tan(np.pi/2 - x)

#transformation from spherical to cartesian coordinates

def sph2cart(theta, phi):
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)
    return x, y, z

#transformation from cartesian to spherical coordinates

def cart2sph(x, y, z):
    theta = np.arccos(z/R)
    phi = np.arctan2(x,y)
    return theta, phi

#define phase angle 

def num(theta, phi):
    return cot(theta/2) * np.sin(phi)

def denom(theta, phi):
    return cot(theta/2) * np.cos(phi) - cot(theta_plus/2)

def denom2(theta, phi):
    return cot(theta/2) * np.cos(phi) - np.tan(theta_plus/2)

def phase(theta, phi):
    phase = np.arctan2(num(theta, phi), denom(theta, phi)) - np.arctan2(num(theta, phi), denom2(theta, phi))
    return phase


#coordinate system

theta, phi = np.linspace(0,  np.pi, 500), np.linspace(0,2 * np.pi, 500)
THETA, PHI = np.meshgrid(theta, phi)
X, Y, Z = sph2cart(THETA, PHI)


fcolors = phase(THETA, PHI)
fmax, fmin = fcolors.max(), fcolors.min()
fcolors = (fcolors - fmin)/(fmax - fmin)
colors = cm.seismic(fcolors)



fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev = 10, azim = -20)

p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors = colors, cmap = cm.seismic)



fig.colorbar(p, location = 'left', shrink = 0.8, aspect = 8, label = r'$\theta$')

plt.tight_layout()
plt.title('Initial phase of two vortices')
plt.savefig('J:/Uni - Physik/Master/6. Semester/Masterarbeit/phase angle.pdf', bbox_inches='tight',  dpi = 300)
plt.show()
