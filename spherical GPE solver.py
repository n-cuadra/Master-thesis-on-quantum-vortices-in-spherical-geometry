import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pyshtools as pysh

#parameters
N = 512 #grid points
R = 1.0 #radius of sphere
theta_plus = np.pi/3 #position of vortex on the upper hemisphere
dt = 1.e-5  #time step
omega = 2 * np.pi * 10 #frequency in Hz
alpha = 5 / omega # linear coefficient
g = -1. / omega   #nonlinear coefficient
dens = 1. #condensate density far from vortices
lmax = N//2 - 1 #maximum degree of spherical harmonics

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

#define initial phase angle 

def num(theta, phi):
    return cot(theta/2) * np.sin(phi)

def denom(theta, phi):
    return cot(theta/2) * np.cos(phi) - cot(theta_plus/2)

def denom2(theta, phi):
    return cot(theta/2) * np.cos(phi) - np.tan(theta_plus/2)

def phase(theta, phi):
    phase = np.arctan2(num(theta, phi), denom(theta, phi)) - np.arctan2(num(theta, phi), denom2(theta, phi))
    return phase

#define initial density 

def density(theta, phi):
    sigma = 1./12.
    density = dens - np.exp(- (theta - theta_plus)**2 / (2 * sigma **2 )) * np.exp(- (phi - np.pi)**2 / (2 * sigma**2)) - np.exp(- (theta - np.pi + theta_plus)**2 / (2 * sigma **2 )) * np.exp(- (phi - np.pi)**2 / (2 * sigma**2))
    
    return density
    
#define norm function

def get_norm(psi):
    norm = 0
    for i in range(N//2):
        for j in range(N):
            norm += np.sin(theta[i]) * dangle**2 * abs(psi[i,j])**2
    return norm[0]
    

#coordinate system

theta, phi = np.linspace(0,  np.pi, N, endpoint = False), np.linspace(0, 2 * np.pi, 2*N, endpoint= False)  #rectangular grid with polar angle theta and azimuthal angle phi
dangle = np.diff(theta)  #grid spacing
THETA, PHI = np.meshgrid(theta, phi) #meshgrid of the above
X, Y, Z = sph2cart(THETA, PHI) #meshgrid in cartesian coordinates



#initialize wavefunction
psi = np.zeros(shape = (N, 2*N), dtype = np.complex128)
for i in range(N):
    for j in range(2*N):
        psi[i,j] = density(theta[i], phi[j]) * np.exp(1.0j * phase(theta[i], phi[j]))
        
   
sh_coeffs = pysh.expand.SHExpandDHC(griddh = psi, norm = 4, sampling = 2) #calculate array of initial spherical harmonic expansion coefficients 

def timestep(sh_coeffs, dt):
    len_l = np.size(sh_coeffs, axis = 1)  #size of coeffs in l and m indices
    for l in range(len_l):
        sh_coeffs[:,l,:] *= np.exp(- 1.0j * alpha * l * (l + 1) * dt / 2)      #timestep of kinetic term
    for m in range(len_l):
        for i in range(2):
            sh_coeffs[i,:,m] *= np.exp(- 1.0j * m * dt * (-1.)**i)  #timestep of rotating term
    clm = pysh.SHCoeffs.from_array(sh_coeffs, normalization= 'ortho', lmax = lmax) #create SHCoeffs instance from array of coefficients
    grid = clm.expand()  #create SHGrid instance from SHCoeff instance
    data = grid.data  #create gridded data from SHGrid instance
    data = data * np.exp(1.0j * g * dt * abs(data)**2) #perform nonlinear timestep
    sh_coeffs = pysh.expand.SHExpandDHC(griddh = data, norm = 4, sampling = 2) #calculate expansion coefficients from the gridded data
    return sh_coeffs


sh_coeffs = timestep(sh_coeffs, dt)



clm = pysh.SHCoeffs.from_array(sh_coeffs, normalization= 'ortho', lmax = lmax)




grid = clm.expand()


fig, ax = grid.plot(show=False, colorbar = 'right')


































































'''

#define all needed derivatives

#derivative in theta direction
def deriv_theta(psi):
    deriv = np.zeros(shape = np.ma.shape(psi), dtype = np.complex128)
    for i in range(1, N//2 - 1):
        deriv[i,:] = (psi[i+1,:] - psi[i,:] ) / dangle
    deriv[0,:] = (psi[1,:] - psi[0,:]) / dangle
    deriv[-1,:] =  (psi[-1,:] - psi[-2,:]) / dangle
    return deriv

#derivative in phi direction
def deriv_phi(psi):
    deriv = np.zeros(shape = np.ma.shape(psi), dtype = np.complex128)
    for j in range(N-1):
        deriv[:,j] = (psi[:,j+1] - psi[:,j]) / dangle
    deriv[:,-1] = deriv[:,0]
    return deriv
         
#second derivative in phi direction   
def second_deriv_phi(psi):
    secondderiv = np.zeros(shape = np.ma.shape(psi), dtype = np.complex128)
    
    for j in range(1, N - 1):
        secondderiv[:,j] = (psi[:,j+1] + psi[:,j-1] - 2. * psi[:,j]) / (dangle * dangle)
        
    secondderiv[:,0] = (psi[:,1] + psi[:,N-1] - 2. * psi[0]) / (dangle * dangle)  
    
    secondderiv[:,N-1] = (psi[:,0] + psi[:,N-2] - 2. * psi[:,N-1]) / (dangle * dangle)
    
    return secondderiv

#Laplacian in spherical coordinates. The Laplacian has two parts that are summed together.
#The left part (lhs) contains the derivative wrt theta, the right (rhs) one contains the derivative wrt phi

def Laplacian(psi):
    rhs = second_deriv_phi(psi)
    for i in range(N//2):
        rhs[i,:] = rhs[i,:] / ( np.sin(theta[i])**2 )
    lhs = deriv_theta(psi)
    for i in range(N//2):
        lhs[i,:] = lhs[i,:] * np.sin(theta[i])
    lhs = deriv_theta(lhs)
    for i in range(N//2):
        lhs[i,:] = lhs[i,:] / np.sin(theta[i])
    result = lhs + rhs 
    norm = get_norm(result)
    result = result/np.sqrt(norm) 
    return result


    

#plot initial density in rectangular coordinates
DENS = density(THETA, PHI)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(THETA, PHI, DENS, cmap = cm.jet)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\phi$')
ax.set_zlabel(r'$\rho$')
ax.set_zticks([ 0, 0.5, 1])
plt.tight_layout()


plt.savefig('J:/Uni - Physik/Master/6. Semester/Masterarbeit/Media/initial density.pdf', bbox_inches='tight',  dpi = 300)
plt.show()


#plot initital density on the sphere


colors = cm.seismic(DENS)



fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev = 10, azim = -200)

p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors = colors, cmap = cm.seismic)



fig.colorbar(p, location = 'left', shrink = 0.8, aspect = 8, label = r'$\rho$')

plt.tight_layout()
plt.title('Initial density of two vortices')
plt.savefig('J:/Uni - Physik/Master/6. Semester/Masterarbeit/Media/initial density on the sphere.pdf', bbox_inches='tight',  dpi = 300)
plt.show()
'''
fg = 2