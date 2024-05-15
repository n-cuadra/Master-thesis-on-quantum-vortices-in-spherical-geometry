import numpy as np
import matplotlib.pyplot as plt


#parameters
N = 512 #grid points
R = 1.0 #radius of sphere
theta_plus = np.pi/3 #position of vortex on the upper hemisphere
dt = 1.e-5  #time step
omega = 2 * np.pi * 10 #frequency in Hz
alpha = 5 / omega # linear coefficient
g = -1. / omega   #nonlinear coefficient
dens = 1. #condensate density far from vortices

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
    density = dens -  np.exp(- (theta - theta_plus)**2/100  ) * np.exp(- phi**2) - np.exp(- (theta + np.pi - theta_plus)**2/100 ) * np.exp(- phi**2 ) 
    return density
            
def d(theta):
    sigma = 1.
    density = dens -   np.exp(-(theta - theta_plus)**2 /(2* sigma**2 ) ) -  np.exp(-(theta - np.pi + theta_plus)**2 /(2* sigma**2 ) ) 
    return density  
    
#define norm function

def get_norm(psi):
    norm = 0
    for i in range(N//2):
        for j in range(N):
            norm += np.sin(theta[i]) * dangle**2 * abs(psi[i,j])**2
    return norm[0]
    

#coordinate system

theta, phi = np.linspace(0,  np.pi, N//2, endpoint = False), np.linspace(0, 2 * np.pi, N, endpoint= False)
dangle = np.diff(theta)
THETA, PHI = np.meshgrid(theta, phi)
X, Y, Z = sph2cart(THETA, PHI)



DENS = density(THETA, PHI)
#PSI = np.sqrt(density(THETA, PHI)) * np.exp(1.0j * phase(THETA, PHI))



#define all needed derivatives

def deriv_theta(psi):
    deriv = np.zeros(shape = np.ma.shape(psi), dtype = np.complex128)
    for i in range(1, N//2 - 1):
        deriv[i,:] = (psi[i+1,:] - psi[i,:] ) / dangle
    deriv[0,:] = (psi[1,:] - psi[0,:]) / dangle
    deriv[-1,:] =  (psi[-1,:] - psi[-2,:]) / dangle
    return deriv

def deriv_phi(psi):
    deriv = np.zeros(shape = np.ma.shape(psi), dtype = np.complex128)
    for j in range(N-1):
        deriv[:,j] = (psi[:,j+1] - psi[:,j]) / dangle
    deriv[:,-1] = deriv[:,0]
    return deriv
            
def second_deriv_phi(psi):
    secondderiv = np.zeros(shape = np.ma.shape(psi), dtype = np.complex128)
    for j in range(1, N - 1):
        secondderiv[:,j] = (psi[:,j+1] + psi[:,j-1] - 2. * psi[:,j]) / (dangle * dangle)
        
    secondderiv[:,0] = (psi[:,1] + psi[:,N-1] - 2. * psi[0]) / (dangle * dangle)  
    
    secondderiv[:,N-1] = (psi[:,0] + psi[:,N-2] - 2. * psi[:,N-1]) / (dangle * dangle)
    return secondderiv

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

#single runge kutta 4 timestep

def rk4(psi, dt):
    k1 = -(1. / (2.j)) * Laplacian(psi)
    k2 = -(1. / (2.j)) * Laplacian(psi + 0.5 * dt * k1)
    k3 = -(1. / (2.j)) * Laplacian(psi + 0.5 * dt * k2)
    k4 = -(1. / (2.j)) * Laplacian(psi + dt * k3)
     
    psinew = psi + alpha / 2. * dt * 1. / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return psinew
    

#nonlinear time step

def nonlinear_timestep(psi, dt):
    psi_timestep = np.zeros(shape = np.ma.shape(psi), dtype = np.complex64)
    for i in range(N//2):
        for j in range(N):
            psi_timestep[i,j] = psi[i,j] * np.exp(1.0j * g * abs(psi[i,j])**2 * dt)
    norm = get_norm(psi_timestep)
    psi_final = psi_timestep/np.sqrt(norm)
    return psi_final


#split stepping

def split_step(psi, dt):
    psinew = rk4(psi, dt)
    psinew2 = nonlinear_timestep(psinew, dt)
    return psinew2
    

#plot function


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(THETA, PHI, DENS)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\phi$')
plt.show()

