import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh

from cartopy import crs
from matplotlib import cm


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


#coordinate system

theta, phi = np.linspace(0,  np.pi, N, endpoint = False), np.linspace(0, 2 * np.pi, 2*N, endpoint= False)  #rectangular grid with polar angle theta and azimuthal angle phi
dangle = np.pi / N  #grid spacing


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
    return cot(theta/2) * np.cos(phi) + cot(theta_plus/2)

def denom2(theta, phi):
    return cot(theta/2) * np.cos(phi) + np.tan(theta_plus/2)

def phase(theta, phi):
    phase = np.arctan2(num(theta, phi), denom(theta, phi)) - np.arctan2(num(theta, phi), denom2(theta, phi))
    return phase


#define initial density 

def density(theta, phi):
    sigma = 1./24.
    density = dens - np.exp(- (theta - theta_plus)**2 / (2 * sigma **2 )) * np.exp(- (phi - np.pi)**2 / (2 * sigma**2)) - np.exp(- (theta - np.pi + theta_plus)**2 / (2 * sigma **2 )) * np.exp(- (phi - np.pi)**2 / (2 * sigma**2))
    
    return density
    
#define norm function

def get_norm(psi):
    norm = 0
    for i in range(N):
        for j in range(2*N):
            norm += np.sin(theta[i]) * dangle**2 * abs(psi[i,j])**2
    return norm


#define derivative wrt azimuthal coordinate

def deriv_phi(psi):
    sh_coeffs = pysh.expand.SHExpandDHC(griddh = psi, norm = 4, sampling = 2) #expand sh coefficients of psi
    len_m = np.size(sh_coeffs, axis = 2) 
    for m in range(len_m):
        for i in range(2):
            sh_coeffs[i,:,m] *= 1.0j * m * (-1.)**i  #spherical harmonics are eigenfunctions of del_phi, so need only to multiply coefficients with i*m to perform derivative
    psi = pysh.expand.MakeGridDHC(sh_coeffs, norm = 4, sampling = 2, extend = False) #back to real space, with the gridded data of the modified coefficients 
    return psi

#define angular Laplacian

def Laplacian(psi):
    sh_coeffs = pysh.expand.SHExpandDHC(griddh = psi, norm = 4, sampling = 2) #expand sh coefficients of psi
    len_l = np.size(sh_coeffs, axis = 1) 
    for l in range(len_l):
        sh_coeffs[:,l,:] *= - l * (l + 1) #spherical harmonics are eigenfunctions of angular Laplacian, so need only to multiply coefficients with -l(l+1) to perform Laplacian
    psi = pysh.expand.MakeGridDHC(sh_coeffs, norm = 4, sampling = 2, extend = False) #back to real space, with the gridded data of the modified coefficients 
    return psi

#calculate energy of condensate (conserved quantitiy)

def energy(psi):
    energy = 0
    Laplace_psi = Laplacian(psi) #array of Laplacian of wavefunction
    deriv_phi_psi = deriv_phi(psi) #array of derivative wrt azimuthal angle of wavefunction
    conj_psi = np.conjugate(psi) #array of complex conjugate of wavefunction
    for i in range(N):
        for j in range(2*N):
            energy += dangle**2 * np.sin(theta[i]) * (  - alpha / 2 * conj_psi[i,j] * Laplace_psi[i,j] + g * abs(psi[i,j])**4 - 1.0j * conj_psi[i,j] * deriv_phi_psi[i,j]  ) #compute the hamiltonian
    energy = np.real(energy)
    return energy
    
#calculate angular momentum of condensate (another conserved quantity).

def ang_momentum(psi):
    mom = 0
    deriv_phi_psi = deriv_phi(psi) #array of derivative wrt azimuthal angle of wavefunction
    conj_psi = np.conjugate(psi) #array of complex conjugate of wavefunction
    for i in range(N):
        for j in range(2*N):
            mom += - 1.0j * dangle**2 * np.sin(theta[i]) * conj_psi[i,j] * deriv_phi_psi[i,j] #compute the angular momentum integral
    mom = np.real(mom)
    return mom
    
    
#initialize wavefunction

psi = np.zeros(shape = (N, 2*N), dtype = np.complex128)
for i in range(N):
    for j in range(2*N):
        psi[i,j] = density(theta[i], phi[j]) * np.exp(1.0j * phase(theta[i], phi[j]))
        
norm = get_norm(psi)
psi  = psi / np.sqrt(norm)

print(get_norm(psi))
print(energy(psi))    
print(ang_momentum(psi)) 

sh_coeffs = pysh.expand.SHExpandDHC(griddh = psi, norm = 4, sampling = 2) #calculate array of initial spherical harmonic expansion coefficients 



#one timestep with split stepping method

def timestep(sh_coeffs, dt):
    len_l = np.size(sh_coeffs, axis = 1)  #size of sh_coeffs array in l and m indices
    for l in range(len_l):
        for m in range(len_l):
            for i in range(2):
                sh_coeffs[i,l,m] *= np.exp(- 1.0j * alpha * l * (l + 1) * dt / 2) * np.exp(- 1.0j * m * dt * (-1.)**i)  #timestep of kinetic and rotating term
    grid = pysh.expand.MakeGridDHC(sh_coeffs, norm = 4, sampling = 2, extend = False) #create gridded data in (N, 2*N) array from coeffs
    grid *= np.exp(1.0j * g * dt * abs(grid)**2) #timestep of nonlinear term
    sh_coeffs = pysh.expand.SHExpandDHC(griddh = grid, norm = 4, sampling = 2) #calculate expansion coefficients from the gridded data
    return sh_coeffs





for q in range(10000): 
    sh_coeffs = timestep(sh_coeffs, dt)
    

data = pysh.expand.MakeGridDHC(sh_coeffs, norm = 4, sampling = 2, extend = False)
print(get_norm(data))
print(energy(data))  
print(ang_momentum(data))     
data = abs(data)**2
coeffs = pysh.expand.SHExpandDH(griddh = data, norm = 4, sampling = 2)

mycmap = cm.seismic
myprojection = crs.Mollweide(central_longitude=180.)

clm = pysh.SHCoeffs.from_array(coeffs, normalization='ortho', lmax = lmax)
grid = clm.expand()
fig, ax = grid.plot(cmap = mycmap, colorbar = 'right',  show = False)

#plt.savefig('J:/Uni - Physik/Master/6. Semester/Masterarbeit/Media/10000.pdf', dpi = 300)






