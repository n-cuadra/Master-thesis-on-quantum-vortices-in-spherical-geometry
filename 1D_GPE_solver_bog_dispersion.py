import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm



#simulation parameters
#############################


L = 10.0                              #length of lattice
N = 1024                           #number of lattice points
x = np.linspace(-L/2, L/2, N)              #lattice
dx = x[1] - x[0]                    #lattice spacing   
dt = 1e-4                           #timestep
k = 2 * np.pi * np.fft.fftfreq(N, d  = dx)    #lattice in fourier space, positive k first, then negative k
k_ord = np.fft.fftshift(k) #ordered k values
T = .1
steps = int(T // dt)
omega = 2 * np.pi * np.fft.fftfreq(steps, d = dt)
omega_ord = np.fft.fftshift(omega)
g = 1.0                            #nonlinear coefficient
epsilon = 0.001 


#function to compute second derivative

def second_deriv(psi):
    secondderiv = np.zeros(N, dtype=np.complex128)
    for i in range(1, N - 1):
        secondderiv[i] = (psi[i+1] + psi[i-1] - 2. * psi[i]) / (dx * dx)
    secondderiv[0]  = (psi[1] + psi[-1] - 2. * psi[0]) / (dx * dx)#the first point on the grid is next to the last one
    secondderiv[-1] = (psi[1] + psi[-2] - 2. * psi[-1]) / (dx * dx) #the last point on the grid is next to the first one
    return secondderiv


def perturbation():
    return epsilon * np.random.random()
    

#initalize wave function 

psi = np.zeros(N, dtype = np.complex128)   


for i in range(N -1):
    psi[i] = 1. + perturbation()
    

#timestep with Runge Kutta 4 

def rk4_step(psi, dt):
    k1 = 1.0j * 0.5 * second_deriv(psi) - 1.0j * g * np.abs(psi)**2 * psi
    k2 = 1.0j * 0.5 * second_deriv(psi + dt * k1 * 0.5) - 1.0j * g * np.abs(psi + dt * k1 * 0.5)**2 * (psi + dt * k1 * 0.5)
    k3 = 1.0j * 0.5 * second_deriv(psi + dt * k2 * 0.5) - 1.0j * g * np.abs(psi + dt * k2 * 0.5)**2 * (psi + dt * k2 * 0.5)
    k4 = 1.0j * 0.5 * second_deriv(psi + dt * k3) - 1.0j * g * np.abs(psi + dt * k3)**2 * (psi + dt * k3)
    
    psi += (k1 + 2. * k2 + 2. * k3 + k4) * dt / 6.
    return psi


#run simulation


psi_x_t = np.zeros((steps, N), dtype = np.complex128) #initialize array in which the wavefunction as a function of x AND t will be stored

for i in range(steps):
    psi_x_t[i, :] = psi
    psi = rk4_step(psi, dt)

    
psi_k_t = np.fft.ifft(psi_x_t, axis = 1)
psi_k_omega = np.fft.fft(psi_k_t, axis = 0)
psi_k_omega = np.fft.fftshift(psi_k_omega)
    
#plot function
data = np.log(np.abs(psi_k_omega))


  
    
plt.imshow(data, cmap = cm.seismic, extent=[k_ord.min(), k_ord.max(), omega_ord.min(), omega_ord.max()],
           aspect='auto', origin='lower')    
    
plt.colorbar()
plt.xlabel('k')
plt.ylabel(r'$\omega$')
plt.title('Dispersion Relation')
plt.show()   
    
    