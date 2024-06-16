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
steps = N
omega = 2 * np.pi * np.fft.fftfreq(steps, d = dt)
omega_ord = np.fft.fftshift(omega)
g = 1.0                            #nonlinear coefficient
epsilon = 0.01 


#function to compute second derivative

def second_deriv(psi):
    secondderiv = np.zeros(N, dtype=np.complex128)
    for i in range(1, N - 1):
        secondderiv[i] = (psi[i+1] + psi[i-1] - 2. * psi[i]) / (dx * dx)
    secondderiv[0]  = (psi[1] + psi[-1] - 2. * psi[0]) / (dx * dx)#the first point on the grid is next to the last one
    secondderiv[-1] = (psi[1] + psi[-2] - 2. * psi[-1]) / (dx * dx) #the last point on the grid is next to the first one
    return secondderiv


def perturbation(x):
    return epsilon * (np.sin(x) + np.cos(x))
    

#initalize wave function 

psi = np.zeros(N, dtype = np.complex128)   

#function of bogoliubov dispersion

def bogoliubov_dispersion(k, g):
    disp = np.sqrt(k**2 / 2 * (k**2 / 2 + 2 * g))
    return disp


for i in range(N):
    psi[i] = 1. + perturbation(x[i])
    

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

    
psi_k_t = np.fft.fft(psi_x_t, axis = 1)
psi_k_omega = np.fft.fft(psi_k_t, axis = 0)
psi_k_omega = np.fft.fftshift(psi_k_omega)

    
#plot function


k_analytic = np.linspace(k_ord.min(), k_ord.max(), 1000)
omega_analytic = bogoliubov_dispersion(k_analytic, g)


data = np.log( np.abs(psi_k_omega) + 1e-8)

plt.plot(k_analytic, omega_analytic, 'w--', lw = 0.8)
plt.plot(k_analytic, -omega_analytic, 'w--', lw = 0.8) 
  
    
mappable = plt.imshow(data, cmap = cm.inferno, extent=[k_ord.min(), k_ord.max(), omega_ord.min(), omega_ord.max()],
           aspect='auto', vmin = np.min(data), vmax = np.max(data) )    
    
plt.colorbar(mappable, label = r'$\log |\Psi(k, \omega)|$')

plt.xlabel('k')
plt.ylabel(r'$\omega$')
plt.title('Dispersion Relation')



plt.show()   

    
    