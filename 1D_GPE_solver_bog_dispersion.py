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
    

#timestep with split stepping

def split_step(psi, dt):
    psi = psi[:-1] #cut off the last point which is the first point
    psi_k = np.fft.fft(psi) #go to fourier space
    psi_k *= np.exp(-1.0j * k**2 * dt * 0.5)  #perform kinetic step there
    psi = np.fft.ifft(psi_k) #back to real space
    psi = np.append(psi, psi[0]) #append the first value at the end again
    psi *= np.exp(1.0j * g * np.abs(psi)**2 * dt) #perform nonlinear step here
    return psi


#run simulation


psi_x_t = np.zeros((steps, N), dtype = np.complex128) #initialize array in which the wavefunction as a function of x AND t will be stored

for i in range(steps):
    psi_x_t[i, :] = psi
    psi = split_step(psi, dt)

    
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

    
    