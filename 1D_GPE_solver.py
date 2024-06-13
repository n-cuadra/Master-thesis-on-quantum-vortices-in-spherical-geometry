import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib import cm



#simulation parameters
#############################


L = 10.0                              #length of lattice
N = 1024                           #number of lattice points
x = np.linspace(-L/2, L/2, N)              #lattice
dx = x[1] - x[0]                    #lattice spacing   
dt = 1e-4                           #timestep
sigma = 1./3.                             #gaussian width
k = 2 * np.pi * np.fft.fftfreq(N - 1, d  = dx)    #lattice in fourier space, positive k first, then negative k
g = 1.0                            #nonlinear coefficient
v = 0.5 #soliton velocity
width = 0.4 #soliton width
epsilon = 1e-4  #strength of bogoliubov excitation

    
#function to compute norm

def get_norm(psi):
    norm = 0
    for i in range(len(psi) - 1):
        norm += dx * np.abs(psi[i])**2
    return norm

#function to compute second derivative

def second_deriv(psi):
    secondderiv = np.zeros(N, dtype=np.complex128)
    for i in range(1, N - 1):
        secondderiv[i] = (psi[i+1] + psi[i-1] - 2. * psi[i]) / (dx * dx)
    secondderiv[0]  = (psi[1] + psi[-2] - 2. * psi[0]) / (dx * dx)#the first point on the grid is next to the second last one
    secondderiv[-1] = secondderiv[0] #the last point on the grid is the same as the first one
    return secondderiv

#function to compute energy

def get_energy(psi):
    energy = 0
    secondderiv = second_deriv(psi)
    psi_conj = np.conjugate(psi)
    for i in range(len(psi) - 1):
        energy += dx * (- psi_conj[i] * secondderiv[i] / 2 + g * np.abs(psi[i])**4)
    energy = np.real(energy)
    return energy

#function to initialize condensate as gaussian

def gaussian(x, x_0, sigma):
    gaussian = 1/(np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - x_0)**2 / 2 / sigma**2) 
    return gaussian

#function to initialize condensate as dark soliton

def dark_soliton(x, x0, v):
    dark_sol = 1.0j * v + np.sqrt(1 - v**2) * np.tanh( (x - x0)  * np.sqrt(1 - v**2))
    return dark_sol

#define bogoliubov type exciations on a uniform density
#this is to test whether the code accurately reproduces the known bogoliubov dispersion

def bog_exc(x, k, epsilon):
    return 1 + epsilon * ( np.exp(1.0j * k * x) + np.exp(-1.0j * k * x) )

#initalize wave function 

psi = np.zeros(N, dtype = np.complex128)   


for i in range(N):
    #psi[i] = gaussian(x[i], x[0] + L/2, sigma) #gaussian
    psi[i] = dark_soliton(x[i], -L/4.,  v) * dark_soliton(x[i], L/4., -v) #two dark solitons
    #psi[i] = bog_exc(x[i], k_ord, epsilon) #bogoliubov excitations
    





#timestep with split stepping

def split_step(psi, dt):
    psi = psi[:-1] #cut off the last point which is the first point
    psi_k = np.fft.fft(psi) #go to fourier space
    psi_k *= np.exp(-1.0j * k**2 * dt * 0.5)  #perform kinetic step there
    psi = np.fft.ifft(psi_k) #back to real space
    psi = np.append(psi, psi[0]) #append the first value at the end again
    psi *= np.exp(1.0j * g * np.abs(psi)**2 * dt) #perform nonlinear step here
    return psi

#timestep with Runge Kutta 4 

def rk4_step(psi, dt):
    k1 = 1.0j * 0.5 * second_deriv(psi) - 1.0j * g * np.abs(psi)**2 * psi
    k2 = 1.0j * 0.5 * second_deriv(psi + dt * k1 * 0.5) - 1.0j * g * np.abs(psi + dt * k1 * 0.5)**2 * (psi + dt * k1 * 0.5)
    k3 = 1.0j * 0.5 * second_deriv(psi + dt * k2 * 0.5) - 1.0j * g * np.abs(psi + dt * k2 * 0.5)**2 * (psi + dt * k2 * 0.5)
    k4 = 1.0j * 0.5 * second_deriv(psi + dt * k3) - 1.0j * g * np.abs(psi + dt * k3)**2 * (psi + dt * k3)
    
    psi += (k1 + 2. * k2 + 2. * k3 + k4) * dt / 6.
    return psi



#run simulation
'''
end = 10000
psi_x_t = np.zeros((end//10, N), dtype = np.complex128) #initialize array in which the wavefunction as a function of x AND t will be stored

for i in range(end):
    #if (i % 10 == 0): #every 10 timesteps append the time evolution to the array
        #ind = i // 10
        #psi_x_t[ind, :] = psi
    psi = rk4_step(psi, dt)
   

'''

#plot function

print(psi[0], psi[-1])
    
energy = get_energy(psi)
norm = get_norm(psi)

helperpos = (np.abs(psi).max())**2 - (np.abs(psi).min())**2
ylimmin = (np.abs(psi).min())**2 - helperpos / 20.
ylimmax = (np.abs(psi).max())**2 + helperpos / 20

textstr = '\n'.join((r'$g = $' + str(g), r'$v = $' + str(v)))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)


fig, ax = plt.subplots()
line, = ax.plot(x, np.abs(psi)**2)
ax.text(0.05, 0.5, textstr, fontsize=8, transform = ax.transAxes, bbox=props)
steptext = ax.text(0.05, -0.15, 'Integration step: 0', transform = ax.transAxes,)
energytext = ax.text(0.05,  -0.25, 'Energy: ' + str(energy), transform = ax.transAxes,)
normtext = ax.text(0.05,  -0.35 , 'Norm: ' + str(norm), transform = ax.transAxes,)
plt.xlim(x[0],x[-1])
plt.ylim(ylimmin, ylimmax)
plt.xlabel(r'x')
plt.ylabel(r'$|\Psi|^2$')
plt.tight_layout()
plt.title('1D GPE')




#animation

steps_per_frame = 4


def animate(i):
    global psi
    for q in range(steps_per_frame):
        psi = rk4_step(psi, dt)
    line.set_ydata(np.abs(psi)**2)
    steptext.set_text('Integration step: %s'%(i * steps_per_frame))
    energytext.set_text('Energy: ' + str(get_energy(psi)))
    normtext.set_text('Norm: ' + str(get_norm(psi)))
    return line, 


def init():
    return line,
 
# run animation
 
ani = animation.FuncAnimation(fig, animate, np.arange(1, 1000), init_func=init, interval=25)

 
#FFwriter=animation.FFMpegWriter(fps=60, extra_args=['-vcodec', 'libx264'])
ani.save('J:/Uni - Physik/Master/6. Semester/Masterarbeit/Media/1D_GPE_rk4_solitons.mp4',  dpi = 300)


plt.show()
