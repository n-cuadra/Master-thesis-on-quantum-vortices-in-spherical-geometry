import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation




#simulation parameters
#############################


L = 10.0                              #length of lattice
N = 1024                           #number of lattice points
x = np.linspace(0, L, N, endpoint=False)              #lattice
dx = L / N                    #lattice spacing   
dt = 1e-5                            #timestep
sigma = 1./12.                             #gaussian width
k = 2 * np.pi * np.fft.fftfreq(N, d  = dx) / L    #lattice in fourier space, positive k first, then negative k
g = 5.0                            #nonlinear coefficient
v = 0.3 #soliton velocity

    
#function to compute norm

def get_norm(psi):
    norm = 0
    for i in range(len(psi)):
        norm += dx * np.abs(psi[i])**2
    return norm

#function to compute second derivative

def second_deriv(psi):
    secondderiv = np.zeros(N, dtype=np.complex128)
    for i in range(1, N - 1):
        secondderiv[i] = (psi[i+1] + psi[i-1] - 2. * psi[i]) / (dx * dx)
    secondderiv[0]  = (psi[1] + psi[-1] - 2. * psi[0]) / (dx * dx)  
    secondderiv[-1] = (psi[0] + psi[-2] - 2. * psi[-1]) / (dx * dx)   
    return secondderiv

#function to compute energy

def get_energy(psi):
    energy = 0
    secondderiv = second_deriv(psi)
    psi_conj = np.conjugate(psi)
    for i in range(N):
        energy += dx * (- psi_conj[i] * secondderiv[i] / 2 + g * np.abs(psi[i])**4)
    energy = np.real(energy)
    return energy

#function to initialize condensate as gaussian

def gaussian(x, x_0, sigma):
    gaussian = 1/(np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - x_0)**2 / 2 / sigma**2) 
    return gaussian

#function to initialize condensate as dark soliton

def dark_soliton(x, x0, v):
    dark_sol = 1.0j * v + np.sqrt(1 - v**2) * np.tanh( (x - x0)  * np.sqrt(1 - v**2) )
    return dark_sol



#initalize wave function 

psi = np.zeros(N, dtype = np.complex128)   


for i in range(N):
    #psi[i] = gaussian(x[i], x[0] + L/2, sigma) #gaussian
    psi[i] = dark_soliton(x[i], L/2,  v) 
    


norm = get_norm(psi)

psi = psi/np.sqrt(norm)




#split stepping

def split_step(psi, dt):
    psi = np.fft.fft(psi) #go to fourier space
    
    psi *= np.exp(-1.0j * k**2 * dt * 0.5)  #perform kinetic step there
    
    psi = np.fft.ifft(psi) #back to real space
    
    psi *= np.exp(1.0j * g * np.abs(psi)**2 * dt) #perform nonlinear step here
    
    psi[-1] = psi[0]
    
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


#for i in range(10000):
    #psi = rk4_step(psi, dt)
    
#plot final function
    
energy = get_energy(psi)
norm = get_norm(psi)

textxpos = x[0] + 0.05
textypos = ( (np.abs(psi)**2).max() - (np.abs(psi)**2).min() ) / 10


fig, ax = plt.subplots()
line, = ax.plot(x, np.abs(psi)**2)
steptext = ax.text(textxpos, -1.5 * textypos, 'Integration step: 0')
energytext = ax.text(textxpos ,  -2.5 * textypos , 'Energy: ' + str(energy))
normtext = ax.text(textxpos ,  -3.5 * textypos , 'Norm: ' + str(norm))
plt.xlim(x[0],x[-1])
plt.ylim((np.abs(psi).min())**2, (np.abs(psi).max())**2)
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
ani.save('J:/Uni - Physik/Master/6. Semester/Masterarbeit/Media/1D_GPE_split_stepping_solitons.mp4',  dpi = 300)


plt.show()
