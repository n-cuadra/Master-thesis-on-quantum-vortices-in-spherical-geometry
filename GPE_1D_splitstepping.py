import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation




#simulation parameters
#############################


L = 1.0                              #length of lattice
N = 256                             #number of lattice points
x = np.linspace(-L/2, L/2, N, endpoint=False)              #lattice
dx = L / N                    #lattice spacing   
dt = 1e-5                             #timestep
psi = np.zeros(N, dtype=np.complex128)                   #initialized wave function
sigma = 1./12.                             #gaussian width
k = 2 * np.pi * np.fft.fftfreq(N, d  = dx)  / L    #lattice in fourier space, positive k first, then negative k
g = 1.0                            #nonlinear coefficient
v = 10. #soliton velocity

    
#function to compute norm

def get_norm(psi):
    norm = 0
    for i in range(len(psi)):
        norm += dx * abs(psi[i])**2
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

def energy(psi):
    energy = 0
    secondderiv = second_deriv(psi)
    psi_conj = np.conjugate(psi)
    for i in range(N):
        energy += dx * (- psi_conj[i] * secondderiv[i] / 2 + g * abs(psi[i])**4)
    energy = np.real(energy)
    return energy

#initalize function as a gaussian

psi = 1/(np.sqrt(2 * np.pi) * sigma) * np.exp(-x**2 / 2 / sigma**2) 


#initalize function as a soliton

#psi = 1/np.cosh(10*x)

norm = get_norm(psi)

psi = psi/np.sqrt(norm)

#linear time step 

def linear_timestep(psi, dt):
    psi = np.fft.fft(psi) #go to fourier space
    
    psi *= np.exp(-1.0j * k**2 * dt / 2) #in fourier space, do the timestep corresponding to kinetic term of GPE
    
    psi = np.fft.ifft(psi) #back to real space
    
    #psi[-1] = psi[0]  #periodic boundary condition
    
    return psi


#nonlinear timestep

def nonlinear_timestep(psi, dt):
    psi *= np.exp(1.0j * g * abs(psi)**2 * dt) #perform timestep for nonlinear term in GPE
    
    #psi[-1] = psi[0] #periodic boundary conditions
    
    return psi


#split stepping

def split_step(psi, dt):
    psinew1 = linear_timestep(psi, dt)
    psinew2 = nonlinear_timestep(psinew1, dt)
    return psinew2
    

#plot function


for i in range(20000):
    psi = split_step(psi, dt)
    
energy = energy(psi)
norm = get_norm(psi)

fig, ax = plt.subplots()
line, = ax.plot(x, abs(psi)**2)
ax.text(- L/2 + 0.05 ,  -1.5 , 'Energy: ' + str(energy))
ax.text(- L/2 + 0.05 ,  -2 , 'Norm: ' + str(norm))
plt.xlim(-L/2,L/2)
plt.ylim((abs(psi).min())**2, (abs(psi).max())**2)
plt.xlabel(r'x')
plt.ylabel(r'$|\Psi|^2$')
plt.title('Solitons in the 1D GPE')
#plt.legend()



#animation

steps_per_frame = 4


def animate(i):
    global psi
    for q in range(steps_per_frame):
        psinew = split_step(psi, dt)
        psi = psinew
    line.set_ydata(abs(psi)**2)
    #steptext.set_text('Integration step: %s'%(i * steps_per_frame))
    return line, 


def init():
    return line,
 
# run animation
 
#ani = animation.FuncAnimation(fig, animate, np.arange(1, 1620), init_func=init, interval=25)

 
#FFwriter=animation.FFMpegWriter(fps=60, extra_args=['-vcodec', 'libx264'])
#ani.save('J:/Uni - Physik/Master/6. Semester/Masterarbeit/Media/1D_GPE_split_stepping_gaussian.mp4', dpi = 300)


plt.show()
