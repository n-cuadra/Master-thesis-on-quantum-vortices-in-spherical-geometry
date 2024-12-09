import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, bicgstab


N = 256 
L = 20 
x = np.linspace(0, L, N, endpoint=False)
dx = L / N
k = 2 * np.pi * np.fft.fftfreq(N, d = dx)

sech = 1 / np.cosh(x - L / 2)

#psi0 = np.ones(shape = np.shape(x), dtype = np.float64)
psi0 = 0.5 / np.cosh(x - (L/2))

def dx2(psi):
    deriv = np.zeros(shape = np.shape(psi), dtype = np.float64)
    deriv[0] = (psi[1] - 2 * psi[0] + psi[-1]) / dx**2
    deriv[-1] = (psi[0] - 2 * psi[-1] + psi[-2]) / dx**2
    for i in range(1, N - 1):
        deriv[i] = (psi[i + 1] - 2 * psi[i] + psi[i - 1]) / dx**2
    return deriv
        
def matvec(v, psi):
    result = np.real(np.fft.ifft(-k**2 * np.fft.fft(v))) + 6 * psi**2 * v - v
    return result

def Functional(psi):
    F = np.real(np.fft.ifft(-k**2 * np.fft.fft(psi))) + 2 * psi**3 - psi
    return F

def NR(psi, epsilon, counter = 0):
    b = Functional(psi)
    norm = np.abs(np.sum(b * dx))
    
    print(counter)
    print(norm)
    
    if (norm < epsilon):
        return psi
    
    
    A = LinearOperator(shape = (N, N), dtype = np.float64, matvec = lambda v : matvec(v, psi))
    
    deltapsi, info = bicgstab(A, b, rtol = 1e-5)
    
    if (info != 0):
        print('Linear solver did not converge. Info: ' + str(info) + '. Counter: ' + str(counter + 1))
        return None
    
    psinew = psi - deltapsi
    
    return NR(psinew, epsilon, counter + 1)


psi = NR(psi0, epsilon = 1e-7)

plt.plot(x, sech, linestyle = '--', label = 'sech')
plt.plot(x, psi, label = 'psi')
#plt.plot(x, Functional(psi), label = 'F')
plt.legend()
