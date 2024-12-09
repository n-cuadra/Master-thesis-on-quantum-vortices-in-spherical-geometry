import numpy as np
import cmocean 
import matplotlib.pyplot as plt
import scienceplots
from scipy.sparse.linalg import LinearOperator, bicgstab, lgmres

#set some parameters for plotting

plt.style.use('science')
plt.rcParams.update({'font.size': 7})
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')


#parameters

N = 256
L = 15 * 2 * np.pi
bg_dens = 1000
dt = 1e-5
vel = 0.05

#initialize grid
dx = L / N
x = np.linspace(0, L, N, endpoint = False)
y = np.linspace(0, L, N, endpoint = False)
X, Y = np.meshgrid(x, y)


k = 2 * np.pi * np.fft.fftfreq(N, d = dx)
kx, ky = np.meshgrid(k, k, indexing = 'ij')

def get_particle_number(psi):
    particle_number = np.sum(np.abs(psi)**2) * dx**2
    return particle_number

def IC_Pade(x, y, xv, yv, q):
    a1, a2 = 11./32., 11./384.
    b1, b2 = 1./3., 11./384.
    r = np.sqrt((x - xv)**2 + (y - yv)**2)
    phase = np.arctan2(x - xv, y - yv)
    f  = np.sqrt((r**2 * (a1 + a2 * r**2)) / (1 + b1 * r**2 + b2 * r**4))
    psi = f * np.exp(1j * q * phase)
    return psi


def Laplacian(psi):
    psik = np.fft.fftn(psi)
    psi = np.fft.ifftn(- (kx**2 + ky**2) * psik)
    return psi


def Laplacian_fd(psi):
    laplace = np.zeros(shape = (N, N), dtype = psi.dtype)
    for i in range(1, N - 1):
        laplace[i,:] = (psi[i+1,:] - 2 * psi[i,:] + psi[i-1,:]) / dx**2
        laplace[:,i] = (psi[:,i+1] - 2 * psi[:,i] + psi[:,i-1]) / dx**2
    laplace[0,:] = (psi[1,:] - 2 * psi[0,:] + psi[-1,:]) / dx**2
    laplace[-1,:] = (psi[0,:] - 2 * psi[-1,:] + psi[-2,:]) / dx**2
    laplace[:,0] = (psi[:,1] - 2 * psi[:,0] + psi[:,-1]) / dx**2
    laplace[:,-1] = (psi[:,0] - 2 * psi[:,-1] + psi[:,-2]) / dx**2
    return laplace    
 
def partialy(psi):
    derivative = np.zeros(shape = (N, N), dtype = psi.dtype)
    for i in range(N - 1):
        derivative[:,i] = (psi[:,i+1] - psi[:,i])/dx
    derivative[:,-1] = (psi[:,0] - psi[:,-1])/dx
    return derivative

   
def partialx(psi):
    derivative = np.zeros(shape = (N, N), dtype = psi.dtype)
    for i in range(N - 1):
        derivative[i,:] = (psi[i + 1,:] - psi[i,:])/dx
    derivative[-1,:] = (psi[0,:] - psi[-1,:])/dx
    return derivative

def timestep(psi, dt):
    psi = np.fft.ifft2(np.exp(-1.0j * dt * (kx**2 + ky**2)) * np.fft.fft2(psi))
    psi = psi * np.exp(-1.0j * dt * np.abs(psi)**2)
    return psi

def imaginary_timestep(psi, dt, particle_number):
    psi = psi * np.exp(- 0.5 * dt * np.abs(psi)**2)
    psik = np.fft.fft2(psi)
    psik = psik * np.exp(- dt * (kx**2 + ky**2))
    psi = np.fft.ifft2(psik)
    psi = psi * np.exp(- 0.5 * dt * np.abs(psi)**2)
    psi = np.sqrt(particle_number) * psi / np.sqrt(get_particle_number(psi))
    return psi

def Functional(psi, vel):
    F = - Laplacian(psi) + np.abs(psi)**2 * psi - psi - 2 * vel * np.fft.ifftn(kx * np.fft.fftn(psi))
    return F


def matvecsimple(v, psi, vel):
    deltapsir = v[:N**2]
    deltapsii = v[N**2:]
        
    deltapsi = deltapsir + 1j * deltapsii
    deltapsi_grid = np.reshape(deltapsi, newshape = (N, N), order = 'C')
    
    yy = - Laplacian(deltapsi_grid) + (2 * np.abs(psi)**2 - 1) * deltapsi_grid - vel * np.fft.ifftn(kx * np.fft.fftn(deltapsi_grid))
    yy = np.ravel(yy, order = 'C')
    
    return np.concatenate((np.real(yy), np.imag(yy)))


def matvec(v, psi, vel):
    deltapsir = v[:N**2]
    deltapsii = v[N**2:]
    deltapsir_grid = np.reshape(deltapsir, newshape = (N, N), order = 'C')
    deltapsii_grid = np.reshape(deltapsii, newshape = (N, N), order = 'C')
    
    J11_deltapsir = - Laplacian(deltapsir_grid) + (3 * np.real(psi)**2 + np.imag(psi)**2 - 1) * deltapsir_grid 
    J12_deltapsii = 2 * np.real(psi) * np.imag(psi) * deltapsii_grid - 2 * vel * partialx(deltapsii_grid)
    result1 = np.ravel(J11_deltapsir + J12_deltapsii, order = 'C')
    
    J21_deltapsir = 2 * np.real(psi) * np.imag(psi) * deltapsir_grid + 2 *  vel * partialx(deltapsir_grid)
    J22_deltapsii = - Laplacian(deltapsii_grid) + (3 * np.imag(psi)**2 + np.real(psi)**2 - 1) * deltapsii_grid 
    result2 = np.ravel(J21_deltapsir + J22_deltapsii, order = 'C')
    
    return np.concatenate((result1, result2))


def matveccomplex(v, psi, vel):
    deltapsi = v[:N**2]
    deltapsic = v[N**2:]
    deltapsi_grid = np.reshape(deltapsi, newshape = (N, N), order = 'C')
    deltapsic_grid = np.reshape(deltapsic, newshape = (N, N), order = 'C')
    
    J11_deltapsi = - Laplacian(deltapsi_grid) + (2 * np.abs(psi)**2 - 1) * deltapsi_grid - vel * np.fft.ifftn(kx * np.fft.fftn(deltapsi_grid))
    J12_deltapsic = psi**2 * deltapsic_grid
    result1 = np.ravel(J11_deltapsi + J12_deltapsic, order = 'C')
    
    J21_deltapsi = np.conj(psi)**2 * deltapsi_grid
    J22_deltapsic = - Laplacian(deltapsic_grid) + (2 * np.abs(psi)**2 - 1) * deltapsic_grid + vel * np.fft.ifftn(kx * np.fft.fftn(deltapsic_grid))
    result2 = np.ravel(J21_deltapsi + J22_deltapsic, order = 'C')
    
    return np.concatenate((result1, result2))


def NRcomplex(psi, vel, tol_NR, tol_ls, counter = 0):
    print(counter)
    F = Functional(psi, vel)
    norm = np.sqrt(np.sum(np.abs(F)**2 * dx**2))
    print(norm)
    
    if (norm < tol_NR):
        print('Convergence achieved after ' + str(counter) + ' iterations!')
        return psi
    
    F_flat = np.ravel(F, order = 'C')
    b = np.concatenate((F_flat, np.conj(F_flat)))
    
    A = LinearOperator(shape = (2*N**2, 2*N**2), matvec = lambda v: matveccomplex(v, psi, vel), dtype = np.complex128)
    
    def callback(xk):
        residual = np.linalg.norm(b - A * xk) / np.linalg.norm(b)
        print(residual)
    
    result, info = bicgstab(A, b, rtol = tol_ls)
    
    if (info != 0):
        print('Linear solver did not converge. Info: ' + str(info) + '. Counter: ' + str(counter + 1))
        return None
    
    deltapsi_flat = result[:N**2]
    deltapsi = np.reshape(deltapsi_flat, newshape = (N, N), order = 'C')    
    psinew = psi - deltapsi
    
    return NRcomplex(psinew, vel, tol_NR, tol_ls, counter + 1)

def NR(psi, vel, tol_NR, tol_ls, counter = 0):
    print(counter)
    F = Functional(psi, vel)
    norm = np.sqrt(np.sum(np.abs(F)**2 * dx**2))
    print(norm)
    
    if (norm < tol_NR):
        print('Convergence achieved after ' + str(counter) + ' iterations!')
        return psi
    
    F_flat = np.ravel(F, order = 'C')
    b = np.concatenate((np.real(F_flat), np.imag(F_flat)))
    
    A = LinearOperator(shape = (2*N**2, 2*N**2), matvec = lambda v: matvecsimple(v, psi, vel), dtype = np.float64)
    
    def callback(xk):
        residual = np.linalg.norm(b - A * xk) / np.linalg.norm(b)
        print(residual)
    
    result, info = lgmres(A, b, rtol = tol_ls, callback = callback)
    
    if (info != 0):
        print('Linear solver did not converge. Info: ' + str(info) + '. Counter: ' + str(counter + 1))
        return None
    
    deltapsir = result[:N**2]
    deltapsii = result[N**2:]
    deltapsi_flat = deltapsir + 1j * deltapsii
    deltapsi = np.reshape(deltapsi_flat, newshape = (N, N), order = 'C')    
    psinew = psi - deltapsi
    plot(psinew)
    return NR(psinew, vel, tol_NR, tol_ls, counter + 1)

    
def plot(psi):
    angle = np.angle(psi)
    dens = np.abs(psi)**2

    #plot wavefunction

    density_cmap = cmocean.cm.thermal
    phase_cmap = cmocean.cm.balance

    fig, ax = plt.subplot_mosaic(
        [['B', 'A']],
        figsize = (10, 5)
    )
    plt.subplots_adjust(wspace=0.2, hspace=0.1)

    mappable = ax['A'].pcolormesh(angle, cmap = phase_cmap, vmin = -np.pi, vmax = np.pi)
    ax['A'].set_xlabel(r'x $[\xi]$')
    ax['A'].set_yticks(ticks = np.linspace(0, N, 11), labels = np.linspace(0, L, 11, dtype = np.int32))
    ax['A'].set_xticks(ticks = np.linspace(0, N, 11), labels = np.linspace(0, L, 11, dtype = np.int32))
    cb = fig.colorbar(mappable, cmap = phase_cmap, label = 'Phase', ax = ax['A'], location = 'bottom')
    cb.ax.set_xticks(ticks = [-np.pi, 0, np.pi], labels = [r'$-\pi$', 0, r'$+\pi$'])


    mappable2 = ax['B'].pcolormesh(dens, cmap = density_cmap, vmin = np.min(dens), vmax = np.max(dens))
    ax['B'].set_xlabel(r'x $[\xi]$')
    ax['B'].set_ylabel(r'y $[\xi]$')
    ax['B'].set_yticks(ticks = np.linspace(0, N, 11), labels = np.linspace(0, L, 11, dtype = np.int32))
    ax['B'].set_xticks(ticks = np.linspace(0, N, 11), labels = np.linspace(0, L, 11, dtype = np.int32))
    fig.colorbar(mappable2, cmap = density_cmap, label = 'Density', ax = ax['B'], location = 'bottom')
    plt.show()
    return None

    
psi1 = IC_Pade(X, Y, L / 2, L / 2 - 1 / (2 * vel), 1)
psi2 = IC_Pade(X, Y, L / 2, L / 2 + 1 / (2 * vel), -1)
#psi3 = IC_Pade(X, Y, 3 * L / 4, L / 2 - 1 / (2 * vel), 1)
#psi4 = IC_Pade(X, Y, 3 * L / 4, L / 2 + 1 / (2 * vel), -1)
psi = psi1 * psi2 
'''
particle_number = get_particle_number(psi)

for _ in range(600):
    psi = imaginary_timestep(psi, dt, particle_number)
    
for _ in range(1000):
    psi = timestep(psi, dt)
'''  


plot(psi)


#%%
psi = NR(psi, vel = -0.05, tol_NR = 1e-5, tol_ls = 1e-3)

plot(psi)







