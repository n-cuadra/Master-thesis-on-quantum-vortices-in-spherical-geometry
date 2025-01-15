import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
import spherical_GPE_functions as sgpe
import spherical_GPE_params as params
import cmocean
import scienceplots
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

#set some parameters for plotting

plt.style.use('science')
plt.rcParams.update({'font.size': 7})
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

psi = np.loadtxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/psi_-6.5.txt', delimiter= ',', dtype = np.complex128)


omega = np.arange(6.5, 7, 0.5)


angmom = np.zeros(len(omega), dtype = np.float64)
energykin = np.zeros(len(omega), dtype = np.float64)
energyint = np.zeros(len(omega), dtype = np.float64)
energyrot = np.zeros(len(omega), dtype = np.float64)
e0 = np.zeros(len(omega), dtype = np.float64)

for i in range(len(omega)):
    psi = sgpe.NR2D(psi, mu = params.mu, g = params.g, omega = -omega[i], epsilon = 9e-5)
    np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/psi_' + str(-omega[i]) + '.txt', psi, delimiter = ',', header = 'NR solution for omega = ' + str(-omega[i]))
    ekin, eint, erot = sgpe.get_energy(psi, params.g, -omega[i])
    angmom[i] = sgpe.get_ang_momentum(psi)
    energykin[i] = ekin 
    energyint[i] = eint 
    energyrot[i] = erot
    e0[i] = params.g * sgpe.get_norm(psi)**2 / (8 * np.pi)
    wf_path = 'E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/wf_' + str(-omega[i]) + '.pdf'
    spectrum_path = 'E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/spectrum_' + str(-omega[i]) + '.pdf'
    spectrum2d_path = 'E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/spectrum2d_' + str(-omega[i]) + '.pdf'
    wf_title = r'Wave function for $\tilde\omega =$ ' + str(-omega[i])
    spectrum_title = r'Spectrum for $\tilde\omega =$ ' + str(-omega[i])
    spectrum2d_title = r'2D Spectrum for $\tilde\omega =$ ' + str(-omega[i])
    sgpe.plot(psi, spectrum_title = spectrum_title, wf_title = wf_title, spectrum2d_title=spectrum2d_title, wf_path=wf_path, spectrum_path=spectrum_path, spectrum2d_path=spectrum2d_path)

np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/e0.txt', e0, delimiter = ',')
np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/ekin.txt', energykin, delimiter = ',')
np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/eint.txt', energyint, delimiter = ',')
np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/erot.txt', energyrot, delimiter = ',')
np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/angular_momentum.txt', angmom, delimiter = ',')

#%%

data = np.loadtxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Measuring the angular velocities of vortices/omegacomplete.txt', delimiter = ',')

omega = -data[0, 29:30]
thetaplus = data[1, 29:30]

print(omega)
energyrot = np.zeros(len(omega), dtype = np.float64)
energyint = np.zeros(len(omega), dtype = np.float64)
energykin = np.zeros(len(omega), dtype = np.float64)
angmom = np.zeros(len(omega), dtype = np.float64)


for i in range(len(omega)):
    psi = sgpe.IC_vortex_dipole(thetaplus[i], np.pi, np.pi - thetaplus[i], np.pi, params.xi, params.bg_dens)
    
    particle_number = sgpe.get_norm(psi)

    
    for _ in range(300):
        psi = sgpe.imaginary_timestep_grid(psi, params.dt, params.g, 0.0, particle_number, keep_phase = False)
    
    sgpe.plot(psi)
    psi = sgpe.NR2D(psi, params.mu, params.g, omega[i], epsilon = 1e-3)
    np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/psi_' + str(omega[i]) + '.txt', psi, delimiter = ',', header = 'NR solution for omega = ' + str(-omega[i]))
    wf_path = 'E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/wf_' + str(omega[i]) + '.pdf'
    sgpe.plot(psi, wf_path=wf_path)
    
    ekin, eint, erot = sgpe.get_energy(psi, params.g, omega[i])
    angmom[i] = sgpe.get_ang_momentum(psi)
    energykin[i] = ekin
    energyint[i] = eint
    energyrot[i] = erot
    
np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/ekin.txt', energykin, delimiter = ',')
np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/eint.txt', energyint, delimiter = ',')
np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/erot.txt', energyrot, delimiter = ',')
np.savetxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/angular_momentum.txt', angmom, delimiter = ',')


#%%


psi = np.loadtxt('E:/Uni - Physik/Master/Masterarbeit/Measurements/Simulations of two vortices/Stationary GPE/testing/psi_-0.708.txt', delimiter= ',', dtype = np.complex128)

epsilon = np.linspace(1e-4, 1e-3, 100)
func = np.zeros(shape = (2, 100), dtype = np.float64)
eta = np.random.rand(params.N, 2 * params.N) + 1j * np.random.rand(params.N, 2 * params.N)
psi0 = psi
F0 = sgpe.Functional(psi0, params.mu, params.g, params.omega)

for i in range(100):
    dpsi = epsilon[i] * eta
    dpsif = np.ravel(dpsi, order = 'C')
    v = np.concatenate((np.real(dpsif), np.imag(dpsif)))
    A = LinearOperator(shape = (4 * params.N**2, 4 * params.N**2), 
                                 matvec = lambda v : sgpe.matvec_NR2D(v, psi0, params.mu, params.g, params.omega), 
                                 dtype = np.float64)
    result = A * v
    fieldr, fieldi = result[:2 * params.N**2], result[2 * params.N**2:]
    field = np.reshape(fieldr + 1j * fieldi, newshape = (params.N, 2 * params.N), order = 'C')
    fieldcoeffs = pysh.expand.SHExpandDH(np.abs(field), norm = 4, sampling = 2)
    Fcoeffs = pysh.expand.SHExpandDH(np.abs(F0 - sgpe.Functional(psi0 + dpsi, params.mu, params.g, params.omega)), norm = 4, sampling = 2)
    func[0, i] = fieldcoeffs[0, 0, 0]
    func[1, i] = Fcoeffs[0, 0, 0]
    print(i)


plt.plot(func[0], func[1])
plt.title(r'$\epsilon$ from $10^{-4}$ to $10^{-3}$')
plt.xlabel(r'$\int |J_{\Psi_0}(\epsilon \eta)|$')
plt.ylabel(r'$\int |F(\Psi_0) - F(\Psi_0 + \epsilon \eta)|$')
plt.savefig('E:/Uni - Physik/Master/Masterarbeit/checkop.pdf', dpi = 300)   