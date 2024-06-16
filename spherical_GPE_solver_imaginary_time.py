import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh

from cartopy import crs
from matplotlib import cm


#parameters
N = 512 #grid points
R = 1.0 #radius of sphere
theta_plus = np.pi / 40 #position of vortex on the upper hemisphere
dt = 1.e-4  #time step
omega = 2 * np.pi * 5 #frequency in Hz
real_dt = dt * 1000 / omega #length of one timestep in real time in ms
alpha = 5 / omega # linear coefficient
g = -1. / omega   #nonlinear coefficient
bg_dens = 1. #condensate density far from vortices
lmax = N//2 - 1 #maximum degree of spherical harmonics
sigma = 1./24. #width of gaussian


#coordinate system

theta, phi = np.linspace(0,  np.pi, N, endpoint = False), np.linspace(0, 2 * np.pi, 2 * N, endpoint = False)  #rectangular grid with polar angle theta and azimuthal angle phi
dangle = np.pi / N  #grid spacing


#DEFINING ALL THE FUNCTIONS WE NEED
#####################################################################

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


#define a model initial density as two upside down gaussians at the positions of the vortices

def gauss_density(theta, phi, sigma):
    density = 1. - np.exp(- (theta - theta_plus)**2 / (2 * sigma **2 )) * np.exp(- (phi - np.pi)**2 / (2 * sigma**2)) - np.exp(- (theta - np.pi + theta_plus)**2 / (2 * sigma **2 )) * np.exp(- (phi - np.pi)**2 / (2 * sigma**2))
    return density

    
#define norm function

def get_norm(psi):
    norm = 0
    for i in range(N):
        for j in range(2*N):
            norm += np.sin(theta[i]) * dangle**2 * np.abs(psi[i,j])**2
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

def get_energy(psi):
    energy = 0
    Laplace_psi = Laplacian(psi) #array of Laplacian of wavefunction
    deriv_phi_psi = deriv_phi(psi) #array of derivative wrt azimuthal angle of wavefunction
    conj_psi = np.conjugate(psi) #array of complex conjugate of wavefunction
    for i in range(N):
        for j in range(2*N):
            energy += dangle**2 * np.sin(theta[i]) * (  - alpha / 2 * conj_psi[i,j] * Laplace_psi[i,j] + g * np.abs(psi[i,j])**4 - 1.0j * conj_psi[i,j] * deriv_phi_psi[i,j]  ) #compute the hamiltonian
    energy = np.real(energy)
    return energy
    
#calculate angular momentum of condensate (another conserved quantity).

def get_ang_momentum(psi):
    mom = 0
    deriv_phi_psi = deriv_phi(psi) #array of derivative wrt azimuthal angle of wavefunction
    conj_psi = np.conjugate(psi) #array of complex conjugate of wavefunction
    for i in range(N):
        for j in range(2*N):
            mom += - 1.0j * dangle**2 * np.sin(theta[i]) * conj_psi[i,j] * deriv_phi_psi[i,j] #compute the angular momentum integral
    mom = np.real(mom)
    return mom


#one timestep with split stepping method where the gridded data (i.e. the wave function) is the input and output


def timestep_grid(psi, dt):
    coeffs = pysh.expand.SHExpandDHC(griddh = psi, norm = 4, sampling = 2)
    len_l = np.size(coeffs, axis = 1)  #size of sh_coeffs array in l and m indices(degree and order)
    for l in range(len_l):
        for m in range(len_l):
            for i in range(2):
                coeffs[i,l,m] *= np.exp(-alpha * l * (l + 1) * dt / 2) * np.exp(-m * dt * (-1.)**i)  #timestep of kinetic and rotating term
    psi = pysh.expand.MakeGridDHC(coeffs, norm = 4, sampling = 2, extend = False) #create gridded data in (N, 2*N) array from coeffs
    psi *= np.exp(-g * dt * np.abs(psi)**2) #timestep of nonlinear term
    
    norm = get_norm(psi)
    psi = psi/np.sqrt(norm) #normalize wavefunction, important to do in every step for imaginary time evolution
    return psi


#initialize wavefunction

psi = np.zeros(shape = (N, 2*N), dtype = np.complex128)

for i in range(N):
    for j in range(2*N):
        psi[i,j] = np.sqrt(gauss_density(theta[i], phi[j], sigma)) * np.exp(1.0j * phase(theta[i], phi[j])) #initalize psi with the upside down gaussian density
        
norm = get_norm(psi)
psi = psi/np.sqrt(norm)
        

#some stuff needed for plotting

mycmap = cm.seismic
myprojection = crs.Mollweide(central_longitude=180.)
gridspec_kw = dict(height_ratios = (1, 1), hspace = 0.5)



#SIMULATION
############################################################

#after the if statement follows the plotting routine to give a plot of the density and phase of the wave function and a plot of the spectrum

end = 10000 #number of steps in simulation

for q in range(end + 1): 
    if (q % (end / 10) == 0):  #plot 10 times during simulation
        time = round(dt * q * 1000 / omega, 2) #real time that has passed at this point in the simulation in ms
        
        
        dens = np.abs(psi)**2 #calculate condensate density
        phase_angle = np.angle(psi) #calculate phase of condensate
        
        norm = get_norm(psi) #calculate norm of condensate
        energy = get_energy(psi) #calculate energy of condensate
        mom = get_ang_momentum(psi) #calculate angular momentum of condensate
        
        coeffs = pysh.expand.SHExpandDH(griddh = psi, norm = 4, sampling = 2) #get sh coefficients for the wave function     
        dens_coeffs = pysh.expand.SHExpandDH(griddh = dens, norm = 4, sampling = 2) #get sh coefficients for the density
        phase_coeffs = pysh.expand.SHExpandDH(griddh = phase_angle, norm = 4, sampling = 2) #get sh coefficients for the phase
        
        clm = pysh.SHCoeffs.from_array(coeffs, normalization='ortho', lmax = lmax) #create a SHCoeffs instance for the wavefunction to plot spectrum (at the bottom)
        dens_clm = pysh.SHCoeffs.from_array(dens_coeffs, normalization='ortho', lmax = lmax) #create a SHCoeffs instance from the coefficient array for the density 
        phase_clm = pysh.SHCoeffs.from_array(phase_coeffs, normalization='ortho', lmax = lmax) #create a SHCoeffs instance from the coefficient array for the phase

        dens_grid = dens_clm.expand() #create a SHGrid instance for the density 
        phase_grid = phase_clm.expand() #create a SHGrid instance for the phase

        #plot

        fig, axes = plt.subplots(2, 1, gridspec_kw = gridspec_kw)
        
        plt.suptitle('Imaginary time evolution of two vortices after ' + str(time) + 'ms')

        #subplot for denstiy

        dens_grid.plot(cmap = mycmap, 
                       colorbar = 'right', 
                       cb_label = 'Density', 
                       xlabel = '', 
                       tick_interval = [90,45], 
                       tick_labelsize = 6, 
                       axes_labelsize = 7,
                       ax = axes[0],  
                       show = False)
        
        cb2 = axes[0].images[-1].colorbar
        cb2.mappable.set_clim(0., np.max(dens))
        
        
        #subplot for phase

        phase_grid.plot(cmap = mycmap, 
                        colorbar = 'right',
                        cb_label = 'Phase',
                        tick_interval = [90,45], 
                        cb_tick_interval = np.pi,
                        tick_labelsize = 6, 
                        axes_labelsize = 7, 
                        ax = axes[1],  
                        show = False)
        
       
        cb2 = axes[1].images[-1].colorbar
        cb2.mappable.set_clim(-np.pi, np.pi)
        cb2.ax.set_yticklabels([r'$-\pi$', 0, r'$+\pi$'])
        
        
        #these lines put a textbox with some simulation parameters into the density plot
        
        angle_frac = int(np.pi/theta_plus)
        
        textstr ='\n'.join((r'$\omega=%.1f$' % (omega, ) + 'Hz', r'$\theta_+ = \pi/$' + str(angle_frac), r'$\Delta t =%.3f$' % (real_dt, ) + 'ms'))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axes[0].text(10, 80, textstr, fontsize=4, verticalalignment='top', bbox=props)
        
        #put the conserved quantities below the plots
        
        axes[1].text(-1, -150, 'Norm = ' + str(norm), fontsize = 'x-small')
        axes[1].text(-1, -180, 'Energy = ' + str(energy), fontsize = 'x-small')
        axes[1].text(-1, -210, 'Angular momentum = ' + str(mom), fontsize = 'x-small')
        
        filename = 'J:/Uni - Physik/Master/6. Semester/Masterarbeit/Media/Simulations of two vortices/' + str(time) + 'ms.pdf'

        plt.savefig(fname = filename, dpi = 300, bbox_inches = 'tight', format = 'pdf')
        
        #plot spectrum
        
        clm.plot_spectrum(unit = 'per_l', show = False)
        
        plt.title('Spectrum after ' + str(time) + 'ms')

        filename = 'J:/Uni - Physik/Master/6. Semester/Masterarbeit/Media/Simulations of two vortices/spectrum_' + str(time) + 'ms.pdf'

        plt.savefig(fname = filename, dpi = 300, bbox_inches = 'tight', format = 'pdf')
        plt.show()
    psi = timestep_grid(psi, dt)        
        
