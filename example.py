import numpy as np
import matplotlib.pyplot as plt
import ADM


#Physical properties of the star
#-------------------------------------------------------------------------------
#Effective temperature, Teff, is in Kelvins
#Stellar mass, Mstar, is in solar mass
#Stellar radius, Rstar, is in solar radius
#Terminal velocity, Vinf, is in km/s
#Mass-loss rate, Mdot, is in solar mass per year
#Polar field strength, Bstar is in Gauss
Teff = 35000.0 
Mstar = 30.0
Rstar = 10.
Vinf = 2500.0
Mdot = 10**(-6.0)
Bstar = 2500.0


#Geometric angles
#-------------------------------------------------------------------------------
#Inclination angle, inc, in degrees
#Magnetic obliquity, beta, in degrees
inc = 30.
beta = 60.
A = inc+beta
B = np.abs(inc-beta)


#Extra parameters
#-------------------------------------------------------------------------------
#Smoothing length, delta
#Vertical offset in differentlia magnitde, dm0 (constant)
delta = 0.1 
dm0 = 0.0


#Calling ADM
#-------------------------------------------------------------------------------
phi = np.linspace(0.,1.,50) #rotational phase
Nx = Ny = Nz = 50 #grid size 
out = ADM.LC(phi, A, B, Nx, Ny, Nz, Teff, Mstar, Rstar, Vinf,  Mdot, Bstar, delta, dm0 )


#Plotting phased light curve
#-------------------------------------------------------------------------------
plt.figure(figsize=(9,6))
plt.plot(phi,out,'k')
plt.plot(phi+1,out,'k')
plt.plot(phi-1,out,'k')
plt.gca().invert_yaxis()
plt.xlabel('Rotational phase',fontsize=14)
plt.ylabel('Differential magnitude [mag]',fontsize=14)
plt.xlim([-0.5,1.5])
plt.show()


