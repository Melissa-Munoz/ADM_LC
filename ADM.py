#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ADM-based light curve synthesis 
# Melissa Munoz
# Updated Dec 2020
#
# See publication Munoz et al. 2020 
# See also Owocki et al. 2006 for more details on the ADM formalism
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#-------------------------------------------------------------------------------
# Library import ---------------------------------------------------------------
#-------------------------------------------------------------------------------

import numpy as np
from scipy.optimize import newton
from scipy import interpolate
from scipy.ndimage.interpolation import rotate
from scipy.interpolate import interp1d
#import matplotlib.pyplot as plt
import time 
from scipy.integrate import simps


#-------------------------------------------------------------------------------
# Some constants ---------------------------------------------------------------
#-------------------------------------------------------------------------------

G = 6.67408*10**(-8)
eV = 1.602*10**(-12)
c = 2.99792458*10**(10)	 
h= 6.6260755*10**(-27)	 
kb = 1.380658*10**(-16)
eV = 1.6021772*10**(-12)	
me = 9.1093897*10**(-28)
mp = 1.6726*10**(-24)
mH = 1.6733*10**(-24)
e = 4.8032068*10**(-10)
alphae = (1.+0.7)/2.
sigmat = 6.6524*10**(-25)
Msol = 1.99*10**(33)
Rsol = 6.96*10**(10)
Lsol = 3.9*10**(33)



#-------------------------------------------------------------------------------
# Line-of-sight angle ----------------------------------------------------------
#-------------------------------------------------------------------------------

def csalpha(phi, beta, inc):
	#Note the degenerancy between the inclination and obliquity
	return np.sin(beta)*np.cos(phi)*np.sin(inc)+np.cos(beta)*np.cos(inc)

def csalpha2(phi, A, B):
	#Degenrancy avoided by reexpressong the above equation with 
	#A = inc + beta,
	#B = |inc - beta|
	return 0.5*(np.cos(B)*(1.+np.cos(phi)) + np.cos(A)*(1.-np.cos(phi)) )



#-------------------------------------------------------------------------------
# ADM auxiliary equations ------------------------------------------------------
#-------------------------------------------------------------------------------

# Wind upflow
#-------------------------------------------------------------------------------

def w(r):
	return 1.-1./r

def vw(r,vinf):
	return vinf*(1.-1./r)

def rhow(r,mu):
	return 2.*np.sqrt(r - 1. + mu**2)*np.sqrt(1.+3.*mu**2)/((r- 1.)*(4.*r - 3. + 3.*mu**2))*(1./r)**(3./2.)


# Hot post-shock 
#-------------------------------------------------------------------------------

def wh(r,rs,mu,mus,Tinf,Teff):
	ws=w(rs)
	Ts=Tinf*ws**2
	return ws/4.*(Th(rs,mu,mus,Tinf,Teff)/Ts)*(B(r,mu,mus))

def vh(r,rs,mu,mus,Tinf,Teff,vinf):
	ws=w(rs)
	Ts=Tinf*ws**2
	return ws*vinf/4.*Th(rs,mu,mus,Tinf,Teff)/Ts*np.sqrt((1.+3.*mu**2)/(1.+3.*mus**2))*(rs/r)**3

def g(mu):
	return np.abs(mu - mu**3 + 3.*mu**5/5. - mu**7/7.)

def TTh(rs,mu,mus,Tinf):
	ws=w(rs)
	Ts=Tinf*ws**2
	return Ts*(g(mu)/g(mus))**(1./3.)

def Th(rs,mu,mus,Tinf,Teff):
	return np.maximum(TTh(rs,mu,mus,Tinf),Teff)

def rhoh(r,rs,mu,mus,Tinf,Teff):
	ws=w(rs)
	Ts=Tinf*ws**2
	return 4.*rhow(rs,mus)*Ts/Th(rs,mu,mus,Tinf,Teff)

# Cooled downflow 
#-------------------------------------------------------------------------------

def wc(r,mu):
	return np.abs(mu)*np.sqrt(1./r)

def vc(r,mu,ve):
	return np.abs(mu)*np.sqrt(1./r)*ve

def rhoc(r,mu,delta):
	return 2.*np.sqrt(r - 1. + mu**2)*np.sqrt(1.+3.*mu**2)/(np.sqrt(mu**2+delta**2/r**2)*(4.*r - 3. + 3.*mu**2))*(1./r)**(2.)

def f(mus,mustar,chiinf):
	rm = 1./(1.-mustar**2)
	rs = rm*(1.-mus**2)
	ws = w(rs)
	ggmus = chiinf/(6.*mustar)*(1.+3.*mustar**2)/(1.+3.*mus**2)*(ws*rs/rm)**4*(rs)**2
	gmus = g(mus)
	return (gmus-ggmus) 



#-------------------------------------------------------------------------------
# ADM --------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def admCAL(Nx, Ny, Nz, RA, Rc, Teff, Tinf, chiinf, delta ):

	#Defining magnetosphere grid size	
	NNx=2*Nx
	NNy=2*Ny
	NNz=2*Nz

	#Defining spatial grids
	XX=np.linspace(-Rc,Rc,NNx)
	YY=np.linspace(-Rc,Rc,NNy)
	ZZ=np.linspace(-Rc,Rc,NNz)	
	X=XX[Nx:NNx]
	Y=YY[Ny:NNy]
	Z=ZZ[Nz:NNz]
	
	#Defining density grids of each component
	Rhoh=np.zeros([Nz,Nx,Ny])
	Rhow=np.zeros([Nz,Nx,Ny])
	Rhoc=np.zeros([Nz,Nx,Ny])

	#Last closed loop
	mustar_RA = np.sqrt(1.-1./RA)
	mustars_RA = np.linspace(0.01,mustar_RA,Nz)
	mus_RA = np.zeros(Nz)
	rs_RA = np.zeros(Nz)
	r_RA = (1.-mustars_RA**2)/(1-mustar_RA**2)
	for i in range(Nz):
		try:
			tempmus = newton(f, 0.3, args=(mustars_RA[i],chiinf,))
		except RuntimeError:	
			tempmus=0.
			#print 'error LC'
		mus_RA[i]=np.abs(tempmus)
		rs_RA[i]=(1.-mus_RA[i]**2)/(1.-mustars_RA[i]**2)
	fs=interp1d( mustars_RA,mus_RA,bounds_error=False, fill_value=0. )

	#Compute ADM in first octant of the magnetosphere
	#Velocity and temperature calculations are commented out because they are not required for the light curve synthesis
	for i in range(0,Nx):
		for j in range(0,Ny):
			p=np.sqrt(X[i]**2+Y[j]**2)
			for k in range(0,Nz):
				r=np.sqrt(p**2+Z[k]**2)
				mu=Z[k]/r
				rRA=(1.-mu**2)/(1-mustar_RA**2)
				if r > 1.05:
					mustar=np.sqrt(1.-(1.-mu**2)/r)
					rm = 1./(1.-mustar**2)
					mus=fs(mustar)
					'''
					try:
						tempmus = newton(f, 0.3, args=(mustar,chiinf,))
					except RuntimeError:	
						tempmus=0.
						#print 'error LC'
					mus = np.abs(tempmus)
					'''
					rs = rm*(1.-mus**2)
					Rhow[k,i,j]=rhow(r,mu)
					#Vw[k,i,j]=w(r)
					#tw[k,i,j]=Teff		
					if r < rRA:
						Rhoc[k,i,j]=rhoc(r,mu,delta)
						#Vc[k,i,j]=wc(r,mu)
						#tc[k,i,j]=Teff
						if r > rs and rs > 1.05 :
							Rhoh[k,i,j]=rhoh(r,rs,mu,mus,Tinf,Teff)
							#Vh[k,i,j]=wh(r,rs,mu,mus,Tinf,Teff)
							#th[k,i,j]=Th(rs,mu,mus,Tinf,Teff)
	
	#Transpose density in remaining octants (axial symmetry)
	Rhoh=np.concatenate([Rhoh[::-1,:,:],Rhoh],axis=0)
	Rhoh=np.concatenate([Rhoh[:,::-1,:],Rhoh],axis=1)
	Rhoh=np.concatenate([Rhoh[:,:,::-1],Rhoh],axis=2)
	Rhoc=np.concatenate([Rhoc[::-1,:,:],Rhoc],axis=0)
	Rhoc=np.concatenate([Rhoc[:,::-1,:],Rhoc],axis=1)
	Rhoc=np.concatenate([Rhoc[:,:,::-1],Rhoc],axis=2)
	Rhow=np.concatenate([Rhow[::-1,:,:],Rhow],axis=0)
	Rhow=np.concatenate([Rhow[:,::-1,:],Rhow],axis=1)
	Rhow=np.concatenate([Rhow[:,:,::-1],Rhow],axis=2)	
	
	return [Rhow, Rhoh, Rhoc]		



#-------------------------------------------------------------------------------
# RT ---------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def LC(phi, A, B, Nx, Ny, Nz, Teff, Mstar, Rstar, Vinf,  Mdot, Bd, delta, dm0 ):
	
	#t0 = time.time()

	#Defining phase grid 
	phi=np.concatenate(([0.],phi))
	PH=len(phi)

	#Defining magnetosphere grid size
	NNx=2*Nx
	NNy=2*Ny
	NNz=2*Nz

	#Conversion of stellar properties into cgs 
	mdot=Mdot*Msol/(365.*24*3600.)
	vinf=Vinf*100000.
	rstar=Rstar*Rsol
	mstar=Mstar*Msol
	ve = np.sqrt(2.*G*mstar/rstar)
	Ve = np.sqrt(2.*G*mstar/rstar)/100000.
	rhowstar = mdot/(4.*np.pi*rstar**2*vinf)
	rhocstar = rhowstar*vinf/ve

	#Some scalling relations. see Owocki 2016
	chiinf = 0.034*(vinf/10.**8)**4*(rstar/10**12)/(Mdot/10**(-6))
	Tinf = 14*10**6*(vinf/10.**8)**2

	#Computing the Alfven radius and closure radius
	Beq=Bd/2.
	eta=(Beq)**2*rstar**2/(mdot*vinf)
	RA=0.3+(eta+0.25)**(0.25)
	Rc = RA # This can be changed occording to the user

	#ADM output (the magnetosphere density components)
	admOUT=admCAL(Nx, Ny, Nz, RA, Rc, Teff, Tinf, chiinf, delta )
	Rhow=admOUT[0]
	Rhoh=admOUT[1]
	Rhoc=admOUT[2]
	rhoh=Rhoh*rhowstar
	rhoc=Rhoc*rhocstar
	rhow=Rhow*rhowstar

	#Defining spatial grids
	XX=np.linspace(-Rc,Rc,NNx)
	YY=np.linspace(-Rc,Rc,NNy)
	ZZ=np.linspace(-Rc,Rc,NNz)
	dX=np.abs(XX[0]-XX[1])
	dY=np.abs(YY[0]-YY[1])
	dZ=np.abs(ZZ[0]-ZZ[1])
	dx=dX*Rstar*Rsol
	dy=dY*Rstar*Rsol
	dz=dZ*Rstar*Rsol

	#Variable setup
	I0=np.ones([NNx,NNx]) #Intensity
	F1=np.zeros(PH) #Flux
	F2=np.zeros(PH) #Core fux
	dm=np.zeros(PH) #Differential magnitude
	for ph in range(PH):
		
		#Rotation of density cube 
		RHO=rhoh+rhoc+rhow
		alpha=np.arccos(csalpha2(phi[ph]*2.*np.pi,np.radians(A),np.radians(B)))
		for k in range(0,NNx):
			tempRHO = rotate(RHO[:,k,:],np.degrees(alpha),reshape=False)
			RHO[:,k,:] = tempRHO

		#Defining 3D meshgrid 
		Z_grid, X_grid, Y_grid = np.meshgrid( ZZ, XX, YY, indexing='xy')
		R_grid = np.sqrt( Z_grid**2 + X_grid**2 + Y_grid**2 )
		RHO[ R_grid<1.0 ] = 0.
		RHO[ (np.sqrt(Z_grid**2+Y_grid**2)<1) & (X_grid<0) ] = 0 #removing occulted regions 

		#Defining 2D meshgrid
		X_grid, Y_grid = np.meshgrid( XX, YY, indexing='xy')
		P_grid = np.sqrt( X_grid**2 + Y_grid**2 )
		I0[ P_grid > 1.0] = 0. #including only core rays

		#Computing electron density, optical depth and attenuated intensity
		ne = RHO*alphae/mp
		dtau = dz*ne*sigmat
		tauinf = simps(dtau,axis=0) 
		Ia = I0*np.exp(-tauinf)

		#Computing emergent flux and core flux
		F1[ph] = simps(simps(Ia,XX*Rstar*Rsol),YY*Rstar*Rsol)
		F2[ph] = simps(simps(I0,XX*Rstar*Rsol),YY*Rstar*Rsol)

	#Differential magnitude from normalized flux
	dm=-2.5*np.log10(F1/F2)
	
	#t1=time.time()
	#total = t1-t0
	#print total
	return dm[1:]+dm0-dm[0]



