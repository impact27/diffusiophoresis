# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:18:26 2017

@author: quentinpeter
"""
import numpy as np
from scipy.linalg import toeplitz
from numpy.linalg import eigvals

class diffusioSim():
    def __init__(self, X, CsaltIN, CsaltOUT, Dprot, 
                 Dsalt, Ddiffph, reservoir=False):
        super().__init__()
        #Init salt concentration
        self._Csalt=np.ones_like(X)*CsaltIN
        self._Csalt[0]=CsaltOUT
        
        #init proteins concentration                         
        self._Cprot=np.zeros_like(X)
        self._Cprot[0]=1 
#        self._Cprot[:nx//10] = np.linspace(1, 0, nx//10)
        
        #New dt
        dt=dx**2/Dprot/2
        #Compute salt dt
        dtsalt=dx**2/Dsalt/2
        #Align with dt
        Nsaltdt = int(np.ceil(dt/dtsalt))
        dtsalt = dt/Nsaltdt
        #Step matrix
        I=np.eye(nx, dtype=float)  
        self._Cx = getCx(nx, dx, reservoir)
        self._Cxx = getCxx(nx, dx, reservoir)
        
        dF = dtsalt*Dsalt*self._Cxx
        self._Fsalt = np.linalg.inv(I-.5*dF)@(I+.5*dF)
        self._Fsalt = np.linalg.matrix_power(self._Fsalt, Nsaltdt)
        self._dt = dt
        self._nx = nx
        self._Ddiffph = Ddiffph
        self._Dprot = Dprot
        self._Dsalt = Dsalt
    
#    @profile
    def advance(self, t):
        I=np.eye(self._nx, dtype=float)
        Nsteps = int(t/self._dt)
        for i in range(Nsteps):
            #update Csalt
            self._Csalt=self._Fsalt@self._Csalt
            #Compute the gradient of ln Csalt and the laplacian
            GlnC = self._Cx@self._Csalt/self._Csalt
            GGlnC = (self._Cxx@self._Csalt/self._Csalt 
                             -(self._Cx@self._Csalt)**2/self._Csalt**2)
            
            dF = self._dt*(self._Dprot*self._Cxx
                     + self._Ddiffph*(
                             GlnC[:, np.newaxis]*self._Cx 
                             + GGlnC[:, np.newaxis]*I))
            #Get new step matrix
#            Fprot=np.linalg.inv(I-.5*dF)@(I+.5*dF)
            Fprot=(I+dF)
#            Do step
            self._Cprot=Fprot@self._Cprot
        print(np.max(np.abs(eigvals(Fprot))),np.max(self._Csalt))
    
    @property 
    def Csalt(self):
        return self._Csalt
    
    @property 
    def Cprot(self):
        return self._Cprot
    
    def _getSaltFlux(self, Dsalt, X):
        """
        Returns f_{i+1/2}
        """
        q = getQs(len(X))
        sigmap = (q[1]-q[0])
        dx = sigmap@X
        f = 1/dx[:, np.newaxis]*(Dsalt*sigmap)
        
        return f
    
    def _getProtFlux(self, Dprot, X, Ddiffph, lnQs, neg):
        """
        Returns f_{i+1/2}
        """
    #    neg = Ddiffph*(CsaltIN - CsaltOUT) < 0
        q = getQs(len(X))
        sigmap = (q[1]-q[0])
        dx = sigmap@X
        f = 1/dx[:, np.newaxis]*(Dprot*sigmap
                            +Ddiffph*(sigmap@lnQs)*q[neg])
        return f
        
    def _getdF(self, X, dt, fp):
        q = getQs(len(X))
        dx = 1/2*(q[1]-q[-1])@X
        dF = dt/dx[:, np.newaxis]*(q[-1]@fp - fp)
        return dF
    
#get differential operators matrix
def getCxx(nx, dx, reservoir=False):
    #Laplacian
    line=np.zeros(nx,dtype=float)
    line[:2]=[-2,1]
    Cxx=toeplitz(line,line) #toeplitz creation of matrice which repeat in diagonal 
    Cxx[0,:]=0
    Cxx[-1,-1]=-1
    Cxx/=dx**2
    #Change border conditions if asked
    if reservoir:
        Cxx[-1,:]=0
        
    return Cxx

def getCx(nx, dx, reservoir=False):
    #get grad y operator   
    udiag1 = np.ones(nx-1)
    udiag2 = np.ones(nx-2)
    Cx = (np.diag(udiag2, -2)
        + np.diag(-8*udiag1, -1)
        + np.diag(8*udiag1, 1)
        + np.diag(-udiag2, 2))
    Cx[0,:]=0
    Cx[1,0] = -7
    Cx[-2:,-1] = 7
    Cx /= (12*dx)
    if reservoir:
        Cx[-1,:]=0
    return Cx

def getQs(nX, boundary='Neumann'):
    """Get matrices to access neibours in y with correct boundary conditions
    
    Parameters
    ----------
    Zgrid:  integer
        Number of Z pixel
    Ygrid:  integer
        Number of Y pixel
    bounday: 'Neumann' or 'Dirichlet'
        constant derivative or value
        
        
    Returns
    -------
    qy:  3d array
        A list of matrices to access [-2, +2] y neighbors
    """
    #Create the q matrices
    q = np.zeros((5, nX, nX))
    for i in range(-2, 3):
        q[i]=np.diag(np.ones(nX-np.abs(i)), i)
    
    #Border
    q[-2, :2, 0]=1
    q[-1, 0, 0]=1
    q[1, -1, -1]=1
    q[2, -2:, -1]=1

    return q

