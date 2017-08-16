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
        
        self._X = X
        
        
        self._Ddiffph = Ddiffph
        self._Dprot = Dprot
        self._Dsalt = Dsalt
        
        self._neg = Ddiffph*(CsaltIN - CsaltOUT) > 0
        q = getQs(len(X))
        sigmap = (q[1]-q[0])
        dx = sigmap@X
        dx[-1] = dx[-2]
        sigmap = 1/dx[:, np.newaxis]*sigmap
        self._sigmap = sigmap
        self._dx = dx
        self._q = q
        fsalt = -Dsalt*sigmap
        self._dFsalt = self._getdF(X, fsalt)

        
        
        
        
        self._dtsalt = np.min(dx)**2/Dsalt/2
        
        
    
    def advance(self, t):
        
        q = self._q
        sigmap = self._sigmap
        dx = self._dx
        neg = self._neg
        Dprot = self._Dprot
        Ddiffph = self._Ddiffph
        while t>0:
            #Get dxlnqs
            dxlnQs = sigmap@self._Csalt/self._Csalt
            
            #Get dt
            dt0 = np.min(dx)**2/Dprot/2
#            dt1 = np.min(dx)/np.abs(Ddiffph)/2/np.max(dxlnQs)
#            dt  = np.min((dt0, dt1))
#            print(dt0, dt1)
            dt = dt0
            
            #Update salt
            Nsaltdt = int(np.ceil(dt/self._dtsalt))
            dtsalt = dt/Nsaltdt
            Fsalt = np.linalg.matrix_power(q[0] + dtsalt*self._dFsalt, Nsaltdt)
            self._Csalt = Fsalt@self._Csalt
            
            #Get dxlnqs
            dxlnQs = sigmap@self._Csalt/self._Csalt
            
            #Update proteins
            fprot = (- Dprot*sigmap + Ddiffph*dxlnQs[:, np.newaxis]*q[int(neg)])
            dF = self._getdF(self._X, fprot)
            
            
            
            
            Cxx = getCxx(len(self._X), np.max(dx))
            Cx = getCx(len(self._X), np.max(dx))
            GlnC = Cx@self._Csalt/self._Csalt
            GGlnC = (Cxx@self._Csalt/self._Csalt 
                             -(Cx@self._Csalt)**2/self._Csalt**2)
            

            dF2 = (self._Dprot*Cxx
                     + self._Ddiffph*(
                             GlnC[:, np.newaxis]*Cx 
                             + GGlnC[:, np.newaxis]*q[0]))
            
            
            F = q[0] + dt*dF2
            self._Cprot = F@self._Cprot
            
            t = t-dt
            
        print(np.max(np.abs(eigvals(F))),np.max(self._Csalt))
    
    @property 
    def Csalt(self):
        return self._Csalt
    
    @property 
    def Cprot(self):
        return self._Cprot
        
    def _getdF(self, X, fp):
        q = getQs(len(X))
        dx = 1/2*(q[1]-q[-1])@X
        dx = np.diff(X)
        dx = np.append(dx, dx[-1])
        dF = 1/dx[:, np.newaxis]*(q[-1]@fp - fp)
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

