# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:18:26 2017

@author: quentinpeter
"""
import numpy as np
from scipy.linalg import toeplitz
from numpy.linalg import eigvals
import matplotlib.pyplot as plt

class diffusioSim():
    def __init__(self, X, CsaltIN, CsaltOUT, Dprot, 
                 Dsalt, Ddiffph, reservoir=False):
        super().__init__()
        
        #init proteins concentration                         
        self._Cprot=np.zeros_like(X)
        self._Cprot[0]=1 
        
        self._X = X
        self._t = 0
        
        self._Cin = CsaltIN
        self._Cout = CsaltOUT
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

        self._L = X[-1] - X[0]
        
        
        
        
        self._dtsalt = np.min(dx)**2/Dsalt/2
        
        
#    @profile
    def advance(self, t):
        
        q = self._q
        sigmap = self._sigmap
        dx = self._dx
        neg = self._neg
        Dprot = self._Dprot
        Ddiffph = self._Ddiffph
        while t>0:

            
            #Get dt
            dt0 = np.min(dx)**2/Dprot/2
#            dt1 = np.min(dx)/np.abs(Ddiffph)/2/np.max(dxlnQs)
#            dt  = np.min((dt0, dt1))
#            print(dt0, dt1)
            dt = dt0
            
            GlnC, GGlnC = getdiffs(self._X,self._t, self._Dsalt, self._L, 
                                   self._Cin, self._Cout)
            
            
            
            
            
            Cxx = getCxx(len(self._X), np.max(dx))
            Cx = getCx(len(self._X), np.max(dx))
            Cx = 1/np.max(dx)/12*(-q[2]+8*q[1]-8*q[-1]+q[-2])
#            Cx = 1/np.max(dx)/4*(q[1]+3*q[0]-5*q[-1]+q[-2])
#            Cx = 1/np.max(dx)/2*(q[1]-q[-1])
#            Cx = 1/np.max(dx)*(q[0]-q[-1])
#            Cx = 1/np.max(dx)*(q[1]-q[0])
            
            Cx[0]=0
#            Cx2[0]=0
            GlnC, GGlnC = getdiffs(self._X,self._t, self._Dsalt, self._L, 
                                   self._Cin, self._Cout)

            dF2 = (self._Dprot*Cxx
                     + self._Ddiffph*(
                             GlnC[:, np.newaxis]*Cx 
                             + GGlnC[:, np.newaxis]*q[0]))
            
            #Update proteins
            dF = self._Dprot*Cxx + Dx(self._X, dt, t, self._Dsalt, self._L, 
                    self._Cin, self._Cout, self._Ddiffph)
            
            dF2[0]=0
            dF[0]=0
            
            
            
            F = q[0] + dt*dF2
            
            assert(len(np.shape(F))==2)
            self._Cprot = F@self._Cprot
            
            t = t-dt
            self._t += dt
#        plt.figure(100)
#        plt.imshow(Cx-Cx2)
#        plt.figure(101)
#        plt.imshow(Cx)
#        plt.figure(102)
#        plt.imshow(Cx2)   
        print(np.max(np.abs(eigvals(F))),np.max(self._Cprot))
    
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
    
def Dx(X, dt, t, D, L, Cin, Cout, Ddp):
    nx = len(X)
    dx = X[1]-X[0]
    q = getQs(nx)
    
    Xp = X[:-1] + dx/2
    dxlnqhalf, _ = getdiffs(Xp, t+dt/2, D, L, Cin, Cout, nN=100)
    #Correct
    up = -Ddp * np.append(dxlnqhalf, 0)
    #Incorrect BUT don't care
    um = -Ddp * np.insert(dxlnqhalf, 0, 0)
    
#    dxlnq, _ = getdiffs(X, t, D, L, Cin, Cout, nN=100)
#    up = -Ddp * dxlnq
#    um = -Ddp * dxlnq
    up = up[:, np.newaxis]
    um = um[:, np.newaxis]
    
    

    sigdx = np.zeros((3, nx, nx))
    for i in range(-1, 2):
        # Fromm choise of sigma
        sigdx[i] = (q[i + 1] - q[i - 1]) / 2


    neg = -Ddp*(Cin-Cout) < 0
    assert not neg

    Dx = (um*q[-1 + neg] - up*q[0 + neg]
          + (.5 - neg) * ((um*sigdx[-1 + neg] - up*sigdx[0 + neg])
                          - dt / dx * (um**2*sigdx[-1 + neg] 
                          - up**2*sigdx[0 + neg])))
    
#    Dx = (um*q[0] - up*q[1])
    Dx /= dx
    return Dx 

def Ct(t, N, D, L):
    return np.exp(-D*(np.pi*(1/2+N)/L)**2*t)

def C(x,t, D, L, nN=100):
    x = x[:, np.newaxis]
    N = np.arange(nN)[np.newaxis, :]
    ret=(1/(N+0.5))*Ct(t, N, D, L)*np.sin((np.pi/2+N*np.pi)/L*x)
    return (2/np.pi)*np.sum(ret, 1)
#@profile 
def dC(x,t, D, L, nN=100):
    x = x[:, np.newaxis]
    N = np.arange(nN)[np.newaxis, :]
    ret = (Ct(t, N, D, L) * np.cos((np.pi/2+N*np.pi)/L*x))
    return (2/L)*np.sum(ret, 1)

def ddC(x,t, D, L, nN=100):
    x = x[:, np.newaxis]
    N = np.arange(nN)[np.newaxis, :]
    ret=(N+0.5)*Ct(t, N, D, L)*np.sin((np.pi/2+N*np.pi)/L*x)
    return (-2*np.pi/L**2)*np.sum(ret, 1)

#@profile
def getdiffs(x,t, D, L, Cin, Cout, nN=100):
    x = x[:, np.newaxis]
    N = np.arange(nN)[np.newaxis, :]
    
    Cts = Ct(t, N, D, L)
    xN = (np.pi/2+N*np.pi)/L*x
    
    cos = np.cos(xN)
    sin = np.sin(xN)
    
    Cs = (2/np.pi)*np.sum((1/(N+0.5))*Cts*sin, 1)
    dCs = (2/L)*np.sum(Cts * cos, 1)
    ddCs = (-2*np.pi/L**2)*np.sum((N+0.5)*Cts*sin, 1)

    dlnCs = dCs/(Cout/(Cin-Cout) + Cs)
    
    return dlnCs, (ddCs/(Cout/(Cin-Cout) + Cs)-(dlnCs)**2)
    
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

def getQs(nX):
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

