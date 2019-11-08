# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:18:26 2017

@author: quentinpeter
"""
import numpy as np
from numpy.linalg import eigvals
import scipy.sparse as sp
import scipy.sparse.linalg


class diffusiophoresisIntegrator():
    def __init__(self, L, nx, CsaltIN, CsaltOUT, Dprot,
                 Dsalt, Ddiffph, reservoir=False, dtfrac=1, nN=100):
        """ Init 
        """

        super().__init__()

        # Not implemented
        assert not reservoir

        # create X with border on the left
        dx = L / nx
        nx = nx + 1
        X = np.linspace(0, L, nx) - dx / 2
        self._X = X
        self._dx = dx
        self._nx = nx
        self._L = L

        # init proteins concentration
        self._Cprot = np.zeros_like(X)
        self._Cprot[0] = 1
        self._t = 0

        self._settings = {"salt in": CsaltIN,
                          "salt out": CsaltOUT,
                          "D protein": Dprot,
                          "D salt": Dsalt,
                          "D diffusiophoresis": Ddiffph,
                          "time step factor": dtfrac}

        self._initsalt(nN)
        self._initDx()


#    @profile
    def advance(self, t):
        """Advance the simulation by a time t"""
        nx = len(self._X)
        I = sp.eye(nx)
        dx = self._dx
        Dprot = self._settings["D protein"]
        dt0 = np.min(dx)**2 / Dprot / self._settings["time step factor"]
        Cxx = Dprot * getCxx(nx, np.max(dx))
        while t > 0:

            # Update proteins
            dF, dt = self._Dx(dt0)
            dF += Cxx

            # Implicit method
            F = sp.linalg.inv(I - dt * dF)
#            F = I + dt*dF

            assert(len(np.shape(F)) == 2)
            self._Cprot = F@self._Cprot

            t = t - dt
            self._t += dt

#        print('eig', np.max(np.abs(sp.linalg.eigs(F))), np.max(self._Cprot))

    @property
    def Cprot(self):
        """Get the current proteins distribution"""
        return self._Cprot

    @property
    def X(self):
        """Get the corresponding X position"""
        return self._X

    def _initDx(self):
        """Prepare variables for Dx calculation"""
        nx = self._nx
        q = getQs(nx)
        
        
        sigdx = (q[[1, 2, 0]] - q[[-1, 0, -2]]) / 2

        self._q = q
        self._sigdx = sigdx

        qp = 1 / 2 * (q[1] + q[0]) 
        qm = 1 / 2 * (q[0] - q[-1])
        line = np.ones(nx)
        line[:2] = 0
        qm = sp.diags(line) @ qm
        qm = qm + q[-1]

        self._qp = qp
        self._qm = qm

#    @profile
    def _Dx(self, dt):
        """Get the Dx matrix"""
        dtfrac = self._settings["time step factor"]
        Ddp = self._settings["D diffusiophoresis"]
        dx = self._dx
        nx = self._nx

        uhalf = -Ddp * self.dxlnq(self._t + dt / 2)
        # Correct
        up = np.zeros(nx)
        up[:-1] = uhalf
        # Incorrect BUT don't care
        um = np.zeros(nx)
        um[1:] = uhalf

        dt2 = dx / np.max(np.abs(up)) / dtfrac
        if dt2 < dt:
            dt = dt2

        fp = sp.diags(up) @ self._qp
        fm = sp.diags(um) @ self._qm

        Dx = (fm - fp) / dx
        
        return Dx, dt


    def _initsalt(self, nN=100):
        """Prepare variables for the salt concentration distribution"""
        L = self._L
        dx = self._dx
        x = self._X[1:] - dx / 2
        x = x[:, np.newaxis]
        N = np.arange(nN)[np.newaxis, :]

        xN = (np.pi / 2 + N * np.pi) / L * x

        cos = np.cos(xN)
        sin = np.sin(xN)

        self._salt_N = N
        self._salt_cos = cos
        self._salt_sin = sin

#    @profile
    def dxlnq(self, t):
        """Get dx ln(qs)"""
        L = self._L
        D = self._settings["D salt"]
        Cin = self._settings["salt in"]
        Cout = self._settings["salt out"]

        N = self._salt_N
        cos = self._salt_cos
        sin = self._salt_sin

        Cts = Ct(t, N, D, L)

        Cs = (2 / np.pi) * np.sum((1 / (N + 0.5)) * Cts * sin, 1)
        dCs = (2 / L) * np.sum(Cts * cos, 1)

        dlnCs = dCs / (Cout / (Cin - Cout) + Cs)

        return dlnCs


def Ct(t, N, D, L):
    return np.exp(-D * (np.pi * (1 / 2 + N) / L)**2 * t)

def getCxx(nx, dx, reservoir=False):
    q = getQs(nx)
    # toeplitz creation of matrice which repeat in diagonal
    Cxx = q[-1] - 2*q[0] + q[1]
    Cxx /= dx**2
    return Cxx


def getQs(nX):
    # Create the q matrices
    q = [0] * 5
    for i in range(-2, 3):
        line = np.ones(nX - np.abs(i))
        if i>=0:
            line[0] = 0
        q[i] = sp.diags(line, i)
        if i == -2:
            line = np.zeros(nX - 2)
            line[0] = 1
            q[i] += sp.diags(line, 2)
        elif i>0:
            for j in np.arange(0, i, 1):
                line = np.zeros(nX - np.abs(j))
                line[-1] = 1
                q[i] += sp.diags(line, j)

    return np.asarray(q)


