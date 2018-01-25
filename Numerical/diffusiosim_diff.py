# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:18:26 2017

@author: quentinpeter
"""
import numpy as np
from scipy.linalg import toeplitz
from numpy.linalg import eigvals


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
        nx = nx + 2
        X = np.linspace(-dx, L, nx) - dx / 2
        self._X = X
        self._dx = dx
        self._nx = nx
        self._L = L

        # init proteins concentration
        self._Cprot = np.zeros_like(X)
        self._Cprot[:2] = 1
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
        I = np.eye(nx)
        dx = self._dx
        Dprot = self._settings["D protein"]
        dt0 = np.min(dx)**2 / Dprot / self._settings["time step factor"]/20
        Cxx = Dprot * getCxx(nx, np.max(dx))
        while t > 0:

            # Update proteins
            dF, dt = self._Dx(dt0)
            dF += Cxx

            # 0th position doesn't move
            dF[0] = 0

            # Implicit method
            F = np.linalg.inv(I - dt * dF)
#            F = I + dt*dF
            assert(len(np.shape(F)) == 2)
            self._Cprot = F@self._Cprot
            t = t - dt
            self._t += dt

        print('eig', np.max(np.abs(eigvals(F))), np.max(self._Cprot))

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

        qp = 1 / 2 * (q[1] - q[0])
        qp[0] = 0
        qp = qp + q[0]
        qm = 1 / 2 * (q[0] - q[-1])
        qm[:2] = 0
        qm = qm + q[-1]

        self._qp = qp
        self._qm = qm

    def _Dx2(self, dt):
        """Alternative method"""
        dx = self._dx
        q = self._q
        D = self._settings["D salt"]
        Cin = self._settings["salt in"]
        Cout = self._settings["salt out"]
        Ddp = self._settings["D diffusiophoresis"]

        GlnC, GGlnC = getdiffs(self._X + dx / 2, self._t,
                               D, self._L, Cin, Cout)
#        Cx = 1/np.max(dx)/12*(-q[2]+8*q[1]-8*q[-1]+q[-2])
    #    Cx = 1/np.max(dx)/4*(q[1]+3*q[0]-5*q[-1]+q[-2])
        Cx = 1 / np.max(dx) / 2 * (q[1] - q[-1])
    #    Cx = 1/np.max(dx)*(q[0]-q[-1])
#        Cx = 1/np.max(dx)*(q[1]-q[0])

        Cx[0] = 0

        return Ddp * (GlnC[:, np.newaxis] * Cx + GGlnC[:, np.newaxis] * q[0]), dt

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
        up[1:-1] = uhalf
        # Incorrect BUT don't care
        um = np.zeros(nx)
        um[2:] = uhalf

        dt2 = dx / np.max(np.abs(up)) / dtfrac
        if dt2 < dt:
            dt = dt2

        up = up[:, np.newaxis]
        um = um[:, np.newaxis]

        fp = up * self._qp
        fm = um * self._qm

        Dx = (fm - fp) / dx

#        neg = -Ddp*(Cin-Cout) < 0
#        assert not neg
#        Dx = (um*q[-1 + neg] - up*q[0 + neg]
#              + (.5 - neg) * ((um*sigdx[-1 + neg] - up*sigdx[0 + neg])
#                              - dt / dx * (um**2*sigdx[-1 + neg]
#                              - up**2*sigdx[0 + neg])))
#
#        Dx = 1/2*(um*(q[-1]+q[0]) - up*(q[0]+q[1]))
#        Dx /= dx
        return Dx, dt
#    @profile

    def _initsalt(self, nN=100):
        """Prepare variables for the salt concentration distribution"""
        L = self._L
        dx = self._dx
        x = self._X[2:] - dx / 2
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
    
    def Csalt(self, t):
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
        
        return Cs


def Ct(t, N, D, L):
    return np.exp(-D * (np.pi * (1 / 2 + N) / L)**2 * t)


def C(x, t, D, L, nN=100):
    x = x[:, np.newaxis]
    N = np.arange(nN)[np.newaxis, :]
    ret = (1 / (N + 0.5)) * Ct(t, N, D, L) * \
        np.sin((np.pi / 2 + N * np.pi) / L * x)
    return (2 / np.pi) * np.sum(ret, 1)


def dC(x, t, D, L, nN=100):
    x = x[:, np.newaxis]
    N = np.arange(nN)[np.newaxis, :]
    ret = (Ct(t, N, D, L) * np.cos((np.pi / 2 + N * np.pi) / L * x))
    return (2 / L) * np.sum(ret, 1)


def ddC(x, t, D, L, nN=100):
    x = x[:, np.newaxis]
    N = np.arange(nN)[np.newaxis, :]
    ret = (N + 0.5) * Ct(t, N, D, L) * np.sin((np.pi / 2 + N * np.pi) / L * x)
    return (-2 * np.pi / L**2) * np.sum(ret, 1)

#@profile


def getdiffs(x, t, D, L, Cin, Cout, nN=100):
    x = x[:, np.newaxis]
    N = np.arange(nN)[np.newaxis, :]

    Cts = Ct(t, N, D, L)
    xN = (np.pi / 2 + N * np.pi) / L * x

    cos = np.cos(xN)
    sin = np.sin(xN)

    Cs = (2 / np.pi) * np.sum((1 / (N + 0.5)) * Cts * sin, 1)
    dCs = (2 / L) * np.sum(Cts * cos, 1)
    ddCs = (-2 * np.pi / L**2) * np.sum((N + 0.5) * Cts * sin, 1)

    dlnCs = dCs / (Cout / (Cin - Cout) + Cs)

    return dlnCs, (ddCs / (Cout / (Cin - Cout) + Cs) - (dlnCs)**2)

# get differential operators matrix


def getCxx(nx, dx, reservoir=False):
    q = getQs(nx)
    # toeplitz creation of matrice which repeat in diagonal
    Cxx = q[-1] - 2*q[0] + q[1]
    Cxx[0, :] = 0
    Cxx[-1, -1] = -1
    Cxx /= dx**2
    # Change border conditions if asked
    if reservoir:
        Cxx[-1, :] = 0

    return Cxx


def getCx(nx, dx, reservoir=False):
    # get grad y operator
    udiag1 = np.ones(nx - 1)
    udiag2 = np.ones(nx - 2)
    Cx = (np.diag(udiag2, -2)
          + np.diag(-8 * udiag1, -1)
          + np.diag(8 * udiag1, 1)
          + np.diag(-udiag2, 2))
    Cx[0, :] = 0
    Cx[1, 0] = -7
    Cx[-2:, -1] = 7
    Cx /= (12 * dx)
    if reservoir:
        Cx[-1, :] = 0
    return Cx


def getQs(nX):
    # Create the q matrices
    q = np.zeros((5, nX, nX))
    for i in range(-2, 3):
        q[i] = np.diag(np.ones(nX - np.abs(i)), i)

    # Border
    q[-2, :2, 0] = 1
    q[-1, 0, 0] = 1
    q[1, -1, -1] = 1
    q[2, -2:, -1] = 1

    return q
