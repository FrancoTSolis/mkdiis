"""Extrapolation of the Fock matrix: commutator DIIS (Pulay) and ADIIS (Hu and Yang).

Both return the extrapolated density together with the extrapolated Fock matrix, since the model-kernel
correction needs the density change of the Roothaan step, D_R - D_bar.
"""
import numpy as np
from scipy.optimize import minimize


class CDIIS:
    """Pulay's DIIS: coefficients with sum 1 that minimize the norm of the extrapolated commutator."""

    def __init__(self, space=8):
        self.space = space
        self.F, self.D, self.E = [], [], []

    def update(self, F, D, err):
        self.F.append(F)
        self.D.append(D)
        self.E.append(err.ravel())
        if len(self.E) > self.space:
            self.F.pop(0), self.D.pop(0), self.E.pop(0)
        n = len(self.E)
        B = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(i + 1):
                B[i, j] = B[j, i] = np.dot(self.E[i], self.E[j])
        B[n, :n] = B[:n, n] = -1.0
        rhs = np.zeros(n + 1)
        rhs[n] = -1.0
        w, v = np.linalg.eigh(B)
        keep = np.abs(w) > 1e-14 * np.abs(w).max()
        c = (v[:, keep] @ ((v[:, keep].T @ rhs) / w[keep]))[:n]
        F_bar = sum(ci * Fi for ci, Fi in zip(c, self.F))
        D_bar = sum(ci * Di for ci, Di in zip(c, self.D))
        return F_bar, D_bar


class ADIIS:
    """Hu and Yang (2010): coefficients on the simplex that minimize the ARH energy model
    f(c) = 2 sum_i c_i <D_i - D_n, F_n> + sum_ij c_i c_j <D_i - D_n, F_j - F_n>."""

    def __init__(self, space=8):
        self.space = space
        self.F, self.D = [], []

    def update(self, F, D):
        self.F.append(F)
        self.D.append(D)
        if len(self.F) > self.space:
            self.F.pop(0), self.D.pop(0)
        n = len(self.F)
        Fn, Dn = self.F[-1], self.D[-1]
        dD = [(Di - Dn).ravel() for Di in self.D]
        dF = [(Fj - Fn).ravel() for Fj in self.F]
        a = np.array([np.dot(x, Fn.ravel()) for x in dD])
        B = np.array([[np.dot(x, y) for y in dF] for x in dD])
        B = 0.5 * (B + B.T)

        def fun(t):
            c = t ** 2 / np.dot(t, t)
            return 2.0 * a @ c + c @ B @ c

        t0 = np.full(n, 0.1)
        t0[-1] = 1.0
        t = minimize(fun, t0, method="BFGS", options=dict(gtol=1e-12, maxiter=500)).x
        c = t ** 2 / np.dot(t, t)
        return sum(ci * Fi for ci, Fi in zip(c, self.F)), sum(ci * Di for ci, Di in zip(c, self.D))
