"""The model kernel: the Hartree response fitted in a small auxiliary basis."""
import numpy as np
from pyscf import df, lib
from pyscf.data import radii
from pyscf.data.elements import charge as elem_charge


class AuxHartreeModel:
    """Hartree kernel fitted in an auxiliary basis, K~[dD]_mn = sum_PQ (mn|P) (J^-1)_PQ sum_ls (Q|ls) dD_ls.

    kind = "spd":  one s, one p and one d Gaussian on every atom heavier than He and one s Gaussian on H and He,
                   all with exponent theta / R_cov^2 (R_cov: covalent radius), as in TDDFT-ris.  MK(spd), the
                   default of the paper.
    kind = "s":    one s Gaussian per atom with the same exponent, MK(s).
    kind = <name>: any PySCF auxiliary basis, e.g. "def2-universal-jfit", MK(J-fit), close to the exact
                   Hartree kernel.

    store = "dense": the three-index integrals as (naux, nao, nao); needed by the low-rank (Woodbury) solver and
                     by the exchange and Fock-form corrections.
    store = "tril":  the lower triangle of AO pairs as (npair, naux), for the conjugate-gradient solver.  screen > 0
                     drops shell pairs whose Gaussian-product envelope exp(-a b R^2 / (a + b)) of the most diffuse
                     primitives is below it.
    exchange=True:   add the hybrid's exact exchange, resolved in the same auxiliary basis (dense store).

    The model is kept in double precision.  The Coulomb metric J of the minimal basis has a condition number of
    1e7-1e11 on molecules, and J^-1 amplifies the rounding of the fitted density to a relative error of the model
    potential of about 4e-8 cond(J)^1/2 in single precision.
    """

    def __init__(self, kind="spd", theta=0.2, lindep=1e-10, store="dense", screen=0.0, exchange=False):
        self.kind, self.theta, self.lindep = kind, theta, lindep
        self.store, self.screen, self.exchange = store, screen, exchange

    def build(self, mol):
        if self.kind in ("s", "spd"):
            basis = {}
            for label, _ in mol._atom:
                z = elem_charge(label)
                alpha = self.theta / radii.COVALENT[z] ** 2
                shells = [[0, [alpha, 1.0]]]
                if self.kind == "spd" and z > 2:
                    shells += [[1, [alpha, 1.0]], [2, [alpha, 1.0]]]
                basis[label] = shells
            auxmol = df.addons.make_auxmol(mol, basis)
        else:
            auxmol = df.addons.make_auxmol(mol, self.kind)
        nao, naux = mol.nao, auxmol.nao
        j2c = auxmol.intor("int2c2e")
        w, v = np.linalg.eigh(j2c)
        keep = w > self.lindep * w.max()
        self.W = (v[:, keep] / w[keep]) @ v[:, keep].T
        self.cond_j = float(w.max() / w[keep].min())
        if self.store == "dense":
            int3c = df.incore.aux_e2(mol, auxmol, "int3c2e", aosym="s1").reshape(nao, nao, naux)
            self.T = np.ascontiguousarray(int3c.transpose(2, 0, 1))
        else:
            self._build_tril(mol, auxmol)
        self.naux = naux
        self._auxmol = auxmol
        return self

    def _build_tril(self, mol, auxmol, block=48):
        nao = mol.nao
        self.nao, self.diag = nao, np.arange(nao) * (np.arange(nao) + 3) // 2
        self.mask = None
        if self.screen > 0:
            # Overlaps are no good here: same-centre s-p pairs have zero overlap but a dipole.
            amin = np.array([mol.bas_exp(i).min() for i in range(mol.nbas)])
            xyz = mol.atom_coords()[[mol.bas_atom(i) for i in range(mol.nbas)]]
            r2 = ((xyz[:, None] - xyz[None]) ** 2).sum(-1)
            env = np.exp(-amin[:, None] * amin[None] / (amin[:, None] + amin[None]) * r2)
            shell_of = np.repeat(np.arange(mol.nbas), np.diff(mol.ao_loc_nr()))
            keep = env[shell_of[:, None], shell_of[None, :]] >= self.screen
            self.mask = np.nonzero(keep[np.tril_indices(nao)])[0]
        npair = nao * (nao + 1) // 2 if self.mask is None else self.mask.size
        self.T = np.empty((npair, auxmol.nao))
        aloc = auxmol.ao_loc_nr()
        for k0 in range(0, auxmol.nbas, block):
            k1 = min(k0 + block, auxmol.nbas)
            blk = df.incore.aux_e2(mol, auxmol, "int3c2e", aosym="s2ij",
                                   shls_slice=(0, mol.nbas, 0, mol.nbas, k0, k1))
            self.T[:, aloc[k0]:aloc[k1]] = blk if self.mask is None else blk[self.mask]
        self.npair = npair

    def _pack(self, dm):
        d = lib.pack_tril(dm + dm.T)
        d[self.diag] *= 0.5
        return d if self.mask is None else d[self.mask]

    def tvec(self, dD_total):
        """(P|dD): the auxiliary-basis projection of a total density change."""
        if self.store == "dense":
            return np.einsum("Pmn,mn->P", self.T, dD_total)
        return self._pack(dD_total) @ self.T

    def apply(self, dD_total):
        """K~[dD] in the AO basis."""
        u = self.W @ self.tvec(dD_total)
        if self.store == "dense":
            return np.einsum("Pmn,P->mn", self.T, u)
        v = self.T @ u
        if self.mask is not None:
            full = np.zeros(self.nao * (self.nao + 1) // 2)
            full[self.mask] = v
            v = full
        return lib.unpack_tril(v)

    def vo(self, Co, Cv):
        """(P|ai) for virtual orbitals Cv and occupied orbitals Co (dense store)."""
        return np.einsum("mv,Pmo->Pvo", Cv, self.T @ Co, optimize=True)
