import numpy as np
import pytest
from pyscf import dft, gto

GLY = """N  -1.876  0.267 0.000; C  -0.584 -0.389 0.000; C   0.590  0.574 0.000; O   0.430  1.783 0.000
O   1.799 -0.010 0.000; H  -2.594 -0.443 0.000; H  -1.997  0.888 0.807; H  -0.505 -1.034 0.880
H  -0.505 -1.034 -0.880; H  2.474  0.692 0.000"""


def acene(n, a=1.40, ch=1.09):
    atoms, h = [], np.sqrt(3) / 2 * a
    xs = [i * 2 * h for i in range(n + 1)]
    for x in xs:
        atoms += [("C", x, a / 2), ("C", x, -a / 2)]
    for i in range(n):
        xm = xs[i] + h
        atoms += [("C", xm, a), ("C", xm, -a), ("H", xm, a + ch), ("H", xm, -a - ch)]
    for x, sgn in [(xs[0], -1), (xs[-1], 1)]:
        for y in (a / 2, -a / 2):
            atoms.append(("H", x + sgn * ch * np.sqrt(3) / 2, y + np.sign(y) * ch / 2))
    return [(s, x, y, 0.0) for s, x, y in atoms]


@pytest.fixture(scope="session")
def gly():
    return gto.M(atom=GLY, basis="def2-svp", verbose=0)


@pytest.fixture(scope="session")
def gly_triplet():
    return gto.M(atom=GLY, basis="def2-svp", spin=2, verbose=0)


@pytest.fixture(scope="session")
def gly_rks(gly):
    mf = dft.RKS(gly, xc="b3lyp")
    mf.kernel()
    return mf


@pytest.fixture(scope="session")
def gly_uks(gly_triplet):
    mf = dft.UKS(gly_triplet, xc="b3lyp")
    mf.kernel()
    return mf


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def sym(rng, n, scale=1e-2):
    d = rng.standard_normal((n, n)) * scale
    return d + d.T
