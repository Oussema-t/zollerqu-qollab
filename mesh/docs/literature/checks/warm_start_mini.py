import numpy as np
from scipy.linalg import expm
X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1]).astype(complex)
RX = lambda t: expm(-1j*t*X/2); RY = lambda t: expm(-1j*t*Y/2); RZ = lambda t: expm(-1j*t*Z/2)
th = lambda c: 2*np.arcsin(np.sqrt(c))
mixA = lambda c, b: RY(th(c)) @ RZ(-2*b) @ RY(-th(c))   # paper Eq.(2) mixer; circuit order ry(-th), rz(-2b), ry(th)
mixB = lambda c, b: RY(-th(c)) @ RZ(-2*b) @ RY(th(c))   # paper Sec. 2.3 mixer; circuit order ry(th), rz(-2b), ry(-th)
k0, k1 = np.array([1, 0], complex), np.array([0, 1], complex)
b = 0.37
assert np.allclose(mixA(0.5, b), expm(+1j*b*X)) and np.allclose(mixA(0.5, b), RX(-2*b))      # 1
assert np.allclose(mixB(0.5, b), expm(-1j*b*X)) and np.allclose(mixB(0.5, b), RX(+2*b))      # 2
for c in (0.1, 0.25, 0.6, 0.75, 0.93):
    psi = RY(th(c)) @ k0
    H = np.array([[2*c-1, -2*np.sqrt(c*(1-c))], [-2*np.sqrt(c*(1-c)), 1-2*c]])
    assert np.isclose(abs(psi[1])**2, c)                                                    # 3
    assert np.allclose(H @ psi, -psi) and np.allclose(mixA(c, b), expm(-1j*b*H))             # 4
    assert np.allclose(mixA(c, b) @ psi, np.exp(1j*b)*psi)                                   # 5
    assert abs(np.vdot(psi, mixB(c, b) @ psi)) < 1 - 1e-3                                    # 6
assert np.allclose(mixB(0.25, np.pi/2) @ RY(th(0.25)) @ k0, -1j*k1)                          # 7
assert np.allclose(mixB(0.75, np.pi/2) @ RY(th(0.75)) @ k0, -1j*k0)                          # 8
print("all 8 warm-start checks pass")
