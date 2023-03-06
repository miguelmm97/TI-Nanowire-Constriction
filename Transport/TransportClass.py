from dataclasses import dataclass, field
import numpy as np
from numpy import pi
from numpy.linalg import inv
from scipy.linalg import expm
import numpy as np
import numba as nb
from numba import njit  # "nopython just in time (compiler)"

nb.set_num_threads(1)

# Constants
hbar = 1e-34                # Planck's constant in Js
nm = 1e-9                   # Conversion from nm to m
ams = 1e-10                 # Conversion from Å to m
e = 1.6e-19                 # Electron charge in C
phi0 = 2 * pi * hbar / e    # Quantum of flux

# Pauli matrices
sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)



def step(x1, x2, sigma=None):
    # Smoothened step function theta(x1-x2)
    if sigma is None: return np.heaviside(x1 - x2, 1)
    else: return 0.5 + (1 / pi) * np.arctan(sigma * (x1 - x2))

def geom_nc(x, x1, x2, r1, r2, sigma=None):
    # x1, r1: Initial widest part
    # x2, r2: Final narrow part
    return r1 + (r2 - r1) * step(x, x2, sigma) + ((r2 - r1) / (x2 - x1)) * (x - x1) * (step(x, x1, sigma) - step(x, x2, sigma))

@njit(parallel=True, cache=True)
def M_Ax(modes, dx, w, h, B_perp=0):

    if w is None or h is None: return 0
    else:
        P = 2 * (w + h)                                                    # Perimeter of the nanostructure at x (in nm)
        r = w / (w + h)                                                    # Aspect ratio of the nanostructure at x (in nm)
        C = - 1j * (nm ** 2 / hbar) * e * B_perp * P / pi ** 2             # Auxiliary constant for the matrix elements
        M = np.zeros(shape=(len(modes), len(modes)), dtype=np.complex128)  # Mode mixing matrix for the vector potential

        if B_perp != 0:
            i = 0
            for n1 in modes:
                for n2 in modes:
                    j = 0
                    if (n1 - n2) % 2 != 0:
                        m = n1 - n2
                        M[i, j] = C * ((-1) ** ((m + 1) / 2)) * np.sin(m * pi * r / 2) / m ** 2
                    j += 1
                i += 1

        return np.kron(sigma_0, M) * dx

@njit(parallel=True, cache=True)
def M_theta(modes, dx, R, dR, w=None, h=None, B_par=0):

    C = 0.5 * (nm ** 2) * e * B_par / hbar
    A_theta = C * R ** 2 if (w is None or h is None) else C * (w * h / pi)  # A_theta = eBa²/2hbar
    M = (np.sqrt(1 + dR ** 2) / R) * np.diag(modes - 0.5 + A_theta)         # 1/R (n-1/2 + eBa²/2hbar) term

    return np.kron(sigma_x, M) * dx

@njit(parallel=True, cache=True)
def M_EV(modes, dx, dR, E, vf):

    M = (1j / vf) * np.sqrt(1 + dR ** 2) * (E * np.eye(len(modes), dtype=np.complex128))  # i ( E delta_nm + V_nm) / vf term
    return np.kron(sigma_z, M) * dx

def transfer_to_scattering(transfer_matrix: 'np.ndarray[np.complex128]'):

    # Transform from the transfer matrix to the scattering matrix
    # transfer_matrix: Transfer matrix to translate to scattering
    # n_modes: Number of modes contributing to transport (N_states/2 because spin momentum locking)

    n_modes = int(transfer_matrix.shape[0] / 2)
    inv_T  = transfer_matrix[0: n_modes, 0: n_modes]  # t^\dagger ^(-1)
    inv_Tp = transfer_matrix[n_modes:, n_modes:]      # t' ^(-1)
    inv_R  = transfer_matrix[n_modes:, 0: n_modes]    # -t'^(-1) r
    inv_Rp = transfer_matrix[0: n_modes, n_modes:]    # r't'^(-1)

    T  = inv(inv_T).T.conj()                          # t
    Tp = inv(inv_Tp)                                  # t'
    R  = - Tp @ inv_R                                 # r
    Rp = inv_Rp @ Tp                                  # r'

    scat_matrix = np.block([[R, Tp], [T, Rp]])        # scattering matrix

    return scat_matrix

def scat_product(s1: 'np.ndarray[np.complex128]', s2: 'np.ndarray[np.complex128]'):

    # Product combining two scattering matrices
    # s1, s2: Scattering matrices to combine
    # n_modes: Number of modes contributing to transport (N_states/2 because spin momentum locking)

    if s1.shape != s2.shape:
        raise ValueError(" Different size for scattering matrices")

    n_modes = int(s1.shape[0] / 2)
    r1, r2   = s1[0: n_modes, 0: n_modes], s2[0: n_modes, 0: n_modes]  # r1, r2
    r1p, r2p = s1[n_modes:, n_modes:], s2[n_modes:, n_modes:]          # r1', r2'
    t1, t2   = s1[n_modes:, 0: n_modes], s2[n_modes:, 0: n_modes]      # t1, t2
    t1p, t2p = s1[0: n_modes, n_modes:], s2[0: n_modes, n_modes:]      # t1', t2'

    r1pr2t1 = inv(np.eye(n_modes) - r1p @ r2) @ t1
    r2r1pt2p = inv(np.eye(n_modes) - r2 @ r1p) @ t2p

    R  = r1 + t1p @ r2 @ r1pr2t1                                       # r
    Rp = r2p + t2 @ r1p @ r2r1pt2p                                     # r'
    T  = t2 @ r1pr2t1                                                  # t
    Tp = t1p @ r2r1pt2p                                                # t'

    scat_matrix = np.block([[R, Tp], [T, Rp]])                         # scattering matrix
    return scat_matrix

def transport_checks(transfer_matrix=None, scat_matrix=None, conservation='off', unitarity='off', completeness='off'):
    """
    Checks current conservation, unitarity and completeness of the transport calculation.

    Observation:
    -----------
    The exact unitarity of the scattering matrix and completeness of reflexion/transmission is lost upon numerical precision.
    It should get better the finer the grid in real space becomes. A somewhat less restrictive check is to calculate the
    different traces associated to these requirements. That's the part that is not commented out.
    The part that is commented out is the full check, which generically is not true.
    """

    n_modes = int(len(scat_matrix[0, :]) / 2) if transfer_matrix is None else int(len(transfer_matrix[0, :]) / 2)
    sigma_z = np.array([[1, 0], [0, -1]])  # Pauli z

    # Conservation of the current
    if transfer_matrix is not None and conservation == 'on':
        check1 = transfer_matrix @ np.kron(sigma_z, np.eye(n_modes)) @ np.conj(transfer_matrix.T)
        if not np.allclose(np.kron(sigma_z, np.eye(n_modes)), check1): raise AssertionError('Transfer matrix does not conserve current')

    # Unitarity of the scattering matrix
    if scat_matrix is not None and unitarity == 'on':
        check2 = scat_matrix @ np.conj(scat_matrix.T)
        # print("Unitarity of S: tr(S^\dagger S) - n_modes= ", np.trace(check2) - (2 * n_modes))
        if not np.allclose(np.eye(2 * n_modes), check2, 1e-15): raise AssertionError('Scattering matrix is not unitary')


    # Completeness of reflection and transmission
    if scat_matrix is not None and completeness == 'on':
        t, r = scat_matrix[n_modes:, 0: n_modes], scat_matrix[0: n_modes, 0: n_modes]
        t_dagger, r_dagger = np.conj(t.T), np.conj(r.T)
        # print("Completeness of reflexion/transmission: n:modes - tr(r^\dagger r) - tr(t^\dagger t) =", n_modes-np.trace(r_dagger @ r)-np.trace(t_dagger @ t))
        if not np.allclose(n_modes-np.trace(r_dagger @ r), np.trace(t_dagger @ t)): raise AssertionError('Reflexion doesnt add up to transmission')

@dataclass
class transport:

    """ Transport calculations on 3dTI nanostructures based on their effective surface theory."""
    vf: float                               # Fermi velocity in meV nm
    B_perp: float                           # Magnetic field perpendicular to the axis of the nanostructure
    B_par: float                            # Magnetic field parallel to the axis of the nanostructure
    geometry: dict = field(init=False)      # Dictionary codifying the geometry
    scattering: dict = field(init=False)
    n_regions: int = field(init=False)      # Number of different regions in the nanostructure

    def __post_init__(self):
        self.geometry = {}
        self.n_regions = 0


    def add_nw(self, x0, xf, mu=0, n_points=None, r=None, w=None, h=None):
        """
        Adds a nanowire to the geometry of the nanostructure.

        Params:
        -------
        x0:    {float} Initial point of the nanowire
        xf:    {float} Final point of the nanowire
        r:     {float} Radius of the nanowire
        w:     {float} Width of the nanowire
        h:     {float} Height of the nanowire

        Return:
        -------
        None, but updates self. geometry

        """

        if self.n_regions != 0 and x0 != self.geometry[self.n_regions - 1]['xf']: raise ValueError('Regions dont match')
        if r is None and (w is None or h is None): raise ValueError('Need to specify r or (w, h)')

        self.geometry[self.n_regions] = {}                                     # Add new region to the geometry
        self.geometry[self.n_regions]['type'] = 'nw'                           # Type of region
        self.geometry[self.n_regions]['x0'] = x0                               # Initial point
        self.geometry[self.n_regions]['xf'] = xf                               # Final point
        if n_points is None: self.geometry[self.n_regions]['dx'] = 100         # X increment
        else: self.geometry[self.n_regions]['dx'] = abs(xf - x0) / n_points    # X increment
        self.geometry[self.n_regions]['n_points'] = n_points                   # Number of points in the discretisation
        self.geometry[self.n_regions]['mu'] = mu                               # Chemical potential

        self.geometry[self.n_regions]['r'] = (w + h) / pi if r is None else r  # Radius
        self.geometry[self.n_regions]['w'] = w                                 # Width
        self.geometry[self.n_regions]['h'] = h                                 # Height
        self.n_regions += 1                                                    # Add new region to the geometry

    def add_nc(self, x0, xf, n_points, mu=0, sigma=None, r1=None, r2=None, w1=None, w2=None, h1=None, h2=None):
        """
        Adds a nanowire to the geometry of the nanostructure.

        Params:
        -------
        x0:         {float} Initial point of the nanocone
        xf:         {float} Final point of the nanocone
        n_points:   {int}   Number of points discretising the cone
        sigma:      {float} Smoothing factor for the step functions (the bigger, the sharper)
        r1:         {float} Initial radius of the nanocone
        w1:         {float} Initial width of the nanocone
        h1:         {float} Initial height of the nanocone
        r2:         {float} Final radius of the nanocone
        w2          {float} Final width of the nanocone
        h2:         {float} Final height of the nanocone

        Return:
        -------
        None, but updates self. geometry

        """
        if self.n_regions != 0 and x0 != self.geometry[self.n_regions - 1]['xf']: raise ValueError('Regions dont match')
        if r1 is None and (w1 is None or h1 is None): raise ValueError('Need to specify r or (w, h)')
        if r2 is None and (w2 is None or h2 is None): raise ValueError('Need to specify r or (w, h)')

        self.geometry[self.n_regions] = {}                                            # Add new region to the geometry
        self.geometry[self.n_regions]['type'] = 'nc'                                  # Type of region
        self.geometry[self.n_regions]['x0'] = x0                                      # Initial point
        self.geometry[self.n_regions]['xf'] = xf                                      # Final point
        self.geometry[self.n_regions]['dx'] = abs(xf - x0)/n_points                   # X increment
        self.geometry[self.n_regions]['n_points'] = n_points                          # Number of points in the discretisation
        self.geometry[self.n_regions]['mu'] = mu  # Chemical potential

        r1 = (w1 + h1) / pi if r1 is None else r1                                     # Initial radius
        r2 = (w2 + h2) / pi if r2 is None else r2                                     # Final radius
        x = np.linspace(x0, xf, n_points)                                             # Discretised region

        self.geometry[self.n_regions]['r'] = geom_nc(x, x0, xf, r1, r2, sigma)        # Radius
        if w1 is None or w2 is None: self.geometry[self.n_regions]['w'] = None        # Width
        else: self.geometry[self.n_regions]['w'] = geom_nc(x, x0, xf, w1, w2, sigma)  # Width
        if h1 is None or h2 is None: self.geometry[self.n_regions]['h'] = None        # Height
        else: self.geometry[self.n_regions]['h'] = geom_nc(x, x0, xf, h1, h2, sigma)  # Height
        self.n_regions += 1                                                           # Add new region to the geometry


    def get_Landauer_conductance(self, l_cutoff, E):

        modes = np.arange(-l_cutoff, l_cutoff+1)
        for i in range(0, self.n_regions):

            if self.geometry[i]['type'] == 'nw':
                M = M_EV(modes, self.geometry[i]['dx'], 0, E + self.geometry[i]['mu'], self.vf)
                M += M_theta(modes, self.geometry[i]['dx'], self.geometry[i]['r'], 0, self.geometry[i]['w'], self.geometry[i]['h'], B_par=self.B_par)
                M += M_Ax(modes, self.geometry[i]['dx'], self.geometry[i]['w'], self.geometry[i]['h'], B_perp=self.B_perp)

                if self.geometry[i]['n_points'] is None:
                    T = expm(M * (self.geometry[i]['xf'] - self.geometry[i]['x0']) / self.geometry[i]['dx'])
                    S = transfer_to_scattering(T) if i == 0 else scat_product(S, transfer_to_scattering(T))
                    if np.isnan(S).any(): raise OverflowError('Length to long to calculate the scattering matrix directly. Need for discretisation!')
                else:
                    dT = expm(M)
                    dS = transfer_to_scattering(dT)
                    for j in range(self.geometry[i]['n_points']):
                        S = dS if (i == 0 and j == 0) else scat_product(S, dS)

            else:
                for j in range(self.geometry[i]['n_points'] - 1):
                    dr = self.geometry[i]['r'][j + 1] - self.geometry[i]['r'][j]
                    try:
                        w = self.geometry[i]['w'][j]; h = self.geometry[i]['h'][j];
                    except TypeError:
                        w = None; h = None
                    M = M_EV(modes, self.geometry[i]['dx'], dr, E, self.vf)
                    M += M_theta(modes, self.geometry[i]['dx'], self.geometry[i]['r'][j], dr, w=w, h=h, B_par=self.B_par)
                    M += M_Ax(modes, self.geometry[i]['dx'], w, h, B_perp=self.B_perp)
                    dT = expm(M)
                    dS = transfer_to_scattering(dT)
                    S = dS if (i == 0 and j == 0) else scat_product(dS, S)
                    # transport_checks(transfer_matrix=dT, scat_matrix=S, conservation='on', unitarity='on', completeness='on')
                    n_modes = int(S.shape[0] / 2)
                    t = S[n_modes:, 0: n_modes]
                    G = np.trace(t.T.conj() @ t)
                    a = 1

        n_modes = int(S.shape[0] / 2)
        t = S[n_modes:, 0: n_modes]
        G = np.trace(t.T.conj() @ t)
        return G







