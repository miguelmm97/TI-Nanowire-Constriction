"""
Status variables dataclass for the Topological Transport project IPR calculation.
These variables are accessed and modified at runtime, and encode
the status of the particular simulation or set up a particular instance of a model.
They depend on the msv variables from the .toml file, so if we want to change their definition,
it should be done from inside this dataclass, the .toml file should only 
contain parameters.
"""

import numpy as np
from numpy import pi
from dataclasses import dataclass, field


@dataclass
class MyStatusVariablesIPR:
    
    msv: dataclass

    # Iterative arrays
    delta_x:               np.ndarray[int, np.dtype(np.float64)] = field(init=False)
    delta_theta:           np.ndarray[int, np.dtype(np.float64)] = field(init=False)
    fermi:                 np.ndarray[int, np.dtype(np.float64)] = field(init=False)

    # Setting up the model
    x:                     np.ndarray[int, np.dtype(np.float64)] = field(init=False)
    Ntheta_grid:           np.ndarray[int, np.dtype(np.float64)] = field(init=False)
    theta:                 np.ndarray[int, np.dtype(np.float64)] = field(init=False)
    r_vec:                 np.ndarray[int, np.dtype(np.float64)] = field(init=False)
    Ntheta_plot:           int = field(init=False)


    # Preallocation
    Vstd_th_2d:            float = field(init=False)
    scatt_states_up:       np.ndarray[int, np.dtype(np.complex128)] = field(init=False)
    scatt_states_down:     np.ndarray[int, np.dtype(np.complex128)] = field(init=False)
    IPR_up:                np.ndarray[int, np.dtype(np.float64)] = field(init=False)
    IPR_down:              np.ndarray[int, np.dtype(np.float64)] = field(init=False)
    fit_params1:           np.ndarray[int, np.dtype(np.float64)] = field(init=False)
    fit_params2:           np.ndarray[int, np.dtype(np.float64)] = field(init=False)

    def pop_vars(self):
        """
        Create and populate variables for the calculation of the IPR
        """
        # Status of the simulation
        self.delta_x            = np.linspace(self.msv.corr_length / 2., 5. * self.msv.corr_length, self.msv.N_deltax)
        self.delta_theta        = (1. / self.msv.r) * np.linspace(self.msv.corr_length / 2., 5. * self.msv.corr_length, self.msv.N_deltatheta)
        self.fermi              = np.linspace(self.msv.fermi_0_IPR, self.msv.fermi_end_IPR, self.msv.fermi_length_IPR)

        # Setting up the model
        self.x                  = np.linspace(0., self.msv.L, self.msv.Nx)
        self.Ntheta_grid        = self.msv.Ntheta_fft if self.msv.dimension == '2d' else 1
        self.Ntheta_plot        = self.msv.Ntheta_fft if self.msv.dimension == '2d' else self.msv.default_Ntheta_plot
        self.theta              = np.linspace(0., 2. * pi, self.Ntheta_plot)
        self.r_vec              = np.repeat(self.msv.r, self.x.shape[0])

        # Preallocation
        self.Vstd_th_2d         = np.sqrt(self.msv.dis_strength / (2. * pi)) * self.msv.vf / self.msv.corr_length
        self.scatt_states_up    = np.zeros((len(self.msv.res_index), len(self.delta_x), self.Ntheta_plot, self.msv.Nx - 1), dtype=np.complex128)
        self.scatt_states_down  = np.zeros((len(self.msv.res_index), len(self.delta_x), self.Ntheta_plot, self.msv.Nx - 1), dtype=np.complex128)
        self.IPR_up             = np.zeros((len(self.msv.res_index), len(self.delta_x)), dtype=np.float64)
        self.IPR_down           = np.zeros((len(self.msv.res_index), len(self.delta_x)), dtype=np.float64)
        self.fit_params1        = np.zeros(self.IPR_up.shape[0], dtype=np.float64)
        self.fit_params2        = np.zeros(self.IPR_up.shape[0], dtype=np.float64)

        return self

