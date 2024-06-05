"""
Status variables dataclass for the Topological Transport project transport calculation.
These variables are accessed and modified at runtime, and encode
the status of the particular simulation or set up a particular instance of a model.
They depend on the msv variables from the .toml file, so if we want to change their definition,
it should be done from inside this dataclass, the .toml file should only
contain parameters.
"""

import numpy as np
from numpy import pi
import h5py
from dataclasses import dataclass, field, fields


@dataclass
class MyStatusVariablesScatt:
    msv: dataclass


    # Setting up the model
    x:                  np.ndarray[int, np.dtype(np.float64)] = field(init=False)
    Ntheta_grid:        np.ndarray[int, np.dtype(np.float64)] = field(init=False)
    theta:              np.ndarray[int, np.dtype(np.float64)] = field(init=False)
    r_vec:              np.ndarray[int, np.dtype(np.float64)] = field(init=False)
    V_real:             np.ndarray[int, np.dtype(np.float64)] = field(init=False)
    V_fft:              np.ndarray[int, np.dtype(np.complex128)] = field(init=False)
    fermi:              np.ndarray[int, np.dtype(np.float64)] = field(init=False)
    Ntheta_plot:        int = field(init=False)

    # Preallocation
    trans_eigenvalues:  np.ndarray[int, np.dtype(np.float64)] = field(init=False)
    scatt_states_up:    np.ndarray[int, np.dtype(np.complex128)] = field(init=False)
    scatt_states_down:  np.ndarray[int, np.dtype(np.complex128)] = field(init=False)
    scatt_density_up:   np.ndarray[int, np.dtype(np.complex128)] = field(init=False)
    scatt_density_down: np.ndarray[int, np.dtype(np.complex128)] = field(init=False)


    def pop_vars(self):
        """
        Create and populate variables for the calculation of the IPR
        """

        # Setting up the model
        self.x                    = np.linspace(0., self.msv.L, self.msv.Nx)
        self.Ntheta_grid          = self.msv.Ntheta_fft if self.msv.dimension_scatt == '2d' else 1
        self.Ntheta_plot          = self.msv.Ntheta_fft if self.msv.dimension_scatt == '2d' else self.msv.default_Ntheta_plot
        self.theta                = np.linspace(0., 2. * pi, self.Ntheta_plot)
        self.r_vec                = np.repeat(self.msv.r, self.x.shape[0])
        self.fermi                = np.linspace(self.msv.fermi_0, self.msv.fermi_end, self.msv.fermi_length)

        # Preallocation
        self.V_real               = np.zeros((len(self.theta), len(self.x)), dtype=np.float64)
        self.V_fft                = np.zeros((len(self.theta), len(self.x)), dtype=np.complex128)
        self.psi_scatt_up         = np.zeros((len(self.theta), self.msv.Nx), dtype=np.complex128)
        self.psi_scatt_down       = np.zeros((len(self.theta), self.msv.Nx), dtype=np.complex128)
        self.scatt_density_up     = np.zeros((len(self.theta), self.msv.Nx), dtype=np.complex128)
        self.scatt_density_down   = np.zeros((len(self.theta), self.msv.Nx), dtype=np.complex128)

        return self

    def load_data_to_sim_var(self, file_path):

        with h5py.File(file_path, 'r') as f:
            for fld in fields(self):
                for dataset in f['Simulation'].keys():
                    if dataset == fld.name:
                        if isinstance(f['Simulation'][dataset][()], bytes):
                            setattr(self, fld.name, f['Simulation'][dataset][()].decode())
                        else:
                            setattr(self, fld.name, f['Simulation'][dataset][()])



