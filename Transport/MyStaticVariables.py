"""
Static variables dataclass for the Topological Transport Project.
These are supposed to be type const parameters, so they should be initialised once and not more.
The main should be able to call them but never change their value again at runtime.
"""
from numpy import array
import h5py
from dataclasses import dataclass, field, fields


def init_my_variable(variable: any, type_check: type, var_name=None) -> any:
    if isinstance(variable, type_check):
        return variable
    else:
        raise TypeError('Input variable type of {} does not match with the expected type!'.format(var_name))



@dataclass
class MyStaticVariables:

    # Model
    vf:                     float = 0.0
    l_cutoff:               int = 0
    n_flux:                 int = 0
    B_perp:                 float = 0.0

    # Disorder potential
    corr_length:            float = 0.0
    dis_strength:           float = 0.0

    # Geometry
    r:                      float = 0.0
    L:                      float = 0.0
    Nx:                     int = 0
    Ntheta_fft:             int = 0
    default_Ntheta_plot:    int = 0

    # Transport calculation
    Nx_transport:             int = 0
    fermi_0:                float = 0.0
    fermi_end:              float = 0.0
    fermi_length:           int = 0
    E_resonance_index:      int = 0
    transmission_eigenval:  int = 0

    # IPR calculation
    fermi_0_IPR:            float = 0.0
    fermi_end_IPR:          float = 0.0
    fermi_length_IPR:       int = 0
    res_index:              list = field(default_factory=lambda: [0])
    x0:                     list = field(default_factory=lambda: [None])
    theta0:                 list = field(default_factory=lambda: [None])
    N_deltax:               int = 0
    N_deltatheta:           int = 0

    # Flags for transport
    load_data_transport:    bool = False
    save_data_transport:    bool = False
    load_file_transport:    str = ''
    calculate_transport:    bool = False
    calculate_scattering:   bool = False
    dimension:              str = ''

    # Flags for IPR
    load_data_IPR:          bool = False
    save_data_IPR:          bool = False
    load_file_IPR:          str = ''
    calculate_IPR:          bool = False
    dimension_IPR:          str = ''


    def pop_vars(self, config_dict, list_as_array=True):
        """
        Populate the members of the dataclass with the elements from config_dict
        """
        # Iterate over the fields
        for f in fields(self):

            # Iterate over the config dictionary
            for var_group in config_dict.keys():
                for var_item in config_dict[var_group].keys():

                    # Assign value to the field from the config dictionary
                    if var_item == f.name:
                        var = init_my_variable(config_dict[var_group][var_item], f.type, var_name=var_item)
                        if list_as_array and f.type == list:
                            setattr(self, f.name, array(var))
                        else:
                            setattr(self, f.name, var)
        return self


    def load_data_to_static_var(self, file_path, load_flags=False):

        if load_flags:
            with h5py.File(file_path, 'r') as f:
                for fld in fields(self):
                    for dataset in f['Parameters'].keys():
                        if dataset == fld.name and fld.name:
                            if isinstance(f['Parameters'][dataset][()], bytes):
                                setattr(self, fld.name, f['Parameters'][dataset][()].decode())
                            else:
                                setattr(self, fld.name, f['Parameters'][dataset][()])

        else:
            list_flags = ['load_data_transport', 'save_data_transport', 'load_file_transport','calculate_transport',
                          'calculate_scattering', 'load_data_IPR', 'save_data_IPR', 'load_file_IPR','calculate_IPR']

            with h5py.File(file_path, 'r') as f:
                for fld in fields(self):
                    for dataset in f['Parameters'].keys():
                        if dataset == fld.name and fld.name not in list_flags:
                            if isinstance(f['Parameters'][dataset][()], bytes):
                                setattr(self, fld.name, f['Parameters'][dataset][()].decode())
                            else:
                                setattr(self, fld.name, f['Parameters'][dataset][()])