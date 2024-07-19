import os
os.nice(19)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import multiprocessing as mp
from pk_sim_base import cosmo_sim_to_file
from itertools import product
import numpy as np
import time
from scipy.stats import qmc
import h5py
from glob import glob

# Defining LatinHyperCube sampler function
# -------------------------------------------------------------------------------------------------
class pk_sim_sample_params:

    def __init__(self,
                params_name,      # List of strings
                params_lbound,    # List of corresponding lower bounds
                params_ubound,    # List of corresponding upper bounds
                N_sample):        # Number of 3d power spectra to be simulated

        self.params_name = params_name
        self.pdim = len(params_name)
        self.sampler = qmc.LatinHypercube(d=self.pdim)
        self.sample = self.sampler.random(n=N_sample)
        self.sample_scaled = qmc.scale(sample=self.sample, l_bounds=params_lbound, u_bounds=params_ubound)

    # Array for Pk simulation input
    def get_array(self):
        return self.sample_scaled
# -------------------------------------------------------------------------------------------------

# Create directory
if not os.path.exists('./output_files/'):
    os.makedirs('./output_files/')

# Initializing latin hyper cube
# -------------------------------------------------------------------------------------------------
N_sim = 100000  # Number of simulation

params_name = ['Omega_b', 'Omega_cdm', 'h', 'n_s', 'm_nu', 'log10_T_heat', 'sigma8', 'alpha_B', 'alpha_M', 'k_screen', 'z_val']
params_lbound = [0.015,      0.18,     0.38, 0.7,  0.003,      7,             0.7,      0.,        0.,        -2,         0.]
params_ubound = [0.1,        0.34,      1., 1.25,   1.5,      8.6,            0.92,     2.5,       3.,   np.log10(2.),   4.5]

if './output_files/raw_lhc_params.npy' in glob('./output_files/*'):
    sim_arr = np.load('./output_files/raw_lhc_params.npy')
    with h5py.File("./output_files/hiclass_pk_simulation.h5py", 'r') as data_func:
        previous_sim_num = len(data_func['Pkmm_lin'][:,0])
else:
    sim_instance = pk_sim_sample_params(params_name, params_lbound, params_ubound, N_sim)
    sim_arr = sim_instance.get_array()
    # Rescaling k_screen to its physical values
    # ------------------------------------
    sim_arr[:,-2] = 10**sim_arr[:,-2]
    # ------------------------------------
    previous_sim_num = 0
    np.save('./output_files/raw_lhc_params.npy', sim_arr)

print('Starting at filenumber ',previous_sim_num)
# -------------------------------------------------------------------------------------------------

# Creating files to store data
# -------------------------------------------------------------------------------------------------
k_num = 200
z_num = 200
pdim = len(params_name)
filenames = glob("./output_files/*")

if "./output_files/hiclass_pk_simulation.h5py" in filenames:
    pass
else:
    with h5py.File("./output_files/hiclass_pk_simulation.h5py", "w") as file_pksim:
        function_label = ['Pkmm_lin', 'Pkmm_nonlin', 'bias_sq', 'eta_of_k', 'mu_of_k']
        for label in function_label:
            creater =  file_pksim.create_dataset(name=label, shape=(0,k_num), maxshape=(None,k_num))
        creater = file_pksim.create_dataset(name='chi_of_z', shape=(0,z_num), maxshape=(None,z_num))

if "./output_files/hiclass_param_dict.h5py" in filenames:
    pass
else:
    with h5py.File("./output_files/hiclass_param_dict.h5py", "w") as file_param_dict:
        label = 'input_params'
        creater = file_param_dict.create_dataset(name=label, shape=(0,pdim), maxshape=(None,pdim))
# -------------------------------------------------------------------------------------------------

# Calling multiprocessing
# -------------------------------------------------------------------------------------------------
CPU_num = 4     # number of used processors
start = time.time()
if __name__ == '__main__':
    with mp.Pool(processes=CPU_num)  as p:
        results = p.starmap(cosmo_sim_to_file, sim_arr[previous_sim_num:,:])
end = time.time()

print("Took me", (end-start)/60, "minutes.")
# -------------------------------------------------------------------------------------------------

