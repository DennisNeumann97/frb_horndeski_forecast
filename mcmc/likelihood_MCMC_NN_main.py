import os
os.nice(19)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import multiprocessing as mp
import time
from likelihood_MCMC_NN_base import likelihood
from likelihood_MCMC_NN_base import cosmology_results
import numpy as np
from copy import deepcopy
import nautilus as ns
import cosmopower as cp
import tensorflow as tf

# Setup variables specific to initialization of the Cl simulation code
# ---------------------------------------------------------------------------------
param_name = ['Omega_b', 'Omega_cdm', 'h', 'n_s', 'm_nu', 'log10_T_heat', 'sigma8', 'alpha_B', 'alpha_M', 'log10_k_screen']
param_val_fid = [np.array(0.04931), np.array(0.2642), np.array(0.674), np.array(0.965), np.array(0.06),
                 np.array(7.8), np.array(0.811), np.array(0.05), np.array(0.05), np.array(-1)]
input_dict_fid = dict(zip(param_name, param_val_fid))

# Directory of trained models and desired output and name appendix of the output file
NN_directory = './../training/trained_models/'
outdir = 'euclid_horndeski_1/'
name = '_euclid_horndeski_1'

# Number of cores
n_cores = 4

# Desired measurement to sample. Must be either 'autolens', 'autofrb' or 'crossfrblens'
which_measurement = 'autolens'


# Creating dictionary of survey params
alpha = 2.5
N_FRB = 1e4
z_bin_num = 4
lensing_survey = 'euclid'
f_sky_frb = 0.7
if lensing_survey == 'euclid':
    f_sky_lens = 0.3
    l_array = [1,5000,600]
elif lensing_survey == 'kids':
    f_sky_lens = 1/40.
    l_array = [1,2000,600]
else:
    pass
redshift_array = [1e-2,4,151]

survey_dict={'alpha':alpha, 'f_sky_frb':f_sky_frb, 'f_sky_lens':f_sky_lens, 'N_FRB':N_FRB,
             'FRB_bin_num':z_bin_num, 'lensing_survey':lensing_survey, 'l_array':l_array,
             'redshift_array':redshift_array}

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

# Loading in models
cp_pkmm_lin = cp.cosmopower_NN(restore=True, restore_filename=NN_directory+'pkmm_lin_model')
cp_pkmm_nonlin = cp.cosmopower_NN(restore=True, restore_filename=NN_directory+'pkmm_nonlin_model')
cp_bias_sq = cp.cosmopower_NN(restore=True, restore_filename=NN_directory+'bias_sq_model')
cp_eta = cp.cosmopower_NN(restore=True, restore_filename=NN_directory+'eta_model')
cp_mu = cp.cosmopower_NN(restore=True, restore_filename=NN_directory+'mu_model')
cp_chiz = cp.cosmopower_NN(restore=True, restore_filename=NN_directory+'chiz_model')

NN_models = {'pkmm_lin': cp_pkmm_lin, 'pkmm_nonlin':cp_pkmm_nonlin,'bias_sq':cp_bias_sq,
           'eta':cp_eta, 'mu':cp_mu, 'chiz':cp_chiz}

class_dict_fid = {'NN_params':input_dict_fid, 'alpha':alpha, 'N_FRB': N_FRB, 'FRB_bin_num': z_bin_num, 'f_sky_frb': f_sky_frb, 'f_sky_lens': f_sky_lens,
                  'lensing_survey':lensing_survey, 'l_array':l_array, 'redshift_array':redshift_array, 'NN_models':NN_models}

cosmo_fid = cosmology_results(**class_dict_fid)
Cl_data_full, l_full = cosmo_fid.tomo_Cl_limber_matrix_noise()

def likelihood_frb(param_dict):
    return -likelihood(param_dict['Omega_b'], param_dict['Omega_cdm'], param_dict['h'],  param_dict['n_s'],
                       param_dict['m_nu'], param_dict['log10_T_heat'], param_dict['sigma8'], param_dict['alpha_B'],
                       param_dict['alpha_M'], param_dict['log10_k_screen'], data=Cl_data_full, models=NN_models, survey_dict=survey_dict)[0]

def likelihood_lens(param_dict):
    return -likelihood(param_dict['Omega_b'], param_dict['Omega_cdm'], param_dict['h'],  param_dict['n_s'],
                       param_dict['m_nu'], param_dict['log10_T_heat'], param_dict['sigma8'], param_dict['alpha_B'],
                       param_dict['alpha_M'], param_dict['log10_k_screen'], data=Cl_data_full, models=NN_models, survey_dict=survey_dict)[1]

def likelihood_frblens(param_dict):
    return -likelihood(param_dict['Omega_b'], param_dict['Omega_cdm'], param_dict['h'],  param_dict['n_s'],
                       param_dict['m_nu'], param_dict['log10_T_heat'], param_dict['sigma8'], param_dict['alpha_B'],
                       param_dict['alpha_M'], param_dict['log10_k_screen'], data=Cl_data_full, models=NN_models, survey_dict=survey_dict)[2]

#param_dict = deepcopy(input_dict_fid)
#param_dict['log10_k_screen'] = [-0.5]
#print(likelihood_lens(param_dict))

params_name = ['Omega_b', 'Omega_cdm', 'h', 'n_s', 'm_nu', 'log10_T_heat', 'sigma8', 'alpha_B', 'alpha_M', 'log10_k_screen']
params_lbound = [0.015,      0.18,     0.38, 0.7,  0.003,      7,             0.7,      0.,        0.,       -2]
params_ubound = [0.1,        0.34,      1., 1.25,   1.5,      8.6,            0.92,     2.5,       3.,   np.log10(2.)]

prior = ns.Prior()
for idx_param, param in enumerate(param_name):
    prior.add_parameter(param, dist=(params_lbound[idx_param], params_ubound[idx_param]))

if __name__=='__main__':
        tstart = time.time()
        if which_measurement == 'autolens':
            sampler = ns.Sampler(prior=prior, likelihood=likelihood_lens, filepath='./output_files/'+outdir+'MCMC_NN_checkpoint'+name+'.hdf5', pool=n_cores)
        elif which_measurement == 'autofrb':
            sampler = ns.Sampler(prior=prior, likelihood=likelihood_frb, filepath='./output_files/'+outdir+'MCMC_NN_checkpoint'+name+'.hdf5', pool=n_cores)
        elif which_measurement == 'crossfrblens':
            sampler = ns.Sampler(prior=prior, likelihood=likelihood_frblens, filepath='./output_files/'+outdir+'MCMC_NN_checkpoint'+name+'.hdf5', pool=n_cores)
        else:
            print('Cannot proceed: Use which_measurement="autolens","autofrb" or "crossfrblens".')

        sampler.run(verbose=True)
        tend = time.time()

points, log_w, log_l = sampler.posterior()

print('Took me ',np.round((tend-tstart)/60),' minutes.')

# Create directory
if not os.path.exists('./output_files/'+outdir):
    os.makedirs('./output_files/'+outdir)
  
np.save('./output_files/'+outdir+'MCMC'+name+'_points', points)
np.save('./output_files/'+outdir+'MCMC'+name+'_log_weights', log_w)
np.save('./output_files/'+outdir+'MCMC'+name+'_log_likelihood', log_l)

# Saving settings to a text file
# Converting fiducial cosmology, prior bounds and other settings to string list
fiducial_list = [str(val) for val in param_val_fid]
priorlist_lbound = [str(val) for val in params_lbound]
priorlist_ubound = [str(val) for val in params_ubound]
general_list = ['alpha: '+str(alpha), 'FRB_num: '+str(N_FRB), 'FRB_binnum: '+str(z_bin_num), 'f_sky_frb: '+str(f_sky_frb),
                'f_sky_lens: '+str(f_sky_lens), 'l_min: '+str(l_array[0]), 'l_max: '+str(l_array[1])]

# Writing to text file
text = ['Settings of the MCMC contained in this folder',
        '--------------------------------------------------',
        'Order of Parameters: '+', '.join(params_name),
        'Fiducial cosmology: '+', '.join(fiducial_list),
        'Lower priorbound: '+', '.join(priorlist_lbound),
        'Upper priorbound: '+', '.join(priorlist_ubound),
        '--------------------------------------------------',
        'General settings: '+', '.join(general_list),
        '--------------------------------------------------',
        'Further comments: set alpha_B=alpha_M=1']
with open('./output_files/'+outdir+'MCMC'+name+'_settings.txt', 'w') as f:
    for line in text:
        f.write(line)
        f.write('\n')
