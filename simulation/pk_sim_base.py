
# Loading in all relevant packages
# ----------------------------------------------------------------------------
from classy import Class                         # Class python wrapper
import numpy as np                               # 1000€ calculator
import matplotlib.pyplot as plt                  # 1000€ drawing board
import matplotlib as mpl
import astropy.constants as const
import astropy.units as u
import pyhmcode
import pyhmcode.halo_profile_utils
import pyccl
import h5py
import os
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
from glob import glob
from copy import deepcopy
import time
# ----------------------------------------------------------------------------

# Define a context manager to suppress stdout and stderr.
# ---------------------------------------------------------------------
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
# ---------------------------------------------------------------------

def cosmo_sim_to_file(Omega_b, Omega_cdm, h, n_s, m_nu, log10_T_heat, sigma8,
                      alpha_B, alpha_M, k_screen, z_val):

    tstart=time.time()
    # Initiating Class
    # ---------------------------------------------------------------------
    params = {'output':'mPk, nCl, tCl, lCl',
            'P_k_max_1/Mpc':150,
            'z_max_pk':10.,
            'T_cmb': 2.7255,
            'Omega_b': Omega_b,
            'Omega_cdm': Omega_cdm,
            'h': h,
            'n_s': n_s,
            'N_ncdm': 3,
            'm_ncdm': "0.,0.,"+str(m_nu),
            'N_eff': 3.046,
            'reio_parametrization': 'reio_camb',
            'z_reio': 7.7,
            'YHe': 0.246,
    # From here initialization of hi_class, see hi_class.ini for explanation
            'Omega_Lambda': 0,
            'Omega_fld': 0,
            'Omega_smg': -1,
            'expansion_model': 'lcdm',
            'gravity_model': 'propto_omega',
                            #x_k, x_b, x_m, x_t, M*^2_ini
            'parameters_smg': '1., '+str(alpha_B)+', '+str(alpha_M)+', 0., 1.',
            'number count contributions':'gr'
    }

    # Creating input dictionary
    # ---------------------------------------------------------------------
    input_names = ['Omega_b', 'Omega_cdm', 'h', 'n_s', 'm_nu', 'log10_T_heat',\
                   'sigma8', 'alpha_B', 'alpha_M', 'k_screen', 'z_val']
    input_values = [Omega_b, Omega_cdm, h, n_s, m_nu, log10_T_heat,\
                    sigma8, alpha_B, alpha_M, k_screen, z_val]
    input_dict = dict(zip(input_names, input_values))
    # ---------------------------------------------------------------------

    cosmo = Class()
    cosmo.set(params)

    try:
        cosmo.compute()
        # ---------------------------------------------------------------------

        # Defining scale and comoving distance array
        # ---------------------------------------------------------------------
        z_arr = np.linspace(0., z_val, 4)
        chi_arr = cosmo.z_of_r(z_arr)[0]

        k_max = params['P_k_max_1/Mpc']
        k_min = 1e-5
        k_num = 200
        k_arr = np.geomspace(k_min, k_max, k_num)
        # ---------------------------------------------------------------------

        # Extracting chi array for seperate training
        # ---------------------------------------------------------------------
        CLASS_background = cosmo.get_background()
        z_bg = CLASS_background['z']
        chi_bg = CLASS_background['comov. dist.']
        chi_bg_int = interp1d(z_bg, chi_bg, fill_value="extrapolate")

        z_max = 4.5
        z_min = 1e-3
        z_num = 200
        z_for_chi = np.geomspace(z_min, z_max, z_num)
        chi_of_z = chi_bg_int(z_for_chi)
        # ---------------------------------------------------------------------

        # Simulating k and z dependent variables coming from Hi-Class
        # ---------------------------------------------------------------------
        Omega_m = cosmo.Omega0_m()
        H0 = cosmo.h()*100
        Pofk = np.zeros(shape=(len(k_arr), len(z_arr)))
        for k_idx, kk in enumerate(k_arr):
            for z_idx, zz in enumerate(z_arr):
                Pofk[k_idx, z_idx] = cosmo.pk_lin(kk, 0.)*cosmo.screened_growth(kk,zz,k_screen)**2
        eta_kz = np.array([cosmo.get_phipsiratio_screened(kk,z_val,k_screen) for kk in k_arr])   # Ratio of bardeen(?) potentials
        mu_kz = np.array([cosmo.get_poissonratio_screened(kk,z_val,k_screen,\
                                ((-2/3*(kk)**2/(1+z_val)/Omega_m/H0**2)*const.c.to('km/s')**2).value)\
                                for kk in k_arr])  # Ratio of lensing potential to density contrast
        # ---------------------------------------------------------------------

        # Calculating and renormalizing sigma8
        # ---------------------------------------------------------------------
        Pk_int = interp1d(k_arr, Pofk[:,0])
        k_new = np.geomspace(k_arr[0], k_arr[-1], 10000)
        Pk_new = Pk_int(k_new)

        kR = k_new*8/h
        weight = 3/kR**3*(np.sin(kR)-kR*np.cos(kR))
        sigma8_old_sq = 1/(2*np.pi**2)*trapz(Pk_new*weight**2*k_new**2, k_new)

        Pofk_renorm = Pofk*sigma8**2/sigma8_old_sq
        # ---------------------------------------------------------------------


        # Simulating nonlinear Pkz correction
        # ---------------------------------------------------------------------
        names = ['sigma8', 'h', 'Omega_m', 'Omega_b', 'Omega_cdm', 'Omega_de', \
                'n_s', 'Neff', 'T_cmb', 'wa', 'w0', 'm_nu']
        values = [cosmo.sigma8(), cosmo.h(), cosmo.Omega0_m(), cosmo.Omega_b(),\
                cosmo.Omega0_cdm(), cosmo.Omega_Lambda(), cosmo.n_s(), cosmo.Neff(),\
                cosmo.T_cmb(), 0., -1., [0., 0., m_nu]]
        dict_cosmo = dict(zip(names, values))

        ccl_cosmology = pyccl.Cosmology(Omega_c=dict_cosmo['Omega_cdm'],
                                    Omega_b=dict_cosmo['Omega_b'], h=dict_cosmo['h'],
                                    n_s=dict_cosmo['n_s'], sigma8=dict_cosmo['sigma8'],
                                    w0=dict_cosmo['w0'], wa=dict_cosmo['wa'],
                                    Neff=dict_cosmo['Neff'], m_nu=dict_cosmo['m_nu'], m_nu_type='list')

        hmcode_cosmology = pyhmcode.halo_profile_utils.ccl2hmcode_cosmo(
                                ccl_cosmo = ccl_cosmology,
                                pofk_lin_k_h = k_arr/h,
                                pofk_lin_z = z_arr,
                                pofk_lin = Pofk_renorm.T*h**3,
                                log10_T_heat=log10_T_heat)

        hmcode_model = pyhmcode.Halomodel(pyhmcode.HMx2020_matter_pressure_w_temp_scaling)

        with suppress_stdout_stderr():
            hmcode_pofk = pyhmcode.calculate_nonlinear_power_spectrum(
                                cosmology=hmcode_cosmology,
                                halomodel=hmcode_model,
                                fields=[pyhmcode.field_matter,
                                        pyhmcode.field_gas])

        bias_sq = np.copy(hmcode_pofk[1, 1].T/hmcode_pofk[0, 0].T)

        bias_threshhold_arg = np.argmin(np.abs(k_arr-1e-2))
        bias_sq[:bias_threshhold_arg,:] = bias_sq[bias_threshhold_arg,:]
        bias_sq /= bias_sq[0,:]

        Pk_matter_lin = Pofk_renorm[:,-1]
        Pk_matter_nonlin = (hmcode_pofk[0,0,-1,:])/cosmo.h()**3
        # ---------------------------------------------------------------------

        # Raise exception if there are NaN values in any array
        # ---------------------------------------------------------------------
        isnan_pklin = np.sum(np.isnan(Pk_matter_lin))
        isnan_pknonlin = np.sum(np.isnan(Pk_matter_nonlin))
        isnan_bias = np.sum(np.isnan(bias_sq[:,-1]))
        isnan_eta = np.sum(np.isnan(eta_kz))
        isnan_mu = np.sum(np.isnan(mu_kz))
        isnan_chiz = np.sum(np.isnan(chi_of_z))
        isnan_all = isnan_pklin + isnan_pknonlin + isnan_bias + isnan_eta + isnan_mu + isnan_chiz

        if (isnan_all > 0):
            raise Exception('NaN-value found in simulated arrays')
        else:
            pass
         # ---------------------------------------------------------------------

        hm_bool = True

    except Exception as e:
        print('HiCLASS or HMCode initialization failed with error')
        print(e)
        print('The following parameters produced the error:')
        print(input_dict)

        hm_bool = False
    # ---------------------------------------------------------------------

    # Appending the results to file
    # ---------------------------------------------------------------------
    if hm_bool:
        try:
            with h5py.File("./output_files/hiclass_pk_simulation.h5py", "r+") as file_pksim:
                with h5py.File("./output_files/hiclass_param_dict.h5py", "r+") as file_param_dict:

                    # Extending the files...
                    # ... for the 5 target functions
                    function_label = ['Pkmm_lin', 'Pkmm_nonlin', 'bias_sq', 'eta_of_k', 'mu_of_k']
                    for label in function_label:
                        reshaper = file_pksim.get(label)
                        reshaper.resize((len(file_pksim[label][:,0])+1, k_num))
                    # ... for the comoving distance of z
                    reshaper = file_pksim.get('chi_of_z')
                    reshaper.resize((len(file_pksim['chi_of_z'][:,0])+1, z_num))
                    # ... for the cosmological parameters
                    reshaper = file_param_dict.get('input_params')
                    reshaper.resize((len(file_param_dict['input_params'][:,0])+1, len(input_values)))

                    # Storing the data
                    file_pksim['Pkmm_lin'][-1,:] = np.log10(Pk_matter_lin)
                    file_pksim['Pkmm_nonlin'][-1,:] = np.log10(Pk_matter_nonlin)
                    file_pksim['bias_sq'][-1,:] = np.log10(bias_sq[:,-1])
                    file_pksim['eta_of_k'][-1,:] = eta_kz
                    file_pksim['mu_of_k'][-1,:] = mu_kz
                    file_pksim['chi_of_z'][-1,:] = chi_of_z
                    file_param_dict['input_params'][-1,:] = np.array(input_values)

        except Exception as e:
            print('Failed Filewrite for whatever reason. Wish me luck!')
            print(e)
    else:
        print('Results will not be saved.')
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    cosmo.struct_cleanup()
    tend = time.time()
    print('Simulation done in '+str(np.round((tend-tstart)/60, 2))+' minutes.')
