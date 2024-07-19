import tensorflow as tf
import cosmopower
from cosmopower import cosmopower_NN
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# print(tf.config.list_physical_devices('GPU'))

# Load in validation data and removing duplicates
# ------------------------------------------------------------
with h5py.File("./../simulation/output_files/validation/hiclass_pk_simulation_validation.h5py", 'r') as data_func:
    Pkmm_lin_validation, unique_idx = np.unique(10**data_func['Pkmm_lin'][:,:],axis=0, return_index=True)
    Pkmm_nonlin_validation = 10**data_func['Pkmm_nonlin'][:,:][unique_idx]
    bias_sq_validation = 10**data_func['bias_sq'][:,:][unique_idx]
    eta_validation = data_func['eta_of_k'][:,:][unique_idx]
    mu_validation = data_func['mu_of_k'][:,:][unique_idx]
    chiz_validation = data_func['chi_of_z'][:,:][unique_idx]
print('Training data shape:', np.shape(Pkmm_lin_validation))

with h5py.File("./../simulation/output_files/validation/hiclass_param_dict_validation.h5py", 'r') as data_param:
    validation_params = data_param['input_params'][:,:][unique_idx]
print('Training parameters shape:', np.shape(validation_params))

validation_array = np.array([Pkmm_lin_validation, Pkmm_nonlin_validation, bias_sq_validation,
                             eta_validation, mu_validation, chiz_validation])
# ------------------------------------------------------------

# Loading in trained neural networks
# ------------------------------------------------------------
cp_nn_chiz = cosmopower_NN(restore=True, restore_filename='./trained_models/chiz_model')
cp_nn_pkmm_lin = cosmopower_NN(restore=True, restore_filename='./trained_models/pkmm_lin_model')
cp_nn_pkmm_nonlin = cosmopower_NN(restore=True, restore_filename='./trained_models/pkmm_nonlin_model')
cp_nn_eta = cosmopower_NN(restore=True, restore_filename='./trained_models/eta_model')
cp_nn_mu = cosmopower_NN(restore=True, restore_filename='./trained_models/mu_model')
cp_nn_bias_sq = cosmopower_NN(restore=True, restore_filename='./trained_models/bias_sq_model')
# ------------------------------------------------------------

# Translating validation parameters in prediction dictionary
# ------------------------------------------------------------
params_name = ['Omega_b', 'Omega_cdm', 'h', 'n_s', 'm_nu', 'log10_T_heat', 'sigma8', 'alpha_B', 'alpha_M', 'log10_k_screen', 'z_val']

prediction_dict = {}
for idx, name in enumerate(params_name):
    if name=='log10_k_screen':
        prediction_dict[name] = np.log10(validation_params[:,idx])
    else:
        prediction_dict[name] = validation_params[:,idx]
# ------------------------------------------------------------

# Getting predicted array
# ------------------------------------------------------------
chiz_cp = cp_nn_chiz.predictions_np(prediction_dict)
pkmm_lin_cp = cp_nn_pkmm_lin.ten_to_predictions_np(prediction_dict)
pkmm_nonlin_cp = cp_nn_pkmm_nonlin.ten_to_predictions_np(prediction_dict)
bias_sq_cp = cp_nn_bias_sq.ten_to_predictions_np(prediction_dict)
eta_cp = cp_nn_eta.predictions_np(prediction_dict)
mu_cp = cp_nn_mu.predictions_np(prediction_dict)

prediction_array = np.array([pkmm_lin_cp, pkmm_nonlin_cp, bias_sq_cp, eta_cp, mu_cp, chiz_cp])
# ------------------------------------------------------------


# Calculating deviation in percent
# ------------------------------------------------------------
ratio = prediction_array/validation_array
dev_mean = np.mean(np.abs(1-ratio), axis=2)
dev_max = np.max(np.abs(1-ratio), axis=2)
# ------------------------------------------------------------

# Making some nice plots
# ------------------------------------------------------------
mpl.rcParams.update({'font.size': 16, 'lines.linewidth': 1.5})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
colors = plt.cm.viridis(np.linspace(0, 0.6, 2))

fig, ax = plt.subplots()
model_names = ['Pkmm_lin', 'Pkmm_nonlin', 'bias_sq', 'eta', 'mu', 'chiz']
model_text = [r'\textbf{f)}', r'\textbf{a)}', r'\textbf{b)}', r'\textbf{c)}', r'\textbf{d)}', r'\textbf{e)}']
model_names_pretty = [r'Pkmm_lin', r'$P_\mathrm{mm}(k)$', r'$b_\mathrm{e}(k)$', r'$\eta(k)$', r'$\mu(k)$', r'$\chi(z)$']
upper_range = [0.05, 0.035, 0.035, 0.01, 0.01, 0.001]
# ------------------------------------------------------------



def make_subplots(fig=None, figsize=(12, 12)):

  if fig is None:
    fig = plt.figure(figsize=figsize)

  gridspec = fig.add_gridspec(6, 4)
  gridspec.update(left=0.1,right=0.9,top=0.965,bottom=0.03,wspace=0.6,hspace=0.7)

  # Create the top two subfigures.
  ax1 = fig.add_subplot(gridspec[:2, :2])
  ax2 = fig.add_subplot(gridspec[:2, 2:])

  # Create the middle two subfigures.
  ax3 = fig.add_subplot(gridspec[2:4, :2])
  ax4 = fig.add_subplot(gridspec[2:4, 2:])

  # Create the bottom subfigure.
  ax5 = fig.add_subplot(gridspec[4:6, 1:3])

  return fig, [ax1, ax2, ax3, ax4, ax5]

fig, axes = make_subplots()

for i in range(5):
    axes[i].hist(dev_max[i+1], bins=80, range=(0,upper_range[i+1]), alpha=0.8, color=colors[0], label=r'maximum '+model_names_pretty[i+1]+r' deviation')
    axes[i].hist(dev_mean[i+1], bins=80, range=(0,upper_range[i+1]), alpha=0.8, color=colors[1], label=r'mean '+model_names_pretty[i+1]+r' deviation')
    axes[i].legend(loc='upper right')
    axes[i].text(0.99, 0.08, model_text[i+1], ha='right', va='top', transform=axes[i].transAxes)
    axes[i].set_xlim([0,upper_range[i+1]])
    axes[i].set_xlabel(r'relative deviation, $\Delta$')
    axes[i].set_ylabel(r'number count, $N$')

# Show the figure.
plt.savefig('./validation_plot/mean_and_max_deviation/deviation_full_hist.png', bbox_inches='tight')




