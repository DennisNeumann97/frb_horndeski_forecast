 # Training a neural network
import tensorflow as tf
import cosmopower
import h5py
import numpy as np
print(tf.config.list_physical_devices('GPU'))

# Load in training data and removing duplicates
# ------------------------------------------------------------
with h5py.File("./../simulation/output_files/training/hiclass_pk_simulation.h5py", 'r') as data_func:
    Pkmm_lin_training, unique_idx = np.unique(data_func['Pkmm_lin'][:,:],axis=0, return_index=True)
    Pkmm_nonlin_training = data_func['Pkmm_nonlin'][:,:][unique_idx]
    bias_sq_training = data_func['bias_sq'][:,:][unique_idx]
    eta_training = data_func['eta_of_k'][:,:][unique_idx]
    mu_training = data_func['mu_of_k'][:,:][unique_idx]
    chiz_training = data_func['chi_of_z'][:,:][unique_idx]
print('Training data shape:', np.shape(Pkmm_lin_training))

with h5py.File("./../simulation/output_files/training/hiclass_param_dict.h5py", 'r') as data_param:
    training_params = data_param['input_params'][:,:][unique_idx]
print('Training parameters shape:', np.shape(training_params))
# ------------------------------------------------------------

# Getting training data and making k_screen and z_val log spaced
# ------------------------------------------------------------
names = ['Omega_b', 'Omega_cdm', 'h', 'n_s', 'm_nu', 'log10_T_heat', 'sigma8', 'alpha_B', 'alpha_M', 'log10_k_screen', 'z_val']
training_dict = {}
for idx, name in enumerate(names):
    if name=='log10_k_screen':
        training_dict[name] = np.log10(training_params[:,idx])
    else:
        training_dict[name] = training_params[:,idx]

model_parameters = names
# ------------------------------------------------------------

# Getting k and z array
# ------------------------------------------------------------
k_max = 150
k_min = 1e-5
k_num = 200
k_arr = np.geomspace(k_min, k_max, k_num)

z_max = 4.5
z_min = 1e-3
z_num = 200
z_arr = np.geomspace(z_min, z_max, z_num)

modes_k = k_arr
modes_z = z_arr
# ------------------------------------------------------------


from cosmopower import cosmopower_NN

cp_nn_chiz = cosmopower_NN(parameters=names,
                      modes=modes_z,
                      n_hidden = [30,70],
                      verbose=True, # useful to understand the different steps in initialisation and training
                      )

with tf.device('/GPU:0'):
    # train
    cp_nn_chiz.train(training_parameters=training_dict,
                training_features=chiz_training,
                filename_saved_model='./trained_models/chiz_model',
                # cooling schedule
                validation_split=0.1,
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                batch_sizes=[512, 512, 512, 512, 512],
                gradient_accumulation_steps = [1, 1, 1, 1, 1],
                # early stopping set up
                patience_values = [100,100,100,100,100],
                max_epochs = [1000,1000,1000,1000,1000],
                )

# cp_nn_pklin = cosmopower_NN(parameters=names,
#                       modes=modes_k,
#                       n_hidden = [100, 100],
#                       verbose=True, # useful to understand the different steps in initialisation and training
#                       )
#
# with tf.device('/GPU:0'):
#     # train
#     cp_nn_pklin.train(training_parameters=training_dict,
#                 training_features=Pkmm_lin_training,
#                 filename_saved_model='./trained_models/pkmm_lin_model',
#                 # cooling schedule
#                 validation_split=0.1,
#                 learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
#                 batch_sizes=[512, 512, 512, 512, 512],
#                 gradient_accumulation_steps = [1, 1, 1, 1, 1],
#                 # early stopping set up
#                 patience_values = [100,100,100,100,100],
#                 max_epochs = [1000,1000,1000,1000,1000],
#                 )

cp_nn_pknonlin = cosmopower_NN(parameters=names,
                      modes=modes_k,
                      n_hidden = [200, 200],
                      verbose=True, # useful to understand the different steps in initialisation and training
                      )

with tf.device('/GPU:0'):
    cp_nn_pknonlin.train(training_parameters=training_dict,
                training_features=Pkmm_nonlin_training,
                filename_saved_model='./trained_models/pkmm_nonlin_model',
                # cooling schedule
                validation_split=0.1,
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                batch_sizes=[256, 256, 256, 256, 256],
                gradient_accumulation_steps = [1, 1, 1, 1, 1],
                # early stopping set up
                patience_values = [100,100,100,100,100],
                max_epochs = [1000,1000,1000,1000,1000],
                )

cp_nn_eta = cosmopower_NN(parameters=names,
                      modes=modes_k,
                      n_hidden = [200, 200],
                      verbose=True, # useful to understand the different steps in initialisation and training
                      )

with tf.device('/GPU:0'):
    cp_nn_eta.train(training_parameters=training_dict,
                training_features=eta_training,
                filename_saved_model='./trained_models/eta_model',
                # cooling schedule
                validation_split=0.1,
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                batch_sizes=[512, 512, 512, 512, 512],
                gradient_accumulation_steps = [1, 1, 1, 1, 1],
                # early stopping set up
                patience_values = [100,100,100,100,100],
                max_epochs = [1000,1000,1000,1000,1000],
                )

cp_nn_mu = cosmopower_NN(parameters=names,
                      modes=modes_k,
                      n_hidden = [200, 200],
                      verbose=True, # useful to understand the different steps in initialisation and training
                      )

with tf.device('/GPU:0'):
    cp_nn_mu.train(training_parameters=training_dict,
                training_features=mu_training,
                filename_saved_model='./trained_models/mu_model',
                # cooling schedule
                validation_split=0.1,
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                batch_sizes=[512, 512, 512, 512, 512],
                gradient_accumulation_steps = [1, 1, 1, 1, 1],
                # early stopping set up
                patience_values = [100,100,100,100,100],
                max_epochs = [1000,1000,1000,1000,1000],
                )
#

cp_nn_bias = cosmopower_NN(parameters=names,
                      modes=modes_k,
                      n_hidden = [200, 200],
                      verbose=True, # useful to understand the different steps in initialisation and training
                      )

with tf.device('/GPU:0'):
    cp_nn_bias.train(training_parameters=training_dict,
                training_features=bias_sq_training,
                filename_saved_model='./trained_models/bias_sq_model',
                # cooling schedule
                validation_split=0.1,
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                batch_sizes=[256, 256, 256, 256, 256],
                gradient_accumulation_steps = [1, 1, 1, 1, 1],
                # early stopping set up
                patience_values = [100,100,100,100,100],
                max_epochs = [1000,1000,1000,1000,1000],
                )
