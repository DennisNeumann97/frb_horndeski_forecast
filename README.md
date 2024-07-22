# frb_horndeski_forcast
Open Source Code used in arxiv:INSERT

## Installing
After cloning the repository, create a new conda environment with the provided YAML file

    cd frb_horndeski_forcast
    conda env create -f env.yml 
    conda activate frb_horndeski_forecast

Then, we need to install the provided $\texttt{hiclass}$ package modified by [Spurio Mancini et al. 2018](https://academic.oup.com/mnras/article/480/3/3725/5063592):

    cd external_packages
    tar -xf hi_class_public_modified.tar.gz
    cd hi_class_public
    
    make clean
    make

## Usage
This repository contains three major parts, which are all computationally expensive: **Simulation**, **Training** and **MCMC sampling**.

### Simulation
Simulation of (non-)linear matter power spectrum $P_{\mathrm{mm}}^{\mathrm{(NL)}}(k,z)$, electron bias $b_\mathrm{e}(k,z)$ , ratio of bardeen potentials $\eta(k,z)$, modified gravity change to the Poisson equation $\mu(k,z)$ and comoving distance. Adjust your prefered settings (like prior bounds, number of cores, number of samples) inside the "pk_sim_main.py" file inside the "simulation" directory, then run

    python simulation/pk_sim_main.py
The code saves the initial parameter suggestion as "raw_lhc_params.npy", and the successfull simulations and their parameters are stored on the fly in "hiclass_pk_simulation.h5py" and "hiclass_param_dict.h5py", respectively.

The training and validation data of our paper is saved on dropbox and can be downloaded by running the following lines from from the parent directory:

    cd simulation/output_files
    wget -O modified_gravity_lambcdm_training_samples.tar.gz "https://www.dropbox.com/scl/fi/ou8yr6ohvrwfhfhsp7usf/modified_gravity_lambcdm_training_samples.tar.gz?rlkey=r3dyhxottr9q1xiplesmnxkw2&st=cu7t4uqe&dl=1" 
    tar -xf modified_gravity_lambcdm_training_samples.tar.gz

### Training
The $\texttt{cosmopower}$ training settings can be adjusted in the "pk_training_script.py" file. Simply run

    python training/pk_training_script.py
to train the model on the data contained in "simulation/output_files/training/". Our pretrained models are stored in "training/trained_models"; feel free to use them.

### MCMC-sampling
Run

    python mcmc/likelihood_MCMC_NN_main.py

with your preferred settings put into the .py file. The script will look for the \texttt{cosmopower} models inside "training/trained_models/" and use them for the sampling. A minimal working example for calling the models is provided in the notebook "trained_cp_call.ipynb".


## References

Feel free to use and adapt this code however you like, but please cite our paper (INSERT) and the corresponding packages you use ([cosmopower](https://github.com/alessiospuriomancini/cosmopower), [hi_class](https://github.com/miguelzuma/hi_class_public), [nautilus](https://github.com/johannesulf/nautilus), [pyccl](https://github.com/LSSTDESC/CCL), [hmcode](https://github.com/alexander-mead/HMcode)).
