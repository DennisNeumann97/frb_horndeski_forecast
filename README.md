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

The training and testing data of our paper is saved on on OneDrive and can be downloaded by running the following lines from from the parent directory:

    cd simulation/output_files
    wget -O modified_gravity_lambcdm_training_samples.tar.gz "https://leidenuniv1-my.sharepoint.com/personal/neumannd_vuw_leidenuniv_nl/_layouts/15/download.aspx?UniqueId=581cb497-096a-4f0d-b62b-0b1ecdf2c280&Translate=false&tempauth=v1.eyJzaXRlaWQiOiI2ZjZlM2Y1MC1kNmVhLTQwNjQtOTQwZS1hN2Q1MDQwNWZhMzkiLCJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvbGVpZGVudW5pdjEtbXkuc2hhcmVwb2ludC5jb21AY2EyYTdmNzYtZGJkNy00ZWMwLTkxMDgtNmIzZDUyNGZiN2M4IiwiZXhwIjoiMTcyNTk3NzM5NSJ9.CgoKBHNuaWQSAjQzEgsIoKmo2vzoqD0QBRomMjAwMToxYzAwOjMzOTplNDAwOmMxNzg6MTQzNzoyOGU4OjlhYjAiFG1pY3Jvc29mdC5zaGFyZXBvaW50KixUSk1telZxckhqWGwxOGxNUk80b2JMSG5SVlNKR1YxbFRWZE9BZ2dwRGIwPTChATgBQhChTuOul6AAkNnc941zqmoCShBoYXNoZWRwcm9vZnRva2VuUghbImttc2kiXWIEdHJ1ZWokMTQwZmVmMTctNDdkOS00MWI0LTgzZTctMzlmMWRjMWI1NzJhcikwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDM2Mjc5MDZmM0BsaXZlLmNvbXoBMMIBKjAjLmZ8bWVtYmVyc2hpcHxuZXVtYW5uZEB2dXcubGVpZGVudW5pdi5ubMgBAQ.o2b-OUaocxi6YuXtJpbKS0eXWxzyFEHDZI3YtYG96Vs"
    tar -xf modified_gravity_lambcdm_training_samples.tar.gz

### Training
The $\texttt{cosmopower}$ training settings can be adjusted in the "pk_training_script.py" file. Simply run

    python training/pk_training_script.py
to train the model on the data contained in "simulation/output_files/training/". Our pretrained models are stored in "training/trained_models"; feel free to use them.

### MCMC-sampling
Run

    python mcmc/likelihood_MCMC_NN_main.py

with your preferred settings put into the .py file. The script will look for the *cosmopower* models inside "training/trained_models/" and use them for the sampling. A minimal working example for calling the models is provided in the notebook "trained_cp_call.ipynb".


## References

Feel free to use and adapt this code however you like, but please cite our paper (INSERT) and the corresponding packages you use ([cosmopower](https://github.com/alessiospuriomancini/cosmopower), [hi_class](https://github.com/miguelzuma/hi_class_public), [nautilus](https://github.com/johannesulf/nautilus), [pyccl](https://github.com/LSSTDESC/CCL), [hmcode](https://github.com/alexander-mead/HMcode)).
