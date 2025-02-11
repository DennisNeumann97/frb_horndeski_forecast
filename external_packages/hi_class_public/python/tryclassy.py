from __future__ import division
from classy import Class
import numpy as np

# Define your cosmology (what is not specified will be set to CLASS default parameters)
params = {
    'output': 'mPk, nCl', #tCl lCl',#tCl',# lCl nCl',
    #'l_max_scalars': 2000,
    #'lensing': 'yes',
    'A_s': 2.3e-9,
    'n_s': 0.9624, 
    'h': 0.6711,
    'omega_b': 0.022068,
    'omega_cdm': 0.12029,
    'number count contributions':'gr',
    'Omega_Lambda': 0,
    'Omega_fld': 0,
    'Omega_smg': -1,
    'expansion_model': 'lcdm',
    'gravity_model': 'propto_omega',
    'P_k_max_h/Mpc': 150,
    'non linear': 'hmcode',
    'nonlinear_verbose':1,
    'halofit_tol_sigma':1e-9,
    'z_pk':4,
    #'feedback model':'emu_dmonly',
    'eta_0': 0.70,
    'c_min': 3.    
                #x_k, x_b, x_m, x_t, M*^2_ini
    #'parameters_smg' : 1., 0.625, 0., 0., 1.,#'Omega_Lambda' : '0',
    #'Omega_fld' : '0',
    #'Omega_smg' : -1}
    }
params['parameters_smg'] = "0.01,-1,2,0.,1"#"1., 1.625, 1., 1., 1."

# Create an instance of the CLASS wrapper
cosmo = Class()

# Set the parameters to the cosmological code
cosmo.set(params)

# Run the whole code. Depending on your output, it will call the
# CLASS modules more or less fast. For instance, without any
# output asked, CLASS will only compute background quantities,
# thus running almost instantaneously.
# This is equivalent to the beginning of the `main` routine of CLASS,
# with all the struct_init() methods called.
cosmo.compute()

#siz = len(np.arange(0.001, 1., 0.05))
#kvec = np.logspace(0.001, 10., 100)
#kvec = []
#for i in range(0,100):
#    kvec.append(np.log(0.001) + (np.log(10.)-np.log(0.001))/99.*i)
#zvec = np.linspace(0.001, 10., 100)
#siz = 100
#kvec = []
#for  i in range(i in range(0,100):
#    kvec.append(np.log(0.001) + (np.log(10.)-np.log(0.001))/99.*i)
kvec = []
for i in range(100):
    kvec.append(np.exp(np.log(0.0001) + i*(np.log(10.0) - np.log(0.001))/99.0))
zvec = np.linspace(0., 4., num=100)


np.savetxt("kvector.txt", kvec)
a = np.zeros(len(kvec))
b = np.zeros(len(kvec))
c = np.zeros(len(kvec))
d = np.zeros(len(kvec))
#c = cosmo.pk(kvec, zvec)

growth = np.zeros((len(kvec), len(zvec)))
growthind = np.zeros(len(zvec))
growthscreened = np.zeros((len(kvec), len(zvec)))
mu = np.zeros((len(kvec), len(zvec)))
gamma= np.zeros((len(kvec), len(zvec)))
mu_screened= np.zeros((len(kvec), len(zvec)))
gamma_screened= np.zeros((len(kvec), len(zvec)))

scalefac = np.zeros(len(zvec))
scalefac = 1./(1.+zvec)

H0squared = cosmo.get_Hubble(0.)*cosmo.get_Hubble(0.)
poissonfactor = H0squared*cosmo.Omega_m()


for i in range(len(kvec)):
    for j in range(len(zvec)):

        

    #a[i] = cosmo.pk_lin(kvec[i],0.)
    #b[i] = cosmo.pk(kvec[i], 0.)  
    #c[i] = cosmo.pk(kvec[i], 1.)  
    #d[i] = cosmo.pk_lin(kvec[i], 1.)  
    #print kvec[i], b[i]
        growth[i][j] = cosmo.get_Growth(kvec[i], zvec[j])
	growthscreened[i][j] = cosmo.screened_growth(kvec[i], zvec[j], 0.1)
        mu[i][j] = cosmo.get_poissonratio(kvec[i],zvec[j])*(-kvec[i]*kvec[i])*(2./3.)*scalefac[j]/poissonfactor
        gamma[i][j] = cosmo.get_phipsiratio(kvec[i],zvec[j])
        a = (-kvec[i]*kvec[i])*(2./3.)*scalefac[j]/poissonfactor
	mu_screened[i][j] = cosmo.get_poissonratio_screened(kvec[i],zvec[j], 0.1, a) #*(-kvec[i]*kvec[i])*(2./3.)*scalefac[j]/poissonfactor
	gamma_screened[i][j] = cosmo.get_phipsiratio_screened(kvec[i],zvec[j], 0.1)

np.savetxt("mu.txt", mu)
np.savetxt("gamma.txt", gamma)
np.savetxt("mu_screened.txt", mu_screened)
np.savetxt("gamma_screened.txt", gamma_screened)

np.savetxt("growth.txt", growth)
np.savetxt("growthscreened.txt", growthscreened)
for i in range(len(zvec)):
    growthind[i] = cosmo.scale_independent_growth_factor(zvec[i])

np.savetxt("growthind.txt", growthind)

#np.savetxt("mu.txt", mu)
#np.savetxt("gamma_txt", gamma)
#np.savetxt("pkhmcode_1_lin.txt", d)
"""
r, dzdr = cosmo.z_of_r(zvec)

kvalue = (200.)/(zvec)
for index in range(siz):
    print cosmo.get_Growth(0.1, zvec[index])

#print cosmo.pk(np.exp(kvec), zvec)

#for index in range(siz):
#    print cosmo.pk(np.exp(kvec[index]), zvec[index])

cosmo.struct_cleanup()
cosmo.empty()
#a = np.zeros((siz, siz), 'float64') 
#for kind in range(siz):
#    for zind in range(siz):
#        a[kind][zind] = cosmo.get_phipsiratio(np.exp(kvec[kind]),zvec[zind]) 

#b = np.zeros((siz, siz), 'float64') 
#for kind in range(siz):
#    for zind in range(siz):
#        b[kind][zind] = cosmo.get_poissonratio(np.exp(kvec[kind]),zvec[zind])*np.exp(kvec[kind])*np.exp(kvec[kind]) 

#np.savetxt('avec.txt', a)
#np.savetxt('bvec.txt', b)
#np.savetxt('cvec.txt', kvec)
#np.savetxt('dvec.txt', zvec)

#for index in range(siz):
#    for index2 in range(siz):
#        print np.exp(kvec[index]), zvec[index2], a[index][index2], b[index][index2] 
"""
#for index in range(siz):
#    print cosmo.get_poissonratio(200./zvec[index], zvec[index])*(-2./3.)/cosmo.get_Hubble(zvec[index])/cosmo.get_Hubble(zvec[index])/cosmo.Omega_m()*(1./(1.+zvec[index]))*(200./zvec[index])*(200./zvec[index])

#a = np.zeros(len(kvec))
#for i in range(len(kvec)):
#    a[i] = cosmo.pk(np.exp(kvec[i]), 0.)
#np.savetxt("pk0mg.txt", a)
