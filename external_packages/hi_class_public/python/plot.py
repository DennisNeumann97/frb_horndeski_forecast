import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.interpolate import UnivariateSpline



data1 = np.loadtxt("potentials.txt")


k = data1[:,0]
phi = data1[:,1]
psi = data1[:,2]
nlphi = data1[:,3]
nlpsi = data1[:,4]

fontsi = 18
fontsi2 = 22


plt.tick_params(labelsize=fontsi)

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')



plt.loglog(k,phi, label=r'$\Phi\Phi$',lw = 2)
plt.loglog(k,psi, label=r'$\Psi\Psi$',lw = 2)
plt.loglog(k,nlphi, label=r'nl $\Phi\Phi$',lw = 2)
plt.loglog(k,nlpsi, label=r'nl $\Psi\Psi$',lw = 2)



plt.xlabel('$k[h/\mathrm{Mpc}]$',fontsize=fontsi)
plt.ylabel(' $P_X(k)$',fontsize=fontsi)



plt.legend(loc='upper right', fontsize = fontsi,frameon=False)
plt.tight_layout()
plt.savefig('Spectra.pdf')
plt.show()
