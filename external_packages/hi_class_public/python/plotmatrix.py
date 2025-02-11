import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.figure()
a = np.loadtxt('avec.txt', unpack=True)
plt.matshow(a, cmap=plt.cm.viridis)#, norm=LogNorm(vmin=0.01, vmax=1))
plt.colorbar()
plt.title('phi/psi')
plt.savefig('matrix.png')

fontsi2 = 22

plt.figure()
b = np.loadtxt('bvec.txt', unpack=True)

plt.matshow(b, cmap=plt.cm.viridis)#, norm=LogNorm(vmin=0.01, vmax=1))
plt.colorbar()

#plt.xscale("log")
#plt.colorbarscale("log")


plt.ylabel(r'$z$',fontsize = fontsi2,labelpad=10)
plt.xlabel(r'$k [\mathrm{Mpc}]$',fontsize = fontsi2,labelpad=10)


plt.title('poissonratioxk^2')
plt.savefig('matrix2.png')
