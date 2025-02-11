import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm


data = np.loadtxt("otf.txt")

sigma = data[:,0]
delta = data[:,1]
prob = data[:,3]

n = 100

sigmamatrix = []
deltamatrix = []
probmatrix = []

sigmaaux = []

fontsi = 18
fontsi2 = 22

plt.tick_params(labelsize=fontsi)


plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')

for x in range(0, n):
	sigmaaux.append(sigma[x*n])

for x in range(0, n):
	sigmamatrix.append(sigmaaux)


for x in range(0, n):
	deltaaux = []
	for y in range(0, n):
		deltaaux.append(delta[y*n+x])
	deltamatrix.append(deltaaux)


for x in range(0, n):
	probaux = []
	for y in range(0, n):
		probaux.append(prob[y*n+x])
                #probaux.append(prob[y*n+x]) 
	probmatrix.append(probaux)



plt.contourf(sigmamatrix,deltamatrix,probmatrix,500)

#plt.contourf(sigmamatrix,deltamatrix,probmatrix,500)

cm = plt.colorbar()
#cm.set_label(r'$k^2\Phi/\delta $',fontsize = fontsi2,labelpad=10)
cm.set_label(r'$\Phi/\Psi$',fontsize = fontsi2,labelpad=10)

plt.xscale("log")
#plt.colorbarscale("log")


plt.ylabel(r'$z$',fontsize = fontsi2,labelpad=10)
plt.xlabel(r'$k [\mathrm{Mpc}]$',fontsize = fontsi2,labelpad=10)
plt.tight_layout()

#plt.xlim(-1., 1.)
#plt.ylim(-.3, 1.4)

plt.savefig('Bardeenpotentials2new.pdf')
#plt.savefig('Poissonequation2new.pdf')
#plt.show()
