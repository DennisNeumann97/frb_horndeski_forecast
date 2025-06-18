from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


kvec = np.loadtxt("kvector.txt")

a = np.loadtxt("pkhmcode_0_lin.txt")
anonlin = np.loadtxt("pkhmcode_0.txt")
b = np.loadtxt("pkhmcode_1_lin.txt")
bnonlin = np.loadtxt("pkhmcode_1.txt")

plt.loglog(kvec, anonlin, label='non linear 0')
plt.loglog(kvec, a, label='linear 0')
plt.loglog(kvec, bnonlin, label='non linear 1')
plt.loglog(kvec, b, label='linear 1')

plt.legend()
plt.savefig("spectrawithhmcode.pdf")

