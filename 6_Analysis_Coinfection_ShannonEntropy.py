# Code developed by Gabriella Dantas Franco and Denise Cammarota
# Exploring the Shannon entropy of fraction of infecteds as a measure of strain diversity

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import scipy.stats as st
from operator import matmul
from scipy.integrate import odeint, simps

from Dyn_strains_M_CI_mut import Dyn_strains_M_CI_mut

# Parameters for simulation
pars = {}
pars['nv'] = 100     # number of variants
mu =  (1./(80*365))
pars['mu'] = (mu) * np.ones(pars['nv'])                     # demography
pars['gamma']= np.linspace(0.1, 0.1, pars['nv'])                # recovery rate (1/days)
pars['beta'] = np.linspace(0.4, 8.0, pars['nv'])*pars['gamma']  # feasible betas
pars['basR0'] = pars['beta']/(pars['gamma']+ mu)                   # R0 number

# Cross-immunity matrix calculation
pars['sigmaprime'] = np.zeros((pars['nv'], pars['nv']))
pars['sigma'] = np.zeros((pars['nv'], pars['nv']))
pars['mutation'] = np.zeros((pars['nv'], pars['nv']))
m = (1/100) # m mutation rate
d = 10 # d typical distance between variants for cross-immunity
for i in range(pars['nv']):
    pars['mutation'][i,i] = -2*m
    for j in range(pars['nv']):
        pars['sigma'][i,j] = np.exp(-((i-j)/d)**2)
        pars['sigmaprime'][i,j] = np.exp(-((i-j)/d)**2)
        if np.abs(i-j)==1:
            pars['mutation'][i,j] = m
    pars['sigmaprime'][i,i] = 0

pars['N'] = 1000000
# Infection starts with 10 individuals being infected at strain i0
pars['I0'] = np.zeros(pars['nv'])
i0 = 20
pars['I0'][i0] = 10
pars['S0']  =  pars['N'] * np.ones(pars['nv']) - np.sum(pars['I0'])

x0 = np.zeros(2*pars['nv'])
x0[: pars['nv']] = pars['S0']/pars['N']
x0[pars['nv'] : 2*pars['nv']] = pars['I0']/pars['N']


#Solving with odeint from scipy
t = np.arange(0.1, 2000, .1)
sol = odeint(Dyn_strains_M_CI_mut, x0, t, args =(pars,))


### Visualization 1
plt.figure(figsize=(10,8))
for i in range(pars['nv']):
    #plt.plot(t, sol[:, i], label = '$S_{%i}$' % i)
    #plt.plot(t, sol1[:, i], label = '$S_{%i}$' % i)
    plt.plot(t[:], sol[:, pars['nv']+i], label = '$I_{%i}$' % (i+1), linewidth = 3)
#plt.legend(loc='best', fontsize = 14)
plt.xlabel('Time (days)', fontsize = 18)
plt.ylabel(r'$I_i$', fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.savefig('figs/6_EpiCurves.pdf')
plt.show()

### Visualization 2

####  Proportion of infecteds

St = np.zeros((pars['nv'], len(t)))
It = np.zeros((pars['nv'], len(t)))
for j in range(pars['nv']):
    St[j] = sol[:, j]
    It[j] = sol[:, pars['nv']+j]

plt.figure(figsize=(20,6))
cmap = 'inferno'
for i in range(pars['nv']):
    plt.fill_between(t, 0, np.sum(It[i:], axis=0)/np.sum(It, axis=0),
                     color = sns.color_palette(cmap, pars['nv'])[i],
                    label = f'Strain %i' % i, edgecolor = 'k', alpha = 0.5)

plt.xlabel('Time (days)', fontsize = 14)
plt.ylabel(r'$\frac{I_i}{I_T}$', fontsize = 14, rotation = 0, labelpad=16)
plt.savefig('figs/6_FreqCurves.pdf')
plt.show()

### Visualization 3 :  Dominating strains

fracStrains = It/np.sum(It, axis=0)
dom = np.where(fracStrains > 0.1)[0]
domt = 0.1*np.where(fracStrains > 0.1)[1]

plt.plot(domt, dom, marker = '.', linestyle='None')
plt.title('Strains that represent more than 10% of infecteds')
plt.xlabel('Time')
plt.ylabel('Strain index')
plt.savefig('figs/6_DomStrains.pdf')
plt.show()

### Visualization 4 : Shannon Entropy

S = np.zeros(len(t))
for i in range(pars['nv']):
    for tt in range(len(t)):
        if fracStrains[i, tt] > 0:
            S[tt] -= fracStrains[i, tt] * np.log2(fracStrains[i, tt])

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time (days)', fontsize = 14)
ax1.set_ylabel(r'$S(t)$', color=color, fontsize = 14)
ax1.plot(t, S, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel(r'$I_T(t) = \sum_i I_i$', color=color, fontsize = 14)  # we already handled the x-label with ax1
ax2.plot(t, np.sum(It, axis=0), color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('figs/6_Shannon.pdf')
plt.show()
