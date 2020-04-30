# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:31:20 2017

Generate fidelity weighting vector with exported forward operator (leadfield), 
inverse operator, and source identities

No MNE-Python required.

@author: rouhinen
"""

from fidelityOpMinimal import make_series, _compute_weights, fidelity_estimation
import os
import matplotlib.pyplot as plt
import numpy as np


"""Load source identities, forward and inverse operators from csv. """
dataPath = 'K:\\palva\\fidelityWeighting\\example data\\s11'

fileSourceIdentities = os.path.join(dataPath, 'sourceIdentities.csv')
fileForwardOperator  = os.path.join(dataPath, 'forwardOperator.csv')
fileInverseOperator  = os.path.join(dataPath, 'inverseOperator.csv')

delimiter = ','
identities = np.genfromtxt(fileSourceIdentities, 
                                    dtype='int32', delimiter=delimiter)         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.
forward = np.matrix(np.genfromtxt(fileForwardOperator, 
                                    dtype='float', delimiter=delimiter))        # sensors x sources
inverse = np.matrix(np.genfromtxt(fileInverseOperator, 
                                    dtype='float', delimiter=delimiter))        # sources x sensors


""" Generate signals for parcels. """
idSet = set(identities)                         # Get unique IDs
idSet = [item for item in idSet if item >= 0]   # Remove negative values (should have only -1 if any)
n_parcels = len(idSet)

n_samples = 10000
n_cut_samples = 40
widths = np.arange(5, 6)

parcelSeries = make_series(n_parcels, n_samples, n_cut_samples, widths)

""" Parcel series to source series. 0 signal for sources not belonging to a parcel. """
sourceSeries = parcelSeries[identities]

sourceSeries[identities < 0] = 0

""" Forward then inverse model source series. """
sourceSeries = np.dot(inverse, np.dot(forward, sourceSeries))

""" Compute weighted inverse operator. """
inverse_w, weights = _compute_weights(sourceSeries, parcelSeries, identities, inverse)



"""   Analyze results   """
""" Check if weighting worked. """
fidelity, cp_PLV = fidelity_estimation(forward, inverse_w, identities)
fidelityO, cp_PLVO = fidelity_estimation(forward, inverse, identities)

""" Create plots. """
fig, ax = plt.subplots()
ax.plot(np.sort(fidelity), color='k', linestyle='--', label='Weighted fidelity, mean: ' + np.str(np.mean(fidelity)))
ax.plot(np.sort(fidelityO), color='k', linestyle='-', label='Original fidelity, mean: ' + np.str(np.mean(fidelityO)))

legend = ax.legend(loc='upper center', shadow=False, fontsize='12')
legend.get_frame()

ax.set_ylabel('Estimated fidelity', fontsize='12')
ax.set_xlabel('Sorted parcels', fontsize='12')

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()




# fidelityLV, cp_PLVLV = fidelity_estimation(forward, inverseLV, identities)   ### TEMP

