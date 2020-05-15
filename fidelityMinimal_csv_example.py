# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:31:20 2017

Generate fidelity weighting vector with exported forward operator (leadfield), 
inverse operator, and source identities

No MNE-Python required.

@author: rouhinen
"""

from fidelityOpMinimal import (make_series, _compute_weights, fidelity_estimation,
                               make_series_with_time_shift)
import os
import matplotlib.pyplot as plt
import numpy as np


"""Load source identities, forward and inverse operators from csv. """
dataPath = 'K:\\palva\\fidelityWeighting\\example data\\s11'

fileSourceIdentities = os.path.join(dataPath, 'sourceIdentities_200.csv')
fileForwardOperator  = os.path.join(dataPath, 'forwardOperator.csv')
fileInverseOperator  = os.path.join(dataPath, 'inverseOperator.csv')

delimiter = ';'
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




""" Do network estimation. Code work in progress. """

parcelSeriesPairs, pairs = make_series_with_time_shift(n_parcels, n_samples)

""" Compute cross-patch PLV values of paired data. """
pim, cp_PLVP = fidelity_estimation(forward, inverse_w, identities, parcel_series=parcelSeriesPairs)
pim, cp_PLVPO = fidelity_estimation(forward, inverse, identities, parcel_series=parcelSeriesPairs)

# Do the cross-patch PLV estimation for unmodeled series
cp_PLVU = np.zeros([n_parcels, n_parcels], dtype=np.complex128)

for t in range(n_samples):
    parcelPLVn = parcelSeriesPairs[:,t] / np.abs(parcelSeriesPairs[:,t]) 
    cp_PLVU += np.outer(parcelPLVn, np.conjugate(parcelPLVn)) /n_samples


# Get truth matrix from the unmodeled series.
cp_PLVUim = np.abs(np.imag(cp_PLVU))
truthMatrix = cp_PLVUim > 0.5

# Delete diagonal from truth and estimated matrices
truthMatrix = truthMatrix[~np.eye(truthMatrix.shape[0],dtype=bool)].reshape(
                                                truthMatrix.shape[0],-1)
cp_PLVP = cp_PLVP[~np.eye(cp_PLVP.shape[0],dtype=bool)].reshape(
                                                cp_PLVP.shape[0],-1)

# Use imaginary PLV for the estimation.
cp_PLVPim = np.abs(np.imag(cp_PLVP))

## True positive and false positive rate estimation.
# Set thresholds from the data. Get as many thresholds as number of parcels.
maxVal = np.max(cp_PLVPim)
thresholds = np.sort(np.ravel(cp_PLVPim))
thresholds = thresholds[0:-1:n_parcels]
thresholds = np.append(thresholds, maxVal)

# Get true positive and false positive rates across thresholds. 
tpRate = np.zeros(len(thresholds), dtype=float)
fpRate = np.zeros(len(thresholds), dtype=float)

for i, threshold in enumerate(thresholds):
    estTrueMat = cp_PLVPim > threshold
    tPos = np.sum(estTrueMat * truthMatrix)
    fPos = np.sum(estTrueMat * np.logical_not(truthMatrix))
    tNeg = np.sum(np.logical_not(estTrueMat) * np.logical_not(truthMatrix))
    fNeg = np.sum(truthMatrix) - tPos
    
    tpRate[i] = tPos / (tPos + fNeg)
    fpRate[i] = fPos / (fPos + tNeg)


plt.plot(fpRate, tpRate)


""" Pseudocode """
### Do the estimation separately with normal inverse operator, and weighted one.
## Inputs are n_iterations, parcellation/source_identities, forward and inverse.



## Create "bins" for X-Axis.
n_bins = 101
binArray = np.linspace(0, 1, n_bins, endpoint=True)


### Create cp-PLV matrix. One without forward-inverse (true or in values). One values with modeling (estimation values)
## Output is n_parcels x n_parcels (N x N). Values is cPLV. Extract iPLV by default.
# Truth matrix from very high threshold (0.5 iPLV). 
# 


### Threshold estimation (p-value)
## Estimate threshold from uncorrelated Real values. Like top 5 % thresholded from non-diagonal values. The threshold will be changed.
# Threshold 1 - p-value of non-diagonal values. This will be the threshold for the simulated matrix values with degree one.

## Define get ROC.
# Input is binary truth array (N x N) and iPLV modeled array (N x N).
# Output is ROC curve.
# TP and FP rates estimated at different thresholds. 


## Cross-Patch PLV (original)
# For each sample, take np.dot(patchSeries, )

