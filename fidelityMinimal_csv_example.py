# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:31:20 2017

Generate fidelity weighting vector with exported forward operator (leadfield), 
inverse operator, and source identities

No MNE-Python required.

@author: rouhinen
"""

from fidelityOpMinimal import (compute_weighted_operator, fidelity_estimation,
                               make_series_paired, collapse_operator)
import glob
import matplotlib.pyplot as plt
import numpy as np


"""Load source identities, forward and inverse operators from csv. """
dataPath = 'C:\\temp\\fWeighting\\csvSubjects_p\\sub (5)'

fileSourceIdentities = glob.glob(dataPath + '\\sourceIdentities_parc2018yeo7_100.csv')[0]
fileForwardOperator  = glob.glob(dataPath + '\\forwardOperatorMEEG.csv')[0]
fileInverseOperator  = glob.glob(dataPath + '\\inverseOperatorMEEG.csv')[0]

delimiter = ';'
identities = np.genfromtxt(fileSourceIdentities, dtype='int32', delimiter=delimiter)         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.
forward = np.matrix(np.genfromtxt(fileForwardOperator, dtype='float', delimiter=delimiter))        # sensors x sources
inverse = np.matrix(np.genfromtxt(fileInverseOperator, dtype='float', delimiter=delimiter))        # sources x sensors

n_samples = 10000

""" Get number of parcels. """
idSet = set(identities)                         # Get unique IDs
idSet = [item for item in idSet if item >= 0]   # Remove negative values (should have only -1 if any)
n_parcels = len(idSet)

""" Compute weighted inverse operator. Use collapsed_inv_w for your inversing."""
inverse_w, weights, cplvs = compute_weighted_operator(forward, inverse, identities, n_samples=n_samples)
collapsed_inv_w = collapse_operator(inverse_w, identities) 


"""   Analyze results   """
""" Check if weighting worked. """
fidelity, cp_PLV = fidelity_estimation(forward, inverse_w, identities)
fidelityO, cp_PLVO = fidelity_estimation(forward, inverse, identities)

""" Create plots. """
fig, ax = plt.subplots()
ax.plot(np.sort(fidelity), color='k', linestyle=':', linewidth=1, label='Weighted fidelity, mean: ' + np.str(np.mean(fidelity)))
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



""" Compute cross-patch PLV values of paired data. """
parcelSeriesPairs, pairs = make_series_paired(n_parcels, n_samples)
_, cp_PLVP = fidelity_estimation(forward, inverse_w, identities, parcel_series=parcelSeriesPairs)
_, cp_PLVPO = fidelity_estimation(forward, inverse, identities, parcel_series=parcelSeriesPairs)

# Do the cross-patch PLV estimation for unmodeled series unpaired data. Baseline.
cp_PLVU = np.zeros([n_parcels, n_parcels], dtype=np.complex128)

for t in range(n_samples):
    parcelPLVn = parcelSeriesPairs[:,t] / np.abs(parcelSeriesPairs[:,t]) 
    cp_PLVU += np.outer(parcelPLVn, np.conjugate(parcelPLVn)) /n_samples


# Get truth matrix from the unforward-inverse modeled paired series.
cp_PLVUim = np.abs(np.imag(cp_PLVU))
truthMatrix = cp_PLVUim > 0.5

# Delete diagonal from truth and estimated matrices
def delete_diagonal(symm_matrix):
    symm_matrix = symm_matrix[~np.eye(symm_matrix.shape[0],dtype=bool)].reshape(
                                                symm_matrix.shape[0],-1)
    return symm_matrix

truthMatrix = delete_diagonal(truthMatrix)
cp_PLVP = delete_diagonal(cp_PLVP)
cp_PLVU = delete_diagonal(cp_PLVU)
cp_PLVPO = delete_diagonal(cp_PLVPO)


# Use imaginary PLV for the estimation.
cp_PLVPim = np.abs(np.imag(cp_PLVP))
cp_PLVPOim = np.abs(np.imag(cp_PLVPO))

## True positive and false positive rate estimation.
# Get true positive and false positive rates across thresholds. 
def get_tp_fp_rates(cp_PLV, truth_matrix):
    # Set thresholds from the data. Get as many thresholds as number of parcels.
    maxVal = np.max(cp_PLVPim)
    thresholds = np.sort(np.ravel(cp_PLVPim))
    thresholds = thresholds[0:-1:int(n_parcels/2)]
    thresholds = np.append(thresholds, maxVal)

    tpRate = np.zeros(len(thresholds), dtype=float)
    fpRate = np.zeros(len(thresholds), dtype=float)

    for i, threshold in enumerate(thresholds):
        estTrueMat = cp_PLV > threshold
        tPos = np.sum(estTrueMat * truthMatrix)
        fPos = np.sum(estTrueMat * np.logical_not(truthMatrix))
        tNeg = np.sum(np.logical_not(estTrueMat) * np.logical_not(truthMatrix))
        fNeg = np.sum(truthMatrix) - tPos
        
        tpRate[i] = tPos / (tPos + fNeg)
        fpRate[i] = fPos / (fPos + tNeg)
    return tpRate, fpRate

tpRateW, fpRateW = get_tp_fp_rates(cp_PLVPim, truthMatrix)
tpRateO, fpRateO = get_tp_fp_rates(cp_PLVPOim, truthMatrix)

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

fig, ax = plt.subplots()
ax.plot(fpRateW, tpRateW, color='k', linestyle=':', linewidth=1, label='Weighted, TPR at FPR 0.15: '
        + np.str(tpRateW[find_nearest_index(fpRateW, 0.15)]))
ax.plot(fpRateO, tpRateO, color='k', linestyle='-', label='Original, TPR at FPR 0.15: ' 
        + np.str(tpRateO[find_nearest_index(fpRateO, 0.15)]))

legend = ax.legend(loc='right', shadow=False, fontsize='12')
legend.get_frame()

ax.set_ylabel('True positive rate', fontsize='12')
ax.set_xlabel('False positive rate', fontsize='12')

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()


