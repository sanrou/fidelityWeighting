# -*- coding: utf-8 -*-
"""
Created 2020.05.22.

Script for doing group level analysis of fidelity and true/false positive rates.
@author: rouhinen
"""


"""Pseudocode for TP / FP ROC code"""

# from __future__ import division

import numpy as np
import os
import matplotlib.pyplot as plt

from fidelityOpMinimal import (make_series, _compute_weights, fidelity_estimation, 
                               make_series_paired)



### Set subjects and directories
"""Load source identities, forward and inverse operators from csv. """
subjectsPath = 'K:\\palva\\fidelityWeighting\\csvSubjects\\'

sourceIdFileName = 'sourceIdentities_200.csv'
delimiter = ';'
n_samples = 10000
n_cut_samples = 40
widths = np.arange(5, 6)


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


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_nearest_tp_semi_bin(binArray, tpRate, fpRate):
    nearestTP = np.zeros(len(binArray))
    for i, fpval in enumerate(binArray):
        index = find_nearest_index(fpRate, fpval)
        nearestTP[i] = tpRate[index]
    return nearestTP


def delete_diagonal(symm_matrix):
    symm_matrix = symm_matrix[~np.eye(symm_matrix.shape[0],dtype=bool)].reshape(
                                                symm_matrix.shape[0],-1)
    return symm_matrix


## Create "bins" for X-Axis. 
n_bins = 101
binArray = np.logspace(-2, 0, n_bins-1, endpoint=True)    # Values from 0.01 to 1
binArray = np.concatenate(([0], binArray))  # Add 0 to beginning



## Get subjects list, and first subject's number of parcels.
subjects = next(os.walk(subjectsPath))[1]
subject = subjects[1]
fileSourceIdentities = os.path.join(subjectsPath, subject, sourceIdFileName)
identities = np.genfromtxt(fileSourceIdentities, 
                                    dtype='int32', delimiter=delimiter)         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.

idSet = set(identities)                         # Get unique IDs
idSet = [item for item in idSet if item >= 0]   # Remove negative values (should have only -1 if any)
n_parcels = len(idSet)

## Initialize arrays
fidWArray = np.zeros((len(subjects), n_parcels), dtype=float)
fidOArray = np.zeros((len(subjects), n_parcels), dtype=float)
tpWArray = np.zeros((len(subjects), n_bins), dtype=float)
tpOArray = np.zeros((len(subjects), n_bins), dtype=float)

### Loop over subjects
## Load files
# Without loop for now
for run_i, subject in enumerate(subjects):

    fileSourceIdentities = os.path.join(subjectsPath, subject, sourceIdFileName)
    fileForwardOperator  = os.path.join(subjectsPath, subject, 'forwardOperator.csv')
    fileInverseOperator  = os.path.join(subjectsPath, subject, 'inverseOperator.csv')
    
    identities = np.genfromtxt(fileSourceIdentities, 
                                        dtype='int32', delimiter=delimiter)         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.
    forward = np.matrix(np.genfromtxt(fileForwardOperator, 
                                        dtype='float', delimiter=delimiter))        # sensors x sources
    inverse = np.matrix(np.genfromtxt(fileInverseOperator, 
                                        dtype='float', delimiter=delimiter))        # sources x sensors
    
    # Create weighted operator
    """ Generate signals for parcels. """
    
    if run_i == 0:
        prior_n_parcels = n_parcels
    else:
        if prior_n_parcels == n_parcels:
            print('Done with subject number ' + str(run_i))
        else:
            print('Mismatch in number of parcels between subjects!')
    
    
    parcelSeries = make_series(n_parcels, n_samples, n_cut_samples, widths)
    
    """ Parcel series to source series. 0 signal for sources not belonging to a parcel. """
    sourceSeries = parcelSeries[identities]
    sourceSeries[identities < 0] = 0
    
    """ Forward then inverse model source series. """
    sourceSeries = np.dot(inverse, np.dot(forward, sourceSeries))
    
    """ Compute weighted inverse operator. Get fidelities from unpaired data. """
    inverse_w, weights = _compute_weights(sourceSeries, parcelSeries, identities, inverse)
    fidelityW, pim = fidelity_estimation(forward, inverse_w, identities)
    fidelityO, pim = fidelity_estimation(forward, inverse, identities)
    
    
    """ Do network estimation. Get cross-patch PLV values from paired data"""
    parcelSeriesPairs, pairs = make_series_paired(n_parcels, n_samples)
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
    truthMatrix = delete_diagonal(truthMatrix)
    cp_PLVP = delete_diagonal(cp_PLVP)
    cp_PLVPO = delete_diagonal(cp_PLVPO)
    # cp_PLVU = delete_diagonal(cp_PLVU)
    
    # Use imaginary PLV for the estimation.
    cp_PLVPim = np.abs(np.imag(cp_PLVP))
    cp_PLVOim = np.abs(np.imag(cp_PLVPO))
    
    ## True positive and false positive rate estimation.
    tpRateW, fpRateW = get_tp_fp_rates(cp_PLVPim, truthMatrix)
    tpRateO, fpRateO = get_tp_fp_rates(cp_PLVOim, truthMatrix)
    
    # Get nearest TP values closest to the FP pair at the "bin" values.
    nearTPW = get_nearest_tp_semi_bin(binArray, tpRateW, fpRateW)
    nearTPO = get_nearest_tp_semi_bin(binArray, tpRateO, fpRateO)
    
    fidWArray[run_i,:] = fidelityW
    fidOArray[run_i,:] = fidelityO
    tpWArray[run_i,:] = nearTPW
    tpOArray[run_i,:] = nearTPO



### Out of for loop. 

fidWAverage = np.average(fidWArray, axis=0)
fidOAverage = np.average(fidOArray, axis=0)
tpWAverage = np.average(tpWArray, axis=0)
tpOAverage = np.average(tpOArray, axis=0)



""" Plot fidelities. """
fig, ax = plt.subplots()
ax.plot(np.sort(fidWAverage), color='k', linestyle='--', label='Weighted fidelity, mean: ' + np.str(np.mean(fidWAverage)))
ax.plot(np.sort(fidOAverage), color='k', linestyle='-', label='Original fidelity, mean: ' + np.str(np.mean(fidOAverage)))

legend = ax.legend(loc='upper center', shadow=False, fontsize='12')
legend.get_frame()

ax.set_ylabel('Estimated fidelity', fontsize='12')
ax.set_xlabel('Sorted parcels', fontsize='12')

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()




""" Plot ROC. """
fig, ax = plt.subplots()
ax.plot(binArray, tpWAverage, color='k', linestyle='--', label='Weighted, TPR at FPR 0.15: '
        + np.str(tpWAverage[find_nearest_index(binArray, 0.15)]))
ax.plot(binArray, tpOAverage, color='k', linestyle='-', label='Original, TPR at FPR 0.15: '
        + np.str(tpOAverage[find_nearest_index(binArray, 0.15)]))

legend = ax.legend(loc='upper center', shadow=False, fontsize='12')
legend.get_frame()

ax.set_ylabel('True positive rate', fontsize='12')
ax.set_xlabel('False positive rate', fontsize='12')

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()


## Get ROC and fidelity
# Unweighted and weighted ROC and fidelity
# Output unsorted fidelities, so that one can get population values at anatomical space.


### Analyze group values
## ROC
# Use fixed false positive values, get nearest true positive values.
## Fidelities
# Get difference values of unweighted and weighted fidelities. Maybe weighted-original fidelity would be good, as very small fidelities could give huge ratio improvements that are not meaningful.
