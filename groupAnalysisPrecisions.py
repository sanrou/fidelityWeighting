# -*- coding: utf-8 -*-
"""
Created 2020.09.28.

Script for doing group level analysis of anatomical precision rates.
@author: rouhinen
"""

import numpy as np
import os
import glob
import matplotlib.pyplot as plt

from fidelityOpMinimal import (make_series_paired, make_series, collapse_operator)



"""Load source identities, forward and inverse operators from csv. """
subjectsPath = 'K:\\palva\\fidelityWeighting\\csvSubjects_p\\'

sourceIdPattern = '\\sourceId*200AFS.csv'    # For collapsing the forward operator.
forwardPattern  = '\\*forward*MEEG.csv'     # Should be the at source space
inversePattern  = '\\weighted_*MEEG_200AFS_noParcelFlip_collapsed.csv'     # Should be collapsed to parcel space. Could be the original, as the collapsing is fast. Though file reading is much faster with collapsed files.

n_iterations = 100     # Takes about 35 minutes per subject with 1000 iterations with parc68.
delimiter = ';'
n_samples = 5000
n_cut_samples = 40
widths = np.arange(5, 6)


def get_precision_rates(cp_PLV, truth_matrix):
    # Set thresholds from the data. Get about 200 thresholds.
    maxVal = np.max(cp_PLV)
    thresholds = np.sort(np.ravel(cp_PLV))
    distance = int(len(thresholds) // 200) + (len(thresholds) % 200 > 0)     # To int, round up.
    thresholds = thresholds[0:-1:distance]
    thresholds = np.append(thresholds, maxVal)
    
    precisions = np.zeros([cp_PLV.shape[0], len(thresholds)], dtype=float)
    
    for i, threshold in enumerate(thresholds):
        estTrueMat = cp_PLV > threshold
        tPos = np.sum(estTrueMat * truth_matrix, axis=1)
        fPos = np.sum(estTrueMat * np.logical_not(truth_matrix), axis=1)
        precisions[:, i] = tPos/(tPos + fPos)

    return precisions, thresholds


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# def get_nearest_tp_semi_bin(binArray, tpRate, fpRate):
#     nearestTP = np.zeros(len(binArray))
#     for i, fpval in enumerate(binArray):
#         index = find_nearest_index(fpRate, fpval)
#         nearestTP[i] = tpRate[index]
#     return nearestTP


def delete_diagonal(symm_matrix):
    symm_matrix = symm_matrix[~np.eye(symm_matrix.shape[0],dtype=bool)].reshape(
                                                symm_matrix.shape[0],-1)
    return symm_matrix


def fidelity_estimation_collapsed(fwd, inv, n_samples = 5000, parcel_series=np.asarray([])):
    ''' Compute fidelity and cross-patch PLV (see Korhonen et al 2014)
    Can be used for exclusion of low-fidelity parcels and parcel pairs with high CP-PLV.
    
    Input arguments: 
    ================
    fwd : Forward operator matrix, ndarray [sensors x parcels]
    inv : Inverse operator matrix, ndarray [parcels x sensors]
        Note that collapsed operators are expected.
    n_samples: int
        the number of samples generated for simulated data
    parcel_series : ndarray, complex [parcels x samples]
        If empty, time series will be generated. Else given series is used.
        Overrides n_samples if given.
        
    Output arguments:
    =================
    fidelity : 1D array.
        Fidelity values for each parcel.
    cpPLV : 2D array
        Cross-patch PLV of the reconstructed time series among all parcel pairs.    
    '''
    
    timeCut = 20
    widths=np.arange(5, 6)
    
    N_parcels = inv.shape[0]
    
    ## Check if source time series is empty. If empty, create time series.
    if parcel_series.size == 0:
        origParcelSeries = make_series(N_parcels, n_samples, timeCut, widths)
    else:
        origParcelSeries = parcel_series
        n_samples = parcel_series.shape[1]
    
    # Forward and inverse model cloned parcel time series
    estimatedParcelSeries = np.dot(inv, np.dot(fwd,parcel_series))
    
    # Do the cross-patch PLV estimation before changing the amplitude to 1. 
    cpPLV = np.zeros([N_parcels, N_parcels], dtype=np.complex128)
    
    for t in range(n_samples):
        parcelPLVn = estimatedParcelSeries[:,t] / np.abs(estimatedParcelSeries[:,t]) 
        cpPLV += np.outer(parcelPLVn, np.conjugate(parcelPLVn)) /n_samples
    
    # Change to amplitude 1, keep angle using Euler's formula.
    origParcelSeries = np.exp(1j*(np.asmatrix(np.angle(origParcelSeries))))   
    estimatedParcelSeries = np.exp(1j*(np.asmatrix(np.angle(estimatedParcelSeries))))
    
    # Estimate parcel fidelity.
    fidelity = np.zeros(N_parcels, dtype=np.float32)  # For the weighted inverse operator
    
    for i in range(N_parcels):
        A = np.ravel(origParcelSeries[i,:])                        # True simulated parcel time series. 
        B = np.ravel(estimatedParcelSeries[i,:])                       # Estimated parcel time series. 
        fidelity[i] = np.abs(np.mean(A * np.conjugate(B)))   # Maybe one should take np.abs() away. Though abs is the value you want.
    
    return fidelity, cpPLV


## Create "bins" for X-Axis (False positive rate). Unnecessary?
n_bins = 101
binArray = np.logspace(-2, 0, n_bins-1, endpoint=True)    # Values from 0.01 to 1
binArray = np.concatenate(([0], binArray))  # Add 0 to beginning


## Get subjects list, and first subject's number of parcels.
subjects = next(os.walk(subjectsPath))[1]
if any('_Population' in s for s in subjects):
    subjects.remove('_Population')

subjectFolder = os.path.join(subjectsPath, subjects[0])
inverseFile = glob.glob(subjectFolder + inversePattern)[0]

inverse = np.genfromtxt(inverseFile, dtype='float64', delimiter=delimiter)         # Parcels x Sensors if the file is collapsed (should be)
n_parcels = inverse.shape[0]    

## Initialize arrays
_, thresholds = get_precision_rates(np.ones((n_parcels, n_parcels-1)), 
                                    np.ones((n_parcels, n_parcels-1)))
threshArray = np.zeros((len(subjects), len(thresholds)), dtype=float)
precisArray = np.zeros((len(subjects), n_parcels, len(thresholds)), dtype=float)
sizeArray = []

### Loop over subjects. Run TP rate estimation multiple times. Insert values to subject x parcels/bins arrays.
for si, subject in enumerate(subjects):
    ## Load files
    subjectFolder = os.path.join(subjectsPath, subject)
    fileSourceIdentities = glob.glob(subjectFolder + sourceIdPattern)[0]
    fileForwardOperator  = glob.glob(subjectFolder + forwardPattern)[0]
    fileInverseOperator  = glob.glob(subjectFolder + inversePattern)[0]
    
    identities = np.genfromtxt(fileSourceIdentities, 
                                        dtype='int32', delimiter=delimiter)         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.
    forward = np.matrix(np.genfromtxt(fileForwardOperator, 
                                        dtype='float', delimiter=delimiter))        # sensors x parcels
    inverse = np.matrix(np.genfromtxt(fileInverseOperator, 
                                        dtype='float', delimiter=delimiter))        # parcels x sensors
    
    n_parcels = inverse.shape[0]
    
    if si == 0:
        prior_n_parcels = n_parcels
        print('Running subject ' + subject)
    else:
        if prior_n_parcels == n_parcels:
            print('Running subject ' + subject)
        else:
            print('Mismatch in number of parcels between subjects!')
    
    # Collapse forward operator.
    forward = collapse_operator(forward, identities, op_type='forward')
    
    """ Do network estimation several times. Get cross-patch PLV values from paired data"""
    threshIter = np.zeros((n_iterations, len(thresholds)), dtype=float)
    precisIter = np.zeros((n_iterations, n_parcels, len(thresholds)), dtype=float)
    
    for ni in range(n_iterations):
        parcelSeriesPairs, pairs = make_series_paired(n_parcels, n_samples)
        # _, cp_PLVP = fidelity_estimation(forward, inverse_w, identities, parcel_series=parcelSeriesPairs)
        _, cp_PLVPO = fidelity_estimation_collapsed(forward, inverse, parcel_series=parcelSeriesPairs)
        
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
        cp_PLVPO = delete_diagonal(cp_PLVPO)
        
        # Use imaginary PLV for the estimation.
        cp_PLVOim = np.abs(np.imag(cp_PLVPO))
        
        ## Precision estimation.
        precisions, thresholds = get_precision_rates(cp_PLVOim, truthMatrix)
        
        precisions[np.isnan(precisions)] = 0    # NaNs to zero.
        precisIter[ni,:,:] = precisions
        threshIter[ni,:] = thresholds
    
    precisArray[si,:,:] = np.mean(precisIter, axis=0)   # Subjects x Parcels x Thresholds
    threshArray[si,:] = np.mean(threshIter, axis=0)

### Statistics. 


precisAverage = np.average(precisArray, axis=0)
precisStd = np.std(precisArray, axis=0)
threshAverage = np.average(threshArray, axis=0)
threshStd = np.std(threshArray, axis=0)


"""   Plots   """
import pandas as pd

parcelList = list(range(0, n_parcels))

# Set global figure parameters, including CorelDraw compatibility (.fonttype)
import matplotlib.pylab as pylab
params = {'legend.fontsize':'7',
          'figure.figsize':(3, 2),
         'axes.labelsize':'7',
         'axes.titlesize':'7',
         'xtick.labelsize':'7',
         'ytick.labelsize':'7',
         'lines.linewidth':'0.5',
         'pdf.fonttype':42,
         'ps.fonttype':42}
pylab.rcParams.update(params)



""" Plot precisions as function of threshold. """
sumPres = np.sum(precisAverage, axis=0)   # Sum across parcels

fig, ax = plt.subplots(1,1)

# Averaged
ax.plot(threshAverage, sumPres, color='black', linestyle='-', 
        label='Max precision: ' + "{:.3f}".format(max(sumPres)))

ax.set_ylabel('Summed precision')
ax.set_xlabel('Threshold (iPLV)')

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

legend = ax.legend(loc='best', shadow=False)
legend.get_frame()

plt.tight_layout(pad=0.1)
plt.show()




""" Plot precisions at max summed precision threhold. """
# Find maximum precision threshold
maxPresIndex = np.argmax(sumPres)
precisAveAtMax = precisAverage[:,maxPresIndex]
precisStdAtMax = precisStd[:,maxPresIndex]

# Sort according to average precision
sorting = np.argsort(precisAveAtMax)
presAveSorted = pd.DataFrame(precisAveAtMax[sorting])
presStdSorted = pd.DataFrame(precisStdAtMax[sorting])

fig, ax = plt.subplots(1,1)

# Average
ax.plot(presAveSorted, color='black', linestyle='-',
        label='Precision, mean: ' + "{:.3f}".format(np.mean(np.asarray(presAveSorted))))

legend = ax.legend(loc='best', shadow=False)
legend.get_frame()

ax.fill_between(parcelList, np.ravel(presAveSorted-presStdSorted), 
                np.ravel(presAveSorted+presStdSorted), color='black', alpha=0.5)

ax.set_ylabel('Precision')
ax.set_xlabel('Parcels, sorted by precision')

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=0.1)
plt.show()

