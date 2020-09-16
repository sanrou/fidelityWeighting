# -*- coding: utf-8 -*-
"""
Created 2020.05.22.

Script for doing group level analysis of fidelity and true/false positive rates.
@author: rouhinen
"""

# from __future__ import division

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib
# from scipy import stats

from fidelityOpMinimal import fidelity_estimation, make_series_paired



"""Load source identities, forward and inverse operators from csv. """
subjectsPath = 'K:\\palva\\fidelityWeighting\\csvSubjects_p\\'

sourceIdPattern = '\\*parc2009_200AFS.csv'
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
subjectFolder = os.path.join(subjectsPath, subjects[0])
sourceIdFile = glob.glob(subjectFolder + sourceIdPattern)[0]

# fileSourceIdentities = os.path.join(subjectsPath, subject, sourceIdPattern)
identities = np.genfromtxt(sourceIdFile, dtype='int32', delimiter=delimiter)         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.

idSet = set(identities)                         # Get unique IDs
idSet = [item for item in idSet if item >= 0]   # Remove negative values (should have only -1 if any)
n_parcels = len(idSet)

## Initialize arrays
fidWArray = np.zeros((len(subjects), n_parcels), dtype=float)
fidOArray = np.zeros((len(subjects), n_parcels), dtype=float)
tpWArray = np.zeros((len(subjects), n_bins), dtype=float)
tpOArray = np.zeros((len(subjects), n_bins), dtype=float)
sizeArray = []

### Loop over subjects. Insert values to subject x parcels/bins arrays.
for run_i, subject in enumerate(subjects):

    ## Load files
    subjectFolder = os.path.join(subjectsPath, subject)
    fileSourceIdentities = glob.glob(subjectFolder + sourceIdPattern)[0]
    fileForwardOperator  = glob.glob(subjectFolder + '\\*forward*MEEG.csv')[0]
    fileInverseOperator  = glob.glob(subjectFolder + '\\*inverse*MEEG.csv')[0]
    fileWeightedOperator  = glob.glob(subjectFolder + '\\*weighted*MEEG*_200AFS.csv')[0]
    
    identities = np.genfromtxt(fileSourceIdentities, 
                                        dtype='int32', delimiter=delimiter)         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.
    forward = np.matrix(np.genfromtxt(fileForwardOperator, 
                                        dtype='float', delimiter=delimiter))        # sensors x sources
    inverse = np.matrix(np.genfromtxt(fileInverseOperator, 
                                        dtype='float', delimiter=delimiter))        # sources x sensors
    inverse_w = np.matrix(np.genfromtxt(fileWeightedOperator, 
                                        dtype='float', delimiter=delimiter))        # sources x sensors
    
    if run_i == 0:
        prior_n_parcels = n_parcels
    else:
        if prior_n_parcels == n_parcels:
            print('Running subject ' + subject)
        else:
            print('Mismatch in number of parcels between subjects!')
    
    """ Get fidelities from unpaired data. """
    # inverse_w, _ = _compute_weights(sourceSeries, parcelSeries, identities, inverse)
    fidelityW, _ = fidelity_estimation(forward, inverse_w, identities)
    fidelityO, _ = fidelity_estimation(forward, inverse, identities)
    
    
    """ Do network estimation. Get cross-patch PLV values from paired data"""
    parcelSeriesPairs, pairs = make_series_paired(n_parcels, n_samples)
    _, cp_PLVP = fidelity_estimation(forward, inverse_w, identities, parcel_series=parcelSeriesPairs)
    _, cp_PLVPO = fidelity_estimation(forward, inverse, identities, parcel_series=parcelSeriesPairs)
    
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
    sizeArray.append(len(identities))   # Approximate head size with number of sources.
    

### Statistics. 

fidWAverage = np.average(fidWArray, axis=0)
fidWStd = np.std(fidWArray, axis=0)
fidOAverage = np.average(fidOArray, axis=0)
fidOStd = np.std(fidOArray, axis=0)
tpWAverage = np.average(tpWArray, axis=0)
tpWStd = np.std(tpWArray, axis=0)
tpOAverage = np.average(tpOArray, axis=0)
tpOStd = np.std(tpOArray, axis=0)




"""   Plots   """
import pandas as pd

parcelList = list(range(0, n_parcels))

# Set font type to be CorelDraw compatible, supposedly. 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Set global figure parameters
import matplotlib.pylab as pylab
params = {'legend.fontsize': '7',
          'figure.figsize': (3, 2),
         'axes.labelsize': '7',
         'axes.titlesize':'7',
         'xtick.labelsize':'7',
         'ytick.labelsize':'7',
         'lines.linewidth':'0.5'}
pylab.rcParams.update(params)


""" Plot Fidelities. """
# Sort according to original fidelity
sorting = np.argsort(fidOAverage)

meansWF = pd.DataFrame(fidWAverage[sorting])
stdsWF = pd.DataFrame(fidWStd[sorting])
meansOF = pd.DataFrame(fidOAverage[sorting])
stdsOF = pd.DataFrame(fidOStd[sorting])

fig, ax = plt.subplots(1,1)

# Weighted
ax.plot(meansWF, color='black', linestyle='-', 
        label='Weighted fidelity, mean: ' + "{:.3f}".format(np.mean(fidWAverage)))
# Original
ax.plot(meansOF, color='black', linestyle='--', 
        label='Original fidelity, mean: ' + "{:.3f}".format(np.mean(fidOAverage)))

legend = ax.legend(loc='best', shadow=False)
legend.get_frame()

ax.fill_between(parcelList, np.ravel(meansWF-stdsWF), np.ravel(meansWF+stdsWF), color='black', alpha=0.5)
ax.fill_between(parcelList, np.ravel(meansOF-stdsOF), np.ravel(meansOF+stdsOF), color='black', alpha=0.3)
ax.set_ylabel('Fidelity')
ax.set_xlabel('Parcels, sorted by original')

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=0.1)
plt.show()


""" Plot ROC, True positives, false positives. """
meansW = pd.DataFrame(np.array([binArray, tpWAverage]).T,columns=['time','mean'])
stdsW = pd.DataFrame(tpWStd)
meansO = pd.DataFrame(np.array([binArray, tpOAverage]).T,columns=['time','mean'])
stdsO = pd.DataFrame(tpOStd)

fig, ax = plt.subplots(1,1)

# Weighted
ax.plot(meansW.iloc[:,0], meansW.iloc[:,1], color='black', linestyle='-', 
        label='Weighted, TPR at FPR 0.15: ' 
        + "{:.3f}".format(tpWAverage[find_nearest_index(binArray, 0.15)]))
# Original
ax.plot(meansO.iloc[:,0], meansO.iloc[:,1], color='black', linestyle='--', 
        label='Original, TPR at FPR 0.15: ' 
        + "{:.3f}".format(tpOAverage[find_nearest_index(binArray, 0.15)]))

legend = ax.legend(loc='right', shadow=False)
legend.get_frame()

ax.fill_between(meansW.iloc[:,0], meansW.iloc[:,1]-stdsW.iloc[:,0], meansW.iloc[:,1]+stdsW.iloc[:,0], color='black', alpha=0.5)
ax.fill_between(meansO.iloc[:,0], meansO.iloc[:,1]-stdsO.iloc[:,0], meansO.iloc[:,1]+stdsO.iloc[:,0], color='black', alpha=0.3)
ax.set_ylabel('True positive rate')
ax.set_xlabel('False positive rate')

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=0.1)
plt.show()




""" Make and plot relative benefits from weighting to fidelity. """
fidRelative = fidWArray / fidOArray
meansR = np.average(fidRelative, axis=0)
stdsR = np.std(fidRelative, axis=0)

# Sort parcels by average, and multiply by 100 to change to percentage.
sortArray = np.argsort(meansR)
meansR = 100* meansR[sortArray] 
stdsR = 100* stdsR[sortArray]

fig, ax = plt.subplots(1,1)

ax.plot(meansR, color='black', linestyle='-', 
        label='Relative fidelity, mean: ' + "{:.3f}".format(np.mean(meansR)))
ax.plot(100*np.ones(n_parcels, dtype=float), color='black', linestyle='-', linewidth=0.3)  # Set a horizontal line at 100 %.

legend = ax.legend(loc='best', shadow=False)
legend.get_frame()

ax.fill_between(parcelList, np.ravel(meansR-stdsR), np.ravel(meansR+stdsR), color='black', alpha=0.5)
ax.set_ylabel('Relative fidelity (%)')
ax.set_xlabel('Parcels, sorted by benefit')


ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=0.1)
plt.ylim(0, 300)
plt.show()



""" Plot relative ROC gain, True positives/True positives by false positives bins. """
# Skip first and last bin because division by zero
meansRR = pd.DataFrame(np.array([binArray[1:-1], 100* (tpWAverage[1:-1] / tpOAverage[1:-1])]).T,
                       columns=['FP bins','TP-relative'])
stdsRR = np.std(100* (tpWArray[:, 1:-1] / tpOArray[:, 1:-1]), axis=0)
stdsRR = pd.DataFrame(np.array([binArray[1:-1], stdsRR]).T, columns=['FP bins','TP-relative'])

fig, ax = plt.subplots(1,1)

ax.plot(meansRR.iloc[:,0], meansRR.iloc[:,1], color='black', linestyle='-', 
        label='Relative TPR at FPR 0.15: ' 
        + "{:.3f}".format(meansRR.iloc[:,1][find_nearest_index(binArray[1:-1], 0.15)]))
# ax.plot(np.ones(len(binArray[1:-1]), dtype=float), color='black', linestyle='-', linewidth=0.3)  # Set a horizontal line at 100 %.
ax.plot([0, 1], [100, 100], color='black', linestyle='-', linewidth=0.3)  # Set a horizontal line at 100 %. [X X] [Y, Y]

legend = ax.legend(loc='best', shadow=False)
legend.get_frame()

ax.fill_between(meansRR.iloc[:,0], meansRR.iloc[:,1]-stdsRR.iloc[:,1],
                meansRR.iloc[:,1]+stdsRR.iloc[:,1], color='black', alpha=0.5)
ax.set_ylabel('Relative true positive rate (%)')
ax.set_xlabel('False positive rate')

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=0.1)
plt.show()
