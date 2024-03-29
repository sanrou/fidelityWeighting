# -*- coding: utf-8 -*-
"""
Created 2020.05.22.

Script for doing group level analysis of fidelity and true/false positive rates.
@author: rouhinen
"""

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from fidelityOpMinimal import fidelity_estimation, make_series_paired, source_fid_to_weights


"""Load source identities, forward and inverse operators from npy. """
subjectsPath = 'C:\\temp\\fWeighting\\fwSubjects_p\\'

sourceIdPattern = '\\sourceIdentities_parc2018yeo7_XYZ.npy'   # XYZ is replaced below.
sourceFidPattern = '\\sourceFidelities_MEEG_parc2018yeo7_XYZ.npy'
savePathBase = "C:\\temp\\fWeighting\\plotDump\\schaeferXYZ "
forwardPattern  = '\\forwardOperatorMEEG.npy'
inversePattern  = '\\inverseOperatorMEEG.npy'
XYZto = '100'

n_samples = 10000
n_cut_samples = 40
widths = np.arange(5, 6)

# Source fidelity to weights settings
exponent = 2
normalize = True
flips = False

# Save and plotting settings
savePDFs = True
tightLayout = True


""" Replace XYZ """
sourceIdPattern = sourceIdPattern.replace('XYZ', XYZto)
sourceFidPattern = sourceFidPattern.replace('XYZ', XYZto)
savePathBase = savePathBase.replace('XYZ', XYZto)


def get_tp_fp_rates(cp_PLV, truth_matrix):
    # Set thresholds from the data. Get about 200 thresholds.
    maxVal = np.max(cp_PLV)
    thresholds = np.sort(np.ravel(cp_PLV))
    distance = int(len(thresholds) // 200) + (len(thresholds) % 200 > 0)     # To int, round up.
    thresholds = thresholds[0:-1:distance]
    thresholds = np.append(thresholds, maxVal)

    tpRate = np.zeros(len(thresholds), dtype=float)
    fpRate = np.zeros(len(thresholds), dtype=float)

    for i, threshold in enumerate(thresholds):
        estTrueMat = cp_PLV > threshold
        tPos = np.sum(estTrueMat * truth_matrix)
        fPos = np.sum(estTrueMat * np.logical_not(truth_matrix))
        tNeg = np.sum(np.logical_not(estTrueMat) * np.logical_not(truth_matrix))
        fNeg = np.sum(truth_matrix) - tPos
        
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

def get_n_parcels(identities):
    idSet = set(identities)                         # Get unique IDs
    idSet = [item for item in idSet if item >= 0]   # Remove negative values (should have only -1 if any)
    n_parcels = len(idSet)
    return n_parcels


## Create "bins" for X-Axis. 
n_bins = 101
binArray = np.logspace(-2, 0, n_bins-1, endpoint=True)    # Values from 0.01 to 1
binArray = np.concatenate(([0], binArray))  # Add 0 to beginning



## Get subjects list, and first subject's number of parcels.
subjects = next(os.walk(subjectsPath))[1]
if any('_Population' in s for s in subjects):
    subjects.remove('_Population')

subjectFolder = os.path.join(subjectsPath, subjects[0])
sourceIdFile = glob.glob(subjectFolder + sourceIdPattern)[0]

identities = np.load(sourceIdFile)         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.

n_parcels = get_n_parcels(identities)

## Initialize arrays
fidWArray = np.zeros((len(subjects), n_parcels), dtype=float)
fidOArray = np.zeros((len(subjects), n_parcels), dtype=float)
tpWArray = np.zeros((len(subjects), n_bins), dtype=float)
tpOArray = np.zeros((len(subjects), n_bins), dtype=float)
sizeArray = []

### Loop over subjects. Insert values to subject x parcels/bins arrays.
for run_i, subject in enumerate(tqdm(subjects)):
    ## Load files
    subjectFolder = os.path.join(subjectsPath, subject)
    fileSourceIdentities = glob.glob(subjectFolder + sourceIdPattern)[0]
    fileForwardOperator  = glob.glob(subjectFolder + forwardPattern)[0]
    fileInverseOperator  = glob.glob(subjectFolder + inversePattern)[0]
    fileSourceFidelities = glob.glob(subjectFolder + sourceFidPattern)[0]
    
    identities = np.load(fileSourceIdentities)         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.
    forward = np.matrix(np.load(fileForwardOperator))  # sensors x sources
    inverse = np.matrix(np.load(fileInverseOperator))  # sources x sensors
    sourceFids = np.load(fileSourceFidelities)         # sources
    # identities = np.genfromtxt(fileSourceIdentities, dtype='int32', delimiter=delimiter)         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.
    # forward = np.matrix(np.genfromtxt(fileForwardOperator, dtype='float', delimiter=delimiter))        # sensors x sources
    # inverse = np.matrix(np.genfromtxt(fileInverseOperator, dtype='float', delimiter=delimiter))        # sources x sensors
    # sourceFids = np.genfromtxt(fileSourceFidelities, dtype='float', delimiter=delimiter)    # sources
    
    weights = source_fid_to_weights(sourceFids, exponent=exponent, normalize=normalize, 
                                    inverse=inverse, identities=identities, flips=flips)
    inverse_w = np.einsum('ij,i->ij', inverse, weights)
    
    n_parcels = get_n_parcels(identities)
    
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
    _, cp_PLVPW = fidelity_estimation(forward, inverse_w, identities, parcel_series=parcelSeriesPairs)
    _, cp_PLVPO = fidelity_estimation(forward, inverse, identities, parcel_series=parcelSeriesPairs)
    
    # Do the cross-patch PLV estimation for unmodeled series
    cp_PLVU = np.zeros([n_parcels, n_parcels], dtype=np.complex128)
    
    for t in range(n_samples):
        parcelPLVn = parcelSeriesPairs[:,t] / np.abs(parcelSeriesPairs[:,t]) 
        cp_PLVU += np.outer(parcelPLVn, np.conjugate(parcelPLVn)) /n_samples
    
    cp_PLVUim = np.abs(np.imag(cp_PLVU))
    
    # Build truth matrix from pairs.
    truthMatrix = np.zeros((n_parcels, n_parcels), dtype=bool)
    for i, pair in enumerate(pairs):
      truthMatrix[pair[0], pair[1]] = True
      truthMatrix[pair[1], pair[0]] = True

    # Delete diagonal from truth and estimated matrices
    truthMatrix = delete_diagonal(truthMatrix)
    cp_PLVPW = delete_diagonal(cp_PLVPW)
    cp_PLVPO = delete_diagonal(cp_PLVPO)
    
    # Use imaginary PLV for the estimation.
    cp_PLVWim = np.abs(np.imag(cp_PLVPW))
    cp_PLVOim = np.abs(np.imag(cp_PLVPO))
    
    ## True positive and false positive rate estimation.
    tpRateW, fpRateW = get_tp_fp_rates(cp_PLVWim, truthMatrix)
    tpRateO, fpRateO = get_tp_fp_rates(cp_PLVOim, truthMatrix)
    
    # Get nearest TP values closest to the FP pair at the "bin" values.
    nearTPW = get_nearest_tp_semi_bin(binArray, tpRateW, fpRateW)
    nearTPO = get_nearest_tp_semi_bin(binArray, tpRateO, fpRateO)
    
    fidWArray[run_i,:] = fidelityW
    fidOArray[run_i,:] = fidelityO
    tpWArray[run_i,:] = nearTPW
    tpOArray[run_i,:] = nearTPO
    sizeArray.append(len(identities))   # Approximate head size with number of sources.
    
print(f'gain of fidelities. Mean/mean {np.mean(fidWArray)/np.mean(fidOArray)}')
print(f'gain of fidelities. Mean(fidelityW/fidelityO) {np.mean(fidWArray/fidOArray)}')


### Statistics. 

fidWAverage = np.average(fidWArray, axis=0)
fidWStd = np.std(fidWArray, axis=0)
fidOAverage = np.average(fidOArray, axis=0)
fidOStd = np.std(fidOArray, axis=0)
tpWAverage = np.average(tpWArray, axis=0)
tpWStd = np.std(tpWArray, axis=0)
tpOAverage = np.average(tpOArray, axis=0)
tpOStd = np.std(tpOArray, axis=0)

### TEMP testing for sort first, then average and SD.
fidWArraySorted = np.sort(fidWArray, axis=1)
fidWAverageSorted = np.average(fidWArraySorted, axis=0)
fidWStdSorted = np.std(fidWArraySorted, axis=0)
fidOArraySorted = np.sort(fidOArray, axis=1)
fidOAverageSorted = np.average(fidOArraySorted, axis=0)
fidOStdSorted = np.std(fidOArraySorted, axis=0)



"""   Plots   """
# Set global figure parameters, including CorelDraw compatibility (.fonttype)
import matplotlib.pylab as pylab
if tightLayout == True:
  params = {'legend.fontsize':'7',
         'figure.figsize':(1.6, 1),
         'axes.labelsize':'7',
         'axes.titlesize':'7',
         'xtick.labelsize':'7',
         'ytick.labelsize':'7',
         'lines.linewidth':'0.5',
         'pdf.fonttype':42,
         'ps.fonttype':42,
         'font.family':'Arial'}
else:   # Looks nice on the screen parameters
  params = {'legend.fontsize':'7',
         'figure.figsize':(3, 2),
         'axes.labelsize':'7',
         'axes.titlesize':'7',
         'xtick.labelsize':'7',
         'ytick.labelsize':'7',
         'lines.linewidth':'0.5',
         'pdf.fonttype':42,
         'ps.fonttype':42,
         'font.family':'Arial'}
pylab.rcParams.update(params)

parcelList = list(range(0, n_parcels))

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
        label='Weighted fidelity,\nmean: ' + "{:.3f}".format(np.mean(fidWAverage)))
# Original
ax.plot(meansOF, color='black', linestyle=':', linewidth=1, 
        label='Original fidelity,\nmean: ' + "{:.3f}".format(np.mean(fidOAverage)))

legend = ax.legend(loc='lower right', shadow=False)
legend.get_frame()

# ax.fill_between(parcelList, np.ravel(meansWF-stdsWF), np.ravel(meansWF+stdsWF), color='black', alpha=0.5)
# ax.fill_between(parcelList, np.ravel(meansOF-stdsOF), np.ravel(meansOF+stdsOF), color='black', alpha=0.3)
ax.set_ylabel('Parcel fidelity')
ax.set_xlabel('Parcels, sorted by original')

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)

# update the view limits and ticks to sensible % values
locs, labels = plt.yticks()
plt.yticks(np.array([0., 0.2, 0.4, 0.6, 0.8]))
ax.set_ylim(0, 0.9)      # Preset y-limits
plt.tight_layout(pad=0.1)
plt.show()
if savePDFs == True:
  fig.savefig(savePathBase + 'Fidelities sort orig.pdf', format='pdf')


### TEMP earlier sorting plot test.
""" Plot Fidelities, sorted according to individual fidelity. """
# Get means and standard deviations for weighted and original inv ops.
meansWFS = pd.DataFrame(fidWAverageSorted, columns=['value'])
stdsWFS = pd.DataFrame(fidWStdSorted, columns=['value'])
meansOFS = pd.DataFrame(fidOAverageSorted, columns=['value'])
stdsOFS = pd.DataFrame(fidOStdSorted, columns=['value'])

fig, ax = plt.subplots(1,1)

# Weighted
ax.plot(meansWFS, color='black', linestyle='-', 
        label='Weighted fidelity,\nmean: ' + "{:.3f}".format(np.mean(meansWFS['value'])))
# Original
ax.plot(meansOFS, color='black', linestyle=':', linewidth=1, 
        label='Original fidelity,\nmean: ' + "{:.3f}".format(np.mean(meansOFS['value'])))

legend = ax.legend(loc='best', shadow=False)
legend.get_frame()

ax.fill_between(parcelList, np.ravel(meansWFS-stdsWFS), np.ravel(meansWFS+stdsWFS), color='black', alpha=0.5)
ax.fill_between(parcelList, np.ravel(meansOFS-stdsOFS), np.ravel(meansOFS+stdsOFS), color='black', alpha=0.3)
ax.set_ylabel('Fidelity')
ax.set_xlabel('Parcels, mean(sort([fidelity]))')

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=0.1)
plt.show()

if savePDFs == True:
  fig.savefig(savePathBase + 'Fidelities sort separate.pdf', format='pdf')


""" Plot ROC, True positives, false positives. """
meansW = pd.DataFrame(np.array([binArray, tpWAverage]).T,columns=['time','mean'])
stdsW = pd.DataFrame(tpWStd)
meansO = pd.DataFrame(np.array([binArray, tpOAverage]).T,columns=['time','mean'])
stdsO = pd.DataFrame(tpOStd)

fig, ax = plt.subplots(1,1)

# Weighted
ax.plot(meansW.iloc[:,0], meansW.iloc[:,1], color='black', linestyle='-', 
        label='Weighted, TPR at\nFPR 0.15: ' 
        + "{:.3f}".format(tpWAverage[find_nearest_index(binArray, 0.15)]))
# Original
ax.plot(meansO.iloc[:,0], meansO.iloc[:,1], color='black', linestyle=':', linewidth=1, 
        label='Original, TPR at\nFPR 0.15: ' 
        + "{:.3f}".format(tpOAverage[find_nearest_index(binArray, 0.15)]))

legend = ax.legend(loc='right', shadow=False)
legend.get_frame()

ax.fill_between(meansW.iloc[:,0], meansW.iloc[:,1]-stdsW.iloc[:,0], meansW.iloc[:,1]+stdsW.iloc[:,0], color='black', alpha=0.5)
ax.fill_between(meansO.iloc[:,0], meansO.iloc[:,1]-stdsO.iloc[:,0], meansO.iloc[:,1]+stdsO.iloc[:,0], color='black', alpha=0.3)
ax.set_ylabel('True positive rate')
ax.set_xlabel('False positive rate')

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=0.1)
plt.show()

if savePDFs == True:
  fig.savefig(savePathBase + 'True false positive rates ROC.pdf', format='pdf')




""" Make and plot relative fidelity benefits from weighting to fidelity. """
fidRArray = fidWArray / fidOArray
meansR = np.average(fidRArray, axis=0)
stdsR = np.std(fidRArray, axis=0)

# Sort parcels by average, and multiply by 100 to change to percentage.
sortArray = np.argsort(meansR)
meansR = 100* meansR[sortArray] 
stdsR = 100* stdsR[sortArray]

fig, ax = plt.subplots(1,1)

ax.plot(meansR, color='black', linestyle='-', 
        label='Relative fidelity,\nmean: ' + "{:.3f}".format(np.mean(meansR)))
ax.plot(100*np.ones(n_parcels, dtype=float), color='black', linestyle='-', linewidth=0.3)  # Set a horizontal line at 100 %.

legend = ax.legend(loc='best', shadow=False)
legend.get_frame()

# ax.fill_between(parcelList, np.ravel(meansR-stdsR), np.ravel(meansR+stdsR), color='black', alpha=0.5)
ax.set_ylabel('Relative fidelity (%)')
ax.set_xlabel('Parcels, sorted by benefit')


ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=0.1)
plt.ylim(90, 150)
plt.show()

if savePDFs == True:
  fig.savefig(savePathBase + 'Fidelities relative.pdf', format='pdf')



""" Plot relative ROC gain, True positives/True positives by false positives bins. """
# Skip first and last bin because division by zero
meansRR = np.mean(100* (tpWArray[:, 1:-1] / tpOArray[:, 1:-1]), axis=0)
meansRR = pd.DataFrame(np.array([binArray[1:-1], meansRR]).T,
                       columns=['FP bins','TP-relative'])
stdsRR = np.std(100* (tpWArray[:, 1:-1] / tpOArray[:, 1:-1]), axis=0)
stdsRR = pd.DataFrame(np.array([binArray[1:-1], stdsRR]).T, columns=['FP bins','TP-relative'])

fig, ax = plt.subplots(1,1)

ax.plot(meansRR.iloc[:,0], meansRR.iloc[:,1], color='black', linestyle='-', 
        label='Relative TPR at\nFPR 0.15: ' 
        + "{:.3f}".format(meansRR.iloc[:,1][find_nearest_index(binArray[1:-1], 0.15)]))
ax.plot([0, 1], [100, 100], color='black', linestyle='-', linewidth=0.3)  # Set a horizontal line at 100 %. [X X] [Y, Y]

legend = ax.legend(loc='best', shadow=False)
legend.get_frame()

# ax.fill_between(meansRR.iloc[:,0], meansRR.iloc[:,1]-stdsRR.iloc[:,1],
#                 meansRR.iloc[:,1]+stdsRR.iloc[:,1], color='black', alpha=0.5)
ax.set_ylabel('Relative TPR (%)')
ax.set_xlabel('False positive rate')

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=0.1)
plt.ylim(90, 150)
plt.show()

if savePDFs == True:
  fig.savefig(savePathBase + 'True false positive rates ROC relative.pdf', format='pdf')




""" Scatter plot weighted and original fidelities per parcel. """
fig, ax = plt.subplots(1,1)
ax.scatter(fidOAverage, fidWAverage, c='black', alpha=0.5, s=10)  ## X, Y.
ax.plot([0,1], [0,1], color='black')    # Draw diagnonal line.
# ax.set_title('PLV')
ax.set_xlabel('PLV, Original inv op')
ax.set_ylabel('PLV, Weighted inv op')

plt.ylim(0, 1)
plt.xlim(0, 1)

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=0.1)
plt.show()

if savePDFs == True:
  fig.savefig(savePathBase + 'Fidelities Orig x Weighted scatter by parcel.pdf', format='pdf')



""" Scatter plot weighted and original ROC at FPR 0.15. """
fig, ax = plt.subplots(1,1)
ax.scatter(np.ravel(tpOArray), np.ravel(tpWArray), c='red', alpha=0.2, s=4)  ## X, Y.
ax.scatter(tpOAverage, tpWAverage, c='black', alpha=0.5, s=10)  ## X, Y.
ax.plot([0,1], [0,1], color='black')
# ax.set_title('PLV')
ax.set_xlabel('TPR, Original inv op')   
ax.set_ylabel('TPR, Weighted inv op')   

plt.ylim(0, 1)
plt.xlim(0, 1)

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=0.1)
plt.show()

if savePDFs == True:
  fig.savefig(savePathBase + 'True positive rates Orig x Weighted scatter by parcel.pdf', format='pdf')




