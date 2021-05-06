# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:07:24 2021
Cross parcellation resolution analysis.
@author: rouhinen
"""

import os
import glob
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from fidelityOpMinimal import make_series

subjectsFolder = 'C:\\temp\\fWeighting\\csvSubjects_p\\'
forwardPattern = '\\forwardOperatorMEEG.csv'
inversePattern = '\\inverseOperatorMEEG.csv'
sourceIdPattern = '\\sourceIdentities_parc2018yeo7_XYZ.csv'
weightedPattern = '\\weights_MEEG_parc2018yeo7_XYZ.csv'
weightedPattern2 = '\\onlyFlips_MEEG_parc2018yeo7_XYZ.csv'

resolutions = ['100', '200', '400', '597', '775', '942']
n_samples = 4000 

delimiter = ';'


def parcel_plv(x, y, source_identities):
    """ Function for computing the complex phase-locking value.
    x : ndarray source series 1
    y : ndarray source series 2
    source_identities : ndarray [sources]
        Expected ids for parcels are 0 to n-1, where n is number of parcels, 
        and -1 for sources that do not belong to any parcel. 
    """
    
    """Collapse the source series into parcel series """
    id_set = set(source_identities)
    id_set = [item for item in id_set if item >= 0]   #Remove negative values (should have only -1 if any)
    n_parcels = len(id_set)
    sourceParcelMatrix = np.zeros((n_parcels,len(source_identities)), dtype=np.int8)
    for i,identity in enumerate(source_identities):
        if identity >= 0:     # Don't place negative values. These should be sources not belonging to any parcel.
            sourceParcelMatrix[identity,i] = 1
    
    parcelSeriesX = np.dot(sourceParcelMatrix, x)
    parcelSeriesY = np.dot(sourceParcelMatrix, y)
    
    """Parcel series to amplitude 1, keep angle using Euler's formula."""
    parcelSeriesX = np.exp(1j*(np.asmatrix(np.angle(parcelSeriesX))))
    parcelSeriesY = np.exp(1j*(np.asmatrix(np.angle(parcelSeriesY))))
    
    cplv = np.zeros((n_parcels), dtype='complex')
    
    for i in range(n_parcels):
        cplv[i] = np.sum((np.asarray(parcelSeriesY[i])) * 
                         np.conjugate(np.asarray(parcelSeriesX[i])))
        
    cplv /= np.shape(x)[1]  # Normalize by n samples.
    return cplv


def parcel_multi_plv(x, y, identitiesX, identitiesY):
    """ Function for computing the complex phase-locking value between different 
    parcellation schemas.
    x : ndarray source series 1, e.g. simulated
    y : ndarray source series 2, e.g. modeled
    identities : ndarray [sources]
        Expected ids for parcels are 0 to n-1, where n is number of parcels, 
        and -1 for sources that do not belong to any parcel. 
        identitiesX for x, 2 for y source series.
    """
    
    """ Get source pairs. Build list of lists with pairs' source indices. """
    sourcePairs = np.zeros((len(identitiesX),2), dtype=int)
    sourcePairs[:,0] = identitiesX
    sourcePairs[:,1] = identitiesY
    
    uniquePairs = np.unique(sourcePairs, axis=0)
    uniquePairs = np.c_[uniquePairs, np.zeros((uniquePairs.shape[0]))]  # Columns: resolution1 ID, resolution2 ID, number sources added.
    
    pairSI = [ [] for _ in range(len(uniquePairs)) ]    # Initialize list of lists
    for i, row in enumerate(sourcePairs):
        pIndex = np.where(np.all(uniquePairs[:,0:2] == row[0:2], axis=1))[0][0]     # Get pair's row index.
        pairSI[pIndex].append(i)
        
    # Drop unassigned from pairs 
    toDelete = []
    for i, row in enumerate(uniquePairs):
        if row[0] <0 or row[1] <0:
            toDelete.append(i)
    uniquePairs = np.delete(uniquePairs, toDelete, axis=0)
    pairSI = np.delete(pairSI, toDelete, axis=0)
    
    
    """ Parcel union sum PLVs to array. Increment n sources. """
    cplvs = np.zeros((uniquePairs.shape[0]), dtype=complex)
    for i, sources in enumerate(pairSI):   # For each pair. 
        if (uniquePairs[i,0] >= 0) & (uniquePairs[i,1] >= 0):   # Only compute if source has an identity in both parcellations.
            seriesX = np.ravel(np.sum(x[sources], axis=0))
            seriesY = np.ravel(np.sum(y[sources], axis=0))
            seriesX = np.exp(1j*(np.angle(seriesX)))    # Normalize to amplitude 1.
            seriesY = np.exp(1j*(np.angle(seriesY)))
            
            cplv = np.sum(seriesX * np.conjugate(seriesY)) / np.shape(x)[1]   # Divide by samples.
            uniquePairs[i,2] += len(sources)
            cplvs[i] += cplv
    
    """ Average parcel fidelities for identitiesX and 2. """
    xFidelities = np.zeros(max(identitiesX)+1)
    yFidelities = np.zeros(max(identitiesY)+1)
    
    for i in range(max(identitiesX)+1):
        rowsI = np.where(uniquePairs[:,0] == i)     # Get all pair rows where parcel is included.
        sourceNWeighted = (cplvs[rowsI] * uniquePairs[rowsI, 2] 
                           / np.sum(uniquePairs[rowsI, 2]))  # Weight by n sources. 
        xFidelities[i] = np.abs(np.sum(np.real(sourceNWeighted)))
    
    for i in range(max(identitiesY)+1):
        rowsI = np.where(uniquePairs[:,1] == i)     # Get all pair rows where parcel is included.
        sourceNWeighted = (cplvs[rowsI] * uniquePairs[rowsI, 2] 
                           / np.sum(uniquePairs[rowsI, 2]))  # Weight by n sources. 
        yFidelities[i] = np.abs(np.sum(np.real(sourceNWeighted)))
    
    return xFidelities, yFidelities, cplvs, uniquePairs




sourceIdPatterns = []
weightedPatterns = []
weightedPatterns2 = []
for i, resolution in enumerate(resolutions):
    sourceIdPatterns.append(sourceIdPattern.replace('XYZ', resolution))
    weightedPatterns.append(weightedPattern.replace('XYZ', resolution))
    weightedPatterns2.append(weightedPattern2.replace('XYZ', resolution))


""" Search subject folders in main folder. """
subjects = next(os.walk(subjectsFolder))[1]
if any('_Population' in s for s in subjects):
    subjects.remove('_Population')

idArray = [ [] for _ in range(len(subjects))]        # Subjects x Resolutions in the end
weightedInvOps = [ [] for _ in range(len(subjects))]    # Subjects x Resolutions
weightedInvOps2 = [ [] for _ in range(len(subjects))]   # Subjects x Resolutions
invOps = []                                             # Subjects
forwards = []                                           # Subjects

""" Load subject data. Perform cross resolution fidelity analysis. """
times = [time.perf_counter()]
## Loop over folders containing subjects. Load operators and source identities.
for i, subject in enumerate(tqdm(subjects)):
    subjectFolder = os.path.join(subjectsFolder, subject)
    times.append(time.perf_counter())
    
    # Load forward and inverse operator matrices
    print(' ' + subject + ' operators being loaded')
    fileForwardOperator  = glob.glob(subjectFolder + forwardPattern)[0]
    fileInverseOperator  = glob.glob(subjectFolder + inversePattern)[0]
    
    forwards.append(np.matrix(np.genfromtxt(fileForwardOperator, 
                              dtype='float', delimiter=delimiter)))        # sensors x sources
    invOps.append(np.matrix(np.genfromtxt(fileInverseOperator, 
                              dtype='float', delimiter=delimiter)))        # sources x sensors
    
    # Loop over resolutions. Load source identities and weights.
    for ii, idPattern in enumerate(sourceIdPatterns):
        fileSourceIdentities = glob.glob(subjectFolder + idPattern)[0]
        identities = np.genfromtxt(fileSourceIdentities, dtype='int32', delimiter=delimiter)         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.
        idArray[i].append(identities)
        
        fileWeighted = glob.glob(subjectFolder + weightedPatterns[ii])[0]
        weights = np.genfromtxt(fileWeighted, dtype=float, delimiter=delimiter)
        weightedInvOp = np.einsum('ij,i->ij', invOps[-1], weights)     # Row-wise multiplication of the current inverse operator with weights.
        weightedInvOps[i].append(weightedInvOp)
        
        fileWeighted2 = glob.glob(subjectFolder + weightedPatterns2[ii])[0]       # Load weights from file
        weights2 = np.genfromtxt(fileWeighted2, dtype=float, delimiter=delimiter)
        weightedInvOps2[i].append(np.einsum('ij,i->ij', invOps[-1], weights2))     # Row-wise multiplication of the original inverse operator with flips -> flipped operator.


maxResolution = max([int(res) for res in resolutions])
plvArray_meth1 = np.zeros((3, len(subjects),len(resolutions),len(resolutions),maxResolution))  # For method 1. First dimension 0 Orig inv, 1 Weighting1, 2 weighting2

plvArray_meth2s = np.zeros((3, len(subjects),len(resolutions),len(resolutions),maxResolution)) # For method 2, simulation resolution. First dimension 0 Orig inv, 1 Weighting1, 2 weighting2
plvArray_meth2m = np.zeros((3, len(subjects),len(resolutions),len(resolutions),maxResolution)) # For method 2, modeling resolution. First dimension 0 Orig inv, 1 Weighting1, 2 weighting2

## Loop over resolutions. Do cross resolution analysis. Method 1 and 2. 
for i1, resolution1 in enumerate(resolutions):
    n_parcels = int(resolution1)
    print('Simulation resolution: ' + str(resolution1))
    for i2, resolution2 in enumerate(resolutions):
        print('\t Modeling resolution: ' + str(resolution2))
        for isub, subject in enumerate(subjects):
            """ Make simulated parcel series at resolution1. Measure plv at
            resolution2. Get fidelity with original and weighted inv ops. """
            simulatedParcelSeries = make_series(n_parcels, n_samples, 
                                            n_cut_samples=40, widths=range(5,6))
            simulatedSourceSeries = simulatedParcelSeries[idArray[isub][i1]]
            modeledSourceSeriesO = np.dot(invOps[isub], np.dot(
                                        forwards[isub],simulatedSourceSeries))
            modeledSourceSeriesW = np.dot(weightedInvOps[isub][i2], np.dot(
                                        forwards[isub],simulatedSourceSeries))
            modeledSourceSeriesW2 = np.dot(weightedInvOps2[isub][i2], np.dot(
                                        forwards[isub],simulatedSourceSeries))
            
            # indicesS = tuple([isub,i1,i2,range(int(resolution1))])
            # indicesM = tuple([isub,i1,i2,range(int(resolution2))])
            indicesSo = tuple([0,isub,i1,i2,range(int(resolution1))])   # indicesS = simulation resolution, indicesM = modeling resolution
            indicesMo = tuple([0,isub,i1,i2,range(int(resolution2))])   # indicesS = simulation resolution, indicesM = modeling resolution
            indicesSw1 = tuple([1,isub,i1,i2,range(int(resolution1))])
            indicesMw1 = tuple([1,isub,i1,i2,range(int(resolution2))])
            indicesSw2 = tuple([2,isub,i1,i2,range(int(resolution1))])
            indicesMw2 = tuple([2,isub,i1,i2,range(int(resolution2))])
            
            plvArray_meth1[indicesMo] = np.abs(np.real(parcel_plv(
                simulatedSourceSeries, modeledSourceSeriesO, idArray[isub][i2])))  # Method 1, original inverse op
            
            plvArray_meth1[indicesMw1] = np.abs(np.real(parcel_plv(
                simulatedSourceSeries, modeledSourceSeriesW, idArray[isub][i2])))  # Weighting 1

            plvArray_meth1[indicesMw2] = np.abs(np.real(parcel_plv(
                simulatedSourceSeries, modeledSourceSeriesW2, idArray[isub][i2])))  # Weighting 2

            plvArray_meth2s[indicesSo], plvArray_meth2m[indicesMo], _,  _ = (
                parcel_multi_plv(simulatedSourceSeries, modeledSourceSeriesO, 
                                 idArray[isub][i1], idArray[isub][i2]))       # Method 2, original inverse op
            
            plvArray_meth2s[indicesSw1], plvArray_meth2m[indicesMw1], _,  _ = (
                parcel_multi_plv(simulatedSourceSeries, modeledSourceSeriesW, 
                                 idArray[isub][i1], idArray[isub][i2]))       # Weighting 1
            
            plvArray_meth2s[indicesSw2], plvArray_meth2m[indicesMw2], _,  _ = (
                parcel_multi_plv(simulatedSourceSeries, modeledSourceSeriesW2, 
                                 idArray[isub][i1], idArray[isub][i2]))       # Weighting 2


# """ Temp save """
np.save('plvArray_meth1', plvArray_meth1)
np.save('plvArray_meth2s', plvArray_meth2s)
np.save('plvArray_meth2m', plvArray_meth2m)

""" Analyze plvArrays. """
def nonZeroMeans(zeroBufferedData, resolutions):
    means = np.zeros((len(resolutions),len(resolutions)))
    for i, resolution1 in enumerate(resolutions):
        for ii, resolution2 in enumerate(resolutions):
            nonZero = zeroBufferedData[:,i,ii,:]
            nonZero = nonZero[nonZero != 0]
            means[i,ii] = round(np.mean(nonZero), 4)
    return means

means_meth1om = nonZeroMeans(plvArray_meth1[0], resolutions)    # Method 1 original modeled resolution fidelities
means_meth1w1m = nonZeroMeans(plvArray_meth1[1], resolutions)   # Method 1 weighting type 1 modeled resolution fidelities
means_meth1w2m = nonZeroMeans(plvArray_meth1[2], resolutions)   # Method 1 weighting type 1 modeled resolution fidelities
means_meth2om = nonZeroMeans(plvArray_meth2m[0], resolutions)   # Method 2 original modeled resolution fidelities
means_meth2w1m = nonZeroMeans(plvArray_meth2m[1], resolutions)  # Method 2 weighting type 1 modeled resolution fidelities
means_meth2w2m = nonZeroMeans(plvArray_meth2m[2], resolutions)  # Method 2 weighting type 2 modeled resolution fidelities
# Simulation resolutions not added.


### Visualization
# Set global figure parameters, including CorelDraw compatibility (.fonttype)
import matplotlib.pylab as pylab
params = {'legend.fontsize':'7',
          'figure.figsize':(7.5, 2.5),
         'axes.labelsize':'7',
         'axes.titlesize':'7',
         'xtick.labelsize':'7',
         'ytick.labelsize':'7',
         'lines.linewidth':'0.5',
         'pdf.fonttype':42,
         'ps.fonttype':42}
pylab.rcParams.update(params)

def heat_plot(data, tickLabels, appendStrings, decimals=2):
    # Data 3D, with first dimension sub-plots.
    columns = len(data)
    
    fig, ax = plt.subplots(1, columns)
    for i, datum in enumerate(data):
        ax[i].imshow(datum[::-1,:])  # Visualize Y-axis down to up.
        
        # Show all ticks...
        ax[i].set_xticks(np.arange(len(tickLabels)))
        ax[i].set_yticks(np.arange(len(tickLabels)))
        # ... and label them with the respective list entries
        ax[i].set_xticklabels(tickLabels)
        ax[i].set_yticklabels(tickLabels[::-1])    # Reverse y-axis labels.
        
        # # Rotate the tick labels and set their alignment.
        # plt.setp(ax[i].get_xticklabels(), rotation=0, ha="right",
        #          rotation_mode="anchor")
        
        # Loop over datum dimensions and create text annotations.
        for ii in range(len(tickLabels)):
            for j in range(len(tickLabels)):
                ax[i].text(j, ii, round(datum[-ii-1, j], decimals), ha="center", va="center", 
                        color="w", fontsize=7)
        
        ax[i].set_title("Mean parcel fidelity, " + appendStrings[i])
        ax[i].set_xlabel('Modeling resolution')
        ax[i].set_ylabel('Simulation resolution')
    
    fig.tight_layout()
    plt.show()

## Method 1, original and weighted. Modeled resolution. Weighting methods 1 and 2.
strOrig = 'original inv'
strW1 = 'unsigned weighted, \n exponent2'
strW2 = 'unsigned weighted, \n exponent8'
heat_plot([means_meth1om, means_meth1w1m, means_meth1w2m], resolutions, 
          [strOrig + ', \n method1', strW1, strW2])

## Method 2, original and weighted. Modeled resolution. Weighting methods 1 and 2.
heat_plot([means_meth2om, means_meth2w1m, means_meth2w2m], resolutions, 
          [strOrig + ', \n method2, modeled resolution', strW1, strW2])

## Percent changes, Weighted - Original; Flips*Orig - Original; Weighted - Flips*Orig
heat_plot([means_meth1w1m/means_meth1om*100, means_meth1w2m/means_meth1om*100, means_meth1w1m/means_meth1w2m*100], 
          resolutions, ['\n' + strW1 + '/' + strOrig + ', method1', '\n' + strW2 + '/' + strOrig + ', method1', 
                        '\n' + strW1 + '/' + strW2 + ', method1'], decimals=1)

heat_plot([means_meth2w1m/means_meth2om*100, means_meth2w2m/means_meth2om*100, means_meth2w1m/means_meth2w2m*100], 
          resolutions, ['\n' + strW1 + '/' + strOrig + ', method2', '\n' + strW2 + '/' + strOrig + ', method2', 
                        '\n' + strW1 + '/' + strW2 + ', method2'], decimals=1)



o = np.mean(means_meth1om)
w1 = np.mean(means_meth1w1m)
w2 = np.mean(means_meth1w2m)


