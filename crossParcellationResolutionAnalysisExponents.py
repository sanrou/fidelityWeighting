# -*- coding: utf-8 -*-
"""
Created on 2021.05.04
Cross parcellation resolution analysis with different exponents.
@author: rouhinen
"""

import os
import glob
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import norm
from fidelityOpMinimal import make_series

subjectsFolder = 'C:\\temp\\fWeighting\\csvSubjects_p\\'
forwardPattern = '\\forwardOperatorMEEG.csv'
inversePattern = '\\inverseOperatorMEEG.csv'
sourceIdPattern = '\\sourceIdentities_parc2018yeo7_XYZ.csv'
weightedPattern = '\\weights_MEEG_parc2018yeo7_XYZ.csv'   # Exponent 2 and signing expected.

resolutions = ['100', '200', '400', '597', '775', '942']
n_samples = 2000 
exponents = [0, 1, 2, 4, 8, 16, 32]

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


def exp2weightToNew(weights, exponent, identities, unsign=False):
  """ From weight = Sign x SourceFidelity**2 to (Sign?) x SourceFidelity**Exponent.
      Keeps parcel weight normals as same. """
  sourceFids = np.abs(weights)**(1/2)
  newWeights = sourceFids**exponent
  newWeights = newWeights if unsign==True else np.sign(weights)*newWeights
  
  id_set = set(identities)
  id_set = [item for item in id_set if item >= 0]   #Remove negative values (should have only -1 if any)
  n_parcels = len(id_set)  # Number of unique IDs with ID >= 0
  
  newWeights_normalized = 1*newWeights
  for parcel in range(n_parcels): # Normalize parcel level norms.
    # Index sources belonging to parcel
    ii = [i for i, source in enumerate(identities) if source == parcel]

    # Normalize per parcel.
    newWeights_normalized[ii] /= norm(newWeights[ii]) / norm(weights[ii])

  return newWeights_normalized



sourceIdPatterns = []
weightedPatterns = []
for i, resolution in enumerate(resolutions):
    sourceIdPatterns.append(sourceIdPattern.replace('XYZ', resolution))
    weightedPatterns.append(weightedPattern.replace('XYZ', resolution))


""" Search subject folders in main folder. """
subjects = next(os.walk(subjectsFolder))[1]
if any('_Population' in s for s in subjects):
    subjects.remove('_Population')

idArray = [ [] for _ in range(len(subjects))]        # Subjects x Resolutions in the end
weightsArray = [ [] for _ in range(len(subjects))]   # Subjects x Resolutions
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
        weightsArray[i].append(np.genfromtxt(fileWeighted, dtype=float, delimiter=delimiter))
        

maxResolution = max([int(res) for res in resolutions])
plvArray_meth1 = np.zeros((len(exponents), len(subjects),len(resolutions),len(resolutions),maxResolution))  # For method 1. 
plvArray_meth2s = np.copy(plvArray_meth1)   # For method 2, simulation resolution. 
plvArray_meth2m = np.copy(plvArray_meth1)   # For method 2, modeling resolution. 

## Loop over resolutions. Do cross resolution analysis. Methods 1 and 2. 
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
        
        for iexp, exponent in enumerate(exponents):
          newWeights = exp2weightToNew(weightsArray[isub][i2], exponent, idArray[isub][i2], unsign=True)  
          weightedInvOp = np.einsum('ij,i->ij', invOps[isub], newWeights)
          modeledSourceSeriesW = np.dot(weightedInvOp, np.dot(forwards[isub],simulatedSourceSeries))
              
          indicesS = tuple([iexp,isub,i1,i2,range(int(resolution1))])   # indicesS = simulation resolution
          indicesM = tuple([iexp,isub,i1,i2,range(int(resolution2))])   # indicesM = modeling resolution
          
          plvArray_meth1[indicesM] = np.abs(np.real(parcel_plv(
              simulatedSourceSeries, modeledSourceSeriesW, idArray[isub][i2])))  # Method 1, original inverse op
          
          plvArray_meth2s[indicesS], plvArray_meth2m[indicesM], _,  _ = (
              parcel_multi_plv(simulatedSourceSeries, modeledSourceSeriesW, 
                               idArray[isub][i1], idArray[isub][i2]))       # Method 2, original inverse op


# """ Temp save """
np.save('plvArray_meth1', plvArray_meth1)
np.save('plvArray_meth2s', plvArray_meth2s)
np.save('plvArray_meth2m', plvArray_meth2m)

""" Analyze plvArrays. """
def nonZeroMeans(zeroBufferedData, exponents, resolutions):
  """ zeroBufferedData shape: exponents x subjects x resolutions x resolutions x maxResolution. """
  means = np.zeros((len(exponents), len(resolutions),len(resolutions)))
  for i1, resolution1 in enumerate(resolutions):
      for i2, resolution2 in enumerate(resolutions):
        for iexp, exponent in enumerate(exponents):
          nonZero = zeroBufferedData[iexp,:,i1,i2,:]
          nonZero = nonZero[nonZero != 0]
          means[iexp,i1,i2] = round(np.mean(nonZero), 4)
  return means

means_meth1 = nonZeroMeans(plvArray_meth1, exponents, resolutions)      # Method 1 modeled resolution
means_meth2m = nonZeroMeans(plvArray_meth2m, exponents, resolutions)    # Method 2 modeled resolution
# means_meth2s = nonZeroMeans(plvArray_meth2s, exponents, resolutions)    # Method 2 simulated resolution
# Simulation resolutions not added.


### Visualization
# Set global figure parameters, including CorelDraw compatibility (.fonttype)
import matplotlib.pylab as pylab
params = {'legend.fontsize':'7',
          'figure.figsize':(17, 2.5),
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


def heat_plot_exp(data, tickLabels, titleStrings, vmin=0.1, vmax=0.6, decimals=2):
    # Data 3D, with first dimension sub-plots.
    columns = len(data)
    
    # Set a threshold where text should be black instead of white. 10 % of the mean.
    middle = (vmax+vmin)/2
    textToKT = middle * 0.15
    
    fig, ax = plt.subplots(1, columns)
    for i, datum in enumerate(data):
        ax[i].imshow(datum[::-1,:], cmap='seismic', vmin=vmin, vmax=vmax)  # Visualize Y-axis down to up.
        
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
                value = round(datum[-ii-1, j], decimals)
                tcolor = "w" if np.abs(value-middle) > textToKT else "k"    # Set text color to white if not near middle threshold, else to black.
                ax[i].text(j, ii, value, ha="center", va="center", 
                        color=tcolor, fontsize=7)
        
        ax[i].set_title(titleStrings[i])
        ax[i].set_xlabel('Modeling resolution')
        ax[i].set_ylabel('Simulation resolution')
    
    fig.tight_layout()
    plt.show()

## Method 1
# strOrig = 'original inv'
# strW1 = 'unsigned weighted, \n exponent2'
# strW2 = 'unsigned weighted, \n exponent8'
# heat_plot([means_meth1om, means_meth1w1m, means_meth1w2m], resolutions, 
#           [strOrig + ', \n method1', strW1, strW2])
meth1Strings = ['Mean fidelity, method1,\n exponent ' + str(exponent) for exponent in exponents]
vmin1 = np.min(means_meth1)
vmax1 = np.max(means_meth1)
heat_plot_exp(means_meth1, resolutions, meth1Strings, vmin=vmin1, vmax=vmax1)

## Method 2
meth2Strings = ['Mean fidelity, method2,\n exponent ' + str(exponent) for exponent in exponents]
vmin2 = np.min(means_meth2m)
vmax2 = np.max(means_meth2m)
heat_plot_exp(means_meth2m, resolutions, meth2Strings, vmin=vmin2, vmax=vmax2)

## Percent changes, Relative to exponent 0.
meth1StringsRel = ['Relative fidelity, method1,\n exponent' + str(exponent) + '/exp0' for exponent in exponents]
maxDiffFromOne = np.max(np.abs(means_meth1/means_meth1[0])) -1
vmin1R = 1-maxDiffFromOne
vmax1R = 1+maxDiffFromOne
heat_plot_exp(means_meth1/means_meth1[0], resolutions, meth1StringsRel, vmin=vmin1R, vmax=vmax1R)

meth2mStringsRel = ['Relative fidelity, method2,\n exponent' + str(exponent) + '/exp0' for exponent in exponents]
maxDiffFromOne = np.max(np.abs(means_meth2m/means_meth2m[0])) -1
vmin2R = 1-maxDiffFromOne
vmax2R = 1+maxDiffFromOne
heat_plot_exp(means_meth2m/means_meth2m[0], resolutions, meth2mStringsRel, vmin=vmin2R, vmax=vmax2R)


### Analyze mean fidelities of confusion matrices. 
def heat_plot_resXexp(data, tickLabels, titleStrings, decimals=2):
    # Data 3D, with first dimension sub-plots.
    # tickLabels first list x-axis, second list y-axis.
    # titleStrings the length of sub-plots.
    columns = len(data)
    
    fig, ax = plt.subplots(1, columns)
    for i, datum in enumerate(data):
        ax[i].imshow(datum[::-1,:])  # Visualize Y-axis down to up.
        
        # Show all ticks...
        ax[i].set_xticks(np.arange(len(tickLabels[0])))
        ax[i].set_yticks(np.arange(len(tickLabels[1])))
        # ... and label them with the respective list entries
        ax[i].set_xticklabels(tickLabels[0])
        ax[i].set_yticklabels(tickLabels[1][::-1])    # Reverse y-axis labels.
        
        # # Rotate the tick labels and set their alignment.
        # plt.setp(ax[i].get_xticklabels(), rotation=0, ha="right",
        #          rotation_mode="anchor")
        
        # Loop over datum dimensions and create text annotations.
        for ii in range(len(tickLabels[0])):
            for j in range(len(tickLabels[1])):
                ax[i].text(ii, j, round(datum[-j-1, ii], decimals), ha="center", va="center",
                        color="w", fontsize=7) # str(-j-1) +'j ii: ' + str(ii)
        
        ax[i].set_title(titleStrings[i])
        ax[i].set_xlabel('Resolution')
        ax[i].set_ylabel('Exponent')
    
    fig.tight_layout()
    plt.show()


## Mean of relative all values
# Method 1
def means_withTriangles(data, resolutions, exponents, verbose=True):
  """ data : exponents x resolutions x resolutions. 
      Output : 4 x resolutions, with first dimension means of whole array, upper triangle without 
      diagonal indices, diagonal, and lower triangle without diagonal indices. 
      Upper diagonal: higher modeling resolution than simulation resolution. """
  decimals = 3
  mean_byExp = np.round([np.mean(means) for means in data], decimals)
  
  if verbose==True:
    for i, mean in enumerate(mean_byExp):
      print(f'Mean fidelity with exponent {exponents[i]} whole {mean}')
  
  # Upper, lower, diagonal means.
  iup = np.triu_indices(len(resolutions), 1)  # Upper triangle without diagonal indices.
  idi = np.diag_indices(len(resolutions))  # Diagonal indices.
  ilo = np.tril_indices(len(resolutions), -1)  # Lower triangle without diagonal indices.
  
  mean_byExp_up = np.round([np.mean(means[iup]) for means in data], decimals)
  mean_byExp_di = np.round([np.mean(means[idi]) for means in data], decimals)
  mean_byExp_lo = np.round([np.mean(means[ilo]) for means in data], decimals)
  
  if verbose==True:
    for i, mean_up in enumerate(mean_byExp_up):
      mean_di = mean_byExp_di[i]
      mean_lo = mean_byExp_lo[i]
      print(f'Mean fidelity with exponent {exponents[i]} upper {mean_up}, diagonal {mean_di}, lower {mean_lo}')
  
  return [mean_byExp, mean_byExp_up, mean_byExp_di, mean_byExp_lo]

means_tri_meth1_rel = means_withTriangles(means_meth1/means_meth1[0], resolutions, exponents)
means_tri_meth2_rel = means_withTriangles(means_meth2m/means_meth2m[0], resolutions, exponents)





a = np.arange(16).reshape(4, 4)
iu = np.triu_indices(4)   # Upper diagonal
a[iu]
iu1 = np.triu_indices(4, 1)   # Upper triangle without diagonal
a[iu1]

il = np.tril_indices(4)   # Lower diagonal.
a[il]

idiag = np.diag_indices(4)  # Diagonal.