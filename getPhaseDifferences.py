# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 16:17:00 2021
Analyze phase consistency before and after flips.
@author: rouhinen
"""


import os
import glob
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import kurtosis
from fidelityOpMinimal import make_series

subjectsFolder = 'C:\\temp\\fWeighting\\csvSubjects_p\\'
forwardPattern = '\\forwardOperatorMEEG.csv'
inversePattern = '\\inverseOperatorMEEG.csv'
sourceIdPattern = '\\sourceIdentities_parc2018yeo7_XYZ.csv'
weightedPattern = '\\weights_MEEG_parc2018yeo7_XYZ.csv'

resolutions = ['100', '200', '400', '597', '775', '942']
n_samples = 3000

delimiter = ';'


def parcel_phase(x, y, identities):
    """ Function for computing the complex phase-locking value between parcel time-series.
    x : ndarray source series 1
    y : ndarray source series 2
    identities : ndarray [sources]
        Expected ids for parcels are 0 to n-1, where n is number of parcels, 
        and -1 for sources that do not belong to any parcel. 
    """
    
    """Collapse the source series into parcel series """
    id_set = set(identities)
    id_set = [item for item in id_set if item >= 0]   #Remove negative values (should have only -1 if any)
    n_parcels = len(id_set)
    sourceParcelMatrix = np.zeros((n_parcels,len(identities)), dtype=np.int8)
    for i,identity in enumerate(identities):
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
    excKurto = kurtosis(np.append(np.angle(cplv), [-np.pi, np.pi], axis=0))     # Get excess kurtosis. Add -pi and pi to angles, so that the tails range to the same values. If all values clump at zero, tails will look heavy for kurtosis(), even though there are no tails.
    return np.angle(cplv), excKurto


def source_phases(x, y, identities, phase_bin_edges):
    """ Function for computing the complex phase-locking value between source time-series.
    x : ndarray source series 1
    y : ndarray source series 2
    identities : ndarray [sources]
        Expected ids for parcels are 0 to n-1, where n is number of parcels, 
        and -1 for sources that do not belong to any parcel. 
    phase_bin_edges : ndarray. Bin edges for np.histogram. Range is expected to go from -pi to pi.
    """
    
    """ Get phase distribution. """
    phases = np.zeros(len(identities), dtype=float)
    for i, parcel in enumerate(identities): 
        if parcel>-1:
            phases[i] = np.angle(np.mean(np.ravel(y[i]) * np.ravel(np.conjugate(x[i]))))   # Phases of single sources
        
    histogram = np.histogram(phases, phase_bin_edges, density=True)  # Get probability densities.
    histogram = histogram[0] / histogram[0].sum()   # density=True doesn't work properly.
    excKurto = kurtosis(np.append(phases, [-np.pi, np.pi], axis=0))     # Get excess kurtosis
    return histogram, excKurto


# Get file paths for source IDs and weights at different resolutions.
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
weightedInvOps = [ [] for _ in range(len(subjects))]    # Subjects x Resolutions
# weightedInvOps2 = [ [] for _ in range(len(subjects))]   # Subjects x Resolutions
invOps = []                                             # Subjects
forwards = []                                           # Subjects

""" Load subject data. """
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
    
    # Loop over resolutions. Load source identities and weights. Get weighted inverse operators.
    for ii, idPattern in enumerate(sourceIdPatterns):
        fileSourceIdentities = glob.glob(subjectFolder + idPattern)[0]
        idArray[i].append(np.genfromtxt(fileSourceIdentities, 
                                  dtype='int32', delimiter=delimiter))         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.
        
        fileWeighted = glob.glob(subjectFolder + weightedPatterns[ii])[0]
        weights = np.genfromtxt(fileWeighted, dtype=float, delimiter=delimiter)
        weightedInvOp = np.einsum('ij,i->ij', invOps[-1], weights)     # Row-wise multiplication of the current inverse operator with weights.
        weightedInvOps[i].append(weightedInvOp)
        # weightedInvOp2 = np.einsum('ij,i->ij', invOps[-1], np.abs(weights))     # Row-wise multiplication of the current inverse operator with weights.
        # weightedInvOps2[i].append(weightedInvOp2)


maxResolution = max([int(res) for res in resolutions])
phaseParcelAr = np.zeros((2, len(subjects),len(resolutions),maxResolution))  # First dimension 0 Orig inv, 1 Weighted.
kurtArrayS = np.zeros((2, len(subjects),len(resolutions)))                # First dimension 0 Orig inv, 1 Weighted.
kurtArrayP = np.zeros((2, len(subjects),len(resolutions)))                # First dimension 0 Orig inv, 1 Weighted.

# Create phase bins.
n_bins = 101
binEdges = np.linspace(-np.pi, np.pi, num=n_bins+1)    # Values from -pi to pi. Edges.
phaseSourceAr = np.zeros((2, len(subjects),len(resolutions),n_bins), dtype=float)  # First dimension 0 Orig inv, 1 Weighted.

## Loop over resolutions. Get mean source phase differences per parcel.
for i1, resolution1 in enumerate(resolutions):
    n_parcels = int(resolution1)
    print('Simulation resolution: ' + str(resolution1))
    for isub, subject in enumerate(subjects):
        """ Make simulated parcel series at resolution1. 
        Get mean phase difference with original and weighted inv ops. """
        simulatedParcelSeries = make_series(n_parcels, n_samples, n_cut_samples=40, widths=range(5,6))
        simulatedSourceSeries = simulatedParcelSeries[idArray[isub][i1]]
        modeledSourceSeriesO = np.dot(invOps[isub], np.dot(forwards[isub],simulatedSourceSeries))
        # modeledSourceSeriesO = np.dot(weightedInvOps2[isub][i1], np.dot(
        #                                                     forwards[isub],simulatedSourceSeries))   # TODO: TEMP
        modeledSourceSeriesW = np.dot(weightedInvOps[isub][i1], np.dot(
                                                            forwards[isub],simulatedSourceSeries))
        
        indOS = tuple([0,isub,i1,range(n_bins)])            # Source level phase differences indices
        indWS = tuple([1,isub,i1,range(n_bins)])
        indOP = tuple([0,isub,i1,range(int(resolution1))])  # Parcel level phase differences indices
        indWP = tuple([1,isub,i1,range(int(resolution1))])
        indOK = tuple([0,isub,i1])                          # Kurtosis indices
        indWK = tuple([1,isub,i1])
        
        phaseSourceAr[indOS], kurtArrayS[indOK] = source_phases(simulatedSourceSeries, modeledSourceSeriesO, 
                                                   idArray[isub][i1], binEdges)  # Original inverse op. Get sources' phase densities.
        
        phaseSourceAr[indWS], kurtArrayS[indWK] = source_phases(simulatedSourceSeries, modeledSourceSeriesW, 
                                                   idArray[isub][i1], binEdges)  # Weighted inverse op

        phaseParcelAr[indOP], kurtArrayP[indOK] = parcel_phase(
            simulatedSourceSeries, modeledSourceSeriesO, idArray[isub][i1])  # Original inverse op. Get parcels' phase differences.
        
        phaseParcelAr[indWP], kurtArrayP[indWK] = parcel_phase(
            simulatedSourceSeries, modeledSourceSeriesW, idArray[isub][i1])  # Weighted inverse op



# """ Save """
# np.save('phaseSourceAr', phaseSourceAr)

""" Analyze parcel series phase arrays. """
## Plot phase bins.
# Mean fidelities of simulated parcellation.
def phasesBinned(zeroBufferedData, resolutions, binEdges):
    histograms = np.zeros((len(resolutions), len(binEdges)-1), dtype=float)
    for i, resolution1 in enumerate(resolutions):
            nonZero = zeroBufferedData[:,i,:]
            nonZero = nonZero[nonZero != 0]
            histograms[i,:] = np.histogram(nonZero, binEdges)[0]
    return histograms

phaseBinsO = phasesBinned(phaseParcelAr[0], resolutions, binEdges)    # Parcel phase differences original inverse operator.
phaseBinsW = phasesBinned(phaseParcelAr[1], resolutions, binEdges)   # Parcel phase differences weighted inverse operator.


""" Visualization. """
# Set global figure parameters, including CorelDraw compatibility (.fonttype)
import matplotlib.pylab as pylab
params = {'legend.fontsize':'7',
          'figure.figsize':(6, 4),
         'axes.labelsize':'7',
         'axes.titlesize':'7',
         'xtick.labelsize':'7',
         'ytick.labelsize':'7',
         'lines.linewidth':'0.5',
         'pdf.fonttype':42,
         'ps.fonttype':42}
pylab.rcParams.update(params)


### Parcel phase differences
def plotPhaseDist(data, binEdges, strings, colors, linestyles, titleAndAxisLabels, plotDensity=True):
    # Data 2D, with first dimension resolutions.
    binCe = np.mean([binEdges[0:-1], binEdges[1:]], axis=0)     # Center points from bin edges.
    fig, ax = plt.subplots()
    for i, datum in enumerate(data):
        if plotDensity == True:
            datum = datum/np.sum(datum)
        ax.plot(binCe, datum, label=strings[i], color=colors[i], 
                linestyle=linestyles[i]) 
        
    ax.set_title(titleAndAxisLabels[0])
    ax.set_xlabel(titleAndAxisLabels[1])
    ax.set_ylabel(titleAndAxisLabels[2])
    
    legend = ax.legend(loc='best', shadow=False)
    legend.get_frame()

    fig.tight_layout()
    plt.show()

## Generate legend strings and line appearance
stringsO = [res + ' Orig' for res in resolutions]
stringsW = [res + ' Weighted' for res in resolutions]
strings = stringsO + stringsW
# colors = ['orchid', 'crimson', 'hotpink', 'plum', 'deeppink', 'magenta']
colors = ['black', 'gray', 'salmon', 'goldenrod', 'olivedrab', 'darkcyan']
colors = colors + colors
linestyles = 6*['-'] + 6*['--']

## Plot probability densities
titleAndAxisLabelsParcel = ['Parcel phase differences', 'Phase difference', 'Probability']
phaseBins = np.append(phaseBinsO, phaseBinsW, axis=0)
plotPhaseDist(phaseBins, binEdges, strings, colors, linestyles, titleAndAxisLabelsParcel)

## Cumulative densities
# Get cumulative density values
def getCumulativeVals2D(array2D):
    cumulativeValues = array2D * 0
    for i, res_values in enumerate(array2D):
        cumulative = res_values * 0
        for ii, n in enumerate(res_values):
            cumulative[ii] = np.sum(res_values[0:ii+1])
        cumulative /= np.max(cumulative)
        cumulativeValues[i,:] = cumulative
    return cumulativeValues
    
titleAndAxisLabelsParcel = ['Parcel phase differences', 'Phase difference', 'Cumulative probability']
cumulativeValues = getCumulativeVals2D(phaseBins)
plotPhaseDist(cumulativeValues, binEdges, strings, colors, linestyles, titleAndAxisLabelsParcel, plotDensity=False)


### Source phase differences
phaseHistMeansO = np.zeros((len(resolutions), n_bins), dtype=float)
phaseHistMeansW = np.zeros((len(resolutions), n_bins), dtype=float)
for i, resolution1 in enumerate(resolutions):
    phaseHistMeansO[i,:] = np.mean(phaseSourceAr[0,:,i,:], axis=0)
    phaseHistMeansW[i,:] = np.mean(phaseSourceAr[1,:,i,:], axis=0)
phaseHistMeans = np.append(phaseHistMeansO, phaseHistMeansW, axis=0)

titleAndAxisLabelsSource = ['Source phase differences', 'Phase difference', 'Probability']
plotPhaseDist(phaseHistMeans, binEdges, strings, colors, linestyles, titleAndAxisLabelsSource)
# Get cumulative values
cumulativeValuesHist = np.append(getCumulativeVals2D(phaseHistMeansO), 
                                 getCumulativeVals2D(phaseHistMeansW), axis=0)

titleAndAxisLabelsSource = ['Source phase differences', 'Phase difference', 'Cumulative probability']
plotPhaseDist(cumulativeValuesHist, binEdges, strings, colors, linestyles, titleAndAxisLabelsSource, plotDensity=False)


""" Kurtosis analysis. Higher kurtosis, the less outliers. """
kurtDiffS = np.zeros(len(resolutions), dtype=float)
kurtDiffP = np.zeros(len(resolutions), dtype=float)
for i, resolution1 in enumerate(resolutions):
    kurtDiffS[i] = np.mean(kurtArrayS[0,:,i] - kurtArrayS[1,:,i], axis=0)
    kurtDiffP[i] = np.mean(kurtArrayP[0,:,i] - kurtArrayP[1,:,i], axis=0)







