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
from fidelityOpMinimal import make_series, source_fid_to_weights

subjectsFolder = 'C:\\temp\\fWeighting\\csvSubjects_p\\'
forwardPattern = '\\forwardOperatorMEEG.csv'
inversePattern = '\\inverseOperatorMEEG.csv'
sourceIdPattern = '\\sourceIdentities_parc2018yeo7_XYZ.csv'
sourceFidPattern = '\\sourceFidelities_MEEG_parc2018yeo7_XYZ.csv'
savePathBase = "C:\\temp\\fWeighting\\plotDump\\schaefer "

resolutions = ['100', '200', '400', '597', '775', '942']
n_samples = 3000

delimiter = ';'

# Weighting options.
exponent = 2
normalize = True
flips = True

# Save and plotting settings
savePDFs = False
# tightLayout = True


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
    return np.angle(cplv)


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
    return histogram


# Get file paths for source IDs and weights at different resolutions.
sourceIdPatterns = []
sourceFidPatterns = []
for i, resolution in enumerate(resolutions):
    sourceIdPatterns.append(sourceIdPattern.replace('XYZ', resolution))
    sourceFidPatterns.append(sourceFidPattern.replace('XYZ', resolution))


""" Search subject folders in main folder. """
subjects = next(os.walk(subjectsFolder))[1]
if any('_Population' in s for s in subjects):
    subjects.remove('_Population')

idArray = [ [] for _ in range(len(subjects))]        # Subjects x Resolutions in the end
weightedInvOps = [ [] for _ in range(len(subjects))]    # Subjects x Resolutions
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
    inverse = np.matrix(np.genfromtxt(fileInverseOperator, dtype='float', delimiter=delimiter))        # sources x sensors
    invOps.append(inverse)
    
    # Loop over resolutions. Load source identities and weights. Get weighted inverse operators.
    for ii, idPattern in enumerate(sourceIdPatterns):
        fileSourceIdentities = glob.glob(subjectFolder + idPattern)[0]
        identities = np.genfromtxt(fileSourceIdentities, dtype='int32', delimiter=delimiter)         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.
        idArray[i].append(identities)
        fileSourceFid = glob.glob(subjectFolder + sourceFidPatterns[ii])[0]
        
        sourceFids = np.genfromtxt(fileSourceFid, dtype=float, delimiter=delimiter)
        weights = source_fid_to_weights(sourceFids, exponent=exponent, normalize=normalize, 
                                    inverse=inverse, identities=identities, flips=flips)
        weightedInvOp = np.einsum('ij,i->ij', inverse, weights)     # Row-wise multiplication of the current inverse operator with weights.
        weightedInvOps[i].append(weightedInvOp)


maxResolution = max([int(res) for res in resolutions])
phaseParcelAr = np.zeros((2, len(subjects),len(resolutions),maxResolution))  # First dimension 0 Orig inv, 1 Weighted.

# Create phase bins.
n_bins = 101
binEdges = np.linspace(-np.pi, np.pi, num=n_bins+1)    # Values from -pi to pi. Edges.
phaseSourceAr = np.zeros((2, len(subjects),len(resolutions),n_bins), dtype=float)  # First dimension 0 Orig inv, 1 Weighted.

## Loop over resolutions. Get mean source phase differences per parcel.
for i1, resolution in enumerate(resolutions):
    n_parcels = int(resolution)
    print('Simulation resolution: ' + str(resolution))
    for isub, subject in enumerate(subjects):
        """ Make simulated parcel series at resolution. 
        Get mean phase difference with original and weighted inv ops. """
        simulatedParcelSeries = make_series(n_parcels, n_samples, n_cut_samples=40, widths=range(5,6))
        simulatedSourceSeries = simulatedParcelSeries[idArray[isub][i1]]
        modeledSourceSeriesO = np.dot(invOps[isub], np.dot(forwards[isub],simulatedSourceSeries))
        modeledSourceSeriesW = np.dot(weightedInvOps[isub][i1], np.dot(
                                                            forwards[isub],simulatedSourceSeries))
        
        indOS = tuple([0,isub,i1,range(n_bins)])           # Source level phase differences indices
        indWS = tuple([1,isub,i1,range(n_bins)])
        indOP = tuple([0,isub,i1,range(int(resolution))])  # Parcel level phase differences indices
        indWP = tuple([1,isub,i1,range(int(resolution))])
        
        phaseSourceAr[indOS] = source_phases(simulatedSourceSeries, modeledSourceSeriesO, 
                                                   idArray[isub][i1], binEdges)  # Original inverse op. Get sources' phase densities.
        
        phaseSourceAr[indWS] = source_phases(simulatedSourceSeries, modeledSourceSeriesW, 
                                                   idArray[isub][i1], binEdges)  # Weighted inverse op

        phaseParcelAr[indOP] = parcel_phase(
            simulatedSourceSeries, modeledSourceSeriesO, idArray[isub][i1])  # Original inverse op. Get parcels' phase differences.
        
        phaseParcelAr[indWP] = parcel_phase(
            simulatedSourceSeries, modeledSourceSeriesW, idArray[isub][i1])  # Weighted inverse op



# """ Save """
# np.save('phaseSourceAr', phaseSourceAr)

""" Analyze parcel series phase arrays. """
## Plot phase bins.
# Mean fidelities of simulated parcellation.
def phasesBinned(zeroBufferedData, resolutions, binEdges):
    histograms = np.zeros((len(resolutions), len(binEdges)-1), dtype=float)
    for i, resolution in enumerate(resolutions):
            nonZero = zeroBufferedData[:,i,:]
            nonZero = nonZero[nonZero != 0]
            histograms[i,:] = np.histogram(nonZero, binEdges)[0]
    return histograms

phaseBinsO = phasesBinned(phaseParcelAr[0], resolutions, binEdges)   # Parcel phase differences original inverse operator.
phaseBinsW = phasesBinned(phaseParcelAr[1], resolutions, binEdges)   # Parcel phase differences weighted inverse operator.


""" Visualization. """
# Set global figure parameters, including CorelDraw compatibility (.fonttype)
import matplotlib.pylab as pylab
params = {'legend.fontsize':'7',
          'figure.figsize':(2.6, 1.8),
         'axes.labelsize':'7',
         'axes.titlesize':'7',
         'xtick.labelsize':'7',
         'ytick.labelsize':'7',
         'lines.linewidth':'0.2',
         'pdf.fonttype':42,
         'ps.fonttype':42,
         'font.family':'Arial'}
pylab.rcParams.update(params)


### Parcel phase differences
def plotPhaseDist(data, binEdges, strings, colors, linestyles, titleAndAxisLabels, plotDensity=True, plotLegend=False):
    # Data 2D, with first dimension resolutions.
    binCe = np.mean([binEdges[0:-1], binEdges[1:]], axis=0)     # Center points from bin edges.
    fig, ax = plt.subplots()
    for i, datum in enumerate(data):
        if plotDensity == True:
            datum = datum/np.sum(datum)
        ax.plot(binCe, datum, label=strings[i], color=colors[i], linestyle=linestyles[i]) 
        
    ax.set_title(titleAndAxisLabels[0])
    ax.set_xlabel(titleAndAxisLabels[1])
    ax.set_ylabel(titleAndAxisLabels[2])
    
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(False)
    
    if plotLegend == True:
      legend = ax.legend(loc='right', shadow=False)
      legend.get_frame()
    
    plt.xticks(np.array([-3., -2.,-1., 0, 1, 2, 3]))
    
    fig.tight_layout()
    plt.show()

## Generate legend strings and line appearance
stringsO = [res + ' Orig' for res in resolutions]
stringsW = [res + ' Flipped' for res in resolutions]
strings = stringsO + stringsW
# colors = ['orchid', 'crimson', 'hotpink', 'plum', 'deeppink', 'magenta']
colors = ['black', 'gray', 'salmon', 'goldenrod', 'olivedrab', 'darkcyan']
colors = colors + colors
linestyles = 6*['-'] + 6*[':']

## Plot probability densities
titleAndAxisLabelsParcel = ['Parcel phase differences', 'Phase difference', 'Probability']
phaseBins = np.append(phaseBinsO, phaseBinsW, axis=0)
plotPhaseDist(phaseBins, binEdges, strings, colors, linestyles, titleAndAxisLabelsParcel)
if savePDFs == True:
  plt.savefig(savePathBase + 'phase differences parcels.pdf', format='pdf')

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
if savePDFs == True:
  plt.savefig(savePathBase + 'phase differences parcels cumulative.pdf', format='pdf')


### Source phase differences
phaseHistMeansO = np.zeros((len(resolutions), n_bins), dtype=float)
phaseHistMeansW = np.zeros((len(resolutions), n_bins), dtype=float)
for i, resolution in enumerate(resolutions):
    phaseHistMeansO[i,:] = np.mean(phaseSourceAr[0,:,i,:], axis=0)
    phaseHistMeansW[i,:] = np.mean(phaseSourceAr[1,:,i,:], axis=0)
phaseHistMeans = np.append(phaseHistMeansO, phaseHistMeansW, axis=0)

titleAndAxisLabelsSource = ['Source phase differences', 'Phase difference', 'Probability']
plotPhaseDist(phaseHistMeans, binEdges, strings, colors, linestyles, titleAndAxisLabelsSource)
if savePDFs == True:
  plt.savefig(savePathBase + 'phase differences sources.pdf', format='pdf')

# Get cumulative values
cumulativeValuesHist = np.append(getCumulativeVals2D(phaseHistMeansO), 
                                 getCumulativeVals2D(phaseHistMeansW), axis=0)

titleAndAxisLabelsSource = ['Source phase differences', 'Phase difference', 'Cumulative probability']
plotPhaseDist(cumulativeValuesHist, binEdges, strings, colors, linestyles, titleAndAxisLabelsSource, plotDensity=False)
if savePDFs == True:
  plt.savefig(savePathBase + 'phase differences sources cumulative.pdf', format='pdf')







