# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:56:21 2020

@author: rouhinen
"""

import numpy as np
from numpy.linalg import norm
from numpy.random import randn
from scipy import signal, stats
import matplotlib.pyplot as plt
import glob

import sys
sys.path.insert(1, "E:\\bluu\git\\fidelityWeighting\\fidelityWeighting")
from fidelityOpMinimal import _compute_weights


""" Set subject directory and file patterns. """
dataPath = 'K:\\palva\\fidelityWeighting\\csvSubjects_p\\sub (5)'

fileSourceIdentities = glob.glob(dataPath + '\\*parc68.csv')[0]
fileForwardOperator  = glob.glob(dataPath + '\\*forwardOperatorMEEG.csv')[0]
fileInverseOperator  = glob.glob(dataPath + '\\*inverseOperatorMEEG.csv')[0]

""" Settings. """
delimiter = ';'
n_samples = 1000
n_cut_samples = 40
widths = np.arange(5, 6)


## Get subjects list, and first subject's number of parcels.
identities = np.genfromtxt(fileSourceIdentities, 
                                    dtype='int32', delimiter=delimiter)         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.
forward = np.matrix(np.genfromtxt(fileForwardOperator, 
                                    dtype='float', delimiter=delimiter))        # sensors x sources
inverse = np.matrix(np.genfromtxt(fileInverseOperator, 
                                    dtype='float', delimiter=delimiter))        # sources x sensors

idSet = set(identities)                         # Get unique IDs
idSet = [item for item in idSet if item >= 0]   # Remove negative values (should have only -1 if any)
n_parcels = len(idSet)

""" Generate signals for parcels. """
s = randn(n_parcels, n_samples+2*n_cut_samples)

for i in np.arange(0, n_parcels):
    s[i, :] = signal.cwt(s[i, :], signal.ricker, widths)

s = signal.hilbert(s)
parcelSeries = s[:, n_cut_samples:-n_cut_samples]


""" Parcel series to source series. 0 signal for sources not belonging to a parcel. """
sourceSeries = parcelSeries[identities]
sourceSeries[identities < 0] = 0

""" Forward then inverse model source series. """
# Values wrapping? Normalize forward and inverse operators to avoid this, norm to 1.
forward = forward / norm(forward)
inverse = inverse / norm(inverse)

sourceSeries = np.dot(inverse, np.dot(forward, sourceSeries))

""" Compute weighted inverse operator. Time series from complex to real.
    Make source series using the weighted operator. """
inverse_w, weights = _compute_weights(sourceSeries, parcelSeries, identities, inverse)
parcelSeries = np.real(parcelSeries)
sourceSeries = np.real(sourceSeries)
sourceSeries_w = np.dot(inverse_w, np.dot(forward, sourceSeries))

""" Collapse source modeled series. """
# Collapse estimated source series to parcel series
sourceParcelMatrix = np.zeros((n_parcels,len(identities)), dtype=np.int8)
for i,identity in enumerate(identities):
    if identity >= 0:     # Don't place negative values. These should be sources not belonging to any parcel.
        sourceParcelMatrix[identity,i] = 1

parcelSeries_eo = np.dot(sourceParcelMatrix, sourceSeries)
parcelSeries_ew = np.dot(sourceParcelMatrix, sourceSeries_w)



""" Plotted parcels and sources selections. """
## Set time and parcels of interest. These are for sub 5 parc68. Source list will not match another subject or parcellation.
# Parcel and source lists. 0 (sTS_L) and 36 (iFocp_L) are low fidelity; 14 (iP_L) 22 (IO_L) and 58 (sP_L) high fidelity. The fidelity goodness is from average levels.
parcelList = [0, 22, 14, 58, 63]
sourceLists = [[942, 1207, 1251], [129, 342, 546], [386, 473, 852, 511, 897], [415, 432, 954, 1019, 1196], [4482, 4477, 4511, 4672, 4749]]
# ## Set time and parcels of interest. These are for sub_3. Source list will not match another subject.
# sourceLists = [[890, 963, 1125], [235, 280, 341], [2891, 2615, 2499], [2468, 2494, 2510]] #[890, 963, 1125] for 0. [235, 280, 341] for 22. [2891, 2615, 2499] for 36. [2468, 2494, 2510] for 54.
timeStart = 60
timeEnd = 160
n_sensors = 5
sensor_cor_dist = 3  # For sensor selection. Indexes sensor output sorted abs(correlation)[-n_sensors*sensor_cor_dist:-sensor_cor_dist]
selections = [2, 4]

parcels = []
sources = []
for i, selection in enumerate(selections):
    parcels.append(parcelList[selection])
    sources.append(sourceLists[selection])
        

""" Normalize time series to same amplitude at time of interest. """
parcelSeries_n = np.einsum('ij,i->ij', parcelSeries, 
                           1/np.max(abs(parcelSeries)[:,timeStart:timeEnd], axis=1))   # Value error with this one for some reason with a normal divide.
parcelSeries_neo = parcelSeries_eo / np.max(abs(parcelSeries_eo)[:,timeStart:timeEnd], axis=1)
parcelSeries_new = parcelSeries_ew / np.max(abs(parcelSeries_ew)[:,timeStart:timeEnd], axis=1)

sourceSeries_n = sourceSeries / np.max(sourceSeries[:,timeStart:timeEnd], axis=1)
sourceSeries_nw = sourceSeries_w / np.max(sourceSeries_w[:,timeStart:timeEnd], axis=1)


""" Make sensor series. Find sensors with highest correlations to parcels of interest. """
sensorSeries = np.real(np.dot(forward, sourceSeries))
# Normalize sensors so that different types of sensors are in the same range.
for i in range(forward.shape[0]):
    sensorSeries[i,:] = sensorSeries[i,:] / max(abs(np.ravel(sensorSeries[i,timeStart:timeEnd])))

## Find correlations separately for selections. May find same sensors for different parcels.
sensors_high = np.zeros((len(selections), n_sensors), dtype=int)
for i, selection in enumerate(selections):
    correl = np.zeros(forward.shape[0])
    for ii in range(forward.shape[0]):
        correl[ii], _ = stats.spearmanr(np.real(parcelSeries[parcels[i], timeStart:timeEnd]), 
                                      np.ravel(np.real(sensorSeries[ii, timeStart:timeEnd])))
    
    ind = np.argpartition(abs(correl), -n_sensors)[-n_sensors*sensor_cor_dist::sensor_cor_dist]
    sensors_high[i,:] = ind[np.argsort(correl[ind])]
    

""" Set global figure parameters, including CorelDraw compatibility (.fonttype) """
import matplotlib.pylab as pylab
params = {'legend.fontsize':'7',
          'figure.figsize':(11, 3),
         'axes.labelsize':'14',
         'axes.titlesize':'14',
         'xtick.labelsize':'14',
         'ytick.labelsize':'14',
         'lines.linewidth':'0.5',
         'pdf.fonttype':42,
         'ps.fonttype':42,
         'font.sans-serif':'Arial'}
pylab.rcParams.update(params)

colors = [['royalblue', 'aqua', 'dodgerblue', 'cadetblue', 'turquoise'], 
          ['orchid', 'crimson', 'hotpink', 'plum', 'deeppink']]



""" Plot time series of selections. Figure 1. """
for ii, parcel in enumerate(selections):
    ## Source series.
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    
    for i, source in enumerate(sources[ii]):
        ax.plot(np.ravel(np.real(sourceSeries_n[source, timeStart:timeEnd])) -2*i, 
                color=colors[ii][i], linestyle='-', label='Original, ' + str(source))
        ax.plot(np.ravel(np.real(sourceSeries_nw[source, timeStart:timeEnd])) -2*i, 
                color=colors[ii][i], linestyle='--', label='Weighted, ' + str(source))
    
    ax.set_ylabel('Source series')
    ax.set_xlabel('Time, samples, parcel ' + str(parcels[ii]))
    
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    legend = ax.legend(loc='best', shadow=False)
    legend.get_frame()
    
    plt.tight_layout(pad=0.1)
    
    
    ## Parcel series
    ax = fig.add_subplot(1, 3, 2)
    
    # Ground truth
    ax.plot(np.ravel(np.real(parcelSeries_n[parcels[ii], timeStart:timeEnd]))-0, 
            color='black', linestyle='-', label='Ground truth')
    # Original inv op
    ax.plot(np.ravel(np.real(parcelSeries_neo[parcels[ii], timeStart:timeEnd]))-1, 
            color='dimgray', linestyle='-', label='Estimated, Original inv op')
    # Weighted
    ax.plot(np.ravel(np.real(parcelSeries_new[parcels[ii], timeStart:timeEnd]))-2, 
            color='dimgray', linestyle='--', label='Estimated, Weighted inv op')
    
    ax.set_ylabel('Parcel series')
    ax.set_xlabel('Time, samples, parcel ' + str(parcels[ii]))
    
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    legend = ax.legend(loc='best', shadow=False)
    legend.get_frame()
    
    plt.tight_layout(pad=0.1)
    
    ## Sensor series
    ax = fig.add_subplot(1, 3, 3)
    
    for i, sensor in enumerate(sensors_high[ii]):
        ax.plot(np.ravel(sensorSeries[sensor, timeStart:timeEnd])-1*i, 
                color='black', linestyle='-', label='High cor., sensor ' + str(sensor))
        
    ax.set_ylabel('Sensor series')
    ax.set_xlabel('Time, samples, parcel ' + str(parcels[ii]))
    
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    legend = ax.legend(loc='best', shadow=False)
    legend.get_frame()
    
    plt.tight_layout(pad=0.1)
    plt.show()




""" Plot time series of subselections. Figure 2. """
## Compute sums of sources selected for visualization.
# Init summation arrays
sumArray_n = np.zeros((len(selections), n_samples), dtype=float)    # Original inv op
sumArray_w = np.zeros((len(selections), n_samples), dtype=float)    # Weighted inv op
for ii, parcel in enumerate(selections):
    sumArray_n[ii,:] = np.sum(sourceSeries_n[sources[ii]], axis=0)
    sumArray_w[ii,:] = np.sum(sourceSeries_nw[sources[ii]], axis=0)

# Normalize summed arrays for visualization. 
sumArray_n = np.einsum('ij,i->ij', sumArray_n, 
                           1/np.max(abs(sumArray_n)[:,timeStart:timeEnd], axis=1))   # Parcel-wise normalization.
sumArray_w = np.einsum('ij,i->ij', sumArray_w, 
                           1/np.max(abs(sumArray_w)[:,timeStart:timeEnd], axis=1))


for ii, parcel in enumerate(selections):
    ## Simulated source series (so parcel series many times) with original inv op modeled series.
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    
    for i, source in enumerate(sources[ii]):
        # Ground truth
        ax.plot(np.ravel(np.real(parcelSeries_n[parcels[ii], timeStart:timeEnd]))-2*i, 
                color=colors[ii][i], linestyle='-', label='Simulated, parcel' + str(parcels[ii]))
        ax.plot(np.ravel(np.real(sourceSeries_n[source, timeStart:timeEnd]))-2*i, 
                color=colors[ii][i], linestyle='--', label='Orig inv op, source' + str(source))
        
    ax.set_ylabel('Source series')
    ax.set_xlabel('Time, samples, parcel ' + str(parcels[ii]))
    
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    legend = ax.legend(loc='best', shadow=False)
    legend.get_frame()
    
    plt.tight_layout(pad=0.1)
    
    ## Simulated source series with weighted inv op modeled series.
    ax = fig.add_subplot(1, 3, 2)
    
    for i, source in enumerate(sources[ii]):
        ax.plot(np.ravel(np.real(parcelSeries_n[parcels[ii], timeStart:timeEnd]))-2*i, 
                color=colors[ii][i], linestyle='-', label='Simulated, parcel' + str(parcels[ii]))
        ax.plot(np.ravel(np.real(sourceSeries_nw[source, timeStart:timeEnd])) -2*i, 
                color=colors[ii][i], linestyle='--', label='Weighted inv op, source' + str(source))
    
    ax.set_ylabel('Source series')
    ax.set_xlabel('Time, samples, parcel ' + str(parcels[ii]))
    
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    legend = ax.legend(loc='best', shadow=False)
    legend.get_frame()
    
    plt.tight_layout(pad=0.1)
    
    ## Summed selected sources series as "parcel" representatives
    ax = fig.add_subplot(1, 3, 3)
    
    # Ground truth and original inv op
    ax.plot(np.ravel(np.real(parcelSeries_n[parcels[ii], timeStart:timeEnd]))-0, 
            color='black', linestyle='-', label='Simulated, parcel' + str(parcels[ii]))
    ax.plot(sumArray_n[ii,timeStart:timeEnd]-0, 
            color='black', linestyle='--', label='Estimated, Sum Orig' + str(parcels[ii]))
    # Ground truth and weighted inv op
    ax.plot(np.ravel(np.real(parcelSeries_n[parcels[ii], timeStart:timeEnd]))-2, 
            color='black', linestyle='-', label='Simulated, parcel' + str(parcels[ii]))
    ax.plot(sumArray_w[ii,timeStart:timeEnd]-2, 
            color='dimgray', linestyle='--', label='Estimated, Sum Weight' + str(parcels[ii]))
    
    ax.set_ylabel('Source sum series')
    ax.set_xlabel('Time, samples, parcel ' + str(parcels[ii]))
    
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    legend = ax.legend(loc='best', shadow=False)
    legend.get_frame()
    
    plt.tight_layout(pad=0.1)
    
