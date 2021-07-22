# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 11:50:50 2021
Plot single subject's iPLV matrix and truth matrix. 
@author: rouhinen
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.colors as mcolors

from fidelityOpMinimal import fidelity_estimation, make_series_paired, source_fid_to_weights


"""Load source identities, forward and inverse operators. """
subjectPath = 'C:\\temp\\fWeighting\\fwSubjects_p\\sub (1)\\'

sourceIdPattern = '\\sourceIdentities_parc2018yeo7_200.npy'
sourceFidPattern = '\\sourceFidelities_MEEG_parc2018yeo7_200.npy'
savePathBase = "C:\\temp\\fWeighting\\plotDump\\schaefer200 "
forwardPattern  = '\\forwardOperatorMEEG.npy'
inversePattern  = '\\inverseOperatorMEEG.npy'
    
n_samples = 1000
n_cut_samples = 40
widths = np.arange(5, 6)

# Source fidelity to weights settings
exponent = 2
normalize = True
flips = False

# Save and plotting settings
savePDFs = False
tightLayout = True





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

def diagonal_to_zero(symm_matrix):
    symm_matrix[np.eye(symm_matrix.shape[0],dtype=bool)] = 0
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


## Load files. Get number of parcels.
fileSourceIdentities = glob.glob(subjectPath + sourceIdPattern)[0]
fileForwardOperator  = glob.glob(subjectPath + forwardPattern)[0]
fileInverseOperator  = glob.glob(subjectPath + inversePattern)[0]
fileSourceFidelities = glob.glob(subjectPath + sourceFidPattern)[0]

identities = np.load(fileSourceIdentities)         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.
forward = np.matrix(np.load(fileForwardOperator))        # sensors x sources
inverse = np.matrix(np.load(fileInverseOperator))        # sources x sensors
sourceFids = np.load(fileSourceFidelities)    # sources

weights = source_fid_to_weights(sourceFids, exponent=exponent, normalize=normalize, 
                                inverse=inverse, identities=identities, flips=flips)
inverse_w = np.einsum('ij,i->ij', inverse, weights)

n_parcels = get_n_parcels(identities)


""" Do network estimation. Get cross-patch complex PLV values from paired data"""
parcelSeriesPairs, pairs = make_series_paired(n_parcels, n_samples, seed=1)
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

# Diagonal to zero for truth and estimated matrices
truthMatrix = diagonal_to_zero(truthMatrix)
cp_PLVPW = diagonal_to_zero(cp_PLVPW)
cp_PLVPO = diagonal_to_zero(cp_PLVPO)

# Use imaginary PLV for the estimation.
cp_PLVWim = np.abs(np.imag(cp_PLVPW))
cp_PLVOim = np.abs(np.imag(cp_PLVPO))




""" Plot """
# Set global figure parameters, including CorelDraw compatibility (.fonttype)
if tightLayout == True:
  params = {'legend.fontsize':'7',
         'figure.figsize':(2*2, 1.5),
         'axes.labelsize':'7',
         'axes.titlesize':'7',
         'xtick.labelsize':'7',
         'ytick.labelsize':'7',
         'lines.linewidth':'0.5',
         'pdf.fonttype':42,
         'ps.fonttype':42,
         'font.family':'Arial'}
else:   # Looks nice on the screen parameters.
  params = {'legend.fontsize':'14',
         'figure.figsize':(2*4, 3),
         'axes.labelsize':'14',
         'axes.titlesize':'14',
         'xtick.labelsize':'14',
         'ytick.labelsize':'14',
         'lines.linewidth':'0.5',
         'pdf.fonttype':42,
         'ps.fonttype':42,
         'font.family':'Arial'}
pylab.rcParams.update(params)

## Custom color map. Modified from https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72
def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(rgb_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        rgb_list: list of rgb values
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(i) for i in rgb_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

rgb_list = [(255, 255, 255), (255, 0, 0)]   # White to red
cmp = get_continuous_cmap(rgb_list)

## Heat maps

def heat_plot(data, titleStrings, vmin=0, vmax=0.6, axisLabel='Parcel', cmap='Reds'):
    # Data 3D, with first dimension sub-plots.
    columns = len(data)
    
    # Make ticks array
    tickArr = np.int32(np.linspace(0, data[0].shape[0], 5))
    tickLabels = [str(value) for value in tickArr]
    
    fig, ax = plt.subplots(1, columns)
    for i, datum in enumerate(data):
        pos = ax[i].imshow(datum[::-1,:], cmap=cmap, vmin=vmin, vmax=vmax)  # Visualize Y-axis down to up.
        
        # Show all ticks...
        ax[i].set_xticks(np.concatenate((tickArr[:-1], datum.shape[1]), axis=None))   # Handle possibly removed diagonal (one less column).
        ax[i].set_yticks(tickArr)
        # ... and label them with the respective list entries
        ax[i].set_xticklabels(tickLabels)
        ax[i].set_yticklabels(tickLabels[::-1])    # Reverse y-axis labels.
        
        ax[i].set_title(titleStrings[i])
        ax[i].set_xlabel(axisLabel)
        ax[i].set_ylabel(axisLabel)
        
        fig.colorbar(pos, ax=ax[i])
    
    fig.tight_layout()
    plt.show()

vmin = 0
vmax = 0.5
heat_plot([cp_PLVOim[0:50,0:50], cp_PLVWim[0:50,0:50]], ['Orig', 'Weighted'], vmin=vmin, vmax=vmax, 
          axisLabel='Parcel subset', cmap=cmp)
if savePDFs == True:
  plt.savefig(savePathBase + 'iPLV raw matrix example subset.pdf', format='pdf')

heat_plot([truthMatrix[0:50,0:50], truthMatrix[0:50,0:50]], ['Truth', 'Truth'], vmin=vmin, vmax=1, 
          axisLabel='Parcel subset', cmap='Greys')
if savePDFs == True:
  plt.savefig(savePathBase + 'truth matrix example subset.pdf', format='pdf')

# Thresholded. Here 0.1 is too low, 0.2 too high. Good examples.
cp_PLVWim_t1 = 1*cp_PLVWim
cp_PLVWim_t2 = 1*cp_PLVWim
cp_PLVWim_t1[cp_PLVWim<0.1] = 0
cp_PLVWim_t2[cp_PLVWim<0.2] = 0
heat_plot([cp_PLVWim_t1[0:50,0:50], cp_PLVWim_t2[0:50,0:50]], ['Threshold 0.1', 'Threshold 0.2'], 
          vmin=vmin, vmax=vmax, axisLabel='Parcel subset', cmap=cmp)
if savePDFs == True:
  plt.savefig(savePathBase + 'iPLV thresholded matrix example subset.pdf', format='pdf')








