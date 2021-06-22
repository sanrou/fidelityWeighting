# -*- coding: utf-8 -*-
"""
Created on 2021.01.26
Get and plot source level fidelity distributions.

@author: rouhinen
"""


import os
import glob
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

subjectsFolder = 'C:\\temp\\fWeighting\\csvSubjects_p\\'
sourceFidPattern = '\\sourceFidelities_MEEG_parc2018yeo7_200.csv'

delimiter = ';'

tightLayout = True
savePDFs = False
savePathBase = "C:\\temp\\fWeighting\\plotDump\\schaefer200 "

""" Search folders in main folder. """
subjects = next(os.walk(subjectsFolder))[1]
if any('_Population' in s for s in subjects):
    subjects.remove('_Population')

""" Get source fidelities. """
fidelities = list()
times = [time.perf_counter()]
## Loop over folders containing subjects.
for i, subject in enumerate(subjects):
    subjectFolder = os.path.join(subjectsFolder, subject)
    
    # Load source fidelities from file
    fileSourceFidelities = glob.glob(subjectFolder + sourceFidPattern)[0]
    
    fidelities.append(np.trim_zeros(np.abs(np.genfromtxt(fileSourceFidelities, 
                                           dtype='float32', delimiter=delimiter))))        # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.
    
# Ravel fidelities.
fidRavel = np.asarray([])
for i, fid in enumerate(fidelities):
    fidRavel = np.concatenate((fidRavel,fid), axis=None)


""" Plot. """
# Histogram settings
n_bins = 20
binEdges = np.linspace(0, 0.9, n_bins+1)


# Set global figure parameters, including CorelDraw compatibility (.fonttype)
import matplotlib.pylab as pylab
import matplotlib.patches as patches
import matplotlib.path as path

### Histogram.
if tightLayout == True:
  params = {'legend.fontsize':'7',
           'figure.figsize':(8, 2),
           'axes.labelsize':'7',
           'axes.titlesize':'7',
           'xtick.labelsize':'7',
           'ytick.labelsize':'7',
           'lines.linewidth':'0.5',
           'pdf.fonttype':42,
           'ps.fonttype':42,
           'font.family':'Arial'}
else:
  params = {'legend.fontsize':'7',
           'figure.figsize':(11, 3),
           'axes.labelsize':'14',
           'axes.titlesize':'14',
           'xtick.labelsize':'14',
           'ytick.labelsize':'14',
           'lines.linewidth':'0.5',
           'pdf.fonttype':42,
           'ps.fonttype':42,
           'font.family':'Arial'}
pylab.rcParams.update(params)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)

## All sources
# Draw histogram. Get counts for normalization.
counts, bins, bars = ax.hist(fidRavel, bins=binEdges, density=True, facecolor='gray', alpha=0.5)  

# Get the corners of the rectangles for the histogram
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + counts

# Histogram needs a (numrects x numsides x 2) numpy array for the path helper
# function to build a compound path (from https://matplotlib.org/3.2.1/gallery/misc/histogram_path.html#sphx-glr-gallery-misc-histogram-path-py)
XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

# Get the Path object
barpath = path.Path.make_compound_path_from_polys(XY)

# make a patch out of it
patch = patches.PathPatch(barpath)
ax.add_patch(patch)

# update the view limits
ax.set_xlim(left[0], right[-1])
ax.set_ylim(0, 4)      # Preset y-limits
# ax.set_ylim(bottom.min(), top.max())    # Dynamic y-limits

# Set axis invisible
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

# Format y-axis to percentage. One could check how to get the histogram's max value. Maybe put that to xmax.
ax.yaxis.set_major_formatter(PercentFormatter(xmax=sum(counts)))

# update the view limits and ticks to sensible % values
locs, labels = plt.yticks()
cmulPer = np.sum(counts)
plt.yticks(np.array([0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])*cmulPer)
# ax.set_ylim(0, 0.2*cmulPer)      # Preset y-limits

ax.set_ylabel('n sources (%)')
ax.set_xlabel('Source fidelity, all sources')
# plt.ylim(0, 3)  # Histogram y values are pretty wacky with density=True.
plt.xticks(np.array([0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8, 0.9]))

plt.tight_layout(pad=0.1)
plt.show()


## Per subject
ax = fig.add_subplot(1, 2, 2)
ax.hist(fidelities, bins=11)  
# Set axis invisible
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

ax.set_ylabel('n sources')
ax.set_xlabel('Source fidelity, by subject')

plt.tight_layout(pad=0.1)
plt.show()

if savePDFs == True:
  fig.savefig(savePathBase + 'Source Fidelity.pdf', format='pdf')



### Example subject source fidelities
# This list is gotten from subjectExamplesFidelityGain.py with percentile 50.
exampleInds = np.array(([3]))
# # This list is gotten from subjectExamplesFidelityGain.py with percentiles 15 50 85.
# exampleInds = np.array(([37,  3, 27]))

## Histogram.
if tightLayout == True:
  params = {'legend.fontsize':'7',
           'figure.figsize':(4, 2),
           'axes.labelsize':'7',
           'axes.titlesize':'7',
           'xtick.labelsize':'7',
           'ytick.labelsize':'7',
           'lines.linewidth':'0.5',
           'pdf.fonttype':42,
           'ps.fonttype':42,
           'font.family':'Arial'}
else:
  params = {'legend.fontsize':'7',
           'figure.figsize':(11, 3),
           'axes.labelsize':'14',
           'axes.titlesize':'14',
           'xtick.labelsize':'14',
           'ytick.labelsize':'14',
           'lines.linewidth':'0.5',
           'pdf.fonttype':42,
           'ps.fonttype':42,
           'font.family':'Arial'}
pylab.rcParams.update(params)

colors = ['red', 'magenta', 'darkcyan']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

## All sources
# Draw histogram. Get counts for normalization.
exampleFids = [fidelities[ind] for ind in exampleInds]
counts, bins, bars = ax.hist(exampleFids, bins=binEdges, density=True, color=colors, alpha=.8, label=colors)  

# Format y-axis to percentage. density=True does not give sum of 1 values. So divide by sum of densities. Works.
ax.yaxis.set_major_formatter(PercentFormatter(xmax=np.sum(counts[0])))

# Set axis invisible
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)

# update the view limits and ticks to sensible % values
locs, labels = plt.yticks()
cmulPer = np.sum(counts[0])
plt.yticks(np.array([0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])*cmulPer)
# ax.set_ylim(0, 0.25*cmulPer)      # Preset y-limits
plt.xticks(np.array([0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8, 0.9]))

ax.set_ylabel('n sources (%)')
ax.set_xlabel('Source fidelity')

plt.tight_layout(pad=0.1)
plt.show()

if savePDFs == True:
  fig.savefig(savePathBase + 'Examples Source Fidelity.pdf', format='pdf')



### Per subject densities
## Make color array by original mean source fidelity
meanSFids = np.zeros((len(fidelities)), dtype=float)
for i, fids in enumerate(fidelities):
  meanSFids[i] = np.mean(fids)

multipliers = meanSFids / np.max(meanSFids)
colorsFids = []
for i, multiplier in enumerate(multipliers):
  colorsFids.append([0.6*multiplier**2]*3)
colorsFids[exampleInds[0]] = [1,0,0]

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)

# Draw histogram. Get counts for normalization.
counts, bins, bars = ax.hist(fidelities, bins=binEdges, density=True, facecolor='gray', alpha=1, histtype='step', color=colorsFids)  

# Get the corners of the rectangles for the histogram
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + counts

# # Histogram needs a (numrects x numsides x 2) numpy array for the path helper
# # function to build a compound path (from https://matplotlib.org/3.2.1/gallery/misc/histogram_path.html#sphx-glr-gallery-misc-histogram-path-py)
# XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

# # Get the Path object
# barpath = path.Path.make_compound_path_from_polys(XY)

# # make a patch out of it
# patch = patches.PathPatch(barpath)
# ax.add_patch(patch)

# update the view limits
ax.set_xlim(left[0], right[-1])
ax.set_ylim(0, 7)      # Preset y-limits
# ax.set_ylim(bottom.min(), top.max())    # Dynamic y-limits

# Set axis invisible
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

# Format y-axis to percentage. One could check how to get the histogram's max value. Maybe put that to xmax.
ax.yaxis.set_major_formatter(PercentFormatter(xmax=np.sum(counts[0])))

# update the view limits and ticks to sensible % values
locs, labels = plt.yticks()
cmulPer = np.sum(counts[0])
plt.yticks(np.array([0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])*cmulPer)
# ax.set_ylim(0, 0.2*cmulPer)      # Preset y-limits

ax.set_ylabel('n sources (%)')
ax.set_xlabel('Source fidelity, all sources')
# plt.ylim(0, 3)  # Histogram y values are pretty wacky with density=True.
plt.xticks(np.array([0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8, 0.9]))

plt.tight_layout(pad=0.1)
plt.show()



