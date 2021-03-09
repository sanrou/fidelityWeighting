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
from fidelityOpMinimal import compute_weighted_operator

subjectsFolder = 'K:\\palva\\fidelityWeighting\\csvSubjects_p\\'
forwardPattern = '\\forwardOperatorMEEG.csv'
inversePattern = '\\inverseOperatorMEEG.csv'
sourceIdPattern = '\\*parc68.csv'

delimiter = ';'



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
    times.append(time.perf_counter())
    
    # Load forward and inverse operator matrices
    print(subject + ' being computed. Tic: ' + str(times[-1]-times[-2]) + ' s')
    fileSourceIdentities = glob.glob(subjectFolder + sourceIdPattern)[0]
    fileForwardOperator  = glob.glob(subjectFolder + forwardPattern)[0]
    fileInverseOperator  = glob.glob(subjectFolder + inversePattern)[0]
    
    identities = np.genfromtxt(fileSourceIdentities, 
                              dtype='int32', delimiter=delimiter)         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.
    forward = np.matrix(np.genfromtxt(fileForwardOperator, 
                              dtype='float', delimiter=delimiter))        # sensors x sources
    inverse = np.matrix(np.genfromtxt(fileInverseOperator, 
                              dtype='float', delimiter=delimiter))        # sources x sensors
    
    _, weights = compute_weighted_operator(forward, inverse, identities)
    
    # Remove zeros out of fidelities (non-parcel sources)
    weights = weights[weights != 0]
    
    # fidelity = |weight|^(1/2)
    fidelities.append(np.abs(weights)**(1/2))

# Ravel fidelities.
fidRavel = np.asarray([])
for i, fid in enumerate(fidelities):
    fidRavel = np.concatenate((fidRavel,fid), axis=None)


""" Plot. """
# Histogram settings
n_bins = 21

# Set global figure parameters, including CorelDraw compatibility (.fonttype)
import matplotlib.pylab as pylab
import matplotlib.patches as patches
import matplotlib.path as path
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

### Histogram.
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)

## All sources
# Draw histogram. Get counts for normalization.
counts, bins, bars = ax.hist(fidRavel, bins=n_bins, density=True, facecolor='gray', alpha=0.5)  

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
ax.set_ylim(bottom.min(), top.max())

# Set axis invisible
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Format y-axis to percentage. One could check how to get the histogram's max value. Maybe put that to xmax.
ax.yaxis.set_major_formatter(PercentFormatter(xmax=sum(counts)))

ax.set_ylabel('n sources (%)')
ax.set_xlabel('Fidelity, all sources')
# plt.ylim(0, 3)  # Histogram y values are pretty wacky with density=True.

plt.tight_layout(pad=0.1)
plt.show()


## Per subject
ax = fig.add_subplot(1, 2, 2)
ax.hist(fidelities, bins=11)  
# Set axis invisible
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Format y-axis to percentage. One could check how to get the histogram's max value. Maybe put that to xmax.
# ax.yaxis.set_major_formatter(PercentFormatter(xmax=sum(counts)))

ax.set_ylabel('n sources')
ax.set_xlabel('Fidelity, by subject')

plt.tight_layout(pad=0.1)
plt.show()


