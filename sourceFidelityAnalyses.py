# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 14:34:19 2021
Load source fidelity arrays. Do analyses.
@author: rouhinen
"""

import numpy as np
import os
import glob

## Resolutions and path patterns
resolutions = ['100', '200', '400', '597', '775', '942']

subjectsFolder = 'C:\\temp\\fWeighting\\fwSubjects_p\\'
sourceFidPattern = '\\sourceFidelities_MEEG_parc2018yeo7_XYZ.npy'

""" Search folders in main folder. """
subjects = next(os.walk(subjectsFolder))[1]
if any('_Population' in s for s in subjects):
    subjects.remove('_Population')

""" Get source fidelities. """
fidelities = [None]*len(subjects)
## Loop over folders containing subjects.
for i, subject in enumerate(subjects):
  subjectFolder = os.path.join(subjectsFolder, subject)
  fidelities[i] = []
  ## Loop over resolutions
  for ii, resolution in enumerate(resolutions):
    # Load source fidelities from file
    fileSourceFidelities = glob.glob(subjectFolder + sourceFidPattern.replace('XYZ', resolution))[0]
    sourceFidelities = np.abs(np.load(fileSourceFidelities))
    fidelities[i].append(sourceFidelities[sourceFidelities!=0])        # Source length vector - zeros. Sources not belonging to a parcel expected to have fidelity value 0.


for ii, resolution in enumerate(resolutions):
  print(f'Minimum {resolution}: {np.min(np.concatenate(fidelities[ii]))}, max: {np.max(np.concatenate(fidelities[ii]))}')
    

### Range of median fidelities for subjects
medians = np.zeros((len(fidelities), len(resolutions)))
for i, subFids in enumerate(fidelities):
  for ii, singleResFids in enumerate(subFids):
    medians[i,ii] = np.median(singleResFids)
  
print(f'Min median for resolutions {resolutions}: {np.min(medians, axis=0)}. Max medians: {np.max(medians, axis=0)}')


### Range of mean source fidelities for individuals
means = np.zeros((len(fidelities), len(resolutions)))
for i, subFids in enumerate(fidelities):
  for ii, singleResFids in enumerate(subFids):
    means[i,ii] = np.mean(singleResFids)
  
print(f'Min mean for resolutions {resolutions}: {np.min(means, axis=0)}. Max means: {np.max(means, axis=0)}')
