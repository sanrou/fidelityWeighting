# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 11:01:50 2021
Load fidelity arrays. Do random analyses. 
@author: rouhinen
"""

import numpy as np

# Settings and paths.
resolutions = ['100', '200', '400', '597', '775', '942']
fidOPath = "C:\\temp\\fWeighting\\numpyArrays\\fidArrays\\schaeferXYZ\\fidOArray.npy"
fidWPath = "C:\\temp\\fWeighting\\numpyArrays\\fidArrays\\schaeferXYZ\\fidWArray.npy"


""" Load fidelity arrays from files. """
fidArrayO = []
fidArrayW = []

for i, resolution in enumerate(resolutions):
  fidArrayO.append(np.load(fidOPath.replace('XYZ', resolution)))
  fidArrayW.append(np.load(fidWPath.replace('XYZ', resolution)))
  

""" Analyses """
### Median gain of parcellation fidelities from weighting
meanFidsO = np.zeros((len(resolutions), fidArrayO[0].shape[0]))
meanFidsW = meanFidsO.copy()
for i, resolution in enumerate(resolutions):
  meanFidsO[i,:] = np.mean(fidArrayO[i], axis=1)  # Mean across parcels
  meanFidsW[i,:] = np.mean(fidArrayW[i], axis=1)
  print(f'Median gain resolution {resolution}: {np.median(meanFidsW[i]/meanFidsO[i])}')





