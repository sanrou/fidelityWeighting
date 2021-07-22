# -*- coding: utf-8 -*-
"""
Created on 2021.05.17
Create source fidelity files for subjects in a project folder. 
Save collapsed operator at the same time optionally.
@author: rouhinen
"""

import os
import glob
import numpy as np
import time
from tqdm import tqdm
from fidelityOpMinimal import compute_weighted_operator, collapse_operator

subjectsFolder = 'C:\\temp\\fWeighting\\fwSubjects_p\\'
forwardPattern = '\\forwardOperatorMEEG.npy'
inversePattern = '\\inverseOperatorMEEG.npy'
sourceIdPattern = '\\sourceIdentities_parc2018yeo7_XYZ.npy'     # XYZ will be replaced by resolution string(s).
newPattern = 'sourceFidelities_MEEG_parc2018yeo7_XYZ.npy'   # Note that '.npy' is replaced by '_collapsed.npy' if saving collapsed operator.
resolutions = ['100', '200', '400', '597', '775', '942']
saveCollapsed = False   # If true, save a collapsed weighted inverse operator (parcels x sensors) with normal weighted inverse operator (sources x sensors).
sourceFlip = False  # If true, flip sources. Used if saving collapsed inverse operators. Sign will be saved on source fidelities regardless of this.
parcelFlip = False  # If true, flip whole parcels. Not recommended.
exponent = 2    # Weighting exponent. Weight = sign * (real(cPLV)**exponent). Used if saving collapsed inverse operators.


""" Build resolution patterns. Replaces XYZ in sourceIdPattern and newPattern 
with resolutions if resolutions is not empty. """
if len(resolutions) > 0:
    sourceIdPatterns = []
    newPatterns = []
    for i, resolution in enumerate(resolutions):
        sourceIdPatterns.append(sourceIdPattern.replace('XYZ', resolution))
        newPatterns.append(newPattern.replace('XYZ', resolution))
else:
    sourceIdPatterns = [sourceIdPattern]
    newPatterns = [newPattern]
    

""" Write weights/weighted operators. """
## Search folders in main folder. 
subjects = next(os.walk(subjectsFolder))[1]
if any('_Population' in s for s in subjects):
    subjects.remove('_Population')
times = [time.perf_counter()]
## Loop over folders containing subjects.
for i, subject in enumerate(tqdm(subjects)):
    subjectFolder = os.path.join(subjectsFolder, subject)
    times.append(time.perf_counter())
    
    # Load forward and inverse operator matrices
    print(' ' + subject + ' being computed.')
    fileForwardOperator  = glob.glob(subjectFolder + forwardPattern)[0]
    fileInverseOperator  = glob.glob(subjectFolder + inversePattern)[0]
    
    forward = np.matrix(np.load(fileForwardOperator))        # sensors x sources
    inverse = np.matrix(np.load(fileInverseOperator))       # sources x sensors
    
    for ii, idPattern in enumerate(sourceIdPatterns):
        fileSourceIdentities = glob.glob(subjectFolder + idPattern)[0]
        
        identities = np.load(fileSourceIdentities)         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.
        
        np.random.seed(0)
        inverse_w, weights, cplvs = compute_weighted_operator(forward, inverse, identities, 
                                        parcel_flip=parcelFlip, exponent=exponent)
        
        # Save source fidelities
        fileFidelities = os.path.join(subjectFolder, newPatterns[ii])
        np.savetxt(fileFidelities, np.real(cplvs), delimiter=';')    # Save real(cplv) as source fidelity metric. To get full weighted inv op use source_fid_to_weights() to obtain weights, then weights x inverse op.
        
        if saveCollapsed == True:
            collapsed_inv_w = collapse_operator(inverse_w, identities) 
            fileWeightedInvCol = newPatterns[ii].replace('.npy', '_collapsed.npy')
            fileWeightedInvCol = os.path.join(subjectFolder, fileWeightedInvCol)
            np.savetxt(fileWeightedInvCol, collapsed_inv_w, delimiter=';')
            



