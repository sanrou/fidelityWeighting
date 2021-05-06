# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:06:53 2020
Create weighted inverse operators for subjects in a project folder. 
Save collapsed operator at the same time.
@author: rouhinen
"""

import os
import glob
import numpy as np
import time
from tqdm import tqdm
from fidelityOpMinimal import compute_weighted_operator, collapse_operator

subjectsFolder = 'C:\\temp\\fWeighting\\csvSubjects_p\\'
forwardPattern = '\\forwardOperatorMEEG.csv'
inversePattern = '\\inverseOperatorMEEG.csv'
sourceIdPattern = '\\sourceIdentities_parc2018yeo7_XYZ.csv'     # XYZ will be replaced by resolution string(s).
newPattern = 'weightsParcelSignFlipped_MEEG_parc2018yeo7_XYZ.csv'   # Note that '.' is replaced by '_collapsed.' if saving collapsed operator.
resolutions = ['100', '200', '400', '597', '775', '942']
saveCollapsed = False   # If true, save a collapsed weighted inverse operator (parcels x sensors) with normal weighted inverse operator (sources x sensors).
weightsInstead = True   # If true, save weights instead of the whole operator.
parcelFlip = False  # If true, flip whole parcels. Not recommended.
exponent = 2    # Weighting exponent. Weight = sign * (real(cPLV)**exponent)

delimiter = ';'


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
    
    forward = np.matrix(np.genfromtxt(fileForwardOperator, 
                              dtype='float', delimiter=delimiter))        # sensors x sources
    inverse = np.matrix(np.genfromtxt(fileInverseOperator, 
                               dtype='float', delimiter=delimiter))       # sources x sensors
    
    for ii, idPattern in enumerate(sourceIdPatterns):
        fileSourceIdentities = glob.glob(subjectFolder + idPattern)[0]
        
        identities = np.genfromtxt(fileSourceIdentities, 
                                  dtype='int32', delimiter=delimiter)         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.
        
        np.random.seed(0)
        inverse_w, weights = compute_weighted_operator(forward, inverse, identities, 
                                        parcel_flip=parcelFlip, exponent=exponent)
        
        if saveCollapsed == True:
            collapsed_inv_w = collapse_operator(inverse_w, identities) 
            fileWeightedInvCol = newPatterns[ii].replace('.', '_collapsed.')
            fileWeightedInvCol = os.path.join(subjectFolder, fileWeightedInvCol)
            np.savetxt(fileWeightedInvCol, collapsed_inv_w, delimiter=';')
            
        fileWeighted = os.path.join(subjectFolder, newPatterns[ii])
        if weightsInstead == False:
            np.savetxt(fileWeighted, inverse_w, delimiter=';')  # Save whole weighted inverse operator
        else:
            np.savetxt(fileWeighted, weights, delimiter=';')    # Save only weights. To get weighted inv op: weights x inverse_orig.
            



