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
from fidelityOpMinimal import compute_weighted_operator, collapse_operator

subjectsFolder = 'K:\\palva\\fidelityWeighting\\csvSubjects_p\\'
forwardPattern = '\\forwardOperatorMEEG.csv'
inversePattern = '\\inverseOperatorMEEG.csv'
sourceIdPattern = '\\sourceIdentities_parc2018yeo7_600_consolidated.csv'
oldPattern = 'inverseOperatorMEEG'
newPattern = 'weighted_invOperatorMEEG_yeo7_600_consolidated'   # _collapsed is added to the collapsed version file name end
saveCollapsed = True   # If true, save a collapsed weighted inverse operator (parcels x sensors) with normal weighted inverse operator (sources x sensors).
parcelFlip = False

delimiter = ';'



## Search folders in main folder. 
# os.chdir(subjectsFolder)
subjects = next(os.walk(subjectsFolder))[1]
if any('_Population' in s for s in subjects):
    subjects.remove('_Population')
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
    
    inverse_w, _ = compute_weighted_operator(forward, inverse, identities, parcel_flip=parcelFlip)
    
    if saveCollapsed == True:
        collapsed_inv_w = collapse_operator(inverse_w, identities) 
        fileWeightedInvCol = fileInverseOperator.replace(oldPattern, newPattern + '_collapsed')
        np.savetxt(fileWeightedInvCol, collapsed_inv_w, delimiter=';')
        
    fileWeightedInv = fileInverseOperator.replace(oldPattern, newPattern)
    np.savetxt(fileWeightedInv, inverse_w, delimiter=';')

