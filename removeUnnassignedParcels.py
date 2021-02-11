# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:53:28 2021
Unassign parcels that are not included for all subjects (0 sources).
Note that the IDs are changed by the reduction! Removed IDs are saved.
@author: rouhinen
"""


import os
import glob
import numpy as np
import time

subjectsFolder = 'K:\\palva\\fidelityWeighting\\csvSubjects_p\\'
sourceIdPattern = '\\*parc2018yeo7_600.csv'
oldPattern = 'parc2018yeo7_600'
newPattern = 'parc2018yeo7_600_reduced'
reducedPattern = 'parc2018yeo7_600_reducedParcels'

delimiter = ';'



## Search folders in main folder. 
subjects = next(os.walk(subjectsFolder))[1]
if any('_Population' in s for s in subjects):
    subjects.remove('_Population')
maxID = 0
sourceIdentities = []
fileSourceIDNew = []
times = [time.perf_counter()]

## Loop over folders containing subjects. Get maximum parcel number. Populate source identities.
for i, subject in enumerate(subjects):
    subjectFolder = os.path.join(subjectsFolder, subject)
    
    # Load original source identities
    fileSourceIdentities = glob.glob(subjectFolder + sourceIdPattern)[0]
    
    identities = np.genfromtxt(fileSourceIdentities, 
                              dtype='int32', delimiter=delimiter)         # Source length vector. Expected ids for parcels are 0 to n-1, where n is number of parcels, and -1 for sources that do not belong to any parcel.
    
    """ Get maximum ID. Continuous IDs are not assumed for each subject, 
    but are at the population level. """
    maxID = max([maxID, max(identities)])
    
    sourceIdentities.append(identities)
    fileSourceIDNew.append(fileSourceIdentities.replace(oldPattern, newPattern))
    
expectedIDs = list(range(maxID+1))
missingIDs = []
## Loop over subjects. Check which parcels have 0 parcels on at least one subject.
for i, identities in enumerate(sourceIdentities):
    """ Get unique parcel IDs, non-negative. """
    idSet = set(identities)                         # Get unique IDs
    idSet = [item for item in idSet if item >= 0]   # Remove negative values (should have only -1 if any)
    
    # If idSet is smaller than idSet, save missing values.
    if len(expectedIDs) > len(idSet):
        missing = list(range(maxID+1))
        for ii, ID in enumerate(idSet):
            missing.remove(ID)
    
        # Concatenate to missingIDs
        missingIDs += missing
    
# Sorted small to large list of parcels with some subjects having 0 sources.
missingIDs = np.sort(list(set(missingIDs)))

## Loop over subjects. Set missing ID to -1.
for i, identities in enumerate(sourceIdentities):
    # Loop over missing IDs.
    for ii, missing in enumerate(missingIDs):
        identities = [ID if ID !=missing else -1 for ID in identities]
        
    ## Loop over missing IDs, large to small.
    for ii, missing in enumerate(missingIDs[::-1]):
        identities = [ID-1 if ID > missing else ID for ID in identities]
        
    np.savetxt(fileSourceIDNew[i], identities, delimiter=delimiter)
    reducedFile = fileSourceIDNew[i].replace(newPattern, reducedPattern)
    np.savetxt(reducedFile, list(missingIDs), delimiter=delimiter)
    
print('Parcel IDs set to -1: ' + str(np.sort(list(missingIDs))))




