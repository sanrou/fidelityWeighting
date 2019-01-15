# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:31:20 2017

Generate fidelity weighting vector

@author: Santeri Rouhinen
"""

import scipy
from scipy import asmatrix, genfromtxt, signal

import numpy as np
from numpy.linalg import norm
from numpy.random import randn


"""Load source identities, forward model, and inverse operators."""
fpath = '/home/puolival/fidelityWeighting'
fname_source_identities = fpath + '/sourceIdentities.csv'
fname_forward_operator = fpath + '/forwardSolution.csv'
fname_inverse_operator = fpath + '/inverseSolution.csv'

"""Expected ids for parcels are 0 to n-1, where n is number of parcels,
and -1 for sources that do not belong to any parcel."""
sourceIdentities = genfromtxt(fname_source_identities, dtype='int32',
                              delimiter=',')

"""Zero as ID doesn't work if parcel not belonging to any parcel is given
zero value. There could be sources not in any parcel. Sparce parcels that
is. Should initialize those to -1 or Nan."""
# sensors x sources
forwardOperator = scipy.matrix(genfromtxt(fname_forward_operator,
                                          dtype='float', delimiter=','))

# sources x sensors
inverseOperator = scipy.matrix(genfromtxt(fname_inverse_operator,
                                          dtype='float', delimiter=','))


"""Generate oscillatory parcel signals."""

def make_series(n_parcels, n_samples, n_cut_samples, widths):
    """Function for generating oscillating parcel signals.

    Input arguments:
    ================
    n_parcels : int
        Number of source-space parcels or labels.
    n_samples : int
        Length of the generated time-series in number of samples.
    n_cut_samples : int
        Number of temporary extra samples at each end of the signal
        for handling edge artefacts.
    widths : ndarray
        Widths to use for the wavelet transform.

    Output arguments:
    =================
    s : ndarray
        Simulated oscillating parcel time-series.
    """
    s = randn(n_parcels, n_samples+2*n_cut_samples)

    for i in np.arange(0, n_parcels):
        s[i, :] = signal.cwt(s[i, :], signal.ricker, widths)

    s = signal.hilbert(s)
    s = s[:, n_cut_samples:-n_cut_samples]

    return s

def compute_weighted_operator(fwd, inv, source_identities):
    """Function for computing a fidelity-weighted inverse operator.

    Input arguments:
    ================
    fwd : ndarray
        The forward operator.
    inv : ndarray
        The original inverse operator.
    source_identities : ndarray
        Vector mapping sources to parcels or labels.

    Output argument:
    ================
    weighted_inv : ndarray
        The fidelity-weighted inverse operator.
    """
    pass

n_parcels = max(sourceIdentities) + 1  # Maybe one should test if unique non-negative values == max+1. This is expected in the code.

time_output = 30000   # Samples. Peaks at about 20 GB ram with 30 000 samples. Using too few samples will give poor results.
time_cut = 20    # Samples to remove from ends to get rid of border effects
widths = scipy.arange(5, 6)     # Original values 1, 31. Higher number wider span.
time_generate = time_output + 2*time_cut
samplesSubset = 10000 + 2*time_cut

parcelTimeSeries = make_series(n_parcels, time_output, time_cut, widths)

"""Clone parcel time series to source time series."""
sourceTimeSeries = parcelTimeSeries[sourceIdentities]
sourceTimeSeries[sourceIdentities < 0] = 0

checkSourceTimeSeries = scipy.real(sourceTimeSeries[:])

"""Forward then inverse model source series."""
sourceTimeSeries = inverseOperator*(forwardOperator * sourceTimeSeries)


"""Change to amplitude 1, keep angle using Euler's formula."""
sourceTimeSeries = scipy.exp(1j*(asmatrix(scipy.angle(sourceTimeSeries))))
parcelTimeSeries = scipy.exp(1j*(asmatrix(scipy.angle(parcelTimeSeries))))


"""Get cPLV needed for flips and weighting."""
cPLVArray = scipy.zeros(len(sourceIdentities), dtype='complex')   # Initialize as zeros (complex). 

for i, identity in enumerate(sourceIdentities):              # Compute cPLV only of parcel source pairs of sources that belong to that parcel. One source belong to only one parcel.
    if (sourceIdentities[i] >= 0):     # Don't compute negative values. These should be sources not belonging to any parcel.
        cPLVArray[i] = scipy.sum((scipy.asarray(parcelTimeSeries[identity])) * scipy.conjugate(scipy.asarray(sourceTimeSeries[i])))

cPLVArray /= time_output    # Normalize by samples. For debugging. Output doesn't change even if you don't do this.


"""Get weights and flip. This could be the output."""
weights = scipy.sign(scipy.real(cPLVArray)) * scipy.real(cPLVArray) ** 2

"""Create weighted inverse operator and normalize the norm of weighted inv op
to match original inv op's norm."""
weightedInvOp = scipy.einsum('ij,i->ij', inverseOperator, weights)      # Multiply sensor dimension in inverseOperator by weight. This one would be the un-normalized operator.

weightsNormalized = scipy.zeros(len(weights))  # Initialize norm normalized weights. Maybe not necessary.
for parcel in range(n_parcels):       # Normalize parcel level norms. 
    ii = [i for i, source in enumerate(sourceIdentities) if source == parcel]    # Index sources belonging to parcel
    weightsNormalized[ii] = weights[ii] * (norm(inverseOperator[ii]) / norm(weightedInvOp[ii]))   # Normalize per parcel.

weightedInvOp = scipy.einsum('ij,i->ij', inverseOperator, weightsNormalized)   # Parcel level normalized operator.

weightedInvOp *= norm(inverseOperator) / norm(scipy.nan_to_num(weightedInvOp))   # Operator level normalized operator. If there are sources not in any parcel weightedInvOp gets Nan values due to normalizations.
weightedInvOp = scipy.nan_to_num(weightedInvOp)


"""Check if weighting worked.

Do correlations between the original time series and the weighted inverse
and normal inverse models.

Make parcel and sensor time series. Separate series to avoid overfitted
estimation.
"""

checkParcelTimeSeries = make_series(n_parcels, samplesSubset, time_cut, widths)

# Change to amplitude 1, keep angle using Euler's formula.
checkParcelTimeSeries = scipy.exp(1j*(asmatrix(scipy.angle(checkParcelTimeSeries))))



## Clone parcel time series to source time series
checkSourceTimeSeries = checkParcelTimeSeries[sourceIdentities]
checkSourceTimeSeries[sourceIdentities < 0] = 0

sensorTimeSeries = forwardOperator * checkSourceTimeSeries


"""Correlations between inversed sensorTimeSeries and sourceTimeSeries. Use
only a time subset as the memory use is quite large."""

# Binary matrix of sources belonging to parcels
sourceParcelMatrix = scipy.zeros((n_parcels, len(sourceIdentities)), dtype=scipy.int8)
for i,identity in enumerate(sourceIdentities):
    if (identity >= 0):     # Don't place negative values. These should be sources not belonging to any parcel.
        sourceParcelMatrix[identity, i] = 1


parcelPLVW = scipy.zeros(n_parcels, dtype=scipy.float32)  # For the weighted inverse operator
parcelPLVO = scipy.zeros(n_parcels, dtype=scipy.float32)  # For the original inverse operator


estimatedSourceSeriesW = weightedInvOp   * sensorTimeSeries     # Weighted and original estimated source time series
estimatedSourceSeriesO = inverseOperator * sensorTimeSeries

"""Change to amplitude 1, keep angle using Euler's formula."""
estimatedSourceSeriesW = scipy.exp(1j*(asmatrix(scipy.angle(estimatedSourceSeriesW))))
estimatedSourceSeriesO = scipy.exp(1j*(asmatrix(scipy.angle(estimatedSourceSeriesO))))

modeledSeriesO = np.einsum('ij,i->ij',(sourceParcelMatrix * estimatedSourceSeriesO),(1./np.sum(sourceParcelMatrix,1)))   # multiply source matrix with modeled series, then divide for each parcel by the number of its sources
modeledSeriesO = scipy.exp(1j*(scipy.asmatrix(scipy.angle(modeledSeriesO))))                                             # normalize  
modeledSeriesW = np.einsum('ij,i->ij',(sourceParcelMatrix * estimatedSourceSeriesW),(1./np.sum(sourceParcelMatrix,1)))   # multiply source matrix with modeled series, then divide for each parcel by the number of its sources
modeledSeriesW = scipy.exp(1j*(scipy.asmatrix(scipy.angle(modeledSeriesW))))                                             # normalize  

parcelPLVW = np.abs(np.real(np.einsum('ik,ik->i',checkParcelTimeSeries,np.conj(modeledSeriesW)/10000)))
parcelPLVO = np.abs(np.real(np.einsum('ik,ik->i',checkParcelTimeSeries,np.conj(modeledSeriesO)/10000)))
CP_PLV_W   = np.abs(np.real(np.einsum('ik,jk->ij',modeledSeriesW,np.conj(modeledSeriesW))/10000))
CP_PLV_O   = np.abs(np.real(np.einsum('ik,jk->ij',modeledSeriesO,np.conj(modeledSeriesO))/10000))
