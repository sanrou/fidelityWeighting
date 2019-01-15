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

def plv(x, y, identities):
    """Change to amplitude 1, keep angle using Euler's formula."""
    x = scipy.exp(1j*(asmatrix(scipy.angle(x))))
    y = scipy.exp(1j*(asmatrix(scipy.angle(y))))

    """Get cPLV needed for flips and weighting."""
    cplv = scipy.zeros(len(identities), dtype='complex')

    for i, identity in enumerate(identities):
        """Compute cPLV only of parcel source pairs of sources that
        belong to that parcel. One source belong to only one parcel."""
        if (identities[i] >= 0):
            cplv[i] = (scipy.sum((scipy.asarray(y[identity])) *
                       scipy.conjugate(scipy.asarray(x[i]))))

    cplv /= np.shape(x)[1] # time_output # Normalize by samples.
    return cplv

def _compute_weights(source_series, parcel_series, identities, inverse):
    cPLVArray = plv(source_series, parcel_series, identities)

    """Get weights and flip. This could be the output."""
    weights = scipy.sign(scipy.real(cPLVArray)) * scipy.real(cPLVArray) ** 2

    """Create weighted inverse operator and normalize the norm of weighted inv op
    to match original inv op's norm."""
    weightedInvOp = scipy.einsum('ij,i->ij', inverse, weights)      # Multiply sensor dimension in inverseOperator by weight. This one would be the un-normalized operator.

    n_parcels = max(identities) + 1
    weightsNormalized = scipy.zeros(len(weights))  # Initialize norm normalized weights. Maybe not necessary.
    for parcel in range(n_parcels):       # Normalize parcel level norms. 
        ii = [i for i, source in enumerate(identities) if source == parcel]    # Index sources belonging to parcel
        weightsNormalized[ii] = weights[ii] * (norm(inverse[ii]) / norm(weightedInvOp[ii]))   # Normalize per parcel.

    weightedInvOp = scipy.einsum('ij,i->ij', inverse, weightsNormalized)   # Parcel level normalized operator.
    weightedInvOp *= norm(inverse) / norm(scipy.nan_to_num(weightedInvOp))   # Operator level normalized operator. If there are sources not in any parcel weightedInvOp gets Nan values due to normalizations.
    weightedInvOp = scipy.nan_to_num(weightedInvOp)
    return weightedInvOp

def _load_data(fname_identities, fname_forward, fname_inverse):
    """Expected ids for parcels are 0 to n-1, where n is number of parcels,
    and -1 for sources that do not belong to any parcel."""
    sourceIdentities = genfromtxt(fname_identities, dtype='int32',
                                  delimiter=',')

    """Zero as ID doesn't work if parcel not belonging to any parcel is given
    zero value. There could be sources not in any parcel. Sparce parcels that
    is. Should initialize those to -1 or Nan."""
    # sensors x sources
    forwardOperator = scipy.matrix(genfromtxt(fname_forward,
                                              dtype='float', delimiter=','))

    # sources x sensors
    inverseOperator = scipy.matrix(genfromtxt(fname_inverse,
                                              dtype='float', delimiter=','))
    return sourceIdentities, forwardOperator, inverseOperator


def compute_weighted_operator(fwd=None, inv=None, source_identities=None):
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

    """Load source identities, forward model, and inverse operators."""
    fpath = '/home/puolival/fidelityWeighting'
    fname_source_identities = fpath + '/sourceIdentities.csv'
    fname_forward = fpath + '/forwardSolution.csv'
    fname_inverse = fpath + '/inverseSolution.csv'

    sourceIdentities, forwardOperator, inverseOperator = _load_data(
        fname_source_identities, fname_forward, fname_inverse)

    """Generate oscillatory parcel signals."""
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

    weightedInvOp = _compute_weights(sourceTimeSeries, parcelTimeSeries, sourceIdentities, inverseOperator)
    return weightedInvOp


