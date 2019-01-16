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

from mne.minimum_norm import prepare_inverse_operator
from mne.minimum_norm.inverse import _assemble_kernel

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
    """Multiply sensor dimension in inverseOperator by weight. This one would be
    the un-normalized operator."""
    weightedInvOp = scipy.einsum('ij,i->ij', inverse, weights)

    n_parcels = max(identities) + 1
    """Initialize norm normalized weights. Maybe not necessary."""
    weightsNormalized = scipy.zeros(len(weights))
    for parcel in range(n_parcels): # Normalize parcel level norms.
        # Index sources belonging to parcel
        ii = [i for i, source in enumerate(identities) if source == parcel]

        # Normalize per parcel.
        weightsNormalized[ii] = weights[ii] * (norm(inverse[ii]) /
                                               norm(weightedInvOp[ii]))

    """Parcel level normalized operator."""
    weightedInvOp = scipy.einsum('ij,i->ij', inverse, weightsNormalized)

    """Operator level normalized operator. If there are sources not in any
    parcel weightedInvOp gets Nan values due to normalizations."""
    weightedInvOp *= norm(inverse) / norm(scipy.nan_to_num(weightedInvOp))
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

def _extract_operator_data(fwd, inv, labels_parc):
    """Function for extracting forward and inverse operator matrices from
    the MNE-Python forward and inverse data structures, and the assembling
    the source identity map.

    Input arguments:
    ================
    fwd : Forward
        The forward operator. An instance of the MNE-Python class Forward.
    inv : Inverse
        The inverse operator. An instance of the MNE-Python class Inverse.
    labels_parc : list
        List of labels belonging to the used parcellation, e.g.
        Desikan-Killiany or Yeo.
    """

    # read and prepare inv op
    invP = prepare_inverse_operator(inv, 1, 1./9, 'MNE')
    # counterpart to forwardOperator, [sources x sensors]
    inv_sol = _assemble_kernel(invP, None, 'MNE', None)[0]

    # get source space
    src = inv.get('src')
    vert_lh, vert_rh = src[0].get('vertno'), src[1].get('vertno')

    # get labels, vertices and src-identities
    src_ident_lh = np.full(len(vert_lh), -1, dtype='int')
    src_ident_rh = np.full(len(vert_rh), -1, dtype='int')

    # find sources that belong to the left HS labels
    for l, label in enumerate(labels_parc[:201]):
        for v in label.vertices:
            src_ident_lh[np.where(vert_lh == v)] = l

    # find sources that belong to the right HS labels
    for l, label in enumerate(labels_parc[201:402]):
        for v in label.vertices:
            src_ident_rh[np.where(vert_rh == v)] = l

    # fix numbers, so that sources in med. wall and unassigned get value -1
    # TODO: replace constants with parcel counts etc.
    src_ident_lh = src_ident_lh -1
    src_ident_lh[src_ident_lh == -2] = -1
    src_ident_rh = src_ident_rh + 200
    src_ident_rh[src_ident_rh == 400] = -1
    src_ident_rh[src_ident_rh == 199] = -1
    src_ident = np.concatenate((src_ident_lh,src_ident_rh))

    #### change variable names
    sourceIdentities = src_ident
    inverseOperator = inv_sol
    forwardOperator = fwd['sol']['data'] # sensors x sources

    return sourceIdentities, forwardOperator, inverseOperator

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


    """Generate oscillatory parcel signals."""

    """Maybe one should test if unique non-negative values == max+1. This
    is expected in the code."""
    n_parcels = max(source_identities) + 1

    """Samples. Peaks at about 20 GB ram with 30 000 samples. Using too few
    samples will give poor results."""
    time_output = 30000

    """Samples to remove from ends to get rid of border effects."""
    time_cut = 20

    """Original values 1, 31. Higher number wider span."""
    widths = scipy.arange(5, 6)
    time_generate = time_output + 2*time_cut
    samplesSubset = 10000 + 2*time_cut

    """Make and clone parcel time series to source time series."""
    fwd, inv = asmatrix(fwd), asmatrix(inv)
    parcelTimeSeries = make_series(n_parcels, time_output, time_cut, widths)
    sourceTimeSeries = parcelTimeSeries[source_identities]
    sourceTimeSeries[source_identities < 0] = 0

    """Forward then inverse model source series."""
    sourceTimeSeries = inv * (fwd * sourceTimeSeries)

    weightedInvOp = _compute_weights(sourceTimeSeries, parcelTimeSeries,
                                     source_identities, inv)
    return weightedInvOp

def weight_inverse_operator(fwd, inv, labels):

    identities, fwd_mat, inv_mat = _extract_operator_data(fwd, inv, labels)

    """If there are bad channels the corresponding rows can be missing
    from the forward matrix. Not sure if the same can occur for the
    inverse."""
    ind = np.asarray([i for i, ch in enumerate(fwd['info']['ch_names'])
                      if ch not in fwd['info']['bads']])
    fwd_mat = fwd_mat[ind, :]

    weighted_inv = compute_weighted_operator(fwd_mat, inv_mat,
                                             identities)
