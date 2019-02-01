# -*- coding: utf-8 -*-
"""Functions for computing fidelity-weighted inverse operator and collapsing
source time-series into parcel time-series.

Created on Fri Jul 21 16:31:20 2017

Generate fidelity weighting vector

@author: Santeri Rouhinen
"""

from __future__ import division

import scipy
from scipy import asmatrix, genfromtxt, signal

import numpy as np
from numpy.linalg import norm
from numpy.random import randn

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

    cplv /= np.shape(x)[1]
    return cplv

def _compute_weights(source_series, parcel_series, identities, inverse):
    cplv_array = plv(source_series, parcel_series, identities)

    """Get weights and flip. This could be the output."""
    weights = scipy.sign(scipy.real(cplv_array)) * scipy.real(cplv_array) ** 2

    """Create weighted inverse operator and normalize the norm of weighted inv op
    to match original inv op's norm."""
    """Multiply sensor dimension in inverseOperator by weight. This one would be
    the un-normalized operator."""
    weighted_inv = scipy.einsum('ij,i->ij', inverse, weights)

    n_parcels = max(identities) + 1
    """Initialize norm normalized weights. Maybe not necessary."""
    weights_normalized = scipy.zeros(len(weights))
    for parcel in range(n_parcels): # Normalize parcel level norms.
        # Index sources belonging to parcel
        ii = [i for i, source in enumerate(identities) if source == parcel]

        # Normalize per parcel.
        weights_normalized[ii] = weights[ii] * (norm(inverse[ii]) /
                                               norm(weighted_inv[ii]))

    """Parcel level normalized operator."""
    weighted_inv = scipy.einsum('ij,i->ij', inverse, weights_normalized)

    """Operator level normalized operator. If there are sources not in any
    parcel weightedInvOp gets Nan values due to normalizations."""
    weighted_inv *= norm(inverse) / norm(scipy.nan_to_num(weighted_inv))
    weighted_inv = scipy.nan_to_num(weighted_inv)

    return weighted_inv

def _load_data(fname_identities, fname_forward, fname_inverse):
    """Expected ids for parcels are 0 to n-1, where n is number of parcels,
    and -1 for sources that do not belong to any parcel."""
    identities = genfromtxt(fname_identities, dtype='int32',
                                  delimiter=',')

    """Zero as ID doesn't work if parcel not belonging to any parcel is given
    zero value. There could be sources not in any parcel. Sparce parcels that
    is. Should initialize those to -1 or Nan."""
    # sensors x sources
    fwd = scipy.matrix(genfromtxt(fname_forward,
                                  dtype='float', delimiter=','))

    # sources x sensors
    inv = scipy.matrix(genfromtxt(fname_inverse,
                       dtype='float', delimiter=','))
    return identities, fwd, inv

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
        List of labels belonging to the used parcellation, e.g. the
        Desikan-Killiany, Destrieux, or Schaefer parcellation.
    """

    # counterpart to forwardOperator, [sources x sensors]
    inv_sol = _assemble_kernel(inv=inv, label=None, method='MNE',
                               pick_ori='normal')[0]

    # get source space
    src = inv.get('src')
    vert_lh, vert_rh = src[0].get('vertno'), src[1].get('vertno')

    # get labels, vertices and src-identities
    src_ident_lh = np.full(len(vert_lh), -1, dtype='int')
    src_ident_rh = np.full(len(vert_rh), -1, dtype='int')

    # find sources that belong to the left HS labels
    n_labels = len(labels_parc)
    for l, label in enumerate(labels_parc[:n_labels//2]):
        for v in label.vertices:
            src_ident_lh[np.where(vert_lh == v)] = l

    # find sources that belong to the right HS labels
    for l, label in enumerate(labels_parc[n_labels//2:n_labels]):
        for v in label.vertices:
            src_ident_rh[np.where(vert_rh == v)] = l

    # fix numbers, so that sources in med. wall and unassigned get value -1
    src_ident_lh = src_ident_lh -1
    src_ident_lh[src_ident_lh == -2] = -1
    src_ident_rh = src_ident_rh + (n_labels // 2) - 1
    src_ident_rh[src_ident_rh == n_labels // 2 - 2] = -1
    src_ident_rh[src_ident_rh == n_labels // 2 - 2] = -1
    src_ident = np.concatenate((src_ident_lh,src_ident_rh))

    # TODO: check that these hard-coded values above work also with other
    # parcellations

    # change variable names
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

    """Make and clone parcel time series to source time series."""
    fwd, inv = asmatrix(fwd), asmatrix(inv)
    parcel_series = make_series(n_parcels, time_output, time_cut, widths)
    source_series = parcel_series[source_identities]
    source_series[source_identities < 0] = 0

    """Forward then inverse model source series."""
    source_series = inv * (fwd * source_series)
    weighted_inv = _compute_weights(source_series, parcel_series,
                                     source_identities, inv)
    return weighted_inv

def weight_inverse_operator(fwd, inv, labels):
    """Compute fidelity-weighted inverse operator.

    Input arguments:
    ================
    fwd : ForwardOperator
        The forward operator. Must be an instance of the MNE-Python
        ForwardOperator class.

    inv : InverseOperator
        The original inverse operator. Must be an instance of the MNE-Python
        InverseOperator class.

    labels : list
        List of labels belonging to the used parcellation. Each label must
        be an instance of the MNE-Python Label class.

    Output arguments:
    =================
    weighted_inv : ndarray
        TODO: could this be returned as an InverseOperator instance?
        (check which fields would need to be updated)
    """

    identities, fwd_mat, inv_mat = _extract_operator_data(fwd, inv, labels)

    """If there are bad channels the corresponding rows can be missing
    from the forward matrix. Not sure if the same can occur for the
    inverse."""
    ind = np.asarray([i for i, ch in enumerate(fwd['info']['ch_names'])
                      if ch not in fwd['info']['bads']])
    fwd_mat = fwd_mat[ind, :]

    """Compute the weighted operator."""
    weighted_inv = compute_weighted_operator(fwd_mat, inv_mat,
                                             identities)

    return weighted_inv

def apply_weighting_evoked(evoked, weighted_inv, labels):
    """Apply fidelity-weighted inverse operator to evoked data.

    Input arguments:
    ================
    evoked : Evoked
        Trial-averaged evoked data. Evoked must be an instance of the
        MNE-Python class Evoked.

    weighted_inv : ndarray [n_sources, n_sensors]
        The fidelity-weighted inverse operator.

    labels: list of Label
        List of labels or parcels belonging to the used parcellation. Each
        item must be an instance of the MNE-Python Label class.

    Output arguments:
    =================
    parcel_series : ndarray [n_parcels, n_samples]
        The parcel time-series.
    """

    """Build matrix mapping sources to parcels."""
    n_parcels = np.max(identities) + 1
    source_to_parcel_map = scipy.zeros((n_parcels, len(identities)),
                                        dtype=scipy.int8)
    for i, identity in enumerate(identities):
        if (identity >= 0):
            source_to_parcel_map[identity, i] = 1

    """Collapse data to parcels."""
    estimated_sources = np.dot(weighted_inv, evoked._data)

    # TODO: check whether this gives the correct result
    # parcel_series = np.dot(source_to_parcel_map, estimated_sources)

