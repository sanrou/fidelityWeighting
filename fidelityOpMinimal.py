# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:04:52 2020

Fork fidelityWeightedVector_from_csv.py and fidelity.py to a standalone minimal
code version of generating fidelity weighted inverse operators.

@author: rouhinen
"""

from __future__ import division

import scipy
from scipy import asmatrix, signal
import numpy as np
from numpy.linalg import norm
from numpy.random import randn



def compute_weighted_operator(fwd_mat, inv_mat, source_identities, n_samples=10000):
    """Function for computing a fidelity-weighted inverse operator.
       Called by weight_inverse_operator.
       
    Input arguments:
    ================
    fwd_mat : ndarray [sensors x sources]
        The forward operator matrix.
    inv_mat : ndarray [sources x sensors]
        The prepared inverse operator matrix.
    source_identities : ndarray [sources]
        Expected ids for parcels are 0 to n-1, where n is number of parcels, 
        and -1 for sources that do not belong to any parcel. 
    n_samples : int
        The number of samples in the simulated data.    
        
    Output argument:
    ================
    weighted_inv : ndarray
        The fidelity-weighted inverse operator.
    """

    """Generate oscillatory parcel signals."""
        
    """Get number of parcels."""
    id_set = set(source_identities)
    id_set = [item for item in id_set if item >= 0]   #Remove negative values (should have only -1 if any)
    n_parcels = len(id_set)  # Number of unique IDs >= 0

    """Samples to remove from ends to get rid of border effects."""
    time_cut = 20

    """Original values 1, 31. Higher number wider span."""
    widths = scipy.arange(5, 6)

    """Make and clone parcel time series to source time series."""
    # np.random.seed(42) if seed == True else print('not seeded')   # Gives a type error for some reason, if there is seed=False in the definition.
    
    parcel_series = make_series(n_parcels, n_samples, time_cut, widths)
    source_series_orig = parcel_series[source_identities]
    source_series_orig[source_identities < 0] = 0

    """Forward then inverse model source series."""
    source_series = np.dot(inv_mat,(np.dot(fwd_mat ,source_series_orig)))
    
    """Compute weights."""
    weighted_inv, weights = _compute_weights(source_series, parcel_series,
                                     source_identities, inv_mat)
    return weighted_inv




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
    
    decim_factor = 5
    s = randn(n_parcels, n_samples*decim_factor+2*n_cut_samples)

    for i in np.arange(0, n_parcels):
        s[i, :] = signal.cwt(s[i, :], signal.ricker, widths)

    s = signal.hilbert(s)
    s = s[:, n_cut_samples:-n_cut_samples]
    s = scipy.signal.decimate(s, decim_factor, axis=1)

    return s



def plv(x, y, source_identities):
    """ Function for computing the complex phase-locking value."""
    
    """Change to amplitude 1, keep angle using Euler's formula."""
    x = scipy.exp(1j*(asmatrix(scipy.angle(x))))
    y = scipy.exp(1j*(asmatrix(scipy.angle(y))))

    """Get cPLV needed for flips and weighting."""
    cplv = scipy.zeros(len(source_identities), dtype='complex')

    for i, identity in enumerate(source_identities):
        """Compute cPLV only of parcel source pairs of sources that
        belong to that parcel. One source belong to only one parcel."""
        if (source_identities[i] >= 0):
            cplv[i] = (scipy.sum((scipy.asarray(y[identity])) *
                       scipy.conjugate(scipy.asarray(x[i]))))

    cplv /= np.shape(x)[1]
    return cplv



def _compute_weights(source_series, parcel_series, source_identities, inv_mat):
    """Function for computing the weights of the weighted inverse operator.
    """
    
    cplv_array = plv(source_series, parcel_series, source_identities)

    """Get weights and flip. This could be the output."""
    weights = scipy.sign(scipy.real(cplv_array)) * scipy.real(cplv_array) ** 2    

    """Create weighted inverse operator and normalize the norm of weighted inv op
    to match original inv op's norm."""
    """Multiply sensor dimension in inverseOperator by weight. This one would be
    the un-normalized operator."""
    weighted_inv = scipy.einsum('ij,i->ij', inv_mat, weights)

    id_set = set(source_identities)
    id_set = [item for item in id_set if item >= 0]   #Remove negative values (should have only -1 if any)
    n_parcels = len(id_set)  # Number of unique IDs with ID >= 0

    """Initialize norm normalized weights. Maybe not necessary."""
    weights_normalized = scipy.zeros(len(weights))
    for parcel in range(n_parcels): # Normalize parcel level norms.
        # Index sources belonging to parcel
        ii = [i for i, source in enumerate(source_identities) if source == parcel]

        # Normalize per parcel.
        weights_normalized[ii] = weights[ii] * (norm(inv_mat[ii]) /
                                                norm(weighted_inv[ii]))

    """Parcel level normalized operator."""
    weighted_inv = scipy.einsum('ij,i->ij', inv_mat, weights_normalized)

    """Operator level normalized operator. If there are sources not in any
    parcel weightedInvOp gets Nan values due to normalizations."""
    weighted_inv *= norm(inv_mat) / norm(scipy.nan_to_num(weighted_inv))
    weighted_inv = scipy.nan_to_num(weighted_inv)

    return weighted_inv, weights

