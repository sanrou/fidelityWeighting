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
from scipy.ndimage.interpolation import shift
import numpy as np
from numpy.linalg import norm
from numpy.random import randn
from random import shuffle



def compute_weighted_operator(fwd_mat, inv_mat, source_identities, n_samples=10000):
    """Function for computing a fidelity-weighted inverse operator.
       Note that only good channels are expected. Parcel level flips are applied.
       
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
        
    """Get number of parcels."""
    id_set = set(source_identities)
    id_set = [item for item in id_set if item >= 0]   #Remove negative values (should have only -1 if any)
    n_parcels = len(id_set)  # Number of unique IDs >= 0

    """Samples to remove from ends to get rid of border effects."""
    time_cut = 20

    """Original values 1, 31. Higher number wider span."""
    widths = np.arange(5, 6)

    """Generate oscillatory parcel signals."""
    parcel_series = make_series(n_parcels, n_samples, time_cut, widths)
    source_series_orig = parcel_series[source_identities]
    source_series_orig[source_identities < 0] = 0

    """Forward then inverse model source series."""
    source_series = np.dot(inv_mat,(np.dot(fwd_mat ,source_series_orig)))
    
    """Compute weights."""  
    weighted_inv, weights = _compute_weights(source_series, parcel_series,
                                     source_identities, inv_mat)
    
    """ Perform parcel flips (multiply by 1 or -1) depending on inverse forward operation result. 
    This is to make evoked responses of neighbouring parcels match in direction. """
    sensor_orig = np.ones((fwd_mat.shape[0], 1))
    inversed = np.dot(weighted_inv, sensor_orig)
    
    for index, parcel in enumerate(id_set): 
        # Index sources (not) belonging to the parcel
        ni = [i for i, source in enumerate(source_identities) if source != parcel]
        ii = [i for i, source in enumerate(source_identities) if source == parcel]
        
        # Forward model parcel's sources using bulk simulated "series". Flip whole parcels.
        fwd_par = 1*fwd_mat
        fwd_par[:,ni] = 0
        sensor_mod = np.dot(fwd_par, inversed)
        parcel_flip = np.sign(sum(sensor_mod))[0,0]
        weighted_inv[ii,:] *= parcel_flip
        weights[ii] *= parcel_flip
        
    return weighted_inv, weights




def make_series(n_parcels, n_samples, n_cut_samples=40, widths=range(5,6)):
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
    """ Function for computing the complex phase-locking value.
    source_identities : ndarray [sources]
        Expected ids for parcels are 0 to n-1, where n is number of parcels, 
        and -1 for sources that do not belong to any parcel. 
    """
    
    """Change to amplitude 1, keep angle using Euler's formula."""
    x = np.exp(1j*(asmatrix(np.angle(x))))
    y = np.exp(1j*(asmatrix(np.angle(y))))

    """Get cPLV needed for flips and weighting."""
    cplv = np.zeros(len(source_identities), dtype='complex')

    for i, identity in enumerate(source_identities):
        """Compute cPLV only of parcel source pairs of sources that
        belong to that parcel. One source belong to only one parcel."""
        if (source_identities[i] >= 0):
            cplv[i] = (np.sum((np.asarray(y[identity])) *
                       np.conjugate(np.asarray(x[i]))))

    cplv /= np.shape(x)[1]
    return cplv



def _compute_weights(source_series, parcel_series, source_identities, inv_mat):
    """Function for computing the weights of the weighted inverse operator.
    source_identities : ndarray [sources]
        Expected ids for parcels are 0 to n-1, where n is number of parcels, 
        and -1 for sources that do not belong to any parcel. 
    
    """
    
    cplv_array = plv(source_series, parcel_series, source_identities)

    """Get weights and flip. This could be the output."""
    weights = np.sign(np.real(cplv_array)) * np.real(cplv_array) ** 2    

    """Create weighted inverse operator and normalize the norm of weighted inv op
    to match original inv op's norm."""
    """Multiply sensor dimension in inverseOperator by weight. This one would be
    the un-normalized operator."""
    weighted_inv = np.einsum('ij,i->ij', inv_mat, weights)

    id_set = set(source_identities)
    id_set = [item for item in id_set if item >= 0]   #Remove negative values (should have only -1 if any)
    n_parcels = len(id_set)  # Number of unique IDs with ID >= 0

    """Initialize norm normalized weights. Maybe not necessary."""
    weights_normalized = np.zeros(len(weights))
    for parcel in range(n_parcels): # Normalize parcel level norms.
        # Index sources belonging to parcel
        ii = [i for i, source in enumerate(source_identities) if source == parcel]

        # Normalize per parcel.
        weights_normalized[ii] = weights[ii] * (norm(inv_mat[ii]) /
                                                norm(weighted_inv[ii]))
        
    """Parcel level normalized operator."""
    weighted_inv = np.einsum('ij,i->ij', inv_mat, weights_normalized)

    """Operator level normalized operator. If there are sources not in any
    parcel weightedInvOp gets Nan values due to normalizations."""
    weighted_inv *= norm(inv_mat) / norm(np.nan_to_num(weighted_inv))
    weighted_inv = np.nan_to_num(weighted_inv)
    
    return weighted_inv, weights




def fidelity_estimation(fwd, inv, source_identities, n_samples = 20000, parcel_series=np.asarray([])):
    ''' Compute fidelity and cross-patch PL60V (see Korhonen et al 2014)
    Can be used for exclusion of low-fidelity parcels and parcel pairs with high CP-PLV.
    
    Input arguments: 
    ================
    fwd : Forward operator matrix, ndarray [sensors x sources]
    inv : Inverse operator matrix, ndarray [sources x sensors]
        Note that only good channels are expected in forward and inverse operators.
    source_identities : ndarray [sources]
        Expected ids for parcels are 0 to n-1, where n is number of parcels, 
        and -1 for sources that do not belong to any parcel. 
    n_samples: int
        the number of samples generated for simulated data
    parcel_series : ndarray, complex [parcels x samples]
        If empty, time series will be generated. Else given series is used.
        Overrides n_samples if given.

        
    Output arguments:
    =================
    fidelity : 1D array.
        Fidelity values for each parcel.
    cpPLV : 2D array
        Cross-patch PLV of the reconstructed time series among all parcel pairs.    
    '''
    
    timeCut = 20
    widths=np.arange(5, 6)
    
    id_set = set(source_identities)
    id_set = [item for item in id_set if item >= 0]   #Remove negative values (should have only -1 if any)
    N_parcels = len(id_set)  # Number of unique IDs >= 0

    ## Check if source time series is empty. If empty, create time series.
    if parcel_series.size == 0:
        origParcelSeries = make_series(N_parcels, n_samples, timeCut, widths)
    else:
        origParcelSeries = parcel_series
        n_samples = parcel_series.shape[1]
    
    ## Clone parcel time series to source time series
    cloneSourceTimeSeries = origParcelSeries[source_identities]
    
    # Forward and inverse model cloned source time series
    estimatedSourceSeries = np.dot(inv, np.dot(fwd,cloneSourceTimeSeries))
    
    # Collapse estimated source series to parcel series
    sourceParcelMatrix = np.zeros((N_parcels,len(source_identities)), dtype=np.int8)
    for i,identity in enumerate(source_identities):
        if identity >= 0:     # Don't place negative values. These should be sources not belonging to any parcel.
            sourceParcelMatrix[identity,i] = 1
    
    estimatedParcelSeries = np.dot(sourceParcelMatrix, estimatedSourceSeries)
    
    # Do the cross-patch PLV estimation before changing the amplitude to 1. 
    cpPLV = np.zeros([N_parcels, N_parcels], dtype=np.complex128)
    
    for t in range(n_samples):
        parcelPLVn = estimatedParcelSeries[:,t] / np.abs(estimatedParcelSeries[:,t]) 
        cpPLV += np.outer(parcelPLVn, np.conjugate(parcelPLVn)) /n_samples
    
    # cpPLV = np.abs(cpPLV)  ### TEMP Removed absolute value out.
    
    # Change to amplitude 1, keep angle using Euler's formula.
    origParcelSeries = np.exp(1j*(np.asmatrix(np.angle(origParcelSeries))))   
    estimatedParcelSeries = np.exp(1j*(np.asmatrix(np.angle(estimatedParcelSeries))))
    
    # Estimate parcel fidelity.
    fidelity   = np.zeros(N_parcels, dtype=np.float32)  # For the weighted inverse operator
    
    for i in range(N_parcels):
        A = np.ravel(origParcelSeries[i,:])                        # True simulated parcel time series. 
        B = np.ravel(estimatedParcelSeries[i,:])                       # Estimated parcel time series. 
        fidelity[i] = np.abs(np.mean(A * np.conjugate(B)))   # Maybe one should take np.abs() away. Though abs is the value you want.
    
    return fidelity, cpPLV



def make_series_paired(n_parcels, n_samples, n_cut_samples=40, widths=range(5,6), time_shift=3):
    """Function for generating oscillating parcel signals with each parcel with
    degree of one.
    
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
        Width to use for the wavelet transform.
    time_shift : int
        Shift in samples.
        
    Output arguments:
    =================
    s_comb : ndarray, complex
        Simulated oscillating parcel time-series. 
        Half are time shifted copies of the first half.
        Each parcel has degree of one.
    pairs : ndarray
        Parcel edge pairs.
    """
    
    decim_factor = 5
    n_parcels_half = np.int(n_parcels/2)
    time_shift = (0, time_shift)  # Do not shift across parcels, only time.
    
    pairs = list(range(0, n_parcels))
    shuffle(pairs)
    pairs = np.reshape(pairs, (n_parcels_half, 2))
    
    # Do signals for half of the parcels. Time shift the other half from the first half.
    s = randn(n_parcels_half, n_samples*decim_factor+2*n_cut_samples)
    
    for i in np.arange(0, n_parcels_half):
        s[i, :] = signal.cwt(s[i, :], signal.ricker, widths)
        
    s_shift = shift(s, time_shift, mode='wrap')
    s = signal.hilbert(s)
    s_shift = signal.hilbert(s_shift)
    s = s[:, n_cut_samples:-n_cut_samples]
    s_shift = s_shift[:, n_cut_samples:-n_cut_samples]
    ### TODO: check if amplitude randomization does anything. This is added for s_shifted in LV code. One could use shuffle on the amplitude and keep phase.
    # Decimate the signals separately.
    s = scipy.signal.decimate(s, decim_factor, axis=1)
    s_shift = scipy.signal.decimate(s_shift, decim_factor, axis=1)
    
    # Slice the generated signals to correct indices.
    s_comb = np.zeros((n_parcels, n_samples), dtype=complex)
    s_comb[pairs[:,0],:] = s
    s_comb[pairs[:,1],:] = s_shift
    return s_comb, pairs



def collapse_operator(operator, identities, op_type='inverse'):
    """Function for collapsing operators from source space to parcel space.
    
    Input
    ----------
    operator : ndarray, 2D
        Forward or inverse operator. In source space.
    identities : ndarray, 1D [sources]
        Expected ids for parcels are 0 to n-1, where n is number of parcels, 
        and -1 for sources that do not belong to any parcel.
    op_type : str
        Operator type defines which dimension will be collapsed 
        ('inverse' = [sources x sensors], 'forward' = [sensors x sources]).

    Output
    -------
    collapsed_operator : ndarray, 2D.

    """
    idSet = set(identities)                         # Get unique IDs
    idSet = [item for item in idSet if item >= 0]   # Remove negative values (should have only -1 if any)
    n_parcels = len(idSet)

    # Make collapse matrix (parcels x sources)
    sourceParcelMatrix = np.zeros((n_parcels,len(identities)), dtype=np.int8)
    for i,identity in enumerate(identities):
        if identity >= 0:     # Don't place negative values. These should be sources not belonging to any parcel.
            sourceParcelMatrix[identity,i] = 1
    
    if op_type == 'forward':
        collapsed_operator = np.dot(operator, sourceParcelMatrix.T)
    else:
        collapsed_operator = np.dot(sourceParcelMatrix, operator)
    
    return collapsed_operator

