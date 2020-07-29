# -*- coding: utf-8 -*-
"""Functions for computing fidelity-weighted inverse operator and collapsing
source time-series into parcel time-series.



Generate fidelity weighting vector
@author: Santeri Rouhinen, Tuomas Puoliväli, Felix Siebenhühner
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
    """Function for computing the weights of the weighted inverse operator."""
    
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



def _load_data(fname_identities, fname_forward, fname_inverse, delimiter=';'):
    """ Function for loading identities, and fwd and inv op matrices. """
    """Expected ids for parcels are 0 to n-1, where n is number of parcels,
    and -1 for sources that do not belong to any parcel."""
    source_identities = genfromtxt(fname_identities, dtype='int32',
                                  delimiter=',')

    """Zero as ID doesn't work if parcel not belonging to any parcel is given
    zero value. There could be sources not in any parcel. Sparce parcels that
    is. Should initialize those to -1 or Nan."""
    # sensors x sources
    fwd = np.matrix(genfromtxt(fname_forward,
                                  dtype='float', delimiter=delimiter))

    # sources x sensors
    inv = np.matrix(genfromtxt(fname_inverse,
                       dtype='float', delimiter=delimiter))
    
    return source_identities, fwd, inv



def _extract_operator_data(fwd, inv_prep, labels, method='dSPM'):
    """Function for extracting forward and inverse operator matrices from
    the MNE-Python forward and inverse data structures, and assembling the
    source identity map.
    
    Input arguments:
    ================
    fwd : ForwardOperator
        The fixed_orientation forward operator. 
        Instance of the MNE-Python class Forward.
    inv_prep : Inverse
        The prepared inverse operator.  
        Instance of the MNE-Python class InverseOperator.
    labels : list
        List of labels belonging to the used parcellation, e.g. the
        Desikan-Killiany, Destrieux, or Schaefer parcellation.
        May not contain 'trash' labels/parcels (unknown or medial wall), those
        should be deleted from the labels array!
    method : str
        The inversion method. Default 'dSPM'.
        Other methods ('MNE', 'sLORETA', 'eLORETA') have not been tested.  
        
    Output arguments:    
    =================   
    source_identities : ndarray
        Vector mapping sources to parcels or labels.
    fwd_mat : ndarray [sensors x sources]
        The forward operator matrix.
    inv_mat : ndarray [sources x sensors]
        The prepared inverse operator matrix.
    
    """
    
    # counterpart to forwardOperator, [sources x sensors]. ### pick_ori None for free, 'normal' for fixed orientation.
    K, noise_norm, vertno, source_nn = _assemble_kernel(
                    inv=inv_prep, label=None, method=method, pick_ori='normal')
    
    # get source space    
    src = inv_prep.get('src')
    vert_lh, vert_rh = src[0].get('vertno'), src[1].get('vertno')
    
    # get labels, vertices and src-identities
    src_ident_lh = np.full(len(vert_lh), -1, dtype='int')
    src_ident_rh = np.full(len(vert_rh), -1, dtype='int')
    
    # find sources that belong to the left hemisphere labels
    n_labels = len(labels)
    for la, label in enumerate(labels[:n_labels//2]):
        for v in label.vertices:
            src_ident_lh[np.where(vert_lh == v)] = la
    
    # find sources that belong to the right hemisphere labels. Add by n left.
    for la, label in enumerate(labels[n_labels//2:n_labels]):
        for v in label.vertices:
            src_ident_rh[np.where(vert_rh == v)] = la
    
    src_ident_rh[np.where(src_ident_rh<0)] = src_ident_rh[np.where(
                                                src_ident_rh<0)] -n_labels/2 
    src_ident_rh = src_ident_rh + (n_labels // 2) 
    source_identities = np.concatenate((src_ident_lh,src_ident_rh))
    
    # extract fwd and inv matrices
    fwd_mat          = fwd['sol']['data'] # sensors x sources

    """If there are bad channels the corresponding rows can be missing
    from the forward matrix. Not sure if the same can occur for the
    inverse. This is not a problem if bad channels are interpolated."""   ### MOVED from weight_inverse_operator, just before """Compute the weighted operator."""
    ind = np.asarray([i for i, ch in enumerate(fwd['info']['ch_names'])
                      if ch not in fwd['info']['bads']])
    fwd_mat = fwd_mat[ind, :]

    
    # noise_norm is used with dSPM and sLORETA. Other methods return null.
    if method != 'dSPM' or method != 'sLORETA':
        noise_norm = 1.
    inv_mat          = K * noise_norm     # sources x sensors 
    
    return source_identities, fwd_mat, inv_mat



def compute_weighted_operator(fwd_mat, inv_mat, source_identities, n_samples=10000):
    """Function for computing a fidelity-weighted inverse operator.
       Called by weight_inverse_operator.
       
    Input arguments:
    ================
    fwd_mat : ndarray [sensors x sources]
        The forward operator matrix.
    inv_mat : ndarray [sources x sensors]
        The prepared inverse operator matrix.
    source_identities : ndarray
        Vector mapping sources to parcels or labels.
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
    widths = np.arange(5, 6)
    
    """Make and clone parcel time series to source time series."""
    parcel_series = make_series(n_parcels, n_samples, time_cut, widths)
    source_series_orig = parcel_series[source_identities]
    source_series_orig[source_identities < 0] = 0
    
    """Forward then inverse model source series."""
    source_series = np.dot(inv_mat,(np.dot(fwd_mat ,source_series_orig)))
    
    """Compute weights."""
    weighted_inv, weights = _compute_weights(source_series, parcel_series,
                                     source_identities, inv_mat)
    return weighted_inv



def weight_inverse_operator(fwd, inv_prep, labels, method='dSPM'):
    """Compute fidelity-weighted inverse operator.
    Input arguments:
    ================
    fwd : ForwardOperator
        The fixed_orientation forward operator. Good channels expected.
        Instance of the MNE-Python ForwardOperator class.
    inv_prep : InverseOperator
        The prepared inverse operator. Good channels expected.
        Instance of the MNE-Python InverseOperator class.
    labels : list
        List of labels belonging to the used parcellation. Each label must
        be an instance of the MNE-Python Label class.
        May not contain 'trash' labels/parcels (unknown or medial wall), those
        should be deleted from the labels array!
    method : str
        The inversion method. Default 'dSPM'.
        Other methods ('MNE', 'sLORETA', 'eLORETA') have not been tested.  
        
    Output arguments:
    =================
    weighted_inv : ndarray
    """
    
    source_identities, fwd_mat, inv_mat = _extract_operator_data(fwd, 
                                            inv_prep, labels, method = method)
    
    """Compute the weighted operator."""
    weighted_inv = compute_weighted_operator(fwd_mat, inv_mat,
                                             source_identities)
    
    return weighted_inv




def apply_weighting_evoked(evoked, fwd, inv_prep, weighted_inv, labels, start=0, stop=None, method = 'dSPM', out_dim = 'parcel'):
    """Apply fidelity-weighted inverse operator to evoked data.    
        ---> it also seems to work on "regular data" just as well!
    Input arguments:
    ================
    evoked : Evoked
        Trial-averaged evoked data. Evoked must be an instance of the
        MNE-Python class Evoked.
    fwd : ForwardOperator
        The fixed_orientation forward operator. 
        Instance of the MNE-Python Forward class.
    inv_prep : InverseOperator
        The prepared inverse operator. Instance of the MNE-Python
        InverseOperator class.                                                                  
    weighted_inv : ndarray [n_sources/n_parcels, n_sensors]
        The fidelity-weighted inverse operator. Can be a collapsed operator.
    labels: list of labels/parcels
        List of labels or parcels belonging to the used parcellation. Each
        item must be an instance of the MNE-Python Label class.
        May not contain 'trash' labels/parcels (unknown or medial wall), those
        should be deleted from the labels array!
    method : str
        The inversion method. Default 'dSPM'.
        Other methods ('MNE', 'sLORETA', 'eLORETA') have not been tested.  
    out_dim : str
        Output mode. Default 'parcel', which means parcel time series output.
        If other than 'parcel', output is source time series.
        
    Output arguments:
    =================
    time_series : ndarray [n_parcels OR sources, n_samples]
        The parcel (default) OR source time-series.
    """
    
    
    if stop==None:
        stop = len(evoked._data[0])
        
    
    source_identities, fwd_mat, inv_mat = _extract_operator_data(
                                        fwd, inv_prep, labels, method = method)
    
    """If there are bad channels the corresponding rows can be missing
    from the forward matrix. Not sure if the same can occur for the
    inverse."""
    ind = np.asarray([i for i, ch in enumerate(fwd['info']['ch_names'])
                      if ch not in fwd['info']['bads']])
    # fwd_mat = fwd_mat[ind, :]     # Not called.
    
    if out_dim == 'parcel':
        n_parcels = len(labels)
        
        if n_parcels == weighted_inv.shape[0]:      # If inverse operator is already in parcel space. 
            time_series = np.dot(weighted_inv, evoked._data[ind, start : stop])

        else:       # Inverse operator in source space.
            """Build matrix mapping sources to parcels."""
            source_to_parcel_map = np.zeros((n_parcels, len(source_identities)),
                                                dtype=np.int8)
            for i, identity in enumerate(source_identities):
                if (identity >= 0):
                    source_to_parcel_map[identity, i] = 1
        
            """Collapse data to parcels."""
            collapsed_inv = np.dot(source_to_parcel_map, weighted_inv)
            time_series = np.dot(collapsed_inv, evoked._data[ind, start : stop])
            # parcel_series = np.dot(source_to_parcel_map, estimated_sources)
            # time_series = parcel_series
    else:
        time_series = np.dot(weighted_inv, evoked._data[ind, start : stop])     # Source space output
    
    return time_series









def fidelity_estimation(fwd, inv_prep, labels, method = 'dSPM', n_samples = 20000):
    ''' Compute fidelity and cross-patch PLV (see Korhonen et al 2014) MNE-Python input.
    Can be used for exclusion of low-fidelity parcels and parcel pairs with high CP-PLV.
    
    Input arguments: 
    ================
    fwd : ForwardOperator
        The fixed_orientation forward operator. 
        Instance of the MNE-Python Forward class.
    inv_prep : InverseOperator
        The prepared inverse operator. Instance of the MNE-Python.
        InverseOperator class.
    labels: list of labels/parcels
        List of labels or parcels belonging to the used parcellation. Each
        item must be an instance of the MNE-Python Label class.
        It is recommended to delete 'trash' labels/parcels (unknown or medial wall)!
    n_samples: int
        the number of samples in the simulated data

    method : str
        The inversion method. Default 'dSPM'. 'MNE' works fine, 'eLORETA' unoptimally.
        
    Output arguments:
    =================
    fidelity : 1D array.
        Fidelity values for each parcel.
    cpPLV : 2D array
        Cross-patch PLV of the reconstructed time series among all parcel pairs.    
    '''
    
    
    source_identities, fwd_mat, inv_mat = _extract_operator_data(fwd, inv_prep, labels, method = method)
    
    fidelity, cpPLV = fidelity_estimation_matrix(fwd_mat, inv_mat, source_identities, n_samples=n_samples)

    return fidelity, cpPLV





def fidelity_estimation_matrix(fwd, inv, source_identities, n_samples = 20000):
    ''' Compute fidelity and cross-patch PL60V (see Korhonen et al 2014) matrix input.
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

        
    Output arguments:
    =================
    fidelity : 1D array.
        Fidelity values for each parcel.
    cpPLV : 2D array
        Cross-patch PLV of the reconstructed time series among all parcel pairs.    
    '''
    
    timeCut = 20
    
    id_set = set(source_identities)
    id_set = [item for item in id_set if item >= 0]   #Remove negative values (should have only -1 if any)
    N_parcels = len(id_set)  # Number of unique IDs >= 0

    widths=np.arange(5, 6)
    
    ## Create signal per parcel
    origParcelSeries = np.real(make_series(N_parcels, n_samples, timeCut, widths))
    
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
    
    # Filter time series with Mexican hat (Ricker). Hilbert transform.
    for i in np.arange(0, N_parcels):
        origParcelSeries[i, :] = signal.cwt(np.ravel(origParcelSeries[i, :]), 
                                                 signal.ricker, widths)
    origParcelSeries = signal.hilbert(origParcelSeries)

    for i in np.arange(0, N_parcels):
        estimatedParcelSeries[i, :] = signal.cwt(np.ravel(estimatedParcelSeries[i, :]), 
                                                 signal.ricker, widths)
    estimatedParcelSeries = signal.hilbert(estimatedParcelSeries)
    
    # Do the cross-patch PLV estimation before changing the amplitude to 1. 
    cpPLV = np.zeros([N_parcels, N_parcels], dtype=np.complex128)
    
    for t in range(n_samples):
        parcelPLVn = estimatedParcelSeries[:,t] / np.abs(estimatedParcelSeries[:,t]) 
        cpPLV += np.outer(parcelPLVn, np.conjugate(parcelPLVn)) /n_samples
    
    cpPLV = np.abs(cpPLV)
    
    # Change to amplitude 1, keep angle using Euler's formula.
    origParcelSeries = np.exp(1j*(np.asmatrix(np.angle(origParcelSeries))))   
    estimatedParcelSeries = np.exp(1j*(np.asmatrix(np.angle(estimatedParcelSeries))))
    
    # Estimate parcel fidelity.
    fidelity   = np.zeros(N_parcels, dtype=np.float32)  # For the weighted inverse operator
    
    for i in range(N_parcels):
        A = np.ravel(origParcelSeries[i,:])                        # True simulated parcel time series. 
        B = np.ravel(estimatedParcelSeries[i,:])                       # Estimated parcel time series. Does ravel hinder the average? Or maybe one should use each source separately.
        fidelity[i] = np.abs(np.mean(A * np.conjugate(B)))   # Maybe one should take np.abs() away. Though abs is the value you want.
    
    return fidelity, cpPLV



def collapse_operator(operator, identities, op_type='inverse'):
    """Function for collapsing operators from source space to parcel space.
    
    Input
    ----------
    operator : ndarray, 2D
        Forward or inverse operator matrix. In source space.
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
