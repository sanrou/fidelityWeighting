"""Script for testing the fidelity weighting.

References:

The source-space simulation code is largely based on example code available
from the MNE-python web page.
"""

# -*- encoding: utf-8 -*-
from fidelity import weight_inverse_operator

import mne
from mne import (apply_forward, convert_forward_solution,
                 read_forward_solution, read_labels_from_annot,
                 SourceEstimate)
from mne.datasets import sample
from mne.minimum_norm import (apply_inverse, read_inverse_operator,
                              prepare_inverse_operator)
from mne.simulation import simulate_sparse_stc

import numpy as np

from surfer import Brain

"""Read forward and inverse operators from disk."""
fpath = sample.data_path()
fpath_meg = fpath + '/MEG/sample'

fname_forward = fpath_meg + '/sample_audvis-meg-oct-6-fwd.fif'
fname_inverse = fpath_meg + '/sample_audvis-meg-oct-6-meg-inv.fif'

fwd = read_forward_solution(fname_forward)
inv = read_inverse_operator(fname_inverse)

"""Force fixed source orientation mode."""
fwd_fixed = convert_forward_solution(fwd, force_fixed=True, use_cps=True)

"""Prepare the inverse operator for use."""
inv = prepare_inverse_operator(inv, 1, 1./9, 'MNE')

"""Read labels from FreeSurfer annotation files."""
subject = 'sample'
subjects_dir = fpath + '/subjects'
parcellation = 'aparc.a2009s'

labels = read_labels_from_annot(subject, subjects_dir=subjects_dir,
                                parc=parcellation)

"""Compute the fidelity-weighted inverse operator."""
fid_inv = weight_inverse_operator(fwd_fixed, inv, labels)

"""Simulate source-space data and project it to the sensors."""
fs = 1000
times = np.arange(0, 300, dtype='float') / fs - 0.1

def data_fun(times):
    """Function to generate random source time courses"""
    rng = np.random.RandomState(42)
    return (50e-9 * np.sin(30. * times) *
            np.exp(- (times - 0.15 + 0.05 * rng.randn(1)) ** 2 / 0.01))

stc = simulate_sparse_stc(fwd_fixed['src'], n_dipoles=5, times=times,
                          random_state=42, data_fun=data_fun)

evoked = apply_forward(fwd=fwd_fixed, stc=stc, info=fwd_fixed['info'],
                       use_cps=True, verbose=True)

"""Project data back to sensor space."""
ind = np.asarray([i for i, ch in enumerate(fwd['info']['ch_names'])
                  if ch not in fwd['info']['bads']])

source_data = np.dot(fid_inv, evoked._data[ind, :])
n_sources = np.shape(source_data)[0]

"""Visualize the inverse modeled data."""
vertices = [inv['src'][0]['vertno'], inv['src'][1]['vertno']]

source_data[source_data == 0] = np.nan
source_data = np.nan_to_num(np.log10(source_data))

stc = SourceEstimate(source_data, vertices, tmin=0.0, tstep=0.001)

#stc.data = np.log10(stc.data)
#stc = apply_inverse(evoked, inv, 1/9., 'MNE')

stc.plot(subject=subject, subjects_dir=subjects_dir, hemi='both',
         time_viewer=True) #, clim={'pos_lims': [np.min(source_data),
#         np.median(source_data), np.max(source_data)]})

raw_input('press enter to exit')


