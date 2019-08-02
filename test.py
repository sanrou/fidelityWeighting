# -*- encoding: utf-8 -*-
"""Script for testing the fidelity-weighting method.

References:

The source-space simulation code is largely based on example code available
from the MNE-python web page.
"""

from __future__ import division

from fidelity import apply_weighting, apply_weighting_evoked

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

"""Settings."""
inversion_method = 'dSPM'

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
inv = prepare_inverse_operator(inv, 1, 1./9, inversion_method)

"""Read labels from FreeSurfer annotation files."""
subject = 'sample'
subjects_dir = fpath + '/subjects'
parcellation = 'aparc.a2009s'

labels = read_labels_from_annot(subject, subjects_dir=subjects_dir,
                                parc=parcellation)

"""Simulate source-space data and project it to the sensors."""
fs = 1000
times = np.arange(0, 300, dtype='float') / fs

def data_fun(times):
    """Function to generate random source time courses"""
    rng = np.random.RandomState(42)
    return (50e-9 * np.sin(30. * times) *
            np.exp(- (times - 0.15 + 0.05 * rng.randn(1)) ** 2 / 0.01))

simulated_stc = simulate_sparse_stc(fwd['src'], n_dipoles=1, times=times,
                                    random_state=42, data_fun=data_fun)

evoked = apply_forward(fwd=fwd_fixed, stc=simulated_stc,
                       info=fwd_fixed['info'], use_cps=True, verbose=True)

"""Apply weighting."""
ind = np.asarray([i for i, ch in enumerate(fwd['info']['ch_names'])
                  if ch not in fwd['info']['bads']])

source_data = apply_weighting(evoked._data[ind, :], fwd_fixed, inv,
                              labels, inversion_method)
n_sources = np.shape(source_data)[0]

"""Visualize dipole locations."""
surf = 'inflated'
brain = Brain(subject_id=subject, subjects_dir=subjects_dir, hemi='both',
              surf=surf)

brain.add_foci(coords=simulated_stc.lh_vertno, coords_as_verts=True,
               hemi='lh', map_surface=surf, color='red')
brain.add_foci(coords=simulated_stc.rh_vertno, coords_as_verts=True,
               hemi='rh', map_surface=surf, color='red')

# TODO: this outputs some vtk error, not sure why. It seems to work anyway

raw_input('press enter to continue')

"""Visualize the inverse modeled data."""
vertices = [fwd_fixed['src'][0]['vertno'], fwd_fixed['src'][1]['vertno']]

stc = SourceEstimate(np.abs(source_data), vertices, tmin=0.0,
                     tstep=0.001) # weighted
stc_orig = apply_inverse(evoked, inv, 1/9., inversion_method) # original

stc.plot(subject=subject, subjects_dir=subjects_dir, hemi='both',
         time_viewer=True, colormap='mne', alpha=0.5, transparent=True)

"""Compute the parcel time-series."""
parcel_series = apply_weighting_evoked(evoked, fwd_fixed, inv,
                                       labels, inversion_method)

raw_input('press enter to exit')
