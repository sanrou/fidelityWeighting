from fidelity import compute_weighted_operator

import mne
from mne import (convert_forward_solution, read_forward_solution,
                 read_labels_from_annot)
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator

"""Read forward and inverse operators from disk."""
fpath = sample.data_path()
fpath_meg = fpath + '/MEG/sample'

fname_forward = fpath_meg + '/sample_audvis-meg-oct-6-fwd.fif'
fname_inverse = fpath_meg + '/sample_audvis-meg-oct-6-meg-inv.fif'

fwd = read_forward_solution(fname_forward)
inv = read_inverse_operator(fname_inverse)

"""Force fixed source orientation mode."""
fwd_fixed = convert_forward_solution(fwd, force_fixed=True)

"""Read labels from FreeSurfer annotation files."""
subject = 'sample'
subjects_dir = fpath + '/subjects'
parcellation = 'aparc.a2009s'

labels = read_labels_from_annot(subject, subjects_dir=subjects_dir,
                                parc=parcellation)

"""Compute the fidelity-weighted inverse operator."""
fid_inv = compute_weighted_operator(fwd_fixed, inv, labels)
