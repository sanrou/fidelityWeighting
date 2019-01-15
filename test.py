import mne
from mne import read_forward_solution
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator

"""Read forward and inverse operators from disk."""
fpath = sample.data_path() + '/MEG/sample'

fname_forward = fpath + '/sample_audvis-meg-oct-6-fwd.fif'
fname_inverse = fpath + '/sample_audvis-meg-oct-6-meg-inv.fif'

fwd = read_forward_solution(fname_forward)
inv = read_inverse_operator(fname_inverse)
