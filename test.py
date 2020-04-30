# -*- encoding: utf-8 -*-
"""Script for demonstrating the fidelity-weighting method.

References:

The source-space simulation code is largely based on example code available
from the MNE-python web page.
"""

from __future__ import division

from fidelity import (apply_weighting_evoked, weight_inverse_operator, 
                      _extract_operator_data, fidelity_estimation,
                      fidelity_estimation_matrix)

from mne import (apply_forward, convert_forward_solution,
                 read_forward_solution, read_labels_from_annot,
                 SourceEstimate)
from mne.datasets import sample
from mne.minimum_norm import (apply_inverse, read_inverse_operator,
                              prepare_inverse_operator)
from mne.simulation import simulate_sparse_stc

import numpy as np
from surfer import Brain
import os
import matplotlib.pyplot as plt


print('Loading sample data...')

"""Settings."""
subject = 'sample'
parcellation = 'aparc'   # Options, aparc.a2009s, aparc. Note that Destrieux parcellation seems to already be optimized, so weighting has only a minor effect.
inversion_method = 'dSPM'   # Options: MNE, dSPM, eLORETA. sLORETA is not well supported.
fpath = sample.data_path()
fpath_meg = os.path.join(fpath, 'MEG', 'sample')
subjects_dir = os.path.join(fpath, 'subjects')


"""Read forward and inverse operators from disk."""
fname_forward = os.path.join(fpath_meg, 'sample_audvis-meg-oct-6-fwd.fif')
fname_inverse = os.path.join(fpath_meg, 'sample_audvis-meg-oct-6-meg-inv.fif')

fwd = read_forward_solution(fname_forward)
inv = read_inverse_operator(fname_inverse)

"""Force fixed source orientation mode."""
fwd_fixed = convert_forward_solution(fwd, force_fixed=True, use_cps=True)

"""Prepare the inverse operator for use."""  # Inverse loads with free orientation
inv = prepare_inverse_operator(inv, 1, 1./9, inversion_method)

"""Read labels from FreeSurfer annotation files."""
labels = read_labels_from_annot(subject, subjects_dir=subjects_dir,
                                parc=parcellation)

"""
Create weighted inverse operator.
"""
weighted_inv = weight_inverse_operator(fwd_fixed, inv, labels, method=inversion_method)


"""   Analyze results   """
""" Check if weighting worked. """
source_identities, fwd_mat, inv_mat = _extract_operator_data(
                            fwd_fixed, inv, labels, method=inversion_method)

fidelity, cp_PLV = fidelity_estimation_matrix(fwd_mat, weighted_inv, source_identities)
fidelityO, cp_PLVO = fidelity_estimation(fwd_fixed, inv, labels, method=inversion_method)




""" Create plots. """
fig, ax = plt.subplots()
ax.plot(np.sort(fidelity), color='k', linestyle='--', label='Weighted fidelity, mean: ' + np.str(np.mean(fidelity)))
ax.plot(np.sort(fidelityO), color='k', linestyle='-', label='Original fidelity, mean: ' + np.str(np.mean(fidelityO)))

legend = ax.legend(loc='upper center', shadow=False, fontsize='12')
legend.get_frame()

ax.set_ylabel('Estimated fidelity', fontsize='12')
ax.set_xlabel('Sorted parcels', fontsize='12')

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()






"""Below is 3D visualization and simulation code."""

print('Simulating data...')

"""Simulate source-space data and project it to the sensors."""
fs = 1000
times = np.arange(0, 300, dtype='float') / fs

def data_fun(times):
    """Function to generate random source time courses"""
    rng = np.random.RandomState(42)
    return (50e-9 * np.sin(30. * times) *
            np.exp(- (times - 0.15 + 0.05 * rng.randn(1)) ** 2 / 0.01))

simulated_stc = simulate_sparse_stc(fwd['src'], n_dipoles=2, times=times,
                                    random_state=42, data_fun=data_fun)     # Gives an error later if n_dipoles = 1, as it puts a source only to the right hemisphere.

evoked = apply_forward(fwd=fwd_fixed, stc=simulated_stc,
                       info=fwd_fixed['info'], use_cps=True, verbose=True)

"""Apply weighting. Use good channels."""
ind = np.asarray([i for i, ch in enumerate(fwd['info']['ch_names'])
                  if ch not in fwd['info']['bads']])


"""Visualize dipole locations."""
surf = 'inflated'
brain = Brain(subject_id=subject, subjects_dir=subjects_dir, hemi='both',
              surf=surf)

brain.add_foci(coords=simulated_stc.lh_vertno, coords_as_verts=True,
                hemi='lh', map_surface=surf, color='red')    
brain.add_foci(coords=simulated_stc.rh_vertno, coords_as_verts=True,
                hemi='rh', map_surface=surf, color='red')
brain.show_view('frontal')

input('Dipoles visualized. Left-hold-drag to rotate. Press enter here to continue.')


source_data = apply_weighting_evoked(evoked, fwd_fixed, inv, weighted_inv,
                              labels, method=inversion_method, out_dim='source')

parcel_series = apply_weighting_evoked(evoked, fwd_fixed, inv, weighted_inv,
                              labels, method=inversion_method, out_dim='parcel')

n_sources = np.shape(source_data)[0]



print('Source level visualization. Close visualization windows to continue.')


"""Visualize the inverse modeled data."""   # Parcel space data visualization would be more interesting.
vertices = [fwd_fixed['src'][0]['vertno'], fwd_fixed['src'][1]['vertno']]

stc = SourceEstimate(np.abs(source_data), vertices, tmin=0.0, tstep=0.001) # weighted
stc_orig = apply_inverse(evoked, inv, 1/9., inversion_method) # original

stc.plot(subject=subject, subjects_dir=subjects_dir, hemi='both',
          time_viewer=True, colormap='mne', alpha=0.5, transparent=True,
          views=['fro'], initial_time=0.150)



### testing code for parcellation visualization.
# Assumes that there are 150 time samples. Take sample at 150 ms. Sort data to lh, then rh. Now interleaved. Drop medial wall ('unknown-lh' and 'unknown-rh').
p_data_sorted = np.ravel(parcel_series[:,150])
p_data_sorted = np.concatenate((p_data_sorted[0:-2:2], p_data_sorted[1:-2:2]))


# Parcel data visualization function.
import nibabel as nib

def plot_4_view(data1,parcel_names,parcellation,
                style='linear',alpha=0.95,
                zmin=None,zmax=None,zmid=None,cmap='auto',show=True,
                filename=None,surface='inflated',null_val=0,
                transparent = True,subj='fsaverage',                
                sub_dir='K:\\palva\\resting_state\\_fsaverage\\'):
    
    '''         
    Plots 1d array of data. Plotted views are lateral and medial on both HS.
    Used brain is fsaverage.
    
    INPUT:            
        data1:        1-dimensional data array, len = # parcels. 
                      1st half must be left HS, 2nd half right.
        parcel_names: Parcel_names, in the same order as the data.                       
        parcellation: Abbreviation, e.g. 'parc2018yeo7_100' or "parc2009'
        style:        'linear': pos. values only, 'divergent': both pos & neg
        alpha:        Transparency value; transparency might look weird.
        zmin:         The minimum value of a linear z-axis, or center of a 
                        divergent axis (thus should be 0)
        zmax:         Maximum value of linear z-axis, or max/-min of div.               
        zmid:         Midpoint of z-axis.
        cmap:         Colormap by name. Default is 'rocket' for linear, and
                        'icefire' for divergent; other recommended options: 
                         'YlOrRd' for linear,  or 'bwr' for divergent.
        show:         If False, plot is closed after creation. 
        filename:     File to save plot as, e.g. 'plot_13.png'
        surface:      Surface type.
        null_val:     Value for unassigned vertices
        transparent:  Whether parcels with minimum value should be transparent.
        
    OUTPUT:
        instance of surfer.Brain, if show==True
    '''
    
    N_parc = len(data1)    # the number of actually used parcels
    if len(parcel_names) != N_parc:
        raise ValueError('Number of parcels != len(data1) ')
    
    
    if parcel_names[0][-3:] != '-lh':
       parcel_names[:N_parc//2] = [p + '-lh' for p in parcel_names[:N_parc//2]]
       parcel_names[N_parc//2:] = [p + '-rh' for p in parcel_names[N_parc//2:]]

    
    hemi = 'split'
        
    #### load parcels
    if parcellation == 'parc2009':                
        aparc_lh_file = sub_dir + '\\' + subj + '\\label\\lh.aparc.a2009s.annot'
        aparc_rh_file = sub_dir + '\\' + subj + '\\label\\rh.aparc.a2009s.annot'
    else:
        aparc_lh_file = sub_dir + '\\' + subj + '\\label\\lh.' + parcellation + '.annot'  
        aparc_rh_file = sub_dir + '\\' + subj + '\\label\\rh.' + parcellation + '.annot' 
        
    labels_lh, ctab, names_lh = nib.freesurfer.read_annot(aparc_lh_file)
    labels_rh, ctab, names_rh = nib.freesurfer.read_annot(aparc_rh_file) 
    
    names_lh  = [str(n)[2:-1] +'-lh' for n in names_lh]
    names_rh  = [str(n)[2:-1] + '-rh' for n in names_rh]
    
    N_label_lh   = len(names_lh)      # number of labels/parcels with unkown and med. wall included
    N_label_rh   = len(names_rh)

    #### map parcels in data to loaded parcels
    indicesL = np.full(N_label_lh,-1)
    indicesR = np.full(N_label_rh,-1)
    
    for i in range(N_parc):
        for j in range(N_label_lh):
            if names_lh[j]==parcel_names[i]:
                indicesL[j]=i 
        for j in range(N_label_rh):
            if names_rh[j]==parcel_names[i]:            
                indicesR[j]=i-N_parc//2     
    indicesL += 1
    indicesR += 1

    
    ## assign values to loaded parcels
    data1L     = np.concatenate(([null_val],data1[:N_parc//2]))
    data1R     = np.concatenate(([null_val],data1[N_parc//2:]))
    data_left  = data1L[indicesL]
    data_right = data1R[indicesR]
    
    ## map parcel values to vertices 
    vtx_data_left = data_left[labels_lh]
    vtx_data_left[labels_lh == -1] = null_val
    vtx_data_right = data_right[labels_rh]
    vtx_data_right[labels_rh == -1] = null_val

    
    if zmin == None:
        zmin = 0
    if zmax == None:
        zmax = np.nanmax(abs(data1))
    if zmid == None:
        zmid = zmax/2

    
    if style == 'linear':           # shows only positive values 
        center = None
    elif style == 'divergent':      # shows positive and negative values
        center =  0
    
       
    #### plot to 4-view Brain
    hemi = 'split'
    brain = Brain(subj, hemi, background = 'white', surf=surface, size=[900,800], 
                  cortex = 'classic', subjects_dir=sub_dir,  views=['lat', 'med']) 
    brain.add_data(vtx_data_left,  zmin, zmax, colormap=cmap, center= center, alpha=alpha, hemi='lh')
    brain.add_data(vtx_data_right, zmin, zmax, colormap=cmap, center= center, alpha=alpha, hemi='rh')
    
    # adjust colorbar
    brain.scale_data_colormap(zmin, zmid, zmax, transparent=transparent, 
                              center=center, alpha=alpha, verbose=None) #data=None, hemi=None, 


    if filename != None:
        brain.save_image(filename) 
    
    if show:
        return brain


""" Get sorted label names. Drop medial wall."""
labels_sorted = []
for label in np.concatenate((labels[0:-2:2], labels[1:-2:2])):
    labels_sorted.append(label.name)

# Plot closes down when the program is called in Anaconda prompt for some reason. Works fine when called directly in editor.
plot_4_view(p_data_sorted,labels_sorted,parcellation,
                style='linear',alpha=0.95,
                zmin=None,zmax=None,zmid=None,cmap='auto',show=True,
                filename=None,surface='inflated',null_val=0,
                transparent = True,subj='sample',                
                sub_dir=subjects_dir)
    
input('Parcel activity at 150 ms visualized. Left-hold-drag to rotate. Press enter to continue.')

