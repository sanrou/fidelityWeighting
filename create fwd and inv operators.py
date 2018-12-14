# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:01:49 2018

@author: Localadmin_fesieben
"""
import mne
import mne.minimum_norm as minnorm
import numpy as np
import copy


# settings
subject      = 'S0116'
subjects_dir = 'M:\\inverse_weighting\\'                                 # where M: is E:\projects on PS5
directory    = subjects_dir + subject
spacing      = 'oct6'
parc         = 'parc2018yeo7_400' 
 
# input files    
bemFile      = directory  + '\\bem\\S0116-5120-bem-sol.fif'                        # the BEM file    
icaFile      = directory  + '\\S0116_set02__220416_tsss_09_trans_MEG_ICA.fif'      # fif file with the time series post-ICA    
corFile      = directory  + '\\mri\\T1-neuromag\\sets\\COR-felix-220416.fif'       # the COR file with the transformation 

# setup source spaces - note: one space per hemisphere is created! 
src          = mne.setup_source_space(subject,spacing=spacing,surface='white',subjects_dir=subjects_dir,n_jobs=2, add_dist=False)    # spacing: ico# or oct#
vert_lh      = src[0].get('vertno')
vert_rh      = src[1].get('vertno')

# read labels (parcels)
labels_parc  = mne.read_labels_from_annot(subject, parc=parc, subjects_dir=subjects_dir) 
   
# get forward operator
info         = mne.io.read_raw_fif(icaFile).info
fwd          = mne.make_forward_solution(info,corFile,src,bemFile,meg=True,eeg=False,n_jobs=1)
fwd          = mne.convert_forward_solution(fwd,force_fixed=True)            # convert to fixed orientation 
fwd_sol      = fwd['sol']['data']                                            # counterpart to forwardOperator, [sensors x sources]
 
# read data and create HF-filtered-covariance matrix
data1        = mne.io.read_raw_fif(icaFile, preload=True)  
data2        = copy.deepcopy(data1)        
data2.filter(l_freq=151,h_freq=249)
cov          = mne.compute_raw_covariance(data2)

# get and write inv. operator
inv          = minnorm.make_inverse_operator(data1.info, fwd, cov, loose=0., depth=None, fixed=True) 
inv1         = minnorm.prepare_inverse_operator(inv,1,1./9.)
inv_sol      = minnorm.inverse._assemble_kernel(inv1, None, 'MNE',None)[0]   # counterpart to forwardOperator, [sources x sensors]

# test on extract of time series
source_ts    = minnorm.apply_inverse_raw(data1, inv, method='MNE',lambda2=1. / 9., start=0, stop=6000)  
label_ts     = mne.extract_label_time_course(source_ts, labels_parc, src,  mode='mean_flip',allow_empty=True,     return_generator=False)

# get source identities
src_ident_lh    = np.full(len(vert_lh), -1)
src_ident_rh    = np.full(len(vert_rh), -1)

for l,label in enumerate(labels_parc[:201]):                   # find sources that belong to the left HS labels
    for v in label.vertices:
        src_ident_lh[np.where(vert_lh == v)]=l 
            
for l,label in enumerate(labels_parc[201:402]):                # find sources that belong to the right HS labels
    for v in label.vertices:
        src_ident_rh[np.where(vert_rh == v)]=l     
    
src_ident_lh                    = src_ident_lh -1              # fix numbers, so that sources in med. wall and unassigned get value -1
src_ident_lh[src_ident_lh==-2]  = -1
src_ident_rh                    = src_ident_rh + 200        
src_ident_rh[src_ident_rh==400] = -1    
src_ident_rh[src_ident_rh==199] = -1 
src_ident                       = np.concatenate((src_ident_lh,src_ident_rh))



# write fif files
fwdFile     = icaFile[:-4]+'-py-fwd.fif'                                    # filename to write the fwd operator into
invFile      = icaFile[:-4]+'-py-inv.fif'
identFile    = icaFile[:-4]+'-py-identities.csv'

mne.write_forward_solution(fwdFile, fwd)
mne.minimum_norm.write_inverse_operator(invFile, inv)
np.savetxt(identFile,src_ident,delimiter=';')


#### also save .csv files
# fwdFile2     = icaFile[:-4]+'-py-fwd.csv'                                    # filename to write the fwd operator into
# invFile2     = icaFile[:-4]+'-py-inv.csv'

# np.savetxt(fwdFile2,fwd_sol)
# np.savetxt(invFile2,inv_sol)
