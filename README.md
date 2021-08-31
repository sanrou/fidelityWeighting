README

# Fidelity-weighting

This is a repository for computing fidelity-weighted inverse operators 
with Python 3 to be used with cortical parcellations (see e.g. [1–3]) 
in neurophysiological research. 
Fidelity: how well simulated source activity is replicated after
forward then inverse modeling the source activity.
The code can also be used to estimate how well your source modeling performs.

## Dependencies

For the minimal version:
- NumPy
- SciPy

MNE-Python is also supported, and requires:
- MNE-Python https://martinos.org/mne/stable/index.html.
- PySurfer https://pysurfer.github.io/
- Matplotlib

# Dataset
https://doi.org/10.5281/zenodo.5291628 has subject files that are used in the development of the code and in the coming article.

# References

## Fidelity weighting publication
In works.

## Cortical parcellations

[1] Desikan RS, Ségonne F, Fischl B, Quinn BT, Dickerson BC, Blacker D, 
Buckner RL, Dale AM, Maguire RP, Hyman BT, Albert MS, Killiany RJ (2006):
An automated labeling system for subdividing the human cerebral cortex
on MRI scans into gyral based regions of interest. *NeuroImage* 31:968–980

[2] Destrieux C, Fischl B, Dale A, Halgren E (2010): Automatic parcellation
of human cortical gyri and sulci using standard anatomical nomenclature.
*NeuroImage* 53(1):1–15.

[3] Schaefer A, Kong R, Gordon EM, Laumann TO, Zuo XN, Holmes AJ,
Eickhoff SB, Yeo BTT (2018): Local-global parcellation of the human 
cerebral cortex from intrinsic functional connectivity MRI. *Cerebral 
Cortex* 28(9):3095–3114.
