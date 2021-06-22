# -*- coding: utf-8 -*-
"""
Created on Mon May 31 11:42:12 2021

Load and display example subjects' parcel fidelities. One median original fidelity subject.
@author: rouhinen
"""


import numpy as np
import matplotlib.pyplot as plt


# Subject fraction list
percentileList = np.array([0.5])
# percentileList = np.array([0.15, 0.5, 0.85])

savePathBase = "C:\\temp\\fWeighting\\plotDump\\"
parcelPlotEnd = 'XYZ Fidelity Orig Weighted examples.pdf'
gainPlotEnd = 'XYZ Fidelities Relative examples.pdf'
subjectsOrigWeightEnd = 'XYZ Fidelities Orig x Weighted scatter examples.pdf'
XYZto = 'schaefer200'
savePDFs = False
tightLayout = True

## Replace XYZ
parcelPlotEnd = parcelPlotEnd.replace('XYZ', XYZto)
subjectsOrigWeightEnd = subjectsOrigWeightEnd.replace('XYZ', XYZto)
gainPlotEnd = gainPlotEnd.replace('XYZ', XYZto)

## Load files. fidXArrays created in fidGroupAnalysis.py. There saved with np.save().
fidWArray = np.load('C:\\temp\\fWeighting\\numpyArrays\\fidArrays\\XYZ\\fidWArray.npy'.
                    replace('XYZ', XYZto))
fidOArray = np.load('C:\\temp\\fWeighting\\numpyArrays\\fidArrays\\XYZ\\fidOArray.npy'.
                    replace('XYZ', XYZto))


## Search example subjects. Build average fidelity arrays.
fidRArray = fidWArray/fidOArray
fidRAverage = np.average(fidRArray, axis=1)
fidWAverage = np.average(fidWArray, axis=1)
fidOAverage = np.average(fidOArray, axis=1)

# Searching values, sort the indices
# ind = np.argsort(fidRAverage)                  # Sorted by average gain
ind = np.argsort(fidOAverage)                  # Sorted by average original fidelity

exampleInds = np.int32(np.round(len(fidRAverage)*percentileList))
exampleInds = ind[exampleInds]

# Indices of subjects not used as examples
notExInds = list(range(len(fidRAverage)))
for i, index in enumerate(exampleInds):
  notExInds.remove(index)


"""   Plots   """
# Set global figure parameters, including CorelDraw compatibility (.fonttype)
import matplotlib.pylab as pylab
if tightLayout == True:
  params = {'legend.fontsize':'7',
         'figure.figsize':(1.8, 1.4),
         'axes.labelsize':'7',
         'axes.titlesize':'7',
         'xtick.labelsize':'7',
         'ytick.labelsize':'7',
         'lines.linewidth':'0.5',
         'pdf.fonttype':42,
         'ps.fonttype':42,
         'font.family':'Arial'}
else:   # Looks nice on the screen parameters
  params = {'legend.fontsize':'7',
         'figure.figsize':(3, 2),
         'axes.labelsize':'7',
         'axes.titlesize':'7',
         'xtick.labelsize':'7',
         'ytick.labelsize':'7',
         'lines.linewidth':'0.5',
         'pdf.fonttype':42,
         'ps.fonttype':42,
         'font.family':'Arial'}
pylab.rcParams.update(params)


colors = ['red', 'magenta', 'darkcyan']


""" Scatter plot weighted and original average fidelities by subject. """
fig, ax = plt.subplots(1,1)
# Plot non-example subjects
ax.scatter(fidOAverage[notExInds], fidWAverage[notExInds], c='black', alpha=0.5, s=10)  ## X, Y.
ax.plot([0,1], [0,1], color='black')
# ax.set_title('PLV')
ax.set_xlabel('Parcel fidelity, Original')
ax.set_ylabel('Parcel fidelity, Weighted')

plt.ylim(0.2, 0.5)
plt.xlim(0.2, 0.5)

# Plot example subjects
for i, exInd in enumerate(exampleInds):
  ax.scatter(fidOAverage[exInd], fidWAverage[exInd], c=colors[i], alpha=0.7, s=10)  ## X, Y.
  
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=0.1)
plt.show()

if savePDFs == True:
  fig.savefig(savePathBase + subjectsOrigWeightEnd, format='pdf')



""" Plot Fidelities. """
fig, ax = plt.subplots(1,1)
for i, subInd in enumerate(exampleInds):
  # Weighted
  fidW = np.sort(fidWArray[subInd,:])
  ax.plot(fidW, color=colors[i], linestyle='-', 
          label=f'Weighted fidelity, mean: {np.round(np.mean(fidW),3)}')
  # Original
  fidO = np.sort(fidOArray[subInd,:])
  ax.plot(fidO, color=colors[i], linestyle=':', linewidth=1, 
          label=f'Original fidelity, mean: {np.round(np.mean(fidO),3)}')
  # ax.fill_between(list(range(len(fidW))), fidO, fidW, color=colors[i], alpha=0.2)
  

# legend = ax.legend(loc='best', shadow=False)
# legend.get_frame()

ax.set_ylabel('Parcel fidelity')
ax.set_xlabel('Parcels, sorted by fidelity')

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=0.1)
plt.show()

if savePDFs == True:
  fig.savefig(savePathBase + parcelPlotEnd, format='pdf')


""" Plot gain """
fig, ax = plt.subplots(1,1)
for i, subInd in enumerate(exampleInds):
  # Relative fidelity
  fidR = np.sort(fidRArray[subInd,:])*100
  ax.plot(fidR, color=colors[i], linestyle='-', 
          label=f'Fidelity gain, mean: {np.round(np.mean(fidR),3)}')

ax.plot(100*np.ones(fidR.shape, dtype=float), color='black', linestyle='-', linewidth=0.3)  # Set a horizontal line at 100 %.
# legend = ax.legend(loc='best', shadow=False)
# legend.get_frame()

ax.set_ylabel('Relative parcel fidelity (%)')
ax.set_xlabel('Parcels, sorted by gain')

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=0.1)
plt.show()

if savePDFs == True:
  fig.savefig(savePathBase + gainPlotEnd, format='pdf')


