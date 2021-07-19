# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 14:18:38 2021
Plot true and false positives from loaded arrays created in fidGroupAnalysis.py.
@author: rouhinen
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd


# Subject fraction list
percentileList = np.array([0.5])
# percentileList = np.array([0.15, 0.5, 0.85])

savePathBase = "C:\\temp\\fWeighting\\plotDump\\"
parcelPlotEnd = 'TP FP Orig Weighted.pdf'
gainPlotEnd = 'TP FP Relative.pdf'
subjectsOrigWeightEnd = 'TP FP Orig x Weighted scatter.pdf'
savePDFs = False
tightLayout = True

## Load files. fidXArrays created in fidGroupAnalysis.py. There saved with np.save().
resolutions = ['100', '200', '400']

tpWArrays = []
tpOArrays = []
for i, resolution in enumerate(resolutions):
    tpWArrays.append(np.load('C:\\temp\\fWeighting\\numpyArrays\\fidArrays\\schaeferXYZ\\tpWArray.npy'.
                    replace('XYZ', resolution)))
    tpOArrays.append(np.load('C:\\temp\\fWeighting\\numpyArrays\\fidArrays\\schaeferXYZ\\tpOArray.npy'.
                    replace('XYZ', resolution)))


## Create "bins" for X-Axis. 
n_bins = 101
binArray = np.logspace(-2, 0, n_bins-1, endpoint=True)    # Values from 0.01 to 1
binArray = np.concatenate(([0], binArray))  # Add 0 to beginning


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx



"""   Plots   """
# Set global figure parameters, including CorelDraw compatibility (.fonttype)
import matplotlib.pylab as pylab
if tightLayout == True:
  params = {'legend.fontsize':'7',
         'figure.figsize':(1.6, 1),
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

colors = ['black', 'gray', 'salmon', 'goldenrod', 'olivedrab', 'darkcyan']

""" Plot ROC, True positives, false positives. """
# Create dataframes with first resolution.
meansW = pd.DataFrame(np.array([binArray, np.average(tpWArrays[0], axis=0)]).T,columns=['bin',resolutions[0]])
meansO = pd.DataFrame(np.array([binArray, np.average(tpOArrays[0], axis=0)]).T,columns=['bin',resolutions[0]])

for i, resolution in enumerate(resolutions[1:]):
  meansW.insert(i+2, resolution, np.array(np.average(tpWArrays[i+1], axis=0)))
  meansO.insert(i+2, resolution, np.array(np.average(tpOArrays[i+1], axis=0)))

fig, ax = plt.subplots(1,1)

for i, resolution in enumerate(resolutions):
  # Weighted
  ax.plot(meansW.iloc[:,0], meansW.iloc[:,i+1], color=colors[i], linestyle='-', label='Weighted ' + resolution)
  # Original
  ax.plot(meansO.iloc[:,0], meansO.iloc[:,i+1], color=colors[i], linestyle=':', linewidth=1, label='Original ' + resolution)

legend = ax.legend(loc='right', shadow=False)
legend.get_frame()

# ax.fill_between(meansW.iloc[:,0], meansW.iloc[:,1]-stdsW.iloc[:,0], meansW.iloc[:,1]+stdsW.iloc[:,0], color='black', alpha=0.5)
# ax.fill_between(meansO.iloc[:,0], meansO.iloc[:,1]-stdsO.iloc[:,0], meansO.iloc[:,1]+stdsO.iloc[:,0], color='black', alpha=0.3)
ax.set_ylabel('True positive rate')
ax.set_xlabel('False positive rate')

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)

plt.yticks(np.array([0., 0.5, 1]))

plt.tight_layout(pad=0.1)
plt.show()

if savePDFs == True:
  fig.savefig(savePathBase + 'True false positive rates ROC.pdf', format='pdf')




""" Plot relative ROC gain, True positives/True positives by false positives bins. """
# Skip first and last bin because division by zero
meansRR = np.mean(100*(tpWArrays[0][:, 1:-1] / tpOArrays[0][:, 1:-1]), axis=0)
meansRR = pd.DataFrame(np.array([binArray[1:-1], meansRR]).T, columns=['FP bins','TP-relative' + resolutions[0]])

for i, resolution in enumerate(resolutions[1:]):
  meansN = np.mean(100*(tpWArrays[i+1][:, 1:-1] / tpOArrays[i+1][:, 1:-1]), axis=0)
  meansRR.insert(i+2, resolution, meansN)

fig, ax = plt.subplots(1,1)

for i, resolution in enumerate(resolutions):
  ax.plot(meansRR.iloc[:,0], meansRR.iloc[:,i+1], color=colors[i], linestyle='-', 
        label='Relative TPR ' + resolution)

ax.plot([0, 1], [100, 100], color='black', linestyle='-', linewidth=0.3)  # Set a horizontal line at 100 %. [X X] [Y, Y]

legend = ax.legend(loc='best', shadow=False)
legend.get_frame()

# ax.fill_between(meansRR.iloc[:,0], meansRR.iloc[:,1]-stdsRR.iloc[:,1],
#                 meansRR.iloc[:,1]+stdsRR.iloc[:,1], color='black', alpha=0.5)
ax.set_ylabel('Relative TPR (%)')
ax.set_xlabel('False positive rate')

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=0.1)
plt.ylim(90, 140)
plt.show()

if savePDFs == True:
  fig.savefig(savePathBase + 'True false positive rates ROC relative.pdf', format='pdf')




# """ Scatter plot weighted and original ROC at FPR 0.15. """
# fig, ax = plt.subplots(1,1)
# ax.scatter(np.ravel(tpOArray), np.ravel(tpWArray), c='red', alpha=0.2, s=4)  ## X, Y.
# ax.scatter(np.average(tpOArrays[index], axis=0), np.average(tpWArrays[index], axis=0), c='black', alpha=0.5, s=10)  ## X, Y.
# ax.plot([0,1], [0,1], color='black')
# # ax.set_title('PLV')
# ax.set_xlabel('TPR, Original inv op')   
# ax.set_ylabel('TPR, Weighted inv op')   

# plt.ylim(0, 1)
# plt.xlim(0, 1)

# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(True)
# ax.spines['left'].set_visible(True)
# ax.spines['right'].set_visible(False)

# plt.tight_layout(pad=0.1)
# plt.show()

# if savePDFs == True:
#   fig.savefig(savePathBase + 'True positive rates Orig x Weighted scatter by parcel.pdf', format='pdf')


