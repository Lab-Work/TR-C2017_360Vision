import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use("seaborn-white")
import matplotlib
matplotlib.rc("font", family="FreeSans", size=18)
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.io import savemat
import csv
from tools import *

SRC = sys.argv[1]

print "Processing "+SRC+"..."
tracks = np.load(SRC+"tracks.npy")
dim = tracks.shape
boxes = tracks[0,:,:]
with open("./orders.config", 'r') as file:
    reader = csv.reader(file)
    for line in reader:
        if line[0] == SRC[:-1]:
            order = [int(item) for item in line[1:]]
            break

for idx in range(dim[1]):
    tracks[:,idx,0] = unwrap(tracks[:,idx,0])
    tracks[:,idx,2] = unwrap(tracks[:,idx,2])
tracks_dup1 = tracks.copy()
tracks_dup2 = tracks.copy()
tracks_dup1[:,:,0].sort(axis=1)
for idx in range(dim[1]):
    for idx1, x1 in enumerate(tracks_dup1[0,:,0]):
        if tracks[0,idx,0] == x1:
            tracks_dup2[:,idx1,:] = tracks[:,idx,:]
tracks = tracks_dup2.copy()

init_boxes = tracks[0,:,:]
tracks_aligned = np.empty((dim[0], 25, 4))
tracks_aligned[:] = np.NAN
tracks_aligned_wrapped = np.empty((dim[0], 25, 4))
for idx in range(dim[1]):
    aligned_idx = order[idx]-1
    tracks_aligned[:,aligned_idx,0] = tracks[:,idx,0]
    tracks_aligned[:,aligned_idx,1] = tracks[:,idx,1]
    tracks_aligned[:,aligned_idx,2] = tracks_aligned[:,aligned_idx,0]+(init_boxes[idx,2]-init_boxes[idx,0])
    tracks_aligned[:,aligned_idx,3] = tracks_aligned[:,aligned_idx,1]+(init_boxes[idx,3]-init_boxes[idx,1])
    tracks_aligned_wrapped[:,aligned_idx,0] = tracks[:,idx,0] % 3840.
    tracks_aligned_wrapped[:,aligned_idx,1] = tracks[:,idx,1]
    tracks_aligned_wrapped[:,aligned_idx,2] = tracks_aligned_wrapped[:,aligned_idx,0]+(init_boxes[idx,2]-init_boxes[idx,0])
    tracks_aligned_wrapped[:,aligned_idx,3] = tracks_aligned_wrapped[:,aligned_idx,1]+(init_boxes[idx,3]-init_boxes[idx,1])
tracks_aligned = tracks_aligned/3840.0*260.0
tracks_aligned[:,:,0] *= -1
tracks_aligned[:,:,2] *= -1

init_boxes = tracks_aligned[0,:,:]
derivatives_x = spline_smoothing(tracks_aligned[:,:,0].copy())
derivatives_y = spline_smoothing(tracks_aligned[:,:,1].copy())
tracks_spline = np.empty((dim[0], 25, 4))
tracks_spline_wrapped = np.empty((dim[0], 25, 4))
for idx in range(25):
    tracks_spline[:,idx,0] = derivatives_x[:,idx,0]
    tracks_spline[:,idx,1] = derivatives_y[:,idx,0]
    tracks_spline[:,idx,2] = tracks_spline[:,idx,0]+(init_boxes[idx,2]-init_boxes[idx,0])
    tracks_spline[:,idx,3] = tracks_spline[:,idx,1]+(init_boxes[idx,3]-init_boxes[idx,1])
    tracks_spline_wrapped[:,idx,0] = (derivatives_x[:,idx,0]/260.*3840.*(-1)) % 3840.
    tracks_spline_wrapped[:,idx,1] = derivatives_y[:,idx,0]/260.*3840.
    tracks_spline_wrapped[:,idx,2] = tracks_spline_wrapped[:,idx,0]+(init_boxes[idx,2]-init_boxes[idx,0])/260.*3840.*(-1)
    tracks_spline_wrapped[:,idx,3] = tracks_spline_wrapped[:,idx,1]+(init_boxes[idx,3]-init_boxes[idx,1])/260.*3840.

if True:
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ax1.plot(tracks_aligned_wrapped[:,:,0],'b')
    ax1.plot(tracks_spline_wrapped[:,:,0],'r')
    ax2.plot(tracks_aligned_wrapped[:,:,1],'b')
    ax2.plot(tracks_spline_wrapped[:,:,1],'r')
    plt.show()

np.save("./data/tracks_ali%s.npy" % SRC[-13:-1], tracks_aligned)
savemat("./data/tracks_ali%s.mat" % SRC[-13:-1], mdict={"tracks_aligned": tracks_aligned})
np.save("./data/tracks_ali_wrp%s.npy" % SRC[-13:-1], tracks_aligned_wrapped)
savemat("./data/tracks_ali_wrp%s.mat" % SRC[-13:-1], mdict={"tracks_aligned_wrapped": tracks_aligned_wrapped})
np.save("./data/tracks_spl%s.npy" % SRC[-13:-1], tracks_spline)
savemat("./data/tracks_spl%s.mat" % SRC[-13:-1], mdict={"tracks_spline": tracks_spline})
np.save("./data/tracks_spl_wrp%s.npy" % SRC[-13:-1], tracks_spline_wrapped)
savemat("./data/tracks_spl_wrp%s.mat" % SRC[-13:-1], mdict={"tracks_spline_wrapped": tracks_spline_wrapped})
#np.save("./data/dynamics%s.npy" % SRC[-13:-1], derivatives_x[:,:,1:])
#savemat("./data/dynamics%s.mat" % SRC[-13:-1], mdict={"dynamics": derivatives_x[:,:,1:]})

issues = (np.count_nonzero(~np.isnan(tracks_aligned[0,:,0])) - 
          np.count_nonzero(~np.isnan(tracks_spline[0,:,0])))
print "Issues:", issues

