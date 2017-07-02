# Authors: Fangyu Wu (fwu10@illinois.edu)
# Date: Dec 30th, 2016
# Script to extract displacements, velocities, and accelerations from panoramic videos

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.style.use("seaborn-white")
import matplotlib
matplotlib.rc("font", family="FreeSans")
import sys
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import ConvexHull
from scipy.signal import convolve2d
from scipy.io import savemat
from tools import *
import copy


# Parse the command line argument
SRC = sys.argv[1]
tracks = np.load("data/tracks_spl_wrp%s.npy" % SRC[-13:-1])
cap = cv2.VideoCapture(SRC+"subtracted_video.avi")
cap0 = cv2.VideoCapture(SRC+"edited_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("visual/demo%s.avi" % SRC[-13:-1],fourcc,30.0,(960,320))
idx = 0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print frame_count
while (idx<frame_count):
    ret, frame = cap.read()
    ret0, frame0 = cap0.read()
    boxes = tracks[idx,:,:]
    for num, box in enumerate(boxes):
        if not np.isnan(np.min(box)):
            cv2.rectangle(frame0, (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), (0,255,255), 2)
            cv2.putText(frame0, str(num+1), (int(box[2])+10, int(box[3])), 
                        cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255), 2)
            if np.max(box) > 3840:
                box[0] -= 3840.
                box[2] -= 3840.
                cv2.rectangle(frame0, (int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), (0,255,255), 2)
                cv2.putText(frame0, str(num+1), (int(box[2])+10, int(box[3])), 
                            cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255), 2)
    frame0 = np.vstack((frame0[:,0:960],frame0[:,960:1920], 
                        frame0[:,1920:2880],frame0[:,2880:3840]))
    cv2.imshow("Tracking", frame0)
    out.write(frame0)
    idx += 1
    #cv2.imshow("Video",frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
cap.release()
cap0.release()
out.release()
cv2.destroyAllWindows()
