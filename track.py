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
from copy import deepcopy

# Initialize the templates using background subtraction
# and Delaunay triangulation.
def init_templates(init_frame, background, THRESH):
    init_frame = match_histogram(init_frame, background)
    background_ = background.astype(np.float)
    init_frame_ = init_frame.astype(np.float)
    foreground = background_ - init_frame_
    foreground = np.sqrt(np.sum(foreground**2,2)).astype(np.uint8)
    ret, foreground_ = cv2.threshold(foreground,48,255,cv2.THRESH_BINARY)

    # Use DBSCAN for image segmentation
    dim = background.shape
    r = dim[1]/(2*np.pi)
    R = r + dim[0]
    dbscan_mask = foreground > THRESH
    dbscan_frame = np.column_stack(np.where(dbscan_mask))
    for num, pixel in enumerate(dbscan_frame):
        xr = pixel[1]
        yr = pixel[0]
        dbscan_frame[num, 0:2] = rec2ann(xr, yr, r, R, dim)
    dbscan = DBSCAN(eps=2.5, min_samples=15)
    dbscan.fit(dbscan_frame)
    labels = dbscan.labels_
    labels_unique = np.unique(labels)

    hulls = []
    clusters = []
    boxes = []
    for label in labels_unique:
        if label != -1:
            mask = labels==label
            cluster = dbscan_frame[mask]
            center = find_center(cluster)
            if len(cluster) > 500:
                for idx, point in enumerate(cluster):
                    xa = point[0]
                    ya = point[1]
                    cluster[idx, 0:2] = ann2rec(xa, ya, r, R, dim)
                if max(cluster[:,0]) - min(cluster[:,0]) > 1920:
                    for idx, point in enumerate(cluster):
                        if cluster[idx, 0] < 1920:
                            cluster[idx, 0] += 3840
                # Find convex hull of each cluster
                #hull = ConvexHull(cluster)
                #hull = [[x+100, y] for x, y in zip(cluster[hull.vertices,0], 
                #                                cluster[hull.vertices,1])]
                # Find concave hull of each cluster
                hull = ConcaveHull(cluster)
                pad_hull = []
                for edge in hull:
                    pad_hull.append([[edge[0][0]+100, edge[0][1]],
                                     [edge[1][0]+100, edge[1][1]]])
                box = find_box(cluster)
                boxes.append(box)
                clusters.append(cluster)
                hulls.append(pad_hull)

    for hull in hulls:
        vertices = np.asarray([edge[0] for edge in hull])
        if np.max(vertices[:,0]) > 3940:
            duplicate = []
            for edge in hull:
                duplicate.append([[edge[0][0]-3840, edge[0][1]],
                                  [edge[1][0]-3840, edge[1][1]]])
            hulls.append(duplicate)
    
    hulls = np.asarray(hulls)
    roi = np.hstack((init_frame[:, -100:], init_frame, init_frame[:, 0:100]))
    frame_coord = np.asarray([(x, y) for x in range(4040) for y in range(80)])
    mask = np.zeros((80, 4040))
    for hull in hulls:
        mask_ = np.transpose(in_hull(frame_coord, hull).reshape(4040, 80))
        mask = np.logical_or(mask, mask_)
    roi[np.logical_not(mask)] = 0
    for box in boxes:
        cv2.rectangle(roi,(box[0]+100,box[1]),(box[2]+100,box[3]),(255,0,0))
    cv2.imwrite(SRC+"ROI.png", roi)
    cv2.imshow("ROI.png", roi)
    cv2.waitKey(0)
    templates = []
    for box in boxes:
        template = roi[box[1]:box[3],box[0]+100:box[2]+100]
        templates.append(template)
        #cv2.imshow("Template", template)
        #cv2.waitKey(1000)
    return templates, boxes

# Track the templates from the previous frame in the next frame 
# and return the updated templates.
def track_templates(frame, templates, boxes, radius):
    padded_frame_bgr = np.hstack((frame[:,-100:], frame, frame[:,:100]))
    padded_frame = cv2.cvtColor(padded_frame_bgr, cv2.COLOR_BGR2GRAY)
    for idx in range(len(boxes)):
        boxes[idx][0] += 100
        boxes[idx][2] += 100
    templates_ = []
    boxes_ = []
    for template, box in zip(templates, boxes):
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        neighbor = padded_frame[int(box[1]-radius[1]):int(box[3]+radius[1]),
                                int(box[0]-radius[0]):int(box[2]+radius[0])]

        box_ = []
        matches = cv2.matchTemplate(neighbor, template, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matches)
        x = max_loc[0]
        y = max_loc[1]
        peak_locs_ = np.asarray([[x-1,y-1],[x  ,y-1],[x+1,y-1],
                                 [x-1,y  ],[x  ,y  ],[x+1,y  ],
                                 [x-1,y+1],[x  ,y+1],[x+1,y+1]])
        peak_vals = []
        peak_locs = []
        for loc in peak_locs_:
            try:
                peak_vals.append(matches[loc[1],loc[0]])
                peak_locs.append(loc)
            except:
                continue
        peak_locs = np.asarray(peak_locs)
        peak_vals = np.asarray(peak_vals)
        peak_vals += abs(np.min(peak_vals))
        peak_vals /= (np.std(peak_vals)+1)
        peak_vals **= 1
        best_loc = np.average(peak_locs,axis=0,weights=peak_vals)

        if False:
            fig = plt.figure()
            ax1 = fig.add_subplot(2,2,1)
            ax1.imshow(neighbor, cmap="gray")
            ax2 = fig.add_subplot(2,2,2)
            ax2.imshow(template, cmap="gray")
            ax3 = fig.add_subplot(2,2,3)
            ax3.imshow(matches, interpolation="nearest", cmap="plasma")
            ax3.plot(peak_locs[:,0], peak_locs[:,1], '+')
            ax3.plot(best_loc[0], best_loc[1], '*')
            ax4 = fig.add_subplot(2,2,4)
            ax4.imshow(matches, interpolation="nearest", cmap="plasma")
            plt.show()

        base = best_loc
        #base = [x, y]
        if base[0]+box[0]-radius[0] >= 100:
            base[0] = base[0]+box[0]-radius[0]
        else:
            base[0] = 4039-(base[0]+box[0]-radius[0])
        base[1] = base[1]+box[1]-radius[1]
        box_.append(base[0]-100)
        box_.append(base[1])
        box_.append(base[0]+(box[2]-box[0])-100)
        box_.append(base[1]+(box[3]-box[1]))
        boxes_.append(box_)
        template_ = padded_frame_bgr[int(box_[1]):int(box_[3]),
                                     int(box_[0]):int(box_[2])]
        templates_.append(template_)
    return templates_, boxes_

# Parse the command line argument
SRC = sys.argv[1]
THRESH = int(sys.argv[2])

init_frame = cv2.imread(SRC+"init_frame.png")
background = cv2.imread(SRC+"background.png")
templates, boxes = init_templates(init_frame, background, THRESH)

tracks = []
cap = cv2.VideoCapture(SRC+"subtracted_video.avi")
idx = 0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while (idx<frame_count):
    idx += 1
    tracks.append(boxes)
    ret, frame = cap.read()
    templates_, boxes = track_templates(frame, templates, deepcopy(boxes), [10,2])
    padded_frame = np.hstack((frame[:,-100:], frame, frame[:,:100]))
    for box in boxes:
        cv2.rectangle(padded_frame, (int(box[0]+100), int(box[1])), 
                      (int(box[2]+100), int(box[3])), (255,0,0), 1)
    cv2.line(padded_frame, (100, 0), (100, 80), (255,0,0), 1)
    cv2.line(padded_frame, (3940, 0), (3940, 80), (0,0,255), 1)
    padded_frame = np.vstack((padded_frame[:,:1010],    padded_frame[:,1010:2020], 
                              padded_frame[:,2020:3030],padded_frame[:,3030:4040]))
    cv2.putText(padded_frame, str(idx), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.imshow("Tracking", padded_frame)
    #cv2.imshow("Video",frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
np.save(SRC+"tracks.npy", tracks)
#savemat("tracks.mat", mdict={"tracks": tracks})
