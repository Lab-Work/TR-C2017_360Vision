import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.style.use("seaborn-white")
import matplotlib
matplotlib.rc("font", family="FreeSans")
from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_fill_holes
from tools import rec2ann, ann2rec, find_center, find_box, in_hull
import sys

def subtract_background(frame, background):
    background_ = background.astype(np.float)
    frame_ = frame.astype(np.float)
    foreground = background_ - frame_
    foreground = np.sqrt(np.sum(foreground**2,2)).astype(np.uint8)
    foreground = np.logical_not(foreground < 14)
    foreground = binary_erosion(foreground)
    foreground = binary_dilation(foreground)
    foreground = binary_fill_holes(foreground)
    foreground = np.logical_not(foreground)
    frame[foreground] = 0
    return frame

# Parse the command line argument
SRC = sys.argv[1]

idx = 1
l = 0
cap = cv2.VideoCapture(SRC+"edited_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(SRC+"subtracted_video.avi",fourcc, 30.0, (3840,80))
while(cap.isOpened()):
    period = 15*30
    time = 0
    background = cv2.imread(SRC+"backgrounds/background_c%02d.png" % l)
    while(time < period):
        ret, frame = cap.read()
        if ret == False:
            break
        idx += 1
        time += 1
        img = subtract_background(frame, background)
        out.write(img)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    l += 1
    if ret == False:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
