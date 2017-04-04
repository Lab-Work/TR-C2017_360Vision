# Created by Fangyu Wu
# June 15th, 2016
import numpy as np
import cv2
import sys
import os

def get_background(cache):
    background = []
    for i, m in enumerate(np.transpose(cache)):
        status = i/num_pixels * 100
        sys.stdout.write("Analyzing cached backgrounds %.2f%%...\r" % status)
        sys.stdout.flush()
        median = np.uint8(np.median(m[np.nonzero(m)]))        
        background.append(median)
    background = np.asarray(background)
    background = np.uint8(background.reshape(img_shape[:2]))
    return background

# Parse the command line argument
SRC = sys.argv[1]
MEM = np.inf

sys.stdout.write("Reading %sedited_video.mp4 ...\n" % SRC)
cap = cv2.VideoCapture(SRC+"edited_video.mp4")
ret, rec1 = cap.read()
prvs = cv2.cvtColor(rec1,cv2.COLOR_BGR2GRAY)
img_shape = rec1.shape
memory_b = []
memory_g = []
memory_r = []
idx = 1
sys.stdout.write("Processing frame %d...\r" % idx)

while(cap.isOpened()):
    period = 15*30
    cache_b = []
    cache_g = []
    cache_r = []
    time = 0
    while(time < period):
        ret, rec2 = cap.read()
        if ret == False:
            break
        next = cv2.cvtColor(rec2,cv2.COLOR_BGR2GRAY)
        time += 1
        idx += 1
        sys.stdout.write("Processing frame %d...\r" % idx)
        sys.stdout.flush()
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 
                                            3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mask = mag > 1.0
        rec_b, rec_g, rec_r = cv2.split(rec2)
        rec_b[mask] = 0
        rec_g[mask] = 0
        rec_r[mask] = 0
        cache_b.append(rec_b.flatten())
        cache_g.append(rec_g.flatten())
        cache_r.append(rec_r.flatten())
        prvs = next
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    memory_b.append(cache_b)
    memory_g.append(cache_g)
    memory_r.append(cache_r)
    if ret == False:
        break
cap.release()
cv2.destroyAllWindows()

sys.stdout.write("\nCaching backgrounds...\n")
memory_b = np.asarray(memory_b)
memory_g = np.asarray(memory_g)
memory_r = np.asarray(memory_r)
num_pixels = float(img_shape[0]*img_shape[1])
for l, cache_b, cache_g, cache_r in zip(range(len(memory_g)), memory_b, memory_g, memory_r):
    sys.stdout.write("Processing cache %02d...\n" % l)
    sys.stdout.write('\n')
    background_b = get_background(cache_b)
    background_g = get_background(cache_g)
    background_r = get_background(cache_r)
    background = cv2.merge((background_b, background_g, background_r))
    print SRC+"Saving background_c%02d.png..." % l
    cv2.imwrite(SRC+"backgrounds/background_c%02d.png" % l, background)
