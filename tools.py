def rec2ann(xr, yr, r, R, dim):
    import numpy as np
    rho = float(yr) + r
    phi = float(xr)/dim[1]*2*np.pi + np.pi/2
    ya = rho*np.cos(phi) + R
    xa = rho*np.sin(phi) + R
    return [round(xa), round(ya)]

def ann2rec(xa, ya, r, R, dim):
    import numpy as np
    rho = np.sqrt((xa-R)**2 + (ya-R)**2)
    phi = np.arctan2(xa-R, ya-R) - np.pi/2
    xr = dim[1]*phi*rho / (2*np.pi*rho)
    if xr < 0:
        xr += dim[1]
    yr = rho - r
    return [round(xr), round(yr)]

def find_center(cluster):
    import numpy as np
    return [np.median(cluster[:,0]), np.median(cluster[:,1])]

def find_box(cluster):
    import numpy as np
    cluster = np.asarray(cluster)
    return [min(cluster[:,0]), min(cluster[:,1]), 
            max(cluster[:,0]), max(cluster[:,1])]

def in_hull(points, hull):
    #print "Deciding if points are inside the hull..."
    import matplotlib.path as path
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    edges = []
    codes = []
    prev = hull[0][0]
    next = hull[0][1]
    edges.append(tuple(prev))
    codes.append(path.Path.MOVETO)
    edges.append(tuple(next))
    codes.append(path.Path.LINETO)
    idx = 2
    while idx < len(hull)+1:
        for edge in hull:
            pt1 = edge[0]
            pt2 = edge[1]
            
            if (pt1[0] == next[0] and
                pt1[1] == next[1] and
                (pt2[0] != prev[0] or
                 pt2[1] != prev[1])):
                prev = pt1
                next = pt2
                edges.append(tuple(next))
                codes.append(path.Path.LINETO)
                #print "break on pt2"
                break
            elif (pt2[0] == next[0] and
                  pt2[1] == next[1] and
                  (pt1[0] != prev[0] or
                   pt1[1] != prev[1])):
                prev = pt2
                next = pt1
                edges.append(tuple(next))
                codes.append(path.Path.LINETO)
                #print "break on pt1"
                break
        idx += 1

    edges.append(hull[0][0])
    codes.append(path.Path.CLOSEPOLY)
    polygon = path.Path(edges,codes)
    mask = polygon.contains_points(points)
    if False:
        mask = mask.reshape(4040,80).T
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        patch = patches.PathPatch(polygon, facecolor="orange", lw=2)
        #ax.add_patch(patch)
        ax.imshow(mask)
        ax.set_xlim(0,4040)
        ax.set_ylim(0,80)
        ax.axis("equal")
        ax.invert_yaxis
        plt.show()
    return mask

def find_mask(hull, box):
    import numpy as np
    dim = [box[2]-box[0], box[3]-box[1]]
    mask = np.asarray([[x, y] for x in np.arange(box[0], box[2]) 
                              for y in np.arange(box[1], box[3])])
    mask = np.transpose(in_hull(mask, hull).reshape(dim[0], dim[1]))
    return mask

def unwrap(wrapped):
    import numpy as np
    import copy
    duration = len(wrapped)
    unwrapped = copy.deepcopy(wrapped)
    tmp = np.array(wrapped)
    for idx in range(duration-1):
        if tmp[idx+1] - tmp[idx] > 3840/2:
            tmp = tmp - 3840
            unwrapped[idx+1] = tmp[idx+1]
            for t in np.arange(2,30*3):
                try:
                    if tmp[idx+t] - tmp[idx+1] < -3840/2:
                        tmp[idx+t] += 3840
                except:
                    pass
        else:
            unwrapped[idx+1] = tmp[idx+1]
    return unwrapped

def contains_list(original_list, sublist):
    for item in original_list:
        if (sublist[0] == item[0] and 
            sublist[1] == item[1]):
            return True
    return False

def ConcaveHull(points):
    #print "Finding concave hull..."
    from scipy.spatial import Delaunay
    import numpy as np
    import collections
    import matplotlib.pyplot as plt
    delaunay = Delaunay(points)
    simplices = delaunay.simplices
    edge_list = []
    for simplex in simplices:
        delta = points[simplex]
        perimeter  = np.sqrt((delta[0,0] - delta[1,0])**2 + 
                            (delta[0,1] - delta[1,1])**2)
        perimeter += np.sqrt((delta[1,0] - delta[2,0])**2 + 
                            (delta[1,1] - delta[2,1])**2)
        perimeter += np.sqrt((delta[0,0] - delta[2,0])**2 + 
                            (delta[0,1] - delta[2,1])**2)
        if perimeter < 30:
            delta = delta.tolist()
            delta = [(pt[0],pt[1]) for pt in delta]
            edge = [delta[0], delta[1]]
            edge.sort()
            edge_list.append(tuple(edge))
            edge = [delta[1], delta[2]]
            edge.sort()
            edge_list.append(tuple(edge))
            edge = [delta[2], delta[0]]
            edge.sort()
            edge_list.append(tuple(edge))
    counter = collections.Counter(edge_list)
    hull = np.asarray([key for (key, value) 
                       in counter.items() if value == 1])
    return hull

def remove_padding(data):
    import numpy as np
    for idx, pos in enumerate(data):
        if pos < 100:
            data[idx] = 3940 - pos
        elif pos >= 3940:
            data[idx] = pos - 3940
        else:
            data[idx] -= 100
    data = unwrap(data)
    return np.asarray(data)

def match_histogram(source, template):
    # Credit to ali_m from:
    # http://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    import numpy as np
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    return interp_t_values[bin_idx].reshape(oldshape).astype(np.uint8)
    
# Smooth data using a 4th orderivatives spline smoother.
def spline_smoothing(data, deg=4, tol=14, detect_outliers=True):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-white")
    import matplotlib
    matplotlib.rc("font", family="FreeSans", size=18)
    from scipy.interpolate import UnivariateSpline
    import statsmodels.api as sm
    import warnings
    warnings.filterwarnings("ignore")

    data = np.pad(data, ((150,150),(0,0)), mode="edge")
    dim = data.shape
    time = np.arange(dim[0])/30.
    # Some possible preprocessing  hacks here
    #data = savgol_filter(data, 125, 4, axis=0)
    #for idx in range(dim[1]):
    #    data[:,idx] = sm.nonparametric.lowess(data[:,idx], time, 
    #                                          frac=30.0/dim[0], 
    #                                          return_sorted=False)

    splines = []
    for idx in range(dim[1]):
        x = time
        y = data[:,idx]
        splines.append(UnivariateSpline(x, y, k=deg, s=tol))
    derivatives = np.asarray([[spline.derivatives(t) for spline in splines] for t in time])
    if False:
        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1)
        ax1.plot(derivatives[150:-150,:,0])
        ax2 = fig.add_subplot(2,2,2)
        ax2.plot(derivatives[150:-150,:,1])
        ax3 = fig.add_subplot(2,2,3)
        ax3.plot(derivatives[150:-150,:,2])
        ax4 = fig.add_subplot(2,2,4)
        ax4.plot(derivatives[150:-150,:,3])
        plt.show(block=False)

    if detect_outliers: # weight-based implementation with bbox
        splines = []
        for idx in range(dim[1]):
            x = time
            y = data[:,idx]
            accs = derivatives[:,idx,2]
            weights = []
            for acc in accs:
                if abs(acc) < 2: # outlier detection
                    weights.append(1)
                elif abs(acc) < 3:
                    weights.append(0.1)
                else:
                    weights.append(0)
            weights = np.asarray(weights)
            splines.append(UnivariateSpline(x, y, w=weights, k=deg, s=tol))
        derivatives = np.asarray([[spline.derivatives(t) for spline in splines] for t in time])
        if False:
            fig = plt.figure()
            ax1 = fig.add_subplot(2,2,1)
            ax1.plot(derivatives[150:-150,:,0])
            ax2 = fig.add_subplot(2,2,2)
            ax2.plot(derivatives[150:-150,:,1])
            ax3 = fig.add_subplot(2,2,3)
            ax3.plot(derivatives[150:-150,:,2])
            ax4 = fig.add_subplot(2,2,4)
            ax4.plot(derivatives[150:-150,:,3])
            plt.show(block=False)

        splines = []
        for idx in range(dim[1]):
            x = time
            y = derivatives[:,idx,0]
            jerks = derivatives[:,idx,3]
            weights = []
            for jerk in jerks:
                if abs(jerk) < 3: # outlier detection
                    weights.append(1)
                elif abs(jerk) < 5:
                    weights.append(0.1)
                else:
                    weights.append(0)
            weights = np.asarray(weights)
            splines.append(UnivariateSpline(x, y, w=weights, k=deg, s=tol))
        derivatives = np.asarray([[spline.derivatives(t) for spline in splines] for t in time])
        if False:
            fig = plt.figure()
            ax1 = fig.add_subplot(2,2,1)
            ax1.plot(derivatives[150:-150,:,0])
            ax2 = fig.add_subplot(2,2,2)
            ax2.plot(derivatives[150:-150,:,1])
            ax3 = fig.add_subplot(2,2,3)
            ax3.plot(derivatives[150:-150,:,2])
            ax4 = fig.add_subplot(2,2,4)
            ax4.plot(derivatives[150:-150,:,3])
            plt.show()

    return derivatives[150:-150,:,:]
