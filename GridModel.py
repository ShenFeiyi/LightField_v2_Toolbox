# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
from scipy.ndimage import filters
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from queue import PriorityQueue as PQ

def BuildGridModel(stp_arr, **kwarg):
    """Build grid model from stop image

    NOTE:
        May get warning:
            OptimizeWarning: Covariance of the parameters could not be estimated
        This is because when estimating lines, there are only two points feeded.
        To ignore ALL warnings, set ignoreWarning = True

    Args:
        stp_arr (numpy.ndarray): A numpy array image with small stops. (other image file that is able to convert to numpy.ndarray is alse acceptable)
        approxSpacing (int, default=200): Approximate spacing between stops.
        filterDiskMult (float, default=1/6): Filter disk radius, relative to spacing of stops.
        imageBoundary (int, default=50): Image boundary width, help remove local maximums that are close to boundary (not real centers).
        debugDisplay (bool, default=False): Debug display, to show whether centers are estimated correctly.
        ignoreWarning (bool, default=False): Whether to ignore warnings.

    Returns:
        invM (numpy.ndarray): 2x2 transformation matrix. (rotation & shear)
        allPeaks (list): A list of peaks detected. (exclude the ones on the edge)
        peakGroups (dict): Two groups, 'H' & 'V'
            peakGroups['H'] (list): Group peaks in horizontal direction, [[p11,p12,...],[],...]
            peakGroups['V'] (list): Group peaks in vertical direction, [[p11,p21,...],[],...]
    """
    approxSpacing = kwarg['approxSpacing'] if 'approxSpacing' in kwarg else 200 # px
    filterDiskMult = kwarg['filterDiskMult'] if 'filterDiskMult' in kwarg else 1/6
    imageBoundary = kwarg['imageBoundary'] if 'imageBoundary' in kwarg else 50 # px
    debugDisplay = kwarg['debugDisplay'] if 'debugDisplay' in kwarg else False
    ignoreWarning = kwarg['ignoreWarning'] if 'ignoreWarning' in kwarg else False

    if ignoreWarning:
        import warnings
        warnings.filterwarnings('ignore')

    def line(x, k=1, b=0):
        # estimated vertical lines can't be perfectly vertical.
        return k*x+b

    try:
        assert isinstance(stp_arr,np.ndarray)
    except AssertionError:
        stp_arr = np.array(stp_arr)

    if len(stp_arr.shape) == 3:
        # RGB 2 gray
        stp_arr = cv.cvtColor(stp_arr, cv.COLOR_BGR2GRAY)

    print('Generating disk mask...') # a white circular mask in the center
    h = np.zeros(stp_arr.shape, dtype='float64') # 0 background
    squareSide = 2*approxSpacing*filterDiskMult+1
    hr = np.zeros((int(squareSide),int(squareSide))) # center part ready to draw a circle
    for i in range(int(squareSide)):
        for j in range(int(squareSide)):
            x2 = (i-approxSpacing*filterDiskMult)**2
            y2 = (j-approxSpacing*filterDiskMult)**2
            r2 = (approxSpacing*filterDiskMult)**2
            if x2 + y2 <= r2:
                hr[i,j] = 1
    diskOffset = np.zeros(2,dtype=int) # x & y offset
    diskOffset[0] = int(round((stp_arr.shape[0]-int(squareSide))/2))
    diskOffset[1] = int(round((stp_arr.shape[1]-int(squareSide))/2))
    h[diskOffset[0]:diskOffset[0]+int(squareSide), diskOffset[1]:diskOffset[1]+int(squareSide)] = hr

    print('Filtering...') # convolve mask and image => cone tips are centers
    h = np.fft.fft2(h)
    stp_arr = np.fft.fft2(stp_arr)
    stp_arr *= h
    stp_arr = np.fft.ifftshift(np.fft.ifft2(stp_arr))
    stp_arr = np.abs(stp_arr)
    stp_arr = (stp_arr-stp_arr.min()) / (stp_arr.max()-stp_arr.min())

    print('Finding peaks...') # local maximum coordinate
    neighborhoodSize = approxSpacing/2
    localMax = filters.maximum_filter(stp_arr, neighborhoodSize)
    peaks = (localMax == stp_arr) # true & false map
    boundary = imageBoundary
    realPeaks = []
    for i in range(boundary,peaks.shape[0]-boundary):
        for j in range(boundary,peaks.shape[1]-boundary):
            if peaks[i,j]:
                realPeaks.append((i,j))

    print('Fitting grid lines...')
    # vertical distance between first one in one line and last one in another line
    # should be larger than groupSpacing
    groupSpacing = approxSpacing/2

    # hori lines
    peakGroupsH = [[realPeaks[0]]]
    cntGroup = 0
    for i in range(len(realPeaks)-1):
        curPeak = realPeaks[i]
        nextPeak = realPeaks[i+1]
        if abs(curPeak[0]-nextPeak[0]) < groupSpacing:
            peakGroupsH[cntGroup].append(nextPeak)
        else:
            cntGroup += 1
            peakGroupsH.append([nextPeak])

    # sort hori lines
    # sort points from left to right (y)
    # sort rows from top to bottom (x)
    qH = PQ()
    while not peakGroupsH == []:
        group = peakGroupsH.pop()
        xs = []
        newGroup = []
        qHpoint = PQ()
        for p in group:
            xs.append(p[0])
            qHpoint.put((p[1], p))
        while not qHpoint.empty():
            _, point = qHpoint.get()
            newGroup.append(point)
        qH.put((np.array(xs).mean(), newGroup))
    while not qH.empty():
        _, newGroup = qH.get()
        peakGroupsH.append(newGroup)

    HLineParams = []
    for group in peakGroupsH:
        if len(group) > 1:
            xs, ys = [], []
            for peak in group:
                xs.append(peak[1])
                ys.append(peak[0])
            # fit data to designated function
            # `popt`, optimal values for the parameters
            # sum of the squared residuals of ``f(xdata, *popt) - ydata`` is minimized
            # `pcov`, estimated covariance of popt
            popt, pcov = curve_fit(line, xs, ys)
            HLineParams.append([popt[0],popt[1]])

    # vert lines
    # a more general method to distinguish groups
    peakGroupsV = [[]]
    cntGroup = 0
    realPeaks_cp = realPeaks.copy() # in case passing a pointer rather than a copy of original content
    while True:
        if not realPeaks_cp == []:
            curPeak = realPeaks_cp.pop(0) # get whatever first one
            peakGroupsV[cntGroup].append(curPeak)
            # do not make change to realPeaks_cp while looping it
            for peak in realPeaks_cp:
                if abs(curPeak[1]-peak[1]) < groupSpacing:
                    peakGroupsV[cntGroup].append(peak)
            # delete what we've got from realPeaks_cp
            for peak in peakGroupsV[cntGroup]:
                if peak in realPeaks_cp:
                    realPeaks_cp.pop(realPeaks_cp.index(peak))
            cntGroup += 1
            peakGroupsV.append([])
        else:
            del realPeaks_cp
            break
    peakGroupsV.pop() # delete last empty list

    # sort vert lines
    # sort points from top to bottom (x)
    # sort cols from left to right (y)
    qV = PQ()
    while not peakGroupsV == []:
        group = peakGroupsV.pop()
        ys = []
        qVpoint = PQ()
        newGroup = []
        for p in group:
            ys.append(p[1])
            qVpoint.put((p[0], p))
        while not qVpoint.empty():
            _, point = qVpoint.get()
            newGroup.append(point)
        qV.put((np.array(ys).mean(), newGroup))
    while not qV.empty():
        _, newGroup = qV.get()
        peakGroupsV.append(newGroup)

    VLineParams = []
    for group in peakGroupsV:
        if len(group) > 1:
            xs, ys = [], []
            for peak in group:
                xs.append(peak[1])
                ys.append(peak[0])
            popt, pcov = curve_fit(line, xs, ys)
            VLineParams.append([popt[0],popt[1]])

    # extract slopes, k
    hk, vk = [], []
    for hp in HLineParams:
        hk.append(hp[0])
    for vp in VLineParams:
        vk.append(vp[0])
    hk = np.array(hk)
    vk = np.array(vk)
    ##print(hk.mean(), hk.var())
    ##print(vk.mean(), vk.var())

    print('Fitting transformation matrix...')
    # matrix
    # H1 = a b * 1 = a ; V1 = a b * 0 = b
    # H2   c d   0   c   V2   c d   1   d
    vecH = np.array((1/np.sqrt(hk.mean()**2+1),hk.mean()/np.sqrt(hk.mean()**2+1)))
    vecV = np.array((1/np.sqrt(vk.mean()**2+1),vk.mean()/np.sqrt(vk.mean()**2+1)))
    if vk.mean() < 0:
        vecV = -vecV
    a, b = vecH[0], vecV[0]
    c, d = vecH[1], vecV[1]
    M = np.array([[a,b],[c,d]])
    invM = np.linalg.inv(M)

    if debugDisplay:
        plt.figure('Debug Display')
        plt.imshow(stp_arr,cmap='gray')
        for ipeaks, peaks in enumerate(peakGroupsH):
            for p in peaks:
                plt.plot(p[1],p[0],'rx')
                plt.text(p[1]-50,p[0]+50,str(ipeaks),color='#22FF22')
        for ipeaks, peaks in enumerate(peakGroupsV):
            for p in peaks:
                plt.plot(p[1],p[0],'rx')
                plt.text(p[1]+30,p[0]-30,str(ipeaks),color='#FFFF00')
        for hp in HLineParams:
            k, b = hp
            x = np.array([0, stp_arr.shape[1]])
            plt.plot(x, k*x+b, 'g')
        for vp in VLineParams:
            k, b = vp
            y = np.array([0, stp_arr.shape[0]])
            plt.plot((y-b)/k, y, 'y')
        plt.show()

    return invM, realPeaks, {'H':peakGroupsH, 'V':peakGroupsV}

def subimageFourCorners(invM, peakGroups):
    """Convert image centers into subimage corners

    Args:
        invM (numpy.ndarray): 2x2 transformation matrix. (rotation & shear)
        peakGroups (dict): Two groups, 'H' & 'V'
            peakGroups['H'] (list): Group peaks in horizontal direction, [[p11,p12,...],[],...]
            peakGroups['V'] (list): Group peaks in vertical direction, [[p11,p21,...],[],...]

    Returns:
        cornerGroups (dict): Similar structure to `peakGroups`
    """
    print('Finding subimage corners...')
    # direction vectors
    M = np.linalg.inv(invM)
    vecH, vecV = np.array([M[0,0], M[1,0]]), np.array([M[0,1], M[1,1]])

    # mean distance in each direction
    H = peakGroups['H']
    V = peakGroups['V']
    disH = []
    disV = []
    for row in H:
        for ipeak in range(len(row)-1):
            curPeak = row[ipeak]
            nextPeak = row[ipeak+1]
            disH.append(np.sqrt((nextPeak[0]-curPeak[0])**2+(nextPeak[1]-curPeak[1])**2))
    for col in V:
        for ipeak in range(len(col)-1):
            curPeak = col[ipeak]
            nextPeak = col[ipeak+1]
            disV.append(np.sqrt((nextPeak[0]-curPeak[0])**2+(nextPeak[1]-curPeak[1])**2))
    Hbar = np.array(disH).mean()
    Vbar = np.array(disV).mean()
    Hstep = Hbar/2
    Vstep = Vbar/2
    Hy, Hx = Hstep * vecH
    Vy, Vx = Vstep * vecV
    Hx, Hy, Vx, Vy = int(Hx), int(Hy), int(Vx), int(Vy)

    # generate subimage corner points
    # horizontal corners
    cornersH = []
    for irow, row in enumerate(H):
        cornersH.append([])
        for p in row:
            x, y = p
            x_up_left = x - (Hx+Vx)
            y_up_left = y - (Hy+Vy)
            cornersH[irow].append((x_up_left,y_up_left))
            if p == row[-1]:
                x_up_right = x - (Hx+Vx)
                y_up_right = y + (Hy+Vy)
                cornersH[irow].append((x_up_right,y_up_right))
        if row == H[-1]:
            cornersH.append([])
            for p in row:
                x, y = p
                x_down_left = x + (Hx+Vx)
                y_down_left = y - (Hy+Vy)
                cornersH[irow+1].append((x_down_left,y_down_left))
                if p == row[-1]:
                    x_down_right = x + (Hx+Vx)
                    y_down_right = y + (Hy+Vy)
                    cornersH[irow+1].append((x_down_right,y_down_right))

    # vertical corners
    cornersV = []
    for icol, col in enumerate(V):
        cornersV.append([])
        for p in col:
            x, y = p
            x_up_left = x - (Hx+Vx)
            y_up_left = y - (Hy+Vy)
            cornersV[icol].append((x_up_left,y_up_left))
            if p == col[-1]:
                x_down_left = x + (Hx+Vx)
                y_down_left = y - (Hy+Vy)
                cornersV[icol].append((x_down_left,y_down_left))
        if col == V[-1]:
            cornersV.append([])
            for p in col:
                x, y = p
                x_up_right = x - (Hx+Vx)
                y_up_right = y + (Hy+Vy)
                cornersV[icol+1].append((x_up_right,y_up_right))
                if p == col[-1]:
                    x_down_right = x + (Hx+Vx)
                    y_down_right = y + (Hy+Vy)
                    cornersV[icol+1].append((x_down_right,y_down_right))

    return {'H':cornersH, 'V':cornersV}

def segmentImage(image, invM, peakGroups):
    """Crop raw image into subimages

    Args:
        image (numpy.ndarray): Image to be cropped. (Height, Width, Color)
        invM (numpy.ndarray): 2x2 transformation matrix. (rotation & shear)
        peakGroups (dict): Two groups, 'H' & 'V'
            peakGroups['H'] (list): Group peaks in horizontal direction, [[p11,p12,...],[],...]
            peakGroups['V'] (list): Group peaks in vertical direction, [[p11,p21,...],[],...]

    Returns:
        subimages (dict): Subimages. {#row:{#col:image (Width,Height,Color)}}
        anchors (dict): Five points in each subimage. {#row:{#col:{points}}}
            points = {'center':p0, 'upperLeft':p1, 'upperRight':p2, 'lowerRight':p3, 'lowerLeft':p4}
    """
    cornerGroups = subimageFourCorners(invM, peakGroups)
    CGH = cornerGroups['H']
    CGV = cornerGroups['V']
    subimages = {}
    anchors = {}

    print('Extracting subimages...')
    for irow, row in enumerate(CGH):
        for icol, point in enumerate(row):
            try:
                p1 = CGH[irow][icol]
                p2 = CGH[irow][icol+1]
                p3 = CGH[irow+1][icol+1]
                p4 = CGH[irow+1][icol]
                # may have different number of points in each row
                # but assuming MLA's rotation angle is small
                # points in each row & col are the same
            except IndexError:
                continue

            if not irow in anchors:
                anchors[irow] = {}
            anchors[irow][icol] = {}
            anchors[irow][icol]['center'] = peakGroups['H'][irow][icol]
            anchors[irow][icol]['upperLeft'] = p1
            anchors[irow][icol]['upperRight'] = p2
            anchors[irow][icol]['lowerRight'] = p3
            anchors[irow][icol]['lowerLeft'] = p4 # (y,x)
            if not irow in subimages:
                subimages[irow] = {}

            p1 = [p1[1],p1[0]]
            p2 = [p2[1],p2[0]]
            p3 = [p3[1],p3[0]]
            p4 = [p4[1],p4[0]]
            pts = np.array([p1,p2,p3,p4])
            x, y, h, w = cv.boundingRect(pts)
            cropped = image[y:y+w, x:x+h].copy()
            pts -= pts.min(axis=0)
            mask = np.zeros(cropped.shape, dtype='uint8')
            cv.drawContours(mask, [pts], -1, (255,255,255), -1, cv.LINE_AA)
            subimage = cv.bitwise_and(cropped, cropped, mask=mask)

            subimages[irow][icol] = subimage

    return subimages, anchors

################################################################
if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, default='/Volumes/GoogleDrive/Other computers/Lab/Feiyi_GoogleDrive/Feiyi_images', help='find images from this folder')
    parser.add_argument('-i', '--index', type=int, default=870, help='stop image file index', required=True)
    params = parser.parse_args()
    params = vars(params)

    root = params['root']

    stp02 = os.path.join(root, str(params['index']).zfill(6)+'.pgm')
    target = os.path.join(root, str(params['index']-1).zfill(6)+'.pgm')

    stp_img = cv.imread(stp02,0) # gray scale
    stp_arr = stp_img

    ## USAGE ##
    invM, allPeaks, peakGroups = BuildGridModel(stp_arr, imageBoundary=30, filterDiskMult=1/6, debugDisplay=True) # debugDisplay will only show a image but not write it
    subimages, anchors = segmentImage(cv.imread(target,0), invM, peakGroups)
    ## USAGE ##

    print(len(allPeaks))

    rMax, cMax = 0, 0
    for row in subimages:
        if row > rMax:
            rMax = row
        for col in subimages[row]:
            if col > cMax:
                cMax = col

    fig = plt.figure()
    for r in range(rMax+1):
        for c in range(cMax+1):
            ax = fig.add_subplot(rMax+1, cMax+1, r*(cMax+1)+c+1)
            try:
                image = subimages[r][c]
                assert isinstance(image, type(np.array([1])))
                ax.imshow(image,cmap='gray')
            except (AssertionError, KeyError):
                pass
    plt.show()
