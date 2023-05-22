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
        stp_arr (numpy.ndarray): A numpy array image. (other data format that is able to convert to numpy.ndarray is alse acceptable)
        approxSpacing (int, default=200): Approximate spacing between stops.
        filterDiskMult (float, default=1/6): Filter disk radius, relative to spacing of stops.
        imageBoundary (int, default=50): Image boundary width, help remove local maximums that are close to boundary (not real centers).
        debugDisplay (bool, default=False): Debug display, to show whether centers are estimated correctly.
        ignoreWarning (bool, default=False): Whether to ignore warnings.

    Returns:
        invM (numpy.ndarray): 2x2 transformation matrix. (rotation & shear)
        allPeaks (list): A list of peaks detected. (exclude the ones on the edge)
        peakArr (numpy.ndarray): Peak array, (row, col, 2)
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
    radius = approxSpacing * filterDiskMult
    h = cv.circle(h, (int(stp_arr.shape[1]/2),int(stp_arr.shape[0]/2)), int(radius), (255,255,255), cv.FILLED)

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
    allPeaks = []
    for i in range(boundary,peaks.shape[0]-boundary):
        for j in range(boundary,peaks.shape[1]-boundary):
            if peaks[i,j]:
                allPeaks.append([i,j])

    print('Fitting grid lines...')
    # vertical distance between first one in one line and last one in another line
    # should be larger than groupSpacing
    groupSpacing = approxSpacing/2

    # hori lines
    peakArrH = [[allPeaks[0]]]
    cntGroup = 0
    for i in range(len(allPeaks)-1):
        curPeak = allPeaks[i]
        nextPeak = allPeaks[i+1]
        if abs(curPeak[0]-nextPeak[0]) < groupSpacing:
            peakArrH[cntGroup].append(nextPeak)
        else:
            cntGroup += 1
            peakArrH.append([nextPeak])

    # sort hori lines
    # sort points from left to right (y)
    # sort rows from top to bottom (x)
    qH = PQ()
    while not peakArrH == []:
        group = peakArrH.pop()
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
        peakArrH.append(newGroup)

    HLineParams = []
    for group in peakArrH:
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
    peakArrV = [[]]
    cntGroup = 0
    allPeaks_cp = allPeaks.copy() # in case passing a pointer rather than a copy of original content
    while True:
        if not allPeaks_cp == []:
            curPeak = allPeaks_cp.pop(0) # get whatever first one
            peakArrV[cntGroup].append(curPeak)
            # do not make change to allPeaks_cp while looping it
            for peak in allPeaks_cp:
                if abs(curPeak[1]-peak[1]) < groupSpacing:
                    peakArrV[cntGroup].append(peak)
            # delete what we've got from allPeaks_cp
            for peak in peakArrV[cntGroup]:
                if peak in allPeaks_cp:
                    allPeaks_cp.pop(allPeaks_cp.index(peak))
            cntGroup += 1
            peakArrV.append([])
        else:
            del allPeaks_cp
            break
    peakArrV.pop() # delete last empty list

    # sort vert lines
    # sort points from top to bottom (x)
    # sort cols from left to right (y)
    qV = PQ()
    while not peakArrV == []:
        group = peakArrV.pop()
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
        peakArrV.append(newGroup)

    VLineParams = []
    for group in peakArrV:
        if len(group) > 1:
            xs, ys = [], []
            for peak in group:
                xs.append(peak[1])
                ys.append(peak[0])
            popt, pcov = curve_fit(line, ys, xs)
            VLineParams.append([popt[0],popt[1]])

    # extract slopes, k
    hk, vk = [], []
    for hp in HLineParams:
        hk.append(hp[0])
    for vp in VLineParams:
        vk.append(vp[0])
    hk = np.array(hk)
    vk = np.array(vk) # hk=k; vk=1/k

    print('Fitting transformation matrix...')
    # matrix
    # H1 = a b * 1 = a ; V1 = a b * 0 = b
    # H2   c d   0   c   V2   c d   1   d
    vecH = np.array((1/np.sqrt(hk.mean()**2+1),hk.mean()/np.sqrt(hk.mean()**2+1))) # x' axis
    vecV = np.array((vk.mean()/np.sqrt(vk.mean()**2+1),1/np.sqrt(vk.mean()**2+1))) # y' axis
    if vk.mean() < 0:
        vecV = -vecV
    a, b = vecH[0], vecV[0]
    c, d = vecH[1], vecV[1]
    M = np.array([[a,b],[c,d]])
    invM = np.linalg.inv(M) # x'=Mx, y'=My

    if debugDisplay:
        print('total ', len(allPeaks), ' peaks')
        print(allPeaks)
        plt.figure('Debug Display')
        plt.imshow(stp_arr,cmap='gray')
        for ipeaks, peaks in enumerate(peakArrH):
            for p in peaks:
                plt.plot(p[1],p[0],'rx')
                plt.text(p[1]-50,p[0]+50,str(ipeaks),color='#22FF22')
        for ipeaks, peaks in enumerate(peakArrV):
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
            plt.plot(k*y+b, y, 'y')
        plt.show()

    return invM, allPeaks, np.array(peakArrH)

def subimageFourCorners(invM, peakArr):
    """Convert image centers into subimage corners

    Args:
        invM (numpy.ndarray): 2x2 transformation matrix. (rotation & shear)
        peakArr (numpy.ndarray): Peak array, (row, col, 2)

    Returns:
        cornerGroups (numpy.ndarray): Similar structure to `peakArr`
    """
    print('Finding subimage corners...')
    # direction vectors
    M = np.linalg.inv(invM)
    vecH, vecV = M[:,0], M[:,1]

    # mean distance in each direction
    H = peakArr
    V = peakArr.transpose((1,0,2)) # swap first two dimensions
    disH = []
    disV = []
    for irow in range(H.shape[0]):
        row = H[irow, :, :]
        for ipeak in range(len(row)-1):
            curPeak = row[ipeak]
            nextPeak = row[ipeak+1]
            disH.append(np.sqrt((nextPeak[0]-curPeak[0])**2+(nextPeak[1]-curPeak[1])**2))
    for icol in range(V.shape[0]):
        col = V[icol, :, :]
        for ipeak in range(len(col)-1):
            curPeak = col[ipeak]
            nextPeak = col[ipeak+1]
            disV.append(np.sqrt((nextPeak[0]-curPeak[0])**2+(nextPeak[1]-curPeak[1])**2))
    Hbar = np.array(disH).mean()
    Vbar = np.array(disV).mean()
    Hstep = Hbar/2
    Vstep = Vbar/2
    Hx, Hy = Hstep * vecH
    Vx, Vy = Vstep * vecV
    if Vy >= 0: # ensure two vectors point at + direction
        Hx, Hy, Vx, Vy = int(Hx), int(Hy), int(Vx), int(Vy)
    else:
        Hx, Hy, Vx, Vy = int(Hx), int(Hy), -int(Vx), -int(Vy)


    # generate subimage corner points
    # horizontal corners
    corners = np.zeros((H.shape[0]+1, H.shape[1]+1, 2), dtype=H.dtype)
    xs = H[:, :, 1] - (Hx+Vx)
    ys = H[:, :, 0] - (Hy+Vy)
    corners[:-1, :-1, 1] = xs
    corners[:-1, :-1, 0] = ys
    # last row
    xs = H[-1, :, 1] - (Hx+Vx)
    ys = H[-1, :, 0] + (Hy+Vy)
    corners[-1, :-1, 1] = xs
    corners[-1, :-1, 0] = ys
    # last col
    xs = H[:, -1, 1] + (Hx+Vx)
    ys = H[:, -1, 0] - (Hy+Vy)
    corners[:-1, -1, 1] = xs
    corners[:-1, -1, 0] = ys
    # last point
    x = H[-1, -1, 1] + (Hx+Vx)
    y = H[-1, -1, 0] + (Hy+Vy)
    corners[-1, -1, 1] = x
    corners[-1, -1, 0] = y

    return corners

def segmentImage(image, invM, peakArr):
    """Crop raw image into subimages

    Args:
        image (numpy.ndarray): Image to be cropped. (Height, Width, Color)
        invM (numpy.ndarray): 2x2 transformation matrix. (rotation & shear)
        peakArr (numpy.ndarray): Peak array, (row, col, 2)

    Returns:
        subimages (dict): Subimages. {#row:{#col:image (Width,Height,Color)}}
        anchors (dict): Five points in each subimage. {#row:{#col:{points}}}
            points = {'center':p0, 'upperLeft':p1, 'upperRight':p2, 'lowerRight':p3, 'lowerLeft':p4}
    """
    cornerGroups = subimageFourCorners(invM, peakArr)
    CGH = cornerGroups
    subimages = {}
    anchors = {}

    print('Extracting subimages...')
    for irow in range(CGH.shape[0]):
        for icol in range(CGH.shape[1]):
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
            anchors[irow][icol]['center'] = peakArr[irow][icol]
            anchors[irow][icol]['upperLeft'] = p1 + np.array([p2[0]-p1[0], 0])
            anchors[irow][icol]['upperRight'] = p2 + np.array([0, p3[1]-p2[1]])
            anchors[irow][icol]['lowerRight'] = p3 + np.array([p4[0]-p3[0], 0])
            anchors[irow][icol]['lowerLeft'] = p4 + np.array([0, p1[1]-p4[1]]) # (y,x)

            p1 = anchors[irow][icol]['upperLeft'][::-1]
            p2 = anchors[irow][icol]['upperRight'][::-1]
            p3 = anchors[irow][icol]['lowerRight'][::-1]
            p4 = anchors[irow][icol]['lowerLeft'][::-1]
            pts = np.array([p1,p2,p3,p4]).astype('int64')
            x, y, h, w = cv.boundingRect(pts)
            cropped = image[y:y+w, x:x+h].copy()
            pts -= pts.min(axis=0)
            mask = np.zeros(cropped.shape, dtype='uint8')
            cv.drawContours(mask, [pts], -1, (255,255,255), -1, cv.LINE_AA)
            subimage = cv.bitwise_and(cropped, cropped, mask=mask)

            if not irow in subimages:
                subimages[irow] = {}
            subimages[irow][icol] = subimage

    return subimages, anchors
