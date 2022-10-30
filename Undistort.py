# -*- coding:utf-8 -*-
import time
import numpy as np
from scipy.interpolate import griddata

def myUndistort(imgArr, undist, center, **kwarg):
    """Undistort an image, same size as original image

    Args:
        imgArr (numpy.ndarray): Image to be undistorted.
        undist (list): Undistortion coefficients, [k1, k2, p1, p2, k3].
        center (tuple): Undistortion center.
        method (str, default='linear'): Interpolation method, {'nearest', 'linear', 'cubic'}.

    Returns:
        u (numpy.ndarray): Undistorted image.
    """

    def _move(xn, yn, rn):
        nonlocal undist, x0, y0, normFactor

        k1, k2, p1, p2, k3 = undist

        xdn = xn*(1 + k1*rn**2 + k2*rn**4 + k3*rn**6) + 2*p1*xn*yn + p2*(rn**2+2*xn**2)
        ydn = yn*(1 + k1*rn**2 + k2*rn**4 + k3*rn**6) + 2*p2*xn*yn + p1*(rn**2+2*yn**2)

        xd = xdn * normFactor + x0
        yd = ydn * normFactor + y0

        return xd, yd

    t0 = time.time()

    method = kwarg['method'] if 'method' in kwarg else 'linear'
    print('Start undistorting... (may take a while)...')

    x0, y0 = center
    shape = imgArr.shape
    undist = list(undist)
    if len(undist) < 5:
        for _ in range(5-len(undist)):
            undist.append(0)

    diag1 = np.sqrt((x0)**2+(y0)**2)
    diag2 = np.sqrt((x0-shape[0])**2+(y0)**2)
    diag3 = np.sqrt((x0)**2+(y0-shape[1])**2)
    diag4 = np.sqrt((x0-shape[0])**2+(y0-shape[1])**2)
    normFactor = max(diag1, diag2, diag3, diag4)

    ys = np.array([i for i in range(imgArr.shape[0])])
    xs = np.array([i for i in range(imgArr.shape[1])])

    yns = (ys-y0)/normFactor
    xns = (xs-x0)/normFactor
    YNS, XNS = np.meshgrid(yns, xns)
    RN = np.sqrt(YNS**2+XNS**2)

    points = np.zeros((imgArr.shape[0]*imgArr.shape[1],2))
    values = np.zeros(imgArr.shape[0]*imgArr.shape[1])
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            xd, yd = _move(XNS[i,j], YNS[i,j], RN[i,j])
            points[i*len(ys)+j, 0] = yd
            points[i*len(ys)+j, 1] = xd
            values[i*len(ys)+j] = imgArr[y,x]

    X, Y = np.meshgrid(xs, ys)
    gd = griddata(points, values, (Y, X), method=method)
    u = gd #.T

    print(time.time()-t0, 'seconds.')

    return u

def myUndistort_prev(imgArr, undist, center, **kwarg):
    """Undistort an image, same size as original image

    Args:
        imgArr (numpy.ndarray): Image to be undistorted.
        undist (list): Undistortion coefficients, [k1, k2, p1, p2, k3].
        center (tuple): Undistortion center.
        method (str, default='linear'): Interpolation method, {'nearest', 'linear', 'cubic'}.

    Returns:
        u (numpy.ndarray): Undistorted image.
    """
    method = kwarg['method'] if 'method' in kwarg else 'linear'
    print('Start undistorting... (may take a while)...')

    ys = np.array([i for i in range(imgArr.shape[0])])
    xs = np.array([i for i in range(imgArr.shape[1])])

    points = np.zeros((imgArr.shape[0]*imgArr.shape[1],2))
    values = np.zeros(imgArr.shape[0]*imgArr.shape[1])
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            xd, yd = _move_prev(x, y, imgArr.shape, undist, center)
            points[i*len(ys)+j, 0] = yd
            points[i*len(ys)+j, 1] = xd
            values[i*len(ys)+j] = imgArr[y,x]

    X, Y = np.meshgrid(xs, ys)
    gd = griddata(points, values, (Y, X), method=method)
    u = gd #.T
    return u

def _move_prev(x, y, shape, undist, center):
    x0, y0 = center
    undist = list(undist)
    if len(undist) < 5:
        for _ in range(5-len(undist)):
            undist.append(0)
    k1, k2, p1, p2, k3 = undist

    diag1 = np.sqrt((x0)**2+(y0)**2)
    diag2 = np.sqrt((x0-shape[0])**2+(y0)**2)
    diag3 = np.sqrt((x0)**2+(y0-shape[1])**2)
    diag4 = np.sqrt((x0-shape[0])**2+(y0-shape[1])**2)
    normFactor = max(diag1, diag2, diag3, diag4)

    xn, yn = (x-x0)/normFactor, (y-y0)/normFactor
    rn = np.sqrt((x-x0)**2+(y-y0)**2)/normFactor

    xdn = xn*(1 + k1*rn**2 + k2*rn**4 + k3*rn**6) + 2*p1*xn*yn + p2*(rn**2+2*xn**2)
    ydn = yn*(1 + k1*rn**2 + k2*rn**4 + k3*rn**6) + 2*p2*xn*yn + p1*(rn**2+2*yn**2)

    xd = xdn * normFactor + x0
    yd = ydn * normFactor + y0

    return xd, yd

if __name__ == '__main__':
    import os
    import time
    import cv2 as cv

    root = '/Volumes/GoogleDrive/Other computers/Lab/Feiyi_GoogleDrive/Feiyi_images'
    imgArr = cv.imread(os.path.join(root,'001120.pgm'),0)
    #undist = [0.225, -0.604, 3.2e-4, 5.64e-3, 0.483]
    #undist = [0.17, -0.5, 0, 0, 0.506]
    #undist = [0.07, 0.015, 0.005, 0.005, 0.103]
    #undist = [0.22, -0.604, 3.2e-4, 5.64e-3, 0.483]
    undist = [0.22, -0.604, -3.2e-4, -5.64e-3, 0.483]
    #undist = [0.232, -0.784, -3.2e-4, -5.64e-3, 0.911]
    undist = [0.5, -0.604, -3.2e-4, -5.64e-3, 0.483]

    t0 = time.time()
    u = myUndistort_prev(imgArr, undist, (623,396))
    print('previous version takes ', time.time()-t0, ' seconds')
    u = np.repeat(u, 3).reshape((u.shape[0],u.shape[1],3)) # gray 2 rgb
    cv.circle(u, (623,396), 5, (0,0,255), 5)
    cv.imwrite(os.path.join('001120_un_prev.png'), u)

    t0 = time.time() # 2 times faster
    u = myUndistort(imgArr, undist, (623,396))
    print('new version takes ', time.time()-t0, ' seconds')
    u = np.repeat(u, 3).reshape((u.shape[0],u.shape[1],3)) # gray 2 rgb
    cv.circle(u, (623,396), 5, (0,0,255), 5)
    cv.imwrite(os.path.join('001120_un.png'), u)
