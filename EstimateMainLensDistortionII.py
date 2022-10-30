# -*- coding:utf-8 -*-
import os
import json
import cv2 as cv
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from queue import PriorityQueue as PQ

from Undistort import myUndistort
from GridModel import BuildGridModel, segmentImage
from Checkerboard import unvignet, getCheckerboardCorners, calibrate, calibOptimize, undistort

from Utilities import prepareFolder, findROI, sortRect, sortPointCloud, ProgramSTOP

def averageDistanceFit(group):
    """ Find the average distance in a group of points and fit it to a Gaussion distribution """
    # group (list): [[p11,p12,...],[p21,...],[prc,...],...] # 2d array may work fine
    # group has to be a rectangle
    rows = len(group)
    cols = len(group[0])
    ds = []
    for r in range(rows):
        for c in range(cols):
            curP = group[r][c]
            try: # right point
                rP = group[r][c+1]
                dr = np.sqrt((curP[0]-rP[0])**2+(curP[1]-rP[1])**2)
                ds.append(dr)
            except IndexError:
                pass
            try: # down point
                dP = group[r+1][c]
                dd = np.sqrt((curP[0]-dP[0])**2+(curP[1]-dP[1])**2)
                ds.append(dd)
            except IndexError:
                pass
    ds = np.array(ds)
    mu, sigma = stats.norm.fit(ds)
    return mu, sigma

if __name__ == '__main__':
    # 01
    ### User Define Area ###
    root = '/Volumes/GoogleDrive/Other computers/Lab/Feiyi_GoogleDrive/Feiyi_images'
    example01 = 1270 # resolution target image before adjustment
    stopImage = 1271 # stop image
    whiteImage = 1272 # white image after adjustment
    example02 = 1273 # resolution target image after adjustment

    calibRange = (1274, 1309) # checkerboard images for central subaperture calibration

    paraTarget = 1310 # parallel checkerboard image

    tiltedTargetRange = (1311, 1321) # real scene

    example01_path = os.path.join(root, str(example01).zfill(6)+'.pgm')
    example02_path = os.path.join(root, str(example02).zfill(6)+'.pgm')
    whiteImage_path = os.path.join(root, str(whiteImage).zfill(6)+'.pgm')
    stopImage_path = os.path.join(root, str(stopImage).zfill(6)+'.pgm')
    calibImagePath = []
    for i in range(calibRange[0], calibRange[1]+1):
        path = os.path.join(root, str(i).zfill(6)+'.pgm')
        calibImagePath.append(path)

    paraTarget_path = os.path.join(root, str(paraTarget).zfill(6)+'.pgm')

    allTiltedTargetPath = []
    for i in range(tiltedTargetRange[0], tiltedTargetRange[1]+1):
        path = os.path.join(root, str(i).zfill(6)+'.pgm')
        allTiltedTargetPath.append(path)

    pixel = 4.65e-6
    ### User Define Area END ###

    Log = 'ImageLog'
    prepareFolder(Log)



    #####################
    """ 2. grid model """
    #####################
    # grid model
    stopArr = cv.imread(stopImage_path,0)
    invM, allPeaks, peakGroups = BuildGridModel(stopArr, imageBoundary=90)
    R, C = 1, 2



    #############################
    """ 3. prepare white image"""
    #############################
    # white image, type = float64
    whiteImage = cv.imread(whiteImage_path,0)
    whiteImage = whiteImage.astype('float64')/whiteImage.max()
    segWhiteImages, _ = segmentImage(whiteImage, invM, peakGroups)



    #######################################
    """ 5. calibrate center subaperture """
    #######################################
    # prepare central subimages
    calibImageArr = []
    # prepare folder
    calibFolder = os.path.join(Log,'subimageCalibration')
    prepareFolder(calibFolder)
    for filename in calibImagePath:
        # unvignet
        image = cv.imread(filename,0)
        # image = unvignet(image, whiteImage)
        subimages, _ = segmentImage(image, invM, peakGroups)
        # write subimages
        try:
            si = subimages[R][C]
            si = unvignet(si, segWhiteImages[R][C])
            initials = filename[-10:-4] # xxx/000123.pgm
            cv.imwrite(os.path.join(calibFolder,initials+'.png'), si)
        except (IndexError, KeyError):
            print('404: Center Subimage Not Found, ', filename)

    # calibrate central subaperture system

    ### User Define Area ###
    checkerboardShape = (6,6)
    squareSize = 7.343e-3
    ### User Define Area END ###

    match = getCheckerboardCorners(
        calibFolder, checkerboardShape,
        squareSize=squareSize, #whiteImage=centerWhiteImage, # have been unvignetted
        extension='png', visualize=True
        )
    RMS, cameraMatrix, dist, rvecs, tvecs = calibrate(match)
    sampleImage = cv.imread(os.path.join(calibFolder,initials+'.png'),0)
    newCameraMatrix, roi = calibOptimize(cameraMatrix, dist, sampleImage.shape)
    K = newCameraMatrix
    print('\nCamera Matrix:\n',K,'\nDistortion coefficients:\n',dist,'\n')

    # undistort
    undistFolder = os.path.join(Log,'undistort')
    prepareFolder(undistFolder)
    ccss_images = os.listdir(calibFolder)
    ccss_images = [name for name in ccss_images if name.endswith('png')]
    for ccss in ccss_images:
        udst = undistort(os.path.join(calibFolder,ccss), cameraMatrix, dist, K)
        cv.imwrite(os.path.join(undistFolder,ccss), udst)



    ############################
    """ 6. check parallelism """
    ############################

    ### User Define Area ###
    paraSquareSize = 7.71e-3
    ### User Define Area END ###

    # crop all subimages
    paraFolder = os.path.join(Log,'paraCheckerboard')
    prepareFolder(paraFolder)
    paraCheckerboard = cv.imread(paraTarget_path,0)
    # paraCheckerboard = unvignet(paraCheckerboard, whiteImage)
    subParas, anchors = segmentImage(paraCheckerboard, invM, peakGroups)

    local_origin_1 = {} # {r:{c:()}} (y,x)
    for r in subParas:
        if not r in local_origin_1:
            local_origin_1[r] = {}
        for c in subParas[r]:
            try:
                assert isinstance(subParas[r][c], type(np.array([0])))
                subParas[r][c] = unvignet(subParas[r][c], segWhiteImages[r][c])
                paraname = paraTarget_path.split('.')[0][-6:]+'r'+str(r).zfill(2)+'c'+str(c).zfill(2)+'.png'
                cv.imwrite(os.path.join(paraFolder,paraname), subParas[r][c])
                local_origin_1[r][c] = anchors[r][c]['upperLeft']
            except AssertionError:
                pass



    ##########################
    """ 7. local undistort """
    ##########################

    ### User Define Area ###
    NumOfPixelsPerBlock = 18
    ### User Define Area END ###

    # using subimages obtained above
    LUFolder = os.path.join(Log,'localUndistortion')
    prepareFolder(LUFolder)

    # save each locally undistorted subimage
    # and merge them back
    local_origin_2 = {} # {r:{c:()}} (y,x)
    # local_std = {} # {r:{c:{'points':array, 'center':center, 'shape':shape}}} (y,x)
    # Cannot find appropriate reference to correctly display standard points
    local_shapes = {}
    local_imagePoints = {} # {r:{c:()}}
    paraCheckerboard_LU = np.zeros(paraCheckerboard.shape,dtype='float64') # (y,x)

    do_not_input_next_time = {}
    exist = False
    if os.path.exists('textLog/position_and_shape_main_lens_dist.json'):
        # load data
        exist = True
        with open('textLog/position_and_shape_main_lens_dist.json', 'r') as file:
            do_not_input_next_time = json.load(file)

    for row in subParas:

        if not row in local_origin_2:
            local_origin_2[row] = {}
        # if not row in local_std:
        #     local_std[row] = {}
        if not row in local_shapes:
            local_shapes[row] = {}
        if not row in local_imagePoints:
            local_imagePoints[row] = {}
        if not exist:
            if not row in do_not_input_next_time:
                do_not_input_next_time[row] = {}

        for col in subParas[row]:

            if not exist:
                if not col in do_not_input_next_time[row]:
                    do_not_input_next_time[row][col] = {}

            paraname = paraTarget_path.split('.')[0][-6:]+'r'+str(row).zfill(2)+'c'+str(col).zfill(2)+'.png'
            udst = undistort(os.path.join(paraFolder,paraname), cameraMatrix, dist, K)
            cv.imwrite(os.path.join(LUFolder,paraname), udst)

            if exist:
                top = do_not_input_next_time[str(row)][str(col)]['top']
                bottom = do_not_input_next_time[str(row)][str(col)]['bottom']
                left = do_not_input_next_time[str(row)][str(col)]['left']
                right = do_not_input_next_time[str(row)][str(col)]['right']
                localShape = do_not_input_next_time[str(row)][str(col)]['shape']
            else:
                image = udst
                # find ROI in center subimage
                print('Select a clear area')
                mouseUp = findROI(image, name='paraCheckerboard', origin=(600,500))
                top, bottom, left, right = sortRect(mouseUp)

                # find image points in center subimage
                while True:
                    localShape = input('Inner corners in center subaperture (row,col): ')
                    if localShape == '':
                        continue
                    elif localShape[0] == '(' and localShape[-1] == ')': # input (m,n)
                        localShape = int(localShape.split(',')[0][1:]), int(localShape.split(',')[1][:-1])
                        break
                    else: # input m,n
                        localShape = int(localShape.split(',')[0]), int(localShape.split(',')[1])
                        break

                # collect data
                do_not_input_next_time[row][col]['top'] = top
                do_not_input_next_time[row][col]['bottom'] = bottom
                do_not_input_next_time[row][col]['left'] = left
                do_not_input_next_time[row][col]['right'] = right
                do_not_input_next_time[row][col]['shape'] = localShape

            local_origin_2[row][col] = (top, left)

            # local_std[row][col] = {}
            # std = np.indices(localShape).T.reshape(-1, 2) # (y,x)
            # std *= NumOfPixelsPerBlock
            # local_std[row][col]['shape'] = localShape
            # local_std[row][col]['points'] = std
            # local_std[row][col]['center'] = std.mean(axis=0)
            local_shapes[row][col] = localShape

            boundary = int(NumOfPixelsPerBlock/2)
            localROI = udst[top-boundary:bottom+boundary, left-boundary:right+boundary]
            imageName = 'localROI'+'_r'+str(row).zfill(2)+'_c'+str(col).zfill(2)+'.png'
            cv.imwrite(os.path.join(LUFolder,imageName),localROI)

            match = getCheckerboardCorners(os.path.join(LUFolder,imageName),local_shapes[row][col])
            imagePoints = match['imagePoints'][0] # (x,y)
            local_imagePoints[row][col] = np.flip(imagePoints) # (y,x)

            y, x = anchors[row][col]['upperLeft']
            h, w = subParas[row][col].shape
            paraCheckerboard_LU[y:y+h, x:x+w] = udst # (y,x)

    # write data
    with open('textLog/position_and_shape_main_lens_dist.json', 'w', encoding='utf-8') as file:
        json.dump(do_not_input_next_time, file, indent=4)

    distCenter = peakGroups['H'][1][2] # (y,x)
    # paraCheckerboard_std = []
    paraCheckerboard_real = []
    drawing_real_points_group = {}
    for r in local_origin_1:

        if not r in drawing_real_points_group:
            drawing_real_points_group[r] = {}

        for c in local_origin_1[r]:
            # std
            # peak = peakGroups['H'][r][c] # (y,x)
            # center = local_std[r][c]['center'] # (y,x)
            # dy, dx = np.array(peak) - np.array(center)
            # std = local_std[r][c]['points'] # (y,x)
            # trans_std = std + np.array([dy,dx])
            # for point in trans_std:
            #     paraCheckerboard_std.append(point)
            # print(
            #     '(row, col) = ({:d}, {:d}), peak(y,x) = ({:.2f}, {:.2f})'
            #     .format(r, c, peak[0]-distCenter[0], peak[1]-distCenter[1])
            #     )

            # real
            o1 = local_origin_1[r][c] # (y,x)
            o2 = local_origin_2[r][c] # (y,x)
            imagePoints = local_imagePoints[r][c] # (y,x)
            trans_real = imagePoints + o1 + o2 + np.array([-boundary, -boundary])
            for point in trans_real:
                paraCheckerboard_real.append(point)

            drawing_real_points_group[r][c] = {}
            drawing_real_points_group[r][c]['group'] = sortPointCloud(trans_real, local_shapes[r][c], boundary)
            # drawing_real_points_group = {row:{col:{'group':array, 'circles':circles}}}

            # print(
            #     'o1(y,x) = ({:d}, {:d}), global o2(y,x) = ({:d}, {:d})'
            #     .format(o1[0]-distCenter[0], o1[1]-distCenter[1], o1[0]-distCenter[0]+o2[0], o1[1]-distCenter[1]+o2[1])
            #     )
            # print(trans_real.mean(axis=0)-np.array(distCenter), end='\n\n')

    # paraCheckerboard_std = np.array(paraCheckerboard_std)
    paraCheckerboard_real = np.array(paraCheckerboard_real)

    # paraCheckerboard_std = paraCheckerboard_std.reshape((int(paraCheckerboard_std.flatten().shape[0]/2), 2))
    paraCheckerboard_real = paraCheckerboard_real.reshape((int(paraCheckerboard_real.flatten().shape[0]/2), 2))



    ######################
    """ manual control """
    ######################
    diag1 = np.sqrt((distCenter[0]-0)**2+(distCenter[1]-0)**2)
    diag2 = np.sqrt((distCenter[0]-0)**2+(distCenter[1]-stopArr.shape[1])**2)
    diag3 = np.sqrt((distCenter[0]-stopArr.shape[0])**2+(distCenter[1]-0)**2)
    diag4 = np.sqrt((distCenter[0]-stopArr.shape[0])**2+(distCenter[1]-stopArr.shape[1])**2)
    normFactor = max(diag1, diag2, diag3, diag4)

    # 02
    fig = plt.figure('circular pattern calibration',figsize=(8,5),dpi=150)
    gs = fig.add_gridspec(nrows=18, ncols=12)
    ax = fig.add_subplot(gs[:-7, :])
    ax.set_facecolor((0, 0, 0))
    ax.set_aspect('equal')
    extent = (-distCenter[1], paraCheckerboard_LU.shape[1]-distCenter[1], paraCheckerboard_LU.shape[0]-distCenter[0], -distCenter[0]) # left right bottom top
    ax.imshow(paraCheckerboard_LU, cmap='gray', extent=extent, origin='upper')
    # for ip, p in enumerate(paraCheckerboard_real):
    #     ax.plot(p[1]-distCenter[1],p[0]-distCenter[0],'r1',alpha=0.5)
    # for ip, p in enumerate(paraCheckerboard_std):
    #     ax.plot(p[1]-distCenter[1], p[0]-distCenter[0], 'gx',alpha=0.75)
    # for r in peakGroups['H']:
    #     for p in r:
    #         ax.plot(p[1]-distCenter[1], p[0]-distCenter[0], 'bo', alpha=0.6)

    def update(val):
        k1 = slider_k1.val
        k2 = slider_k2.val
        p1 = slider_p1.val
        p2 = slider_p2.val
        k3 = slider_k3.val
        r = slider_r.val * np.pi/180
        R = np.array([[np.cos(r),np.sin(r)],[-np.sin(r),np.cos(r)]])
        mus, sigmas = [], []

        y0, x0 = distCenter
        yn0, xn0 = y0/normFactor, x0/normFactor
        for row in drawing_real_points_group:
            for col in drawing_real_points_group[row]:

                pointGroup = drawing_real_points_group[row][col]['group'].copy()
                circles = drawing_real_points_group[row][col]['circles'].copy()
                t = drawing_real_points_group[row][col]['txt']

                normPs = pointGroup/normFactor

                for r in range(len(pointGroup)):
                    for c in range(len(pointGroup[r])):
                        yn, xn = normPs[r][c]
                        rn = yn**2 + xn**2
                        xdn = xn*(1 + k1*rn + k2*rn**2 + k3*rn**3) + 2*p1*xn*yn + p2*(rn+2*xn**2)
                        ydn = yn*(1 + k1*rn + k2*rn**2 + k3*rn**3) + p1*(rn+2*yn**2) + 2*p2*xn*yn
                        DNR = np.matmul(R,np.array([[xdn],[ydn]]))

                        circles[r][c].center = DNR[0]*normFactor, DNR[1]*normFactor
                        pointGroup[r][c] = np.array([DNR[1]*normFactor, DNR[0]*normFactor]).reshape((2,))

                mu, sigma = averageDistanceFit(pointGroup)
                mus.append(mu)
                sigmas.append(sigma)
                t.set_text(r'$\mu$,$\sigma$={:.2f},{:.2f}'.format(mu, sigma))
        txt.set_text(r'$\mu$,$\sigma$={:.6f},{:.12f}'.format(np.array(mus).mean(), np.array(sigmas).mean()))
        fig.canvas.draw_idle()

    def k1p(val):
        global step
        k1 = slider_k1.val
        slider_k1.set_val(k1+step)
        fig.canvas.draw()
        update(1)

    def k1m(val):
        global step
        k1 = slider_k1.val
        slider_k1.set_val(k1-step)
        fig.canvas.draw()
        update(1)

    def k2p(val):
        global step
        k2 = slider_k2.val
        slider_k2.set_val(k2+step)
        fig.canvas.draw()
        update(1)

    def k2m(val):
        global step
        k2 = slider_k2.val
        slider_k2.set_val(k2-step)
        fig.canvas.draw()
        update(1)

    def p1p(val):
        global step
        p1 = slider_p1.val
        slider_p1.set_val(p1+step)
        fig.canvas.draw()
        update(1)

    def p1m(val):
        global step
        p1 = slider_p1.val
        slider_p1.set_val(p1-step)
        fig.canvas.draw()
        update(1)

    def p2p(val):
        global step
        p2 = slider_p2.val
        slider_p2.set_val(p2+step)
        fig.canvas.draw()
        update(1)

    def p2m(val):
        global step
        p2 = slider_p2.val
        slider_p2.set_val(p2-step)
        fig.canvas.draw()
        update(1)

    def k3p(val):
        global step
        k3 = slider_k3.val
        slider_k3.set_val(k3+step)
        fig.canvas.draw()
        update(1)

    def k3m(val):
        global step
        k3 = slider_k3.val
        slider_k3.set_val(k3-step)
        fig.canvas.draw()
        update(1)

    from matplotlib.patches import Circle
    from matplotlib.widgets import Slider, Button

    dist = [0.010087439945092657 , -0.014629947380462131 , -0.00285971173644475 , 0.008047700754975979 , 0.011167353008464885]
    step = 1e-6

    sl1 = fig.add_subplot(gs[-6, :-4])
    sl2 = fig.add_subplot(gs[-5, :-4])
    sl3 = fig.add_subplot(gs[-4, :-4])
    sl4 = fig.add_subplot(gs[-3, :-4])
    sl5 = fig.add_subplot(gs[-2, :-4])
    sl6 = fig.add_subplot(gs[-1, :-4])
    slider_k1 = Slider(sl1, 'k1', 5e-3, 1.1e-2, dist[0])
    slider_k2 = Slider(sl2, 'k2', -5e-2, 0, dist[1])
    slider_p1 = Slider(sl3, 'p1', -4e-3, -2e-3, dist[2])
    slider_p2 = Slider(sl4, 'p2', 7e-3, 9e-3, dist[3])
    slider_k3 = Slider(sl5, 'k3', 0, 5e-2, dist[4])
    slider_r = Slider(sl6, 'r', -45, 45, 0)
    slider_k1.on_changed(update)
    slider_k2.on_changed(update)
    slider_p1.on_changed(update)
    slider_p2.on_changed(update)
    slider_k3.on_changed(update)
    slider_r.on_changed(update)
    txt = ax.text(-600, -400, r'$\mu$,$\sigma$', color='g', fontsize=8)
    b11 = fig.add_subplot(gs[-6, -2])
    b21 = fig.add_subplot(gs[-5, -2])
    b31 = fig.add_subplot(gs[-4, -2])
    b41 = fig.add_subplot(gs[-3, -2])
    b51 = fig.add_subplot(gs[-2, -2])
    b_k1p = Button(b11, 'k1+')
    b_k2p = Button(b21, 'k2+')
    b_p1p = Button(b31, 'p1+')
    b_p2p = Button(b41, 'p2+')
    b_k3p = Button(b51, 'k3+')
    b_k1p.on_clicked(k1p)
    b_k2p.on_clicked(k2p)
    b_p1p.on_clicked(p1p)
    b_p2p.on_clicked(p2p)
    b_k3p.on_clicked(k3p)
    b12 = fig.add_subplot(gs[-6, -1])
    b22 = fig.add_subplot(gs[-5, -1])
    b32 = fig.add_subplot(gs[-4, -1])
    b42 = fig.add_subplot(gs[-3, -1])
    b52 = fig.add_subplot(gs[-2, -1])
    b_k1m = Button(b12, 'k1-')
    b_k2m = Button(b22, 'k2-')
    b_p1m = Button(b32, 'p1-')
    b_p2m = Button(b42, 'p2-')
    b_k3m = Button(b52, 'k3-')
    b_k1m.on_clicked(k1m)
    b_k2m.on_clicked(k2m)
    b_p1m.on_clicked(p1m)
    b_p2m.on_clicked(p2m)
    b_k3m.on_clicked(k3m)

    y0, x0 = distCenter
    for row in drawing_real_points_group:
        for col in drawing_real_points_group[row]:
            pointGroup = drawing_real_points_group[row][col]['group']
            drawing_real_points_group[row][col]['circles'] = {}
            for r in range(len(pointGroup)):

                if not r in drawing_real_points_group[row][col]['circles']:
                    drawing_real_points_group[row][col]['circles'][r] = {}

                for c in range(len(pointGroup[r])):
                    y, x = pointGroup[r][c]
                    r2 = (y-y0)**2 + (x-x0)**2
                    circle = Circle((x-x0, y-y0), 5)
                    ax.add_patch(circle)
                    drawing_real_points_group[row][col]['circles'][r][c] = circle
                    pointGroup[r][c] = np.array([y-y0,x-x0]).reshape((2,))

            drawing_real_points_group[row][col]['txt'] = ax.text(
                local_origin_1[row][col][1]-x0, local_origin_1[row][col][0]-y0, 
                r'$\mu$,$\sigma$', color='r', fontsize=6
                )

    # for ipeak, peak in enumerate(paraCheckerboard_real):
    #     y, x = peak
    #     y0, x0 = distCenter
    #     r2 = (x-x0)**2 + (y-y0)**2
    #     circle = Circle((x-x0, y-y0), 5)
    #     ax.add_patch(circle)
    #     circles.append((circle, r2, (x-x0,y-y0)))

    plt.show()
    print(slider_k1.val, ',', slider_k2.val, ',', slider_p1.val, ',', slider_p2.val, ',', slider_k3.val)
