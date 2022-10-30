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
    drawing_real_points_group = {}
    for r in local_origin_1:

        if not r in drawing_real_points_group:
            drawing_real_points_group[r] = {}

        for c in local_origin_1[r]:

            # real
            o1 = local_origin_1[r][c] # (y,x)
            o2 = local_origin_2[r][c] # (y,x)
            imagePoints = local_imagePoints[r][c] # (y,x)
            trans_real = imagePoints + o1 + o2 + np.array([-boundary, -boundary])

            drawing_real_points_group[r][c] = {}
            drawing_real_points_group[r][c]['group'] = sortPointCloud(trans_real, local_shapes[r][c], boundary)
            # drawing_real_points_group = {row:{col:{'group':array, }}}



    #####################
    """ automatic fit """
    #####################
    diag1 = np.sqrt((distCenter[0]-0)**2+(distCenter[1]-0)**2)
    diag2 = np.sqrt((distCenter[0]-0)**2+(distCenter[1]-stopArr.shape[1])**2)
    diag3 = np.sqrt((distCenter[0]-stopArr.shape[0])**2+(distCenter[1]-0)**2)
    diag4 = np.sqrt((distCenter[0]-stopArr.shape[0])**2+(distCenter[1]-stopArr.shape[1])**2)
    normFactor = max(diag1, diag2, diag3, diag4)

    dist = [0.010087439945092657 , -0.014629947380462131 , -0.00285971173644475 , 0.008047700754975979 , 0.011167353008464885] # 16.915962928916038 0.25880913034748315 9.431887493249747e-10
    #dist = [0.00941983527796843 , -0.015563944177533741 , -0.002846716998398536 , 0.007961610615419813 , 0.013116563715396936] # 16.891359693294124 0.25906407800982406 1.569955021540892e-08
    step = 1e-5
    flags = [1,1,1,1,1]
    k1, k2, p1, p2, k3 = dist

    y0, x0 = distCenter
    drawing_real_points_group_center = {}
    for row in drawing_real_points_group:

        if not row in drawing_real_points_group_center:
            drawing_real_points_group_center[row] = {}

        for col in drawing_real_points_group[row]:
            points = drawing_real_points_group[row][col]['group'].copy()
            points = points - np.array(distCenter)
            drawing_real_points_group_center[row][col] = points
            del points

    mus, sigmas = [], []

    for row in drawing_real_points_group_center:
        for col in drawing_real_points_group_center[row]:

            points = drawing_real_points_group_center[row][col].copy()
            normPs = points/normFactor

            ps = np.zeros((normPs.shape[0], normPs.shape[1], 2))
            for r in range(normPs.shape[0]):
                for c in range(normPs.shape[1]):
                    yn, xn = normPs[r, c]
                    rn = yn**2 + xn**2
                    xdn = xn*(1 + k1*rn + k2*rn**2 + k3*rn**3) + 2*p1*xn*yn + p2*(rn+2*xn**2)
                    ydn = yn*(1 + k1*rn + k2*rn**2 + k3*rn**3) + p1*(rn+2*yn**2) + 2*p2*xn*yn
                    yd, xd = ydn*normFactor, xdn*normFactor
                    ps[r, c, 0] = yd
                    ps[r, c, 1] = xd

            mu, sigma = averageDistanceFit(ps)
            mus.append(mu)
            sigmas.append(sigma)
            del points

    mubar = np.array(mus).mean()
    sigmabar = np.array(sigmas).mean()

    e = 1e-8 # precision 1e-17
    dist_i = 0
    try:
        while True:
            dist[dist_i] += flags[dist_i]*step
            k1, k2, p1, p2, k3 = dist

            ### same as above ###
            mus, sigmas = [], []

            for row in drawing_real_points_group_center:
                for col in drawing_real_points_group_center[row]:

                    points = drawing_real_points_group_center[row][col].copy()
                    normPs = points/normFactor

                    ps = np.zeros((normPs.shape[0], normPs.shape[1], 2))
                    for r in range(normPs.shape[0]):
                        for c in range(normPs.shape[1]):
                            yn, xn = normPs[r, c]
                            rn = yn**2 + xn**2
                            xdn = xn*(1 + k1*rn + k2*rn**2 + k3*rn**3) + 2*p1*xn*yn + p2*(rn+2*xn**2)
                            ydn = yn*(1 + k1*rn + k2*rn**2 + k3*rn**3) + p1*(rn+2*yn**2) + 2*p2*xn*yn
                            yd, xd = ydn*normFactor, xdn*normFactor
                            ps[r, c, 0] = yd
                            ps[r, c, 1] = xd

                    mu, sigma = averageDistanceFit(ps)
                    mus.append(mu)
                    sigmas.append(sigma)
                    del points

            m = np.array(mus).mean()
            s = np.array(sigmas).mean()
            ### same as above ###

            mubar = m
            error = np.abs(s-sigmabar)

            if s > sigmabar:
                flags[dist_i] = -flags[dist_i]
                #print('dist:', dist, 'flags:', flags, 'error:', error, 's > sigmabar', dist_i)
            else:
                pass
                #print('dist:', dist, 'flags:', flags, 'mubar:', mubar, 'sigmabar:', sigmabar, 'error:', error, 's<=sigmabar', dist_i)
            sigmabar = s

            if error < e*4**(len(dist)-dist_i):
                dist_i += 1
                print(dist_i, end=' ')
                if dist_i == len(dist):
                    dist_i = 0

            if error < e:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print(dist, '#', mubar, sigmabar, error)
