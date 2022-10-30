# -*- coding:utf-8 -*-
import os
import json
import cv2 as cv
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import griddata

from makeGIF import create_gif
from Undistort import myUndistort
from GridModel import BuildGridModel, segmentImage
from Checkerboard import unvignet, getCheckerboardCorners, calibrate, calibOptimize, undistort
from Utilities import prepareFolder, saveDict, loadDict, KeyFinder, ProgramSTOP
from Utilities import SystemParamReader, GO, findSubPixPoint, findROI, sortRect, sortPointCloud

def StraightLine(x, k=1, b=0):
    return k*x+b



########################
""" 0. program start """
########################
Reader = SystemParamReader()
data = Reader.read()

KF = KeyFinder(data)
Log = KF.find('Log')
txtLog = KF.find('txtLog')

SysData = loadDict(os.path.join(txtLog, 'SystemParamsFullEditionReadOnly.json'))
CalibrationResult = loadDict(os.path.join(txtLog, 'CalibrationResult.json'))

everything_in_one_dict = {0:SysData, 1:CalibrationResult}
KF = KeyFinder(everything_in_one_dict)

depthFolder = KF.find('depthFolder')

anchors = KF.find('anchors')
ROWS = KF.find('ROWS')
COLS = KF.find('COLS')
info = KF.find('correspondence')

calibSquareSize = KF.find('calibSquareSize')
paraSquareSize = KF.find('paraSquareSize')

cameraMatrix = KF.find('cameraMatrix')
cameraMatrix = np.array(cameraMatrix['value']).reshape(cameraMatrix['shape'])
K = KF.find('K')
K = np.array(K['value']).reshape(K['shape'])
dist = KF.find('dist')
dist = np.array(dist['value']).reshape(dist['shape'])

undist = KF.find('undist')
margin = KF.find('margin')
undistCenter = KF.find('undistCenter')
centerAperture = loadDict(os.path.join(txtLog, 'Center.json'))
R, C = centerAperture['R'], centerAperture['C']

whiteImage = KF.find('whiteImages')
whiteImage = np.load(whiteImage['whiteImage'])

invM = KF.find('invM')
invM = np.array(invM['value']).reshape(invM['shape'])
allPeaks = KF.find('allPeaks')
peakGroups = KF.find('peakGroups')

initDepth = KF.find('initDepth')
lastDepth = KF.find('lastDepth')
depthInterval = KF.find('depthInterval')
depthTargetPath = KF.find('depth', data=SysData['ImagePaths']['otherTargetPath'])

depthGIF = 'depth.gif'
if not os.path.exists(depthGIF):
    d = depthTargetPath.copy()
    reverseDepthTargetPath = depthTargetPath.copy()
    d.reverse()
    create_gif(d+reverseDepthTargetPath, 'depth.gif', 1/60)
    del d, reverseDepthTargetPath

ProgramResult = {}
# ProgramResult = {
#     'localCalib':{
#         index:[index+R00C00+ext, ...]
#     },
#     'globalCalib':{
#         index:{
#             'image':...,
#             'subimages':[index+R00C00+ext, ...]
#         }
#     },
#     'VIP':{
#         index:{
#             'sum':...,
#             'single':[index+R00C00+ext, ...]
#         }
#     },
#     'ROI':ROIfilename
# }



######################################
""" 1. calibrate & local undistort """
######################################
prepareFolder(depthFolder)
prepareFolder(os.path.join(depthFolder, 'calibration'))
ProgramResult['localCalib'] = {}
for depthTarget in depthTargetPath:
    image = cv.imread(depthTarget, 0)
    image = unvignet(image, whiteImage)
    subDepth, _ = segmentImage(image, invM, peakGroups)

    initials = depthTarget[-10:-4]
    ProgramResult['localCalib'][int(initials)] = []
    for row in range(ROWS):
        for col in range(COLS):

            img = subDepth[row][col]
            filename = initials + 'R' + str(row).zfill(2) + 'C' + str(col).zfill(2)
            cv.imwrite(os.path.join(depthFolder, 'calibration', filename+'.thumbnail.jpg'), (255*img/img.max()).astype('uint8'))
            np.save(os.path.join(depthFolder, 'calibration', filename+'.npy'), img.astype('float64'))

            img = undistort(os.path.join(depthFolder, 'calibration', filename+'.npy'), cameraMatrix, dist, K)
            cv.imwrite(os.path.join(depthFolder, 'calibration', filename+'.thumbnail.jpg'), (255*img/img.max()).astype('uint8'))
            np.save(os.path.join(depthFolder, 'calibration', filename+'.npy'), img.astype('float64'))

            ProgramResult['localCalib'][int(initials)].append(os.path.join(depthFolder, 'calibration', filename+'.npy'))



###########################
""" 2. global undistort """
###########################
prepareFolder(os.path.join(depthFolder, 'globalUndist'))
ProgramResult['globalCalib'] = {}
sampleImage = image
for index in ProgramResult['localCalib']:
    ProgramResult['globalCalib'][index] = {}
    ProgramResult['globalCalib'][index]['subimages'] = []

    # local undistort
    depth_LU = np.zeros(sampleImage.shape, dtype='float64')
    for filename in ProgramResult['localCalib'][index]:
        row, col = int(filename[-9:-7]), int(filename[-6:-4]) # xxx/xxx/000123R01C02.npy
        img = np.load(filename)
        y, x = anchors[row][col]['upperLeft']
        h, w = img.shape
        depth_LU[y:y+h, x:x+w] += img

    # global undistort
    depth_GU = myUndistort(depth_LU, undist, undistCenter, method='linear')
    cv.imwrite(os.path.join(depthFolder, 'globalUndist', str(index).zfill(6)+'.thumbnail.jpg'), (255*depth_GU/depth_GU.max()).astype('uint8'))
    np.save(os.path.join(depthFolder, 'globalUndist', str(index).zfill(6)+'.npy'), depth_GU.astype('float64'))
    ProgramResult['globalCalib'][index]['image'] = os.path.join(depthFolder, 'globalUndist', str(index).zfill(6)+'.npy')

    subDepthGUs, _ = segmentImage(depth_GU, invM, peakGroups)
    for row in subDepthGUs:
        for col in subDepthGUs[row]:
            img = subDepthGUs[row][col]
            filename = str(index).zfill(6) + 'R' + str(row).zfill(2) + 'C' + str(col).zfill(2)
            cv.imwrite(os.path.join(depthFolder, 'globalUndist', filename+'.thumbnail.jpg'), (255*img/img.max()).astype('uint8'))
            np.save(os.path.join(depthFolder, 'globalUndist', filename+'.npy'), img.astype('float64'))

            ProgramResult['globalCalib'][index]['subimages'].append(os.path.join(depthFolder, 'globalUndist', filename+'.npy'))



##############################
""" 3. virtual image plane """
##############################
prepareFolder(os.path.join(depthFolder, 'VIP'))
ProgramResult['VIP'] = {}
dir_dict = {'upper':(-1,0), 'lower':(1,0), 'left':(0,-1), 'right':(0,1)}
for index in ProgramResult['globalCalib']:
    ProgramResult['VIP'][index] = {}
    ProgramResult['VIP'][index]['single'] = []

    depth_VI = np.zeros(sampleImage.shape, dtype='float64')
    for filename in ProgramResult['globalCalib'][index]['subimages']:
        img = np.load(filename)
        row, col = int(filename[-9:-7]), int(filename[-6:-4]) # xxx/xxx/000123R01C02.npy

        totalShift = np.zeros(2, dtype='float64')
        steps = GO((row, col), (R, C), info)
        curR, curC = row, col
        for step in steps:
            shift = info[curR][curC][step]['shift']
            totalShift[0] -= shift[0]
            totalShift[1] -= shift[1]
            dr, dc = dir_dict[step]
            curR += dr
            curC += dc
        dy, dx = totalShift

        if (dy == int(dy)) and (dx == int(dx)):
            h, w = img.shape
            y0, x0 = anchors[row][col]['upperLeft']
            single = np.zeros(sampleImage.shape, dtype='float64')
            single[int(y0+dy+margin):int(y0+dy+h-margin), int(x0+dx+margin):int(x0+dx+w-margin)] += img[margin:-margin, margin:-margin]
            depth_VI += single

            filename = filename[-16:-4]
            cv.imwrite(os.path.join(depthFolder, 'VIP', filename+'.thumbnail.jpg'), (255*single/single.max()).astype('uint8'))
            np.save(os.path.join(depthFolder, 'VIP', filename+'.npy'), single)
            ProgramResult['VIP'][index]['single'].append(os.path.join(depthFolder, 'VIP', filename+'.npy'))

        else:
            points = np.zeros((img.shape[0]-2*margin, img.shape[1]-2*margin, 2))
            values = np.zeros((img.shape[0]-2*margin, img.shape[1]-2*margin))
            for x in range(img.shape[1]-2*margin):
                for y in range(img.shape[0]-2*margin):
                    points[y, x, 0] = anchors[row][col]['upperLeft'][0] + y + margin + dy
                    points[y, x, 1] = anchors[row][col]['upperLeft'][1] + x + margin + dx
                    values[y, x] = img[y+margin, x+margin]

            points = points.reshape(((img.shape[0]-2*margin)*(img.shape[1]-2*margin), 2))
            values = values.reshape((img.shape[0]-2*margin)*(img.shape[1]-2*margin))

            ys = np.array([i for i in range(depth_VI.shape[0])])
            xs = np.array([i for i in range(depth_VI.shape[1])])

            X, Y = np.meshgrid(xs, ys)
            gd = griddata(points, values, (Y, X), method='linear', fill_value=0)

            single = gd.astype('float64')
            depth_VI += single

            filename = filename[-16:-4]
            cv.imwrite(os.path.join(depthFolder, 'VIP', filename+'.thumbnail.jpg'), (255*single/single.max()).astype('uint8'))
            np.save(os.path.join(depthFolder, 'VIP', filename+'.npy'), single)
            ProgramResult['VIP'][index]['single'].append(os.path.join(depthFolder, 'VIP', filename+'.npy'))

    cv.imwrite(os.path.join(depthFolder, 'VIP', str(index).zfill(6)+'.thumbnail.jpg'), (255*depth_VI/depth_VI.max()).astype('uint8'))
    np.save(os.path.join(depthFolder, 'VIP', str(index).zfill(6)+'.npy'), depth_VI)
    ProgramResult['VIP'][index]['sum'] = os.path.join(depthFolder, 'VIP', str(index).zfill(6)+'.npy')

VIP_GIF = []
for index in ProgramResult['VIP']:
    path = ProgramResult['VIP'][index]['sum']
    path = path.split('.')[0] + '.thumbnail.jpg'
    VIP_GIF.append(path)
reVIP_GIF = VIP_GIF.copy()
reVIP_GIF.reverse()
create_gif(VIP_GIF+reVIP_GIF, os.path.join(depthFolder, 'VIP', depthGIF), 1/60)
del VIP_GIF, reVIP_GIF

saveDict(os.path.join(txtLog, 'onAxisPart1Result.json'), ProgramResult)



raise ProgramSTOP
# go to part 2
############################
""" 4. ROI at each depth """
############################
# on_axis_point.json = {
#     index:{
#         'center':{
#             'upper':,
#             'lower':,
#             'left':,
#             'right':,
#             'shape':,
#             'filename':
#         }
#     }
# }
prepareFolder(os.path.join(depthFolder, 'ROI'))
ROIfilename = os.path.join(txtLog, 'on_axis_point.json')
if os.path.exists(ROIfilename):
    ROIinfo = loadDict(ROIfilename)
    ProgramResult['ROI'] = ROIfilename
else:
    print('Select 1. origin, 2. x axis, 3. y axis, respectively')
    print('If it does not have origin inside, leave it empty. (press enter/return directly)')

    ROIinfo = {}
    y0, x0 = undistCenter

    while True:
        centerShape = input('According to the nearest image, what is the inner corner shape (row, col):')
        if not centerShape == '':
            break
    if centerShape[0] == '(' and centerShape[-1] == ')': # input (m,n)
        centerShape = int(centerShape.split(',')[0][1:]), int(centerShape.split(',')[1][:-1])
    else: # input m,n
        centerShape = int(centerShape.split(',')[0]), int(centerShape.split(',')[1])

    for index in ProgramResult['VIP']:
        # each depth image
        ROIinfo[index] = {}

        # in center view, find true center
        for filename, totalShift in ProgramResult['VIP'][index]['single']:
            row, col = int(filename[-9:-7]), int(filename[-6:-4]) # xxx/xxx/000123R01C02.npy
            if (row == R) and (col == C):
                img = np.load(filename)

                shift_int = (int(totalShift[0]), int(totalShift[1]))
                for path in ProgramResult['globalCalib'][index]['subimages']:
                    if path.split(os.sep)[-1] == filename.split(os.sep)[-1]:
                        small_img = np.load(path)

                h, w = small_img.shape
                y, x = shift_int
                click_img = img[y:y+h, x:x+w]
                click_img = np.repeat(click_img, 3).reshape(click_img.shape[0], click_img.shape[1], 3)

                mouseUp = findROI(click_img)
                top, bottom, left, right = sortRect(mouseUp)
                img_ROI = img[y:y+h, x:x+w]
                img_ROI = img_ROI[top-margin:bottom+margin, left-margin:right+margin]
                img_ROI_filename = str(index).zfill(6) + 'R' + str(row).zfill(2) + 'C' + str(col).zfill(2)
                cv.imwrite(os.path.join(depthFolder, 'ROI', img_ROI_filename+'.thumbnail.jpg'), (255*img_ROI/img_ROI.max()).astype('uint8'))
                np.save(os.path.join(depthFolder, 'ROI', img_ROI_filename+'.npy'), img_ROI.astype('float64'))

                ROIinfo[index]['center'] = {
                    'upper': y + top,
                    'lower': y + bottom,
                    'left': x + left,
                    'right': x + right,
                    'shape':centerShape,
                    'filename': os.path.join(depthFolder, 'ROI', img_ROI_filename+'.npy')
                }

                break

    # estimate true center
    lines = {'points':[]}
    for row in range(centerShape[0]):
        lines['points'].append([])
        for col in range(centerShape[1]):
            lines['points'][row].append([])

    for index in ROIinfo:
        shape = ROIinfo[index]['center']['shape']
        match = getCheckerboardCorners(np.load(ROIinfo[index]['center']['filename']), shape, extension='npy')
        try:
            imagePoints = match['imagePoints'][0]
            imagePoints = sortPointCloud(imagePoints, shape, 2*margin)
            for row in range(shape[0]):
                for col in range(shape[1]):
                    lines['points'][row][col].append(imagePoints[row, col])
        except IndexError:
            pass

    lines['lines'] = {} # k, b, x0 => x1
    for row in range(centerShape[0]):
        for col in range(centerShape[1]):
            points = lines['points'][row][col]
            # to be continued



        for filename in ProgramResult['VIP'][index]['single']:
            # find center view
            row, col = int(filename[-9:-7]), int(filename[-6:-4]) # xxx/xxx/000123R01C02.npy

            if (row == R) and (col == C):
                y0, x0 = undistCenter
                img = np.load(filename)
                corners = findSubPixPoint(img, center=(y0,x0))

                if not corners == []:
                    origin, xaxis, yaxis = corners
                    xUnit = xaxis - origin if xaxis[1] > origin[1] else origin - xaxis
                    yUnit = yaxis - origin if yaxis[0] > origin[0] else origin - yaxis

                    # solve linear equation
                    # / y1 y2 \ / a1 \ = / y \
                    # \ x1 x2 / \ a2 /   \ x /
                    A = np.array([[xUnit[0], yUnit[0]],[xUnit[1], yUnit[1]]])
                    B = np.array([[y0 - origin[0]],[x0 - origin[1]]])
                    solve = np.matmul(np.linalg.inv(A), B)
                    a1, a2 = solve

                    ROIinfo[index]['coordinates'] = (a1, a2)
                    break

        for filename in ProgramResult['VIP'][index]['single']:
            # each single view
            row, col = int(filename[-9:-7]), int(filename[-6:-4]) # xxx/xxx/000123R01C02.npy
            img = np.load(filename)

            corners = findSubPixPoint(img)
            if not corners == []:
                origin, xaxis, yaxis = corners
                xUnit = xaxis - origin if xaxis[1] > origin[1] else origin - xaxis
                yUnit = yaxis - origin if yaxis[0] > origin[0] else origin - yaxis

                a1, a2 = ROIinfo[index]['coordinates']
                A = np.array([[xUnit[0],yUnit[0]],[xUnit[1],yUnit[1]]])
                B = np.matmul(A, np.array([[a1], [a2]]))

                onAxisPoint = origin + B.reshape(2)
                ROIinfo[index]['view'] = {'row':row, 'col':col, 'point':(onAxisPoint[0], onAxisPoint[1])}

    saveDict(ROIfilename, ROIinfo)
    ProgramResult['ROI'] = ROIfilename



















