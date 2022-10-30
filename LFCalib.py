# -*- coding:utf-8 -*-
import os
import json
import cv2 as cv
import numpy as np
from queue import PriorityQueue as PQ
from scipy.interpolate import griddata

from makeGIF import create_gif
from Undistort import myUndistort
from GridModel import BuildGridModel, segmentImage
from Checkerboard import unvignet, getCheckerboardCorners, calibrate, calibOptimize, undistort

from Utilities import prepareFolder, saveDict, loadDict, KeyFinder, ProgramSTOP, hsv2rgb
from Utilities import findROI, sortRect, sortPointCloud, Disparity, GO, intersect, SystemParamReader

############################
""" 0. image preparation """
############################

Reader = SystemParamReader()
SysData = Reader.read()

# RawImageFiles
root = SysData['RawImageFiles']['root']

# System
M_MLA = SysData['System']['M_MLA']
f_MLA = SysData['System']['f_MLA']
pixel = SysData['System']['pixel']
p_MLA = SysData['System']['p_MLA']
z_paraCheckerboard = SysData['System']['z_paraCheckerboard']
f_main = SysData['System']['f_main']

f_MLA /= pixel
p_MLA /= pixel
f_main /= pixel
z_paraCheckerboard /= pixel
zp_paraCheckerboard = 1/(1/(z_paraCheckerboard)+1/(f_main))

# Checkerboard
checkerboardShape = SysData['Checkerboard']['CalibCheckerboardShape']
calibSquareSize = SysData['Checkerboard']['calibSquareSize']
paraSquareSize = SysData['Checkerboard']['paraSquareSize']
undist = SysData['Checkerboard']['undist']
number_of_pixels_per_block = SysData['Checkerboard']['number_of_pixels_per_block']
margin = SysData['Checkerboard']['margin']

initAngle = SysData['Checkerboard']['initAngle']
lastAngle = SysData['Checkerboard']['lastAngle']
tiltAngleInterval = SysData['Checkerboard']['tiltAngleInterval']

initDepth = SysData['Checkerboard']['initDepth']
lastDepth = SysData['Checkerboard']['lastDepth']
depthInterval = SysData['Checkerboard']['depthInterval']

# Folder
Log = SysData['Folder']['Log']
txtLog = SysData['Folder']['txtLog']

whiteFolder = SysData['Folder']['whiteFolder']
demoFolder = SysData['Folder']['demoFolder']
calibFolder = SysData['Folder']['calibFolder']
undistFolder = SysData['Folder']['undistFolder']
paraFolder = SysData['Folder']['paraFolder']
LUFolder = SysData['Folder']['LUFolder']
GUFolder = SysData['Folder']['GUFolder']
tiltFolder = SysData['Folder']['tiltFolder']
pairFolder = SysData['Folder']['pairFolder']
VIPFolder = SysData['Folder']['VIPFolder']
opticFolder = SysData['Folder']['opticFolder']

# ImagePaths
example01_path = SysData['ImagePaths']['example01_path']
example02_path = SysData['ImagePaths']['example02_path']
whiteImage_path = SysData['ImagePaths']['whiteImage_path']
stopImage_path = SysData['ImagePaths']['stopImage_path']
paraTarget_path = SysData['ImagePaths']['paraTarget_path']
calibImagePath = SysData['ImagePaths']['calibImagePath']
otherTargetPath = SysData['ImagePaths']['otherTargetPath']

prepareFolder(Log)



########################################
""" 1. check objective lens movement """
########################################
if os.path.exists('check.gif') and os.path.exists(os.path.join(txtLog, 'Center.json')):
    RC = loadDict(os.path.join(txtLog, 'Center.json'))
    R, C = RC['R'], RC['C']
else:
    checkGIF = 'check.gif'
    checkImageList = [example01_path, example02_path]
    create_gif(checkImageList, checkGIF, 0.25)
    print('Please focus on the defects / patterns in the image.')
    print('Also count which row & column the central subaperture is. (start from 1)')

    while True:
        RC = input('Row & Column of the central subaperture (#row, #col): ')
        if not RC == '':
            break

    if RC[0] == '(' and RC[-1] == ')': # input (m,n)
        R, C = int(RC.split(',')[0][1:]), int(RC.split(',')[1][:-1])
    else: # input m,n
        R, C = int(RC.split(',')[0]), int(RC.split(',')[1])
    R -= 1
    C -= 1

    saveDict(os.path.join(txtLog, 'Center.json'), {'R':R, 'C':C})



#####################
""" 2. grid model """
#####################
# grid model
#
# rectangular shape is expected (each row/col has the same number of points)
#
# input:    stopImage_path
#
# output:   invM, allPeaks, peakGroups
#
stopArr = cv.imread(stopImage_path,0).astype('float64')
invM, allPeaks, peakGroups = BuildGridModel(stopArr)



#############################
""" 3. prepare white image"""
#############################
# white image, type = float64, [0,1]
#
# input:    whiteImage_path, invM, peakGroups
#
# output:   whiteImage, segWhiteImages, anchors
#
prepareFolder(whiteFolder)
whiteImage = cv.imread(whiteImage_path,0).astype('float64')
whiteImage = whiteImage/whiteImage.max()
segWhiteImages, anchors = segmentImage(whiteImage, invM, peakGroups)
cv.imwrite(os.path.join(whiteFolder, 'white.thumbnail.jpg'), (255*whiteImage).astype('uint8'))
np.save(os.path.join(whiteFolder, 'white.npy'), whiteImage.astype('float64'))
SysData['ImagePaths']['whiteImages'] = {}
SysData['ImagePaths']['whiteImages']['whiteImage'] = os.path.join(whiteFolder, 'white.npy')
for row in segWhiteImages:
    for col in segWhiteImages[row]:
        cv.imwrite(os.path.join(whiteFolder, 'white_R'+str(row).zfill(2)+'_C'+str(col).zfill(2)+'.thumbnail.jpg'), (255*segWhiteImages[row][col]).astype('uint8'))
        np.save(os.path.join(whiteFolder, 'white_R'+str(row).zfill(2)+'_C'+str(col).zfill(2)+'.npy'), segWhiteImages[row][col].astype('float64'))
        SysData['ImagePaths']['whiteImages']['R'+str(row).zfill(2)+'C'+str(col).zfill(2)] = os.path.join(whiteFolder, 'white_R'+str(row).zfill(2)+'_C'+str(col).zfill(2)+'.npy')



#######################
""" 4. segment demo """
#######################
example = cv.imread(example02_path,0).astype('float64')
example = unvignet(example, whiteImage)
subimages, _ = segmentImage(example, invM, peakGroups)
prepareFolder(demoFolder)
for r in subimages:
    for c in subimages[r]:
        try:
            assert isinstance(subimages[r][c], type(np.array([0])))
            cv.imwrite(os.path.join(demoFolder,'r'+str(r).zfill(2)+'c'+str(c).zfill(2)+'.thumbnail.jpg'), subimages[r][c].astype('uint8'))
        except AssertionError:
            pass



#######################################
""" 5. calibrate center subaperture """
#######################################
#
# input:    calibImagePath, whiteImage, invM, peakGroups, R, C
#
# output:   RMS, cameraMatrix, K, dist
#
prepareFolder(calibFolder)
for filename in calibImagePath:
    # unvignet
    image = cv.imread(filename,0).astype('float64')
    image = unvignet(image, whiteImage)
    subimages, _ = segmentImage(image, invM, peakGroups)
    try:
        initials = filename[-10:-4] # xxx/000123.pgm
        cv.imwrite(os.path.join(calibFolder, initials+'.thumbnail.jpg'), subimages[R][C].astype('uint8'))
        np.save(os.path.join(calibFolder, initials+'.npy'), subimages[R][C].astype('float64'))
    except (IndexError, KeyError):
        print('404: Center Subimage Not Found, ', filename)

# calibrate central subaperture system
match = getCheckerboardCorners(calibFolder, checkerboardShape, squareSize=calibSquareSize, extension='npy', visualize=True)
RMS, cameraMatrix, dist, rvecs, tvecs = calibrate(match)
sampleImage = cv.imread(os.path.join(calibFolder,initials+'.thumbnail.jpg'),0)
newCameraMatrix, _ = calibOptimize(cameraMatrix, dist, sampleImage.shape)
K = newCameraMatrix.copy()
print('\nCamera Matrix:\n',K,'\nDistortion coefficients:\n',dist,'\n')

# undistort
prepareFolder(undistFolder)
ccss_images = os.listdir(calibFolder)
ccss_images = [name for name in ccss_images if name.endswith('npy')]
for ccss in ccss_images:
    udst = undistort(os.path.join(calibFolder,ccss), cameraMatrix, dist, K)
    cv.imwrite(os.path.join(undistFolder,ccss.split('.')[0]+'.thumbnail.jpg'), udst.astype('uint8'))



############################
""" 6. check parallelism """
############################
#
# input:    margin, paraTarget_path, whiteImage, invM, peakGroups, anchors
#
# output:   paraCheckerboard, subParas
#
# crop all subimages
prepareFolder(paraFolder)
paraCheckerboard = cv.imread(paraTarget_path,0).astype('float64')
paraCheckerboard = unvignet(paraCheckerboard, whiteImage)
subParas, _ = segmentImage(paraCheckerboard, invM, peakGroups)
for r in subParas:
    for c in subParas[r]:
        try:
            assert isinstance(subParas[r][c], type(np.array([0])))
            paraname = paraTarget_path.split('.')[0][-6:]+'_R'+str(r).zfill(2)+'_C'+str(c).zfill(2)
            cv.imwrite(os.path.join(paraFolder, paraname+'.thumbnail.jpg'), subParas[r][c].astype('uint8'))
            np.save(os.path.join(paraFolder, paraname+'.npy'), subParas[r][c].astype('float64'))
        except AssertionError:
            pass

# find ROI in center subimage
print('Check parallelism of "parallel" target')
if os.path.exists(os.path.join(txtLog,'centerSubParaTargetROI.json')):
    data = loadDict(os.path.join(txtLog,'centerSubParaTargetROI.json'))
    top, bottom, left, right = data['top'], data['bottom'], data['left'], data['right']
    centerShape = data['shape']
else:
    mouseUp = findROI(subParas[R][C], name='paraCheckerboard')
    top, bottom, left, right = sortRect(mouseUp)

    # find image points in center subimage
    while True:
        centerShape = input('Inner corners in center subaperture (row,col): ')
        if not centerShape == '':
            break

    if centerShape[0] == '(' and centerShape[-1] == ')': # input (m,n)
        centerShape = int(centerShape.split(',')[0][1:]), int(centerShape.split(',')[1][:-1])
    else: # input m,n
        centerShape = int(centerShape.split(',')[0]), int(centerShape.split(',')[1])

    data = {'top':top, 'bottom':bottom, 'left':left, 'right':right, 'shape':centerShape}
    saveDict(os.path.join(txtLog,'centerSubParaTargetROI.json'), data)

centerSubParaTargetROI = subParas[R][C][top-margin:bottom+margin, left-margin:right+margin]
cv.imwrite(os.path.join(paraFolder,'centerSubParaTargetROI.thumbnail.jpg'), centerSubParaTargetROI.astype('uint8'))
np.save(os.path.join(paraFolder,'centerSubParaTargetROI.npy'), centerSubParaTargetROI.astype('float64'))

match = getCheckerboardCorners(os.path.join(paraFolder,'centerSubParaTargetROI.npy'), centerShape, extension='npy')
imagePoints = match['imagePoints'][0]

# generate object points for center subimage
objectPoints = np.zeros((centerShape[0]*centerShape[1], 3), dtype='float64')
objectPoints[:, :2] = np.indices(centerShape).T.reshape(-1, 2)
objectPoints *= paraSquareSize

# estimate rvec & tvec
retval, rvec, tvec = cv.solvePnP(objectPoints, imagePoints, K, dist)
# R vector to R matrix
rmat, jacobian = cv.Rodrigues(rvec)
print('\nRotation matrix for "parallel" checkerboard:\n', rmat, '\n')

data = {}
data['ReprojectionError'] = RMS
data['cameraMatrix'] = [
    [K[0,0],K[0,1],K[0,2]],
    [K[1,0],K[1,1],K[1,2]],
    [K[2,0],K[2,1],K[2,2]]
    ]
data['dist'] = [dist[0,0],dist[0,1],dist[0,2],dist[0,3],dist[0,4]]
data['R'] = [
    [rmat[0,0],rmat[0,1],rmat[0,2]],
    [rmat[1,0],rmat[1,1],rmat[1,2]],
    [rmat[2,0],rmat[2,1],rmat[2,2]]
    ]
saveDict(os.path.join(txtLog, 'CameraParams.json'), data)



##########################
""" 7. local undistort """
##########################
#
# input:    paraCheckerboard, subParas, cameraMatrix, dist, K, anchors
#
# output:   paraCheckerboard_LU
#
# using subimages obtained above
prepareFolder(LUFolder)
# save each locally undistorted subimage
# and merge them back
paraCheckerboard_LU = np.zeros(paraCheckerboard.shape,dtype='float64')
for row in subParas:
    for col in subParas[row]:
        paraname = paraTarget_path.split('.')[0][-6:]+'_R'+str(row).zfill(2)+'_C'+str(col).zfill(2)
        udst = undistort(os.path.join(paraFolder, paraname+'.npy'), cameraMatrix, dist, K)
        cv.imwrite(os.path.join(LUFolder, paraname+'.thumbnail.jpg'), udst.astype('uint8'))

        y, x = anchors[row][col]['upperLeft']
        h, w = subParas[row][col].shape
        paraCheckerboard_LU[y:y+h, x:x+w] = paraCheckerboard_LU[y:y+h, x:x+w] + udst



###########################
""" 8. global undistort """
###########################
#
# input:    peakGroups, R, C, paraCheckerboard_LU, undist
#
# output:   paraCheckerboard_GU, subGUs
#
prepareFolder(GUFolder)
undistCenter = peakGroups['H'][R][C]
undistCenter = (undistCenter[1], undistCenter[0])
paraCheckerboard_GU = myUndistort(paraCheckerboard_LU, undist, undistCenter, method='linear')
cv.imwrite(os.path.join(GUFolder,'GU.thumbnail.jpg'), paraCheckerboard_GU.astype('uint8'))
np.save(os.path.join(GUFolder,'GU.npy'), paraCheckerboard_GU.astype('float64'))

SysData['ImagePaths']['subGUs'] = {}
SysData['ImagePaths']['subGUs']['paraCheckerboard_GU'] = os.path.join(GUFolder,'GU.npy')
subGUs, _ = segmentImage(paraCheckerboard_GU, invM, peakGroups)
for r in subGUs:
    for c in subGUs[r]:
        try:
            assert isinstance(subGUs[r][c], type(np.array([0])))
            filename = 'GU'+'_R'+str(r).zfill(2)+'_C'+str(c).zfill(2)
            cv.imwrite(os.path.join(GUFolder, filename+'.thumbnail.jpg'), subGUs[r][c].astype('uint8'))
            np.save(os.path.join(GUFolder, filename+'.npy'), subGUs[r][c].astype('float64'))
            SysData['ImagePaths']['subGUs']['R'+str(r).zfill(2)+'C'+str(c).zfill(2)] = os.path.join(GUFolder, filename+'.npy')
        except AssertionError:
            pass



#########################################
""" 9. common area in adjacent views """
########################################
#
# input:    paraTarget_path, margin, subGUs
#
# output:   do_not_input_again
#
prepareFolder(pairFolder)
ROWS, COLS = len(peakGroups['H']), len(peakGroups['V'])

do_not_input_again = {}

initials = paraTarget_path[-10:-4] # xxx/000123.pgm
maatchingFileName = 'position_and_shape_pixel_to_pixel_'+initials+'.json'

if os.path.exists(os.path.join(txtLog, maatchingFileName)):
    # load data
    do_not_input_again = loadDict(os.path.join(txtLog, maatchingFileName))

    for row in range(ROWS):
        for col in range(COLS):
            for direction, shiftView in [('right', (0,1)), ('lower', (1,0))]:
                if direction in do_not_input_again[row][col]:
                    curROI = do_not_input_again[row][col][direction]['cur']
                    top, bottom, left, right = curROI['top'], curROI['bottom'], curROI['left'], curROI['right']
                    ROI1 = subGUs[row][col][top-margin:bottom+margin, left-margin:right+margin]
                    filename = do_not_input_again[row][col][direction]['cur']['filename'].split('.')[0]
                    cv.imwrite(os.path.join(pairFolder,filename+'.thumbnail.jpg'),ROI1.astype('uint8'))
                    np.save(os.path.join(pairFolder,filename+'.npy'),ROI1.astype('uint8'))

                    nextROI = do_not_input_again[row][col][direction]['next']
                    top, bottom, left, right = nextROI['top'], nextROI['bottom'], nextROI['left'], nextROI['right']
                    ROI2 = subGUs[row+shiftView[0]][col+shiftView[1]][top-margin:bottom+margin, left-margin:right+margin]
                    filename = do_not_input_again[row][col][direction]['next']['filename'].split('.')[0]
                    cv.imwrite(os.path.join(pairFolder,filename+'.thumbnail.jpg'),ROI2.astype('uint8'))
                    np.save(os.path.join(pairFolder,filename+'.npy'),ROI2.astype('uint8'))

else:
    # do_not_input_again = {
    #     row:{
    #         col:{
    #             'right':{
    #                 'shape':shape,
    #                 'cur':{'top':top, 'bottom':bottom, 'left':left, 'right':right, 'filename':filename},
    #                 'next':{'top':top, 'bottom':bottom, 'left':left, 'right':right, 'filename':filename}
    #                 }
    #             'lower':{
    #                 'shape':shape,
    #                 'cur':{'top':top, 'bottom':bottom, 'left':left, 'right':right, 'filename':filename},
    #                 'next':{'top':top, 'bottom':bottom, 'left':left, 'right':right, 'filename':filename}
    #             }
    #         }
    #     }
    # }

    # calibrate adjacent subapertures
    for row in range(ROWS):

        if not row in do_not_input_again:
            do_not_input_again[row] = {}

        for col in range(COLS):

            if not col in do_not_input_again[row]:
                do_not_input_again[row][col] = {}

            if (row==ROWS-1) and (not col==COLS-1):
                # last row except last one, consider right side only
                RIGHT, LOWER = True, False
            elif (not row==ROWS-1) and (col==COLS-1):
                # last column except last one, consider lower side only
                RIGHT, LOWER = False, True
            elif (row==ROWS-1) and (col==COLS-1):
                # lower-right subaperture, do nothing
                RIGHT, LOWER = False, False
            else:
                # do both right side and lower side
                RIGHT, LOWER = True, True

            if RIGHT:
                curImage = subGUs[row][col]
                nextImage = subGUs[row][col+1]
                print('Finding corresponding points in image[{:d},{:d}] and [{:d},{:d}]'.format(row+1,col+1,row+1,col+2))

                do_not_input_again[row][col]['right'] = {}

                cv.namedWindow('reference')
                cv.imshow('reference', nextImage.astype('uint8'))
                cv.moveWindow('reference', 300, 0)
                mouseUp = findROI(curImage)
                top, bottom, left, right = sortRect(mouseUp)
                ROI1 = curImage[top-margin:bottom+margin, left-margin:right+margin]
                filename = initials+'_R'+str(row).zfill(2)+'_C'+str(col).zfill(2)+'_ROI_R1'
                cv.imwrite(os.path.join(pairFolder,filename+'.thumbnail.jpg'),ROI1.astype('uint8'))
                np.save(os.path.join(pairFolder,filename+'.npy'), ROI1.astype('float64'))

                do_not_input_again[row][col]['right']['cur'] = {}
                do_not_input_again[row][col]['right']['cur']['top'] = top
                do_not_input_again[row][col]['right']['cur']['bottom'] = bottom
                do_not_input_again[row][col]['right']['cur']['left'] = left
                do_not_input_again[row][col]['right']['cur']['right'] = right
                do_not_input_again[row][col]['right']['cur']['filename'] = filename+'.npy'

                cv.namedWindow('reference')
                cv.imshow('reference', curImage.astype('uint8'))
                cv.moveWindow('reference', 300, 0)
                mouseUp = findROI(nextImage)
                top, bottom, left, right = sortRect(mouseUp)
                ROI2 = nextImage[top-margin:bottom+margin, left-margin:right+margin]
                filename = initials+'_R'+str(row).zfill(2)+'_C'+str(col).zfill(2)+'_ROI_R2'
                cv.imwrite(os.path.join(pairFolder,filename+'.thumbnail.jpg'),ROI2.astype('uint8'))
                np.save(os.path.join(pairFolder,filename+'.npy'), ROI2.astype('float64'))

                do_not_input_again[row][col]['right']['next'] = {}
                do_not_input_again[row][col]['right']['next']['top'] = top
                do_not_input_again[row][col]['right']['next']['bottom'] = bottom
                do_not_input_again[row][col]['right']['next']['left'] = left
                do_not_input_again[row][col]['right']['next']['right'] = right
                do_not_input_again[row][col]['right']['next']['filename'] = filename+'.npy'

                while True:
                    matchShape = input('Inner corners of matching area (row, col):')
                    if not matchShape == '':
                        break

                if matchShape[0] == '(' and matchShape[-1] == ')': # input (m,n)
                    matchShape = int(matchShape.split(',')[0][1:]), int(matchShape.split(',')[1][:-1])
                else: # input m,n
                    matchShape = int(matchShape.split(',')[0]), int(matchShape.split(',')[1])

                do_not_input_again[row][col]['right']['shape'] = matchShape

            if LOWER:
                curImage = subGUs[row][col]
                nextImage = subGUs[row+1][col]
                print('Finding corresponding points in image[{:d},{:d}] and [{:d},{:d}]'.format(row+1,col+1,row+2,col+1))

                do_not_input_again[row][col]['lower'] = {}

                cv.namedWindow('reference')
                cv.imshow('reference', nextImage.astype('uint8'))
                cv.moveWindow('reference', 300, 0)
                mouseUp = findROI(curImage)
                top, bottom, left, right = sortRect(mouseUp)
                ROI1 = curImage[top-margin:bottom+margin, left-margin:right+margin]
                filename = initials+'_R'+r.zfill(2)+'_C'+c.zfill(2)+'_ROI_L1'
                cv.imwrite(os.path.join(pairFolder,filename+'.thumbnail.jpg'),ROI1.astype('uint8'))
                np.save(os.path.join(pairFolder,filename+'.npy'), ROI1.astype('float64'))

                do_not_input_again[row][col]['lower']['cur'] = {}
                do_not_input_again[row][col]['lower']['cur']['top'] = top
                do_not_input_again[row][col]['lower']['cur']['bottom'] = bottom
                do_not_input_again[row][col]['lower']['cur']['left'] = left
                do_not_input_again[row][col]['lower']['cur']['right'] = right
                do_not_input_again[row][col]['lower']['cur']['filename'] = filename+'.npy'

                cv.namedWindow('reference')
                cv.imshow('reference', curImage.astype('uint8'))
                cv.moveWindow('reference', 300, 0)
                mouseUp = findROI(nextImage)
                top, bottom, left, right = sortRect(mouseUp)
                ROI2 = nextImage[top-margin:bottom+margin, left-margin:right+margin]
                filename = initials+'_R'+r.zfill(2)+'_C'+c.zfill(2)+'_ROI_L2'
                cv.imwrite(os.path.join(pairFolder,filename+'.thumbnail.jpg'),ROI2.astype('uint8'))
                np.save(os.path.join(pairFolder,filename+'.npy'), ROI2.astype('float64'))

                do_not_input_again[row][col]['lower']['next'] = {}
                do_not_input_again[row][col]['lower']['next']['top'] = top
                do_not_input_again[row][col]['lower']['next']['bottom'] = bottom
                do_not_input_again[row][col]['lower']['next']['left'] = left
                do_not_input_again[row][col]['lower']['next']['right'] = right
                do_not_input_again[row][col]['lower']['next']['filename'] = filename+'.npy'

                while True:
                    matchShape = input('Inner corners of matching area (row, col):')
                    if not matchShape == '':
                        break

                if matchShape[0] == '(' and matchShape[-1] == ')': # input (m,n)
                    matchShape = int(matchShape.split(',')[0][1:]), int(matchShape.split(',')[1][:-1])
                else: # input m,n
                    matchShape = int(matchShape.split(',')[0]), int(matchShape.split(',')[1])

                do_not_input_again[row][col]['lower']['shape'] = matchShape

    # write data
    saveDict(os.path.join(txtLog, maatchingFileName), do_not_input_again)



#####################################
""" 10. find corresponding points """
#####################################
#
# input:    do_not_input_again, margin, anchors, ROWS, COLS
#
# output:   info
#
# info = do_not_input_again = {
#     row:{
#         col:{
#             'right':{
#                 'shape':shape,
#                 'cur':{'top':top, 'bottom':bottom, 'left':left, 'right':right, 'filename':filename},
#                 'next':{'top':top, 'bottom':bottom, 'left':left, 'right':right, 'filename':filename},
#                 'shift':shift # (dy, dx), next -> cur, cur = next + shift
#                 }
#             'lower':{
#                 'shape':shape,
#                 'cur':{'top':top, 'bottom':bottom, 'left':left, 'right':right, 'filename':filename},
#                 'next':{'top':top, 'bottom':bottom, 'left':left, 'right':right, 'filename':filename},
#                 'shift':shift # (dy, dx), next -> cur, cur = next + shift
#             }
#             'left':{
#                 'shift':shift
#             }
#             'upper':{
#                 'shift':shift
#             }
#         }
#     }
# }

info = loadDict(os.path.join(txtLog, maatchingFileName))
fail_to_detect = []

for row in range(ROWS):
    for col in range(COLS):
        for direction, shiftView in [('right',(0,1)), ('lower',(1,0))]:
            if direction in info[row][col]:
                shape = info[row][col][direction]['shape']
                curImageInfo = info[row][col][direction]['cur']
                nextImageInfo = info[row][col][direction]['next']

                curMatch = getCheckerboardCorners(os.path.join(pairFolder,curImageInfo['filename']),shape,visualize=True,extension='npy')
                nextMatch = getCheckerboardCorners(os.path.join(pairFolder,nextImageInfo['filename']),shape,visualize=True,extension='npy')

                try:
                    curImagePoints = curMatch['imagePoints'][0]
                    nextImagePoints = nextMatch['imagePoints'][0]
                except IndexError:

                    # try one more time
                    shape = (shape[0]-1, shape[1]-1) # Check visualization result, may have shift affect
                    curMatch = getCheckerboardCorners(os.path.join(pairFolder,curImageInfo['filename']),shape,visualize=True,extension='npy')
                    nextMatch = getCheckerboardCorners(os.path.join(pairFolder,nextImageInfo['filename']),shape,visualize=True,extension='npy')

                    try:
                        curImagePoints = curMatch['imagePoints'][0]
                        nextImagePoints = nextMatch['imagePoints'][0]
                    except IndexError:
                        fail_to_detect.append(curImageInfo['filename'])
                        break # go to next column

                curImagePoints = sortPointCloud(np.flip(curImagePoints), shape, margin) # y,x
                nextImagePoints = sortPointCloud(np.flip(nextImagePoints), shape, margin) # y,x

                curOrig1 = anchors[row][col]['upperLeft'] # y,x
                curOrig2 = (info[row][col][direction]['cur']['top'], info[row][col][direction]['cur']['left']) # y,x

                nextOrig1 = anchors[row+shiftView[0]][col+shiftView[1]]['upperLeft'] # y,x
                nextOrig2 = (info[row][col][direction]['next']['top'], info[row][col][direction]['next']['left']) # y,x

                curPointsGlobal = curImagePoints + np.array(curOrig1) + np.array(curOrig2)
                nextPointsGlobal = nextImagePoints + np.array(nextOrig1) + np.array(nextOrig2)

                curPointsGlobal_mean = curPointsGlobal.reshape((int(len(curPointsGlobal.flatten())/2),2)).mean(axis=0)
                nextPointsGlobal_mean = nextPointsGlobal.reshape((int(len(nextPointsGlobal.flatten())/2),2)).mean(axis=0)

                # cur = next + shift
                shift = (curPointsGlobal_mean[0]-nextPointsGlobal_mean[0], curPointsGlobal_mean[1]-nextPointsGlobal_mean[1]) # y, x
                info[row][col][direction]['shift'] = tuple(np.array(shift))

print('Fail to detect corners in ', len(fail_to_detect), ' subimages in ', paraTarget_path)
print('They are ', fail_to_detect, '\n')

for row in range(ROWS):
    for col in range(COLS):
        for direction, shiftView, opposite in [('right',(0,1),'left'), ('lower',(1,0),'upper')]:
            if direction in info[row][col]:
                try:
                    shift = info[row][col][direction]['shift']
                    shift = tuple(-np.array(shift))
                    info[row+shiftView[0]][col+shiftView[1]][opposite] = {'shift':shift}
                except KeyError:
                    pass

filename = maatchingFileName.split('.')[0] + '_with_shift_read_only.json'
saveDict(os.path.join(txtLog, filename), info)



###############################
""" 11. tilted checkerboard """
###############################
#
# input:    allTiltTargetPath, whiteImage, invM, peakGroups, cameraMatrix, dist, K, anchors, undist, undistCenter
#
# output:   tiltTarget_GU
#
prepareFolder(tiltFolder)
allTiltTargetPath = otherTargetPath['tilt']
for tiltPath in allTiltTargetPath:
    tiltTarget = cv.imread(tiltPath,0).astype('float64')
    tiltTarget = unvignet(tiltTarget, whiteImage)
    tiltTarget_LU = np.zeros(tiltTarget.shape, dtype=tiltTarget.dtype)
    subTilt, _  = segmentImage(tiltTarget, invM, peakGroups)
    for r in subTilt:
        for c in subTilt[r]:
            try:
                assert isinstance(subTilt[r][c], type(np.array([0])))
                initials = tiltPath[-10:-4]
                filename = initials + '_R' + str(r).zfill(2) + '_C' + str(c).zfill(2)
                cv.imwrite(os.path.join(tiltFolder, filename+'.thumbnail.jpg'), subTilt[r][c].astype('uint8'))
                np.save(os.path.join(tiltFolder, filename+'.npy'), subTilt[r][c].astype('float64'))

                udst = undistort(os.path.join(tiltFolder, filename+'.thumbnail.jpg'), cameraMatrix, dist, K)
                y, x = anchors[r][c]['upperLeft']
                h, w = subTilt[r][c].shape
                tiltTarget_LU[y:y+h, x:x+w] = tiltTarget_LU[y:y+h, x:x+w] + udst
            except AssertionError:
                pass
    cv.imwrite(os.path.join(tiltFolder,initials+'_LU.thumbnail.jpg'), tiltTarget_LU.astype('uint8'))
    print(tiltPath, end=' ')
    tiltTarget_GU = myUndistort(tiltTarget_LU, undist, undistCenter, method='linear')
    cv.imwrite(os.path.join(tiltFolder,initials+'_GU.thumbnail.jpg'), tiltTarget_GU.astype('uint8'))



########################################
""" 12. generate virtual image plane """
########################################
#
# input:    paraCheckerboard, subGUs, segWhiteImages, anchors, ROWS, COLS, info
#
# output:   subVIs
#
prepareFolder(VIPFolder)

# 12.1. to demonstrate which part is used
commonArea = np.zeros((paraCheckerboard.shape[0],paraCheckerboard.shape[1],4), dtype='float64')
commonArea[:,:,-1] = 75*(255/100) # alpha channel, 0-255, 0-100%

for row in range(ROWS):
    for col in range(COLS):
        for direction, shiftView in [('right', (0,1)), ('lower', (1,0))]:
            if direction in info[row][col]:
                shape = info[row][col][direction]['shape']
                curImageInfo = info[row][col][direction]['cur']
                nextImageInfo = info[row][col][direction]['next']
                if not 'shift' in info[row][col][direction]:
                    continue
                else:
                    # sensor
                    curOrig1 = anchors[row][col]['upperLeft'] # y,x
                    curOrig2 = (info[row][col][direction]['cur']['top'], info[row][col][direction]['cur']['left']) # y,x
                    curOrig = np.array(curOrig1) + np.array(curOrig2)
                    curImage = np.load(os.path.join(pairFolder,curImageInfo['filename']))

                    nextOrig1 = anchors[row+shiftView[0]][col+shiftView[1]]['upperLeft'] # y,x
                    nextOrig2 = (info[row][col][direction]['next']['top'], info[row][col][direction]['next']['left']) # y,x
                    nextOrig = np.array(nextOrig1) + np.array(nextOrig2)
                    nextImage = np.load(os.path.join(pairFolder,nextImageInfo['filename']))

                    commonArea[curOrig[0]:curOrig[0]+curImage.shape[0], curOrig[1]:curOrig[1]+curImage.shape[1], 0] = curImage.astype('float64')
                    commonArea[curOrig[0]:curOrig[0]+curImage.shape[0], curOrig[1]:curOrig[1]+curImage.shape[1], 1] = curImage.astype('float64')
                    commonArea[nextOrig[0]:nextOrig[0]+nextImage.shape[0], nextOrig[1]:nextOrig[1]+nextImage.shape[1], 2] = nextImage.astype('float64')

commonArea *= (255/commonArea.max())
cv.imwrite(os.path.join(VIPFolder, 'commonArea.thumbnail.jpg'), commonArea.astype('uint8'))

# 12.2.
paraCheckerboard_VI = np.zeros(paraCheckerboard.shape, dtype='float64')
whiteImage_VI = np.zeros(paraCheckerboard.shape, dtype='float64')

# 12.2.1. for images stored in variables
VIs = [paraCheckerboard_VI, whiteImage_VI]
subimages_ready_to_use = [subGUs, segWhiteImages] # corresponds with VIs
VInames = ['virtualImage.thumbnail.jpg', 'whiteImage.thumbnail.jpg']

dir_dict = {'upper':(-1,0), 'lower':(1,0), 'left':(0,-1), 'right':(0,1)}

for row in range(ROWS):
    for col in range(COLS):

        totalShift = np.array([0, 0], dtype='float64')
        steps = GO((row,col), (R,C), info)
        curR, curC = row, col
        for step in steps:
            shift = info[curR][curC][step]['shift']
            totalShift[0] -= shift[0]
            totalShift[1] -= shift[1]
            dr, dc = dir_dict[step]
            curR += dr
            curC += dc

        dy, dx = totalShift
        print(row, col, totalShift, steps)

        if (totalShift[0] == int(totalShift[0])) and (totalShift[1] == int(totalShift[1])):
            for i in range(len(VIs)):
                VI = VIs[i]
                seg = subimages_ready_to_use[i]
                h, w = seg[row][col].shape
                y0, x0 = anchors[row][col]['upperLeft']
                VI[int(y0+dy+margin):int(y0+dy+h-margin), int(x0+dx+margin):int(x0+dx+w-margin)] += seg[row][col][margin:-margin, margin:-margin].astype('float64')
        else:
            print('Interpolating subimage...')
            for i in range(len(VIs)):
                VI = VIs[i]
                seg = subimages_ready_to_use[i]

                points = np.zeros((seg[row][col].shape[0]-2*margin, seg[row][col].shape[1]-2*margin, 2))
                values = np.zeros((seg[row][col].shape[0]-2*margin, seg[row][col].shape[1]-2*margin))
                for x in range(seg[row][col].shape[1]-2*margin):
                    for y in range(seg[row][col].shape[0]-2*margin):
                        points[y, x, 0] = anchors[row][col]['upperLeft'][0] + y + margin + dy
                        points[y, x, 1] = anchors[row][col]['upperLeft'][1] + x + margin + dx
                        values[y, x] = seg[row][col][y+margin,x+margin]

                points = points.reshape(((seg[row][col].shape[0]-2*margin)*(seg[row][col].shape[1]-2*margin), 2))
                values = values.reshape((seg[row][col].shape[0]-2*margin)*(seg[row][col].shape[1]-2*margin))

                ys = np.array([i for i in range(VI.shape[0])])
                xs = np.array([i for i in range(VI.shape[1])])

                X, Y = np.meshgrid(xs, ys)
                gd = griddata(points, values, (Y, X), method='linear', fill_value=0)

                VI += gd.astype('float64')

for i in range(len(VIs)):
    VI = VIs[i]
    VI *= (255/VI.max())
    cv.imwrite(os.path.join(VIPFolder, VInames[i]), VI.astype('uint8'))

# 12.2.2. for images that needs to load from folder
subVIs = {}
for ii, tiltPath in enumerate(allTiltTargetPath):

    print(tiltPath)
    initials = tiltPath[-10:-4]
    tiltTarget_VI = np.zeros(paraCheckerboard.shape, dtype='float64')
    subVIs[tiltPath] = {}

    for row in range(ROWS):
        subVIs[tiltPath][row] = {}
        for col in range(COLS):

            filename = initials + '_R' + str(row).zfill(2) + '_C' + str(col).zfill(2)
            subTilt = np.load(os.path.join(tiltFolder, filename+'.npy'))

            totalShift = np.array([0, 0], dtype='float64')
            steps = GO((row,col), (R,C), info)
            curR, curC = row, col
            for step in steps:
                shift = info[curR][curC][step]['shift']
                totalShift[0] -= shift[0]
                totalShift[1] -= shift[1]
                dr, dc = dir_dict[step]
                curR += dr
                curC += dc

            dy, dx = totalShift

            if (totalShift[0] == int(totalShift[0])) and (totalShift[1] == int(totalShift[1])):
                h, w = subTilt.shape
                y0, x0 = anchors[row][col]['upperLeft']
                tiltTarget_VI[int(y0+dy+margin):int(y0+dy+h-margin), int(x0+dx+margin):int(x0+dx+w-margin)] += subTilt[margin:-margin, margin:-margin].astype('float64')
                subVIs[tiltPath][row][col] = np.zeros(tiltTarget_VI.shape, dtype='float64')
                subVIs[tiltPath][row][col][int(y0+dy+margin):int(y0+dy+h-margin), int(x0+dx+margin):int(x0+dx+w-margin)] += subTilt[margin:-margin, margin:-margin].astype('float64')
            else:
                points = np.zeros((seg[row][col].shape[0]-2*margin, seg[row][col].shape[1]-2*margin, 2))
                values = np.zeros((subTilt.shape[0]-2*margin, subTilt.shape[1]-2*margin))
                for x in range(subTilt.shape[1]-2*margin):
                    for y in range(subTilt.shape[0]-2*margin):
                        points[y, x, 0] = anchors[row][col]['upperLeft'][0] + y + margin + dy
                        points[y, x, 1] = anchors[row][col]['upperLeft'][1] + x + margin + dx
                        values[y, x] = subTilt[y+margin,x+margin]

                points = points.reshape(((seg[row][col].shape[0]-2*margin)*(seg[row][col].shape[1]-2*margin), 2))
                values = values.reshape((subTilt.shape[0]-2*margin)*(subTilt.shape[1]-2*margin))

                ys = np.array([i for i in range(tiltTarget_VI.shape[0])])
                xs = np.array([i for i in range(tiltTarget_VI.shape[1])])

                X, Y = np.meshgrid(xs, ys)
                gd = griddata(points, values, (Y, X), method='linear', fill_value=0)

                tiltTarget_VI += gd.astype('float64')

                subVIs[tiltPath][row][col] = gd

    tiltTarget_VI *= (255/tiltTarget_VI.max())
    cv.imwrite(os.path.join(VIPFolder, initials+'.thumbnail.jpg'), tiltTarget_VI.astype('uint8'))



###################################
""" 13. save calibration result """
###################################

CalibrationResult = {}
CalibrationResult['invM'] = {}
CalibrationResult['invM']['value'] = list(invM.reshape(-1))
CalibrationResult['invM']['shape'] = invM.shape
CalibrationResult['allPeaks'] = allPeaks
CalibrationResult['peakGroups'] = peakGroups
CalibrationResult['anchors'] = anchors
CalibrationResult['undistCenter'] = undistCenter
CalibrationResult['ROWS'] = ROWS
CalibrationResult['COLS'] = COLS

CalibrationResult['CCSS'] = {}
CalibrationResult['CCSS']['cameraMatrix'] = {}
CalibrationResult['CCSS']['cameraMatrix']['value'] = list(cameraMatrix.reshape(-1))
CalibrationResult['CCSS']['cameraMatrix']['shape'] = cameraMatrix.shape
CalibrationResult['CCSS']['K'] = {}
CalibrationResult['CCSS']['K']['value'] = list(K.reshape(-1))
CalibrationResult['CCSS']['K']['shape'] = K.shape
CalibrationResult['CCSS']['dist'] = {}
CalibrationResult['CCSS']['dist']['value'] = list(dist.reshape(-1))
CalibrationResult['CCSS']['dist']['shape'] = dist.shape

CalibrationResult['correspondence'] = info

saveDict(os.path.join(txtLog, 'CalibrationResult.json'), CalibrationResult)
saveDict(os.path.join(txtLog, 'SystemParamsFullEditionReadOnly.json'), SysData)












