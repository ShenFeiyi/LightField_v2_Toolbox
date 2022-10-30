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
from Utilities import SystemParamReader, GO, findSubPixPoint, findROI, sortRect, sortPointCloud, hsv2rgb

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

ProgramResult = loadDict(os.path.join(txtLog, 'onAxisPart1Result.json'))

############################
""" 4. ROI at each depth """
############################
# on_axis_point.json = { ROIinfo
# 	  'trueCenter':[ybar, xbar],
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

    for index in ProgramResult['VIP']:
	    # in center view, find true center
	    for filename in ProgramResult['VIP'][index]['single']:
	        row, col = int(filename[-9:-7]), int(filename[-6:-4]) # xxx/xxx/000123R01C02.npy
	        if (row == R) and (col == C):
	            img = np.load(filename)

	            upper, lower = ROIinfo[index]['center']['upper'], ROIinfo[index]['center']['lower']
	            left, right = ROIinfo[index]['center']['left'], ROIinfo[index]['center']['right']
	            img_ROI = img[upper-margin:lower+margin, left-margin:right+margin]
	            img_ROI_filename = str(index).zfill(6) + 'R' + str(row).zfill(2) + 'C' + str(col).zfill(2)
	            cv.imwrite(os.path.join(depthFolder, 'ROI', img_ROI_filename+'.thumbnail.jpg'), (255*img_ROI/img_ROI.max()).astype('uint8'))
	            np.save(os.path.join(depthFolder, 'ROI', img_ROI_filename+'.npy'), img_ROI.astype('float64'))

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
        for filename in ProgramResult['VIP'][index]['single']:
            row, col = int(filename[-9:-7]), int(filename[-6:-4]) # xxx/xxx/000123R01C02.npy
            if (row == R) and (col == C):
                img = np.load(filename)

                for path in ProgramResult['globalCalib'][index]['subimages']:
                    if path.split(os.sep)[-1] == filename.split(os.sep)[-1]:
                        small_img = np.load(path)

                h, w = small_img.shape
                y, x = anchors[R][C]['upperLeft']
                click_img = small_img
                click_img = np.repeat(click_img, 3).reshape(click_img.shape[0], click_img.shape[1], 3)
                click_img = (255*click_img/click_img.max()).astype('uint8')

                mouseUp = findROI(click_img)
                top, bottom, left, right = sortRect(mouseUp)
                img_ROI = img[y+top-margin:y+bottom+margin, x+left-margin:x+right+margin]
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
centerShape = ROIinfo[index]['center']['shape']
for row in range(centerShape[0]):
    lines['points'].append([])
    for col in range(centerShape[1]):
        lines['points'][row].append([])

for index in ProgramResult['VIP']:
    upper, left = ROIinfo[index]['center']['upper'], ROIinfo[index]['center']['left']
    shape = ROIinfo[index]['center']['shape']
    match = getCheckerboardCorners(ROIinfo[index]['center']['filename'], shape, extension='npy')
    try:
        imagePoints = match['imagePoints'][0]
        imagePoints = sortPointCloud(imagePoints, shape, 2*margin)
        for row in range(shape[0]):
            for col in range(shape[1]):
                lines['points'][row][col].append(imagePoints[row, col] + np.array([upper, left]) - np.array([margin, margin]))
    except IndexError:
        pass

lines['lines'] = [] # k, b, x0 => x1
for row in range(centerShape[0]):
    for col in range(centerShape[1]):
        points = np.array(lines['points'][row][col])
        xs = points[:, 1]
        ys = points[:, 0]
        if (ys.max()-ys.min()) > 3*(xs.max()-xs.min()):
        	# popt, pcov = curve_fit(StraightLine, ys, xs)
        	# b = -b/k
        	# k = 1/k
        	k, b = None, None
        else:
	        popt, pcov = curve_fit(StraightLine, xs, ys)
	        k, b = popt
        lines['lines'].append({'k':k, 'b':b, 'x0':xs.min(), 'x1':xs.max()})

intersectPoints = []
for l1 in lines['lines']:
	k1 = l1['k']
	if not k1 is None:

		for l2 in lines['lines']:
			k2 = l2['k']
			if (not k2 is None) and (not l1 == l2):

				b1, b2 = l1['b'], l2['b']
				x, y = np.matmul(np.linalg.inv(np.array([[-k1,1],[-k2,1]])), np.array([[b1],[b2]]))
				intersectPoints.append(np.array([x, y]))

intersectPoints = np.array(intersectPoints)
xmin, xmax = intersectPoints[:,0].min(), intersectPoints[:,0].max()
ymin, ymax = intersectPoints[:,1].min(), intersectPoints[:,1].max()
step = 1e-6
xbar, ybar = xmin, ymin
dmin = np.inf
# xRange, yRange = np.arange(xmin, xmax, step), np.arange(ymin, ymax, step)
xRange, yRange = np.arange(631.803-500*step, 631.803+500*step, step), np.arange(337.425-500*step, 337.425+500*step, step)
print('xrange', (xRange.min(), xRange.max()), 'yrange', (yRange.min(), yRange.max()))
for x in xRange:
	for y in yRange:
		d = 0
		for line in lines['lines']:
			k = line['k']
			if not k is None:
				b = line['b']
				d += np.abs((k*x-y+b)/np.sqrt(1+k**2))
		if d < dmin:
			dmin = d
			xbar, ybar = x, y
print('xbar', xbar, 'ybar', ybar)
print('error', np.array(undistCenter) - np.array([xbar, ybar]))
ROIinfo['trueCenter'] = [ybar, xbar]
saveDict(os.path.join(txtLog, 'on_axis_point.json'), ROIinfo)

# below
# presenting work

giflines = []
for index in ProgramResult['VIP']:
	filename = str(index).zfill(6) + 'R' + str(R).zfill(2) + 'C' + str(C).zfill(2)

	img = np.load('ImageLog/depth/VIP/'+filename+'.npy')
	img = np.repeat(img, 3).reshape(img.shape[0], img.shape[1], 3)
	img = (255*img/img.max()).astype('uint8')
	cv.circle(img, (int(xbar), int(ybar)), 3, (0,0,255), 2)

	for row in range(centerShape[0]):
	    for col in range(centerShape[1]):
	    	points = lines['points'][row][col]
	    	for ii, p in enumerate(points):
	    		h, s, v = 300*ii/len(points), 0.9, 0.9
	    		r, g, b = hsv2rgb(h, s, v)
		    	cv.circle(img, (int(p[1]), int(p[0])), 1, (b, g, r), 1)

	for line in lines['lines']:
		k = line['k']
		b = line['b']
		if not k is None:
			x0 = line['x0']
			x1 = line['x1']
			y0 = k*x0+b
			y1 = k*x1+b
			cv.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (255,255,255), 1)

	cv.circle(img, undistCenter, 3, (255,127,0), 2)
	img = img[
		anchors[R][C]['upperLeft'][0]+margin:anchors[R][C]['lowerRight'][0]-margin,
		anchors[R][C]['upperLeft'][1]+margin:anchors[R][C]['lowerRight'][1]-margin
		]
	cv.imwrite('ImageLog/depth/'+filename+'.jpg', img)
	giflines.append('ImageLog/depth/'+filename+'.jpg')

regiflines = giflines.copy()
regiflines.reverse()
create_gif(giflines+regiflines, 'ImageLog/depth/lines.gif', 1/30)
for file in giflines:
	os.remove(file)
del regiflines, giflines






























