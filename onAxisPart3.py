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
# ROIinfo = on_axis_point.json = {
# 	'trueCenter':[ybar, xbar],
# 	index:{
# 		'center':{
# 			'upper':,
# 			'lower':,
# 			'left':,
# 			'right':,
# 			'shape':,
# 			'filename':
# 		}
# 		'views':[{
# 					'row':row,
# 					'col':col,
# 					'o':(x, y),
# 					'x': xpoint,
# 					'y': ypoint
# 		}, ...]
# 	}
# }
prepareFolder(os.path.join(depthFolder, 'ROI'))
ROIfilename = os.path.join(txtLog, 'on_axis_point.json')
ROIinfo = loadDict(ROIfilename)
ProgramResult['ROI'] = ROIfilename

dir_dict = {'upper':(-1,0), 'lower':(1,0), 'left':(0,-1), 'right':(0,1)}
trueCenter = ROIinfo['trueCenter'] # y, x

for index in ProgramResult['VIP']:
	if not 'views' in ROIinfo[index]:

		ROIinfo[index]['views'] = []
		for filename in ProgramResult['VIP'][index]['single']:
			row, col = int(filename[-9:-7]), int(filename[-6:-4]) # xxx/xxx/012345R01C02.npy

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
			dy, dx = -totalShift

			centerInThisView = (trueCenter[0]+dy, trueCenter[1]+dx)
			centerInThisView -= np.array(anchors[row][col]['upperLeft'])

			for path in ProgramResult['globalCalib'][index]['subimages']:
				if path.split(os.sep)[-1] == filename.split(os.sep)[-1]:
					img = np.load(path)
					break
			click_img = np.repeat(img, 3).reshape(img.shape[0], img.shape[1], 3).astype('uint8')
			cv.circle(click_img, (int(centerInThisView[1]), int(centerInThisView[0])), 1, (0,0,255), 1)

			corners = findSubPixPoint(click_img)
			if len(corners) == 3:
				origin, xaxis, yaxis = corners
				ROIinfo[index]['views'].append(
						{
							'row':row, 'col':col,
							'o':[float(origin[0]), float(origin[1])], # x, y
							'x':[float(xaxis[0]), float(xaxis[1])],
							'y':[float(yaxis[0]), float(yaxis[1])]
							}
					)

saveDict(ROIfilename, ROIinfo)

for index in ProgramResult['VIP']:
	for view in ROIinfo[index]['views']:
		row, col = view['row'], view['col']
		for key in ['o', 'x', 'y']:
			view[key] = np.array([view[key][1], view[key][0]]) + np.array(anchors[row][col]['upperLeft'])

# determine on axis point coordinate in center view
solutions = {'a1':[], 'a2':[]} # each depth, unique
for index in ProgramResult['VIP']:
	for view in ROIinfo[index]['views']:
		row, col = view['row'], view['col']
		if (row == R) and (col == C):

			o, x, y = view['o'], view['x'], view['y'] # y, x
			xUnit = np.array(x) - np.array(o) if x[1] > o[1] else np.array(o) - np.array(x)
			yUnit = np.array(y) - np.array(o) if y[0] > o[0] else np.array(o) - np.array(y)
			vecCenter = np.array(trueCenter) - np.array(o) # y, x

			A = np.array([[xUnit[0], yUnit[0]], [xUnit[1], yUnit[1]]])
			B = np.array([[vecCenter[0]], [vecCenter[1]]])

			solve = np.matmul(np.linalg.inv(A), B)
			a1, a2 = float(solve[0]), float(solve[1])
			solutions['a1'].append(a1)
			solutions['a2'].append(a2)

			break

# from matplotlib import pyplot as plt
# fig = plt.figure(figsize=(12,9))
# ax = fig.add_subplot(1,1,1)
# ax.scatter(solutions['a1'], solutions['a2'])
# ax.set_title('Estimated Center Coordinates under H&V Basis', fontsize=16)
# ax.set_xlabel(r'$\hat{x}$', fontsize=16)
# ax.set_ylabel(r'$\hat{y}$', fontsize=16)
# ax.axis('equal')
# plt.text(solutions['a1'][-1]-0.01, solutions['a2'][-1], 'near', fontsize=16)
# plt.text(solutions['a1'][0]+0.01, solutions['a2'][0], 'far', fontsize=16)
# plt.show()
# raise ProgramSTOP

# according to the coordinate, find on axis point in other view
trueCenterInOtherViews = {} # {row:{col:[...]}}
for ii, index in enumerate(ProgramResult['VIP']):
	a1, a2 = solutions['a1'][ii], solutions['a2'][ii]

	for view in ROIinfo[index]['views']:
		row, col = view['row'], view['col']

		if not row in trueCenterInOtherViews:
			trueCenterInOtherViews[row] = {}
		if not col in trueCenterInOtherViews[row]:
			trueCenterInOtherViews[row][col] = []

		o, x, y = view['o'], view['x'], view['y'] # y, x
		xUnit = np.array(x) - np.array(o) if x[1] > o[1] else np.array(o) - np.array(x)
		yUnit = np.array(y) - np.array(o) if y[0] > o[0] else np.array(o) - np.array(y)

		center = a1 * xUnit + a2 * yUnit + np.array(o)
		trueCenterInOtherViews[row][col].append(center) # global, y, x

# from matplotlib import pyplot as plt
# fig = plt.figure(figsize=(12,9))
# ax = fig.add_subplot(1,1,1)
# ax.scatter(trueCenter[1], trueCenter[0], color='r', marker='x')
# for ii, rc in enumerate([(0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]):
# 	h, s, v = 300*ii/9, 0.9, 0.9
# 	r, g, b = hsv2rgb(h, s, v)
# 	color = '#' + hex(r)[2:] + hex(g)[2:] + hex(b)[2:]
# 	p = np.array(trueCenterInOtherViews[rc[0]][rc[1]])
# 	ax.scatter(p[:,1], p[:,0], color=color)
# ax.axis('equal')
# plt.show()
# raise ProgramSTOP

# from matplotlib import pyplot as plt
# fig = plt.figure(figsize=(12,9))
# ax = fig.add_subplot(1,1,1)
# for ii, index in enumerate(ProgramResult['VIP']):
# 	for view in ROIinfo[index]['views']:
# 		ax.scatter(view['o'][1], view['o'][0])
# plt.show()
# raise ProgramSTOP

# compute disparity, trueCenter & on axis point in other view
disparity = {} # {row:{col:array}}
for row in trueCenterInOtherViews:
	disparity[row] = {}
	for col in trueCenterInOtherViews[row]:

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

		c0 = np.array(trueCenter)
		c1s = np.array(trueCenterInOtherViews[row][col]) + totalShift
		d = c1s - c0

		disparity[row][col] = d

# from matplotlib import pyplot as plt
# fig = plt.figure(figsize=(12,9))
# ax = fig.add_subplot(1,1,1)
# for ii, rc in enumerate([(0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]):
# 	p = np.array(disparity[rc[0]][rc[1]])
# 	for jj, pp in enumerate(p):
# 		h, s, v = 300*ii/9, 0.9*jj/len(p), 0.9
# 		r, g, b = hsv2rgb(h, s, v)
# 		color = '#' + hex(r)[2:] + hex(g)[2:] + hex(b)[2:]
# 		ax.scatter(pp[1], pp[0], color=color)
# ax.axis('equal')
# plt.show()

# from matplotlib import pyplot as plt
# fig = plt.figure(figsize=(12,9))
# ax = fig.add_subplot(1,1,1)
# for ii, rc in enumerate([(0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]):
# 	h, s, v = 300*ii/9, 0.9, 0.9
# 	r, g, b = hsv2rgb(h, s, v)
# 	color = '#' + hex(r)[2:] + hex(g)[2:] + hex(b)[2:]
# 	p = np.array(disparity[rc[0]][rc[1]]) + np.array(trueCenter)
# 	ax.scatter(p[:,1], p[:,0], color=color)
# ax.scatter(trueCenter[1], trueCenter[0], color='r', marker='x')
# ax.axis('equal')
# plt.show()

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(12,9))
gs = fig.add_gridspec(nrows=1, ncols=16)
ax = fig.add_subplot(gs[:, :15])
colorbar = fig.add_subplot(gs[:, -1])
for ii, rc in enumerate([(0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]):
	for jj, dd in enumerate(disparity[rc[0]][rc[1]]):
		p = np.array(dd) + np.array(trueCenter)

		h, s, v = 300*jj/len(disparity[rc[0]][rc[1]]), 0.9, 0.9
		r, g, b = hsv2rgb(h, s, v)
		color = '#' + hex(r)[2:] + hex(g)[2:] + hex(b)[2:]

		ax.scatter(p[1], p[0], color=color)
ax.scatter(trueCenter[1], trueCenter[0], color='r', marker='x')
ax.axis('equal')

for ii in range(210, 600):
	h, s, v = 300*(ii-210)/(-210+600), 0.9, 0.9
	r, g, b = hsv2rgb(h, s, v)
	color = '#' + hex(r)[2:] + hex(g)[2:] + hex(b)[2:]

	colorbar.plot([0,1], [ii,ii], color=color)

plt.show()


















