# -*- coding:utf-8 -*-
import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, least_squares

from Checkerboard import getCheckerboardCorners
from GridModel import BuildGridModel, segmentImage

from Utilities import ProgramSTOP, SystemParamReader, KeyFinder, prepareFolder, saveDict, loadDict, StraightLine
from Utilities import findROI, find_corners_manual, getRect, arrangePointCloud, averageDistanceFit, averageDistanceFit_advance, Rodrigue2Euler

from OptiClass import OptiParams
from CheckerboardClass import CheckerboardParams

from FeaturePoint import FeatPoint, RayTraceModule
from NewImageClass import LFImage, WhiteImage, DepthStack

class LFCam:
    """Light Field Camera

    Attributes Lookup Table:
        self.
            optiSystem (class OptiParams): M_MLA, f_MLA, pixel, p_MLA, z_focus, z_focus_bias
            cbParams (class CheckerboardParams): initDepth, lastDepth, depthInterval, depthRepeat, tilt, CalibCheckerSize, CalibCheckerShape, number_of_pixels_per_checker, margin
            exampleImage (class LFImage): image, name
            whiteImage (class WhiteImage): image[0-1], name, unvignet
            paraCheckerStack (class DepthStack): imageStack = {depth: LFImage}

            invM (numpy.ndarray): 2x2 array
            peaks (numpy.ndarray): (nrow x ncol x 2) array

            featPoints (dict): {depth: {'AA': class RayTraceModule, ...}, ...}
                class RayTraceModule: project, minCircle
                class FeatPoint: rawPoints

            pinhole_camera_matrix (numpy.ndarray): Camera matrix. (main lens + MLA) ### LOCAL Variable
            pinhole_camera_dist (numpy.ndarray): Distortion coefficients. (main lens + MLA) ### LOCAL Variable

            MLA_distortion: Distortion coefficients. (MLA only)
            main_lens_distortion: Distortion coefficients. (main lens only)

    """
    def __init__(self, SysParamFile, **kwargs):
        self.Reader = SystemParamReader(filename=SysParamFile)
        self.SysParams = self.Reader.read()
        self.Finder = KeyFinder(self.SysParams)

        # Path to RawImageFiles
        self.RawImageFiles = self.Finder.find('RawImageFiles')

        # Optical System
        optiSystem = self.Finder.find('System')
        self.optiSystem = OptiParams(optiSystem)

        # Checkerboard
        cbParams = self.Finder.find('Checkerboard')
        self.cbParams = CheckerboardParams(cbParams, coeffi=self.optiSystem.coeffi)

        # Folder
        self.Folder = self.Finder.find('Folder')
        self.ImgLog = self.Folder['ImgLog']
        self.txtLog = self.Folder['txtLog']
        self.rawFeatPointFolder = os.path.join(self.txtLog, self.Folder['rawFeatPoints'])

        # Filenames
        self.filenames = self.Finder.find('Filenames')
        self.RowColInfoFilename = os.path.join(self.txtLog, self.filenames['RowColInfo'])

        # ImagePaths
        self.ImagePaths = self.Finder.find('ImagePaths')

        # Read Images
        self.PoorRes = kwargs.get('PoorRes', False) # image quality is poor
        self.exampleImage = LFImage('example', cv.imread(self.ImagePaths['example'], cv.IMREAD_GRAYSCALE))
        self.whiteImage = WhiteImage(cv.imread(self.ImagePaths['whiteImage'], cv.IMREAD_GRAYSCALE))

        N_images = np.round(len(self.ImagePaths['depth'])/self.cbParams.depthRepeat).astype(int)
        imgShape = (N_images, self.exampleImage.image.shape[0], self.exampleImage.image.shape[1])
        images = np.zeros(imgShape, dtype='float64')

        for ii in range(0, len(self.ImagePaths['depth']), self.cbParams.depthRepeat):
            oneSlice = np.zeros(self.exampleImage.image.shape, dtype='float64')
            for jj in range(self.cbParams.depthRepeat):
                img = cv.imread(self.ImagePaths['depth'][ii+jj], cv.IMREAD_GRAYSCALE)
                oneSlice += img
            oneSlice /= self.cbParams.depthRepeat
            if self.PoorRes:
                oneSlice = self.whiteImage.unvignet(oneSlice, clip=50)
            else:
                oneSlice = self.whiteImage.unvignet(oneSlice, clip=255)
            images[int(ii/self.cbParams.depthRepeat), :, :] = oneSlice
        self.paraCheckerStack = DepthStack(self.cbParams, images)

        self.calibImages = []
        for path in self.ImagePaths['calibrate']:
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if self.PoorRes:
                img = self.whiteImage.unvignet(img, clip=200)
            self.calibImages.append(img)

    def __repr__(self):
        data = 'LFCam\n'
        data += 'Profile ' + self.Reader.filename + '\n\n'
        data += self.optiSystem.__repr__() + '\n'
        data += self.cbParams.__repr__()
        return data

    def initFolder(self, **kwargs):
        """Initialize folder
        Args:
            kwargs
                folder (str): Specify a folder to initialize
        """
        if not 'folder' in kwargs:
            prepareFolder(self.ImgLog)
            if not os.path.exists(self.txtLog):
                prepareFolder(self.txtLog)
            if not os.path.exists(self.rawFeatPointFolder):
                prepareFolder(self.rawFeatPointFolder)
        else:
            prepareFolder(kwargs['folder'])

    def showExample(self):
        """Show example image
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax1.set_title('Example Image')
        ax1.set_aspect('equal')
        ax1.imshow(self.exampleImage.image, cmap='gray')
        ax2 = fig.add_subplot(1,2,2)
        ax2.set_title('White Image')
        ax2.set_aspect('equal')
        ax2.imshow(self.whiteImage.image, cmap='gray')
        plt.show()

    def getRowColInfo(self):
        """Input row & col info

        Self attribute created:
            row_center (int): Index of on-axis aperture.
            row_total (int): Total row of views.
            col_center (int): Index of on-axis aperture.
            col_total (int): Total col of views.
        """
        options = ['row', 'col']
        while True:
            try:
                print('Input in this format\n# of center row/# of total rows\n# of center col/# of total cols\n')
                for option in options:
                    info = input('# of center ' + option + '/# of total ' + option + 's: ')
                    info = info.split('/')
                    assert len(info) == 2
                    exec(option + '_center = int(info[0])')
                    exec(option + '_total = int(info[1])')
                    print('# of center ' + option + ' = ', int(info[0]))
                    print('# of total ' + option + 's = ', int(info[1]))
                break
            except AssertionError:
                print('Please follow the required format\n')

        self.row_center, self.row_total = locals()['row_center']-1, locals()['row_total']
        self.col_center, self.col_total = locals()['col_center']-1, locals()['col_total']
        data = {
            'R': self.row_center,
            'ROWS': self.row_total,
            'C': self.col_center,
            'COLS': self.col_total
        }
        saveDict(self.RowColInfoFilename, data)

    def _generate_preliminary_grid_model(self, **kwargs):
        """Generate a preliminary grid model

        Args: refer to `BuildGridModel` in `GridModel.py`

        Returns:
            invM & peaks, as described in `self.generateGridModel`
        """
        keys = list(kwargs.keys())
        execute_string = 'invM, _, peaks = BuildGridModel'
        # execute_string += '(self.stopImage.image,'
        execute_string += '(self.whiteImage.image,'
        for k in keys:
            value = kwargs[k]
            execute_string += (k + f'={value},')
        execute_string = execute_string[:-1]
        execute_string += ')'
        exec(execute_string)
        return locals()['invM'], locals()['peaks']

    def _generate_accurate_grid_model(self, **kwargs):
        """Generate an accurate grid model using feature points in center view

        Args:
            kwargs
                step (float, default=1e-3): Optimization step length.
                minSlope (float, default=1e-3): When dG/dx & dG/dy < minSlope, stop optimization
                maxTurns (int, default=1e9): When optimization turns > maxTurns, stop optimization
                showStep (bool, default=False): Print each step in command line (could be long and slow)

        Note:
            Optimization would stop when one of the conditions is satisfied.

        Returns:
            invM & peaks, as described in `self.generateGridModel`
        """
        step = kwargs['step'] if 'step' in kwargs else 1e-3
        minSlope = kwargs['minSlope'] if 'minSlope' in kwargs else 1e-3
        maxTurns = kwargs['maxTurns'] if 'maxTurns' in kwargs else int(1e9)
        showStep = kwargs['showStep'] if 'showStep' in kwargs else False

        # collect points in center view
        fp_in_center = {} # {'RC':points}
        # for each depth
        for curDepth in self.featPoints:
            # check every feature point
            for fpName in self.featPoints[curDepth]:
                # check each `rawPoint` in `FeatPoint`
                for point_view in self.featPoints[curDepth][fpName].fp.rawPoints:
                    point, view = point_view
                    # if it is the center view, add this point
                    if list(view) == [self.row_center, self.col_center]:
                        if not fpName in fp_in_center:
                            fp_in_center[fpName] = [point]
                        else:
                            fp_in_center[fpName].append(point)

        # straight line fit
        lines = {} # {name:[A,B,C]}
        for pName in fp_in_center:
            points = np.array(fp_in_center[pName])
            if points.shape[0] > 2: # at least 3 points
                xs, ys = points[:,1], points[:,0]
                if (xs.max()-xs.min()) > (ys.max()-ys.min()):
                    # y = kx + b
                    popt, _ = curve_fit(StraightLine, xs, ys)
                    k, b = popt
                    A, B, C = k, -1, b # Ax + By + C = 0
                else:
                    # x = ky + b
                    popt, _ = curve_fit(StraightLine, ys, xs)
                    k, b = popt
                    A, B, C = -1, k, b
                lines[pName] = [A, B, C]

        # find point with minimum distance
        y, x = self.peaks[self.row_center, self.col_center]
        turn = 0
        while True:
            # calculation
            d = 0
            dgdx, dgdy = 0, 0
            for name in lines:
                A, B, C = lines[name]
                d += np.sqrt((A*x+B*y+C)**2/(A**2+B**2))
                dgdx += (2*A*(A*x+B*y+C)/(A**2+B**2))
                dgdy += (2*B*(A*x+B*y+C)/(A**2+B**2))
            if showStep:
                print(
                    str(turn).zfill(len(str(maxTurns))),
                    'x {:.6f}'.format(x), 'y {:.6f}'.format(y),
                    'D {:.6f}'.format(d),
                    'dG/dx {:.6f}'.format(dgdx), 'dG/dy {:.6f}'.format(dgdy)
                    )
            # whether quit optimization
            if (abs(dgdx) < minSlope) and (abs(dgdy) < minSlope):
                print('Slope small enough!')
                break
            if turn > maxTurns:
                print('Optimize for a long time!')
                break
            # update
            turn += 1
            x += (-dgdx * step)
            y += (-dgdy * step)
        print(str(turn).zfill(len(str(maxTurns))), 'D {:.6f}'.format(d), 'dG/dx {:.6f}'.format(dgdx), 'dG/dy {:.6f}'.format(dgdy))
        print('accurate center x {:.2f} y {:.2f}'.format(x, y))

        # generate accurate centers
        centerArr = np.zeros((self.row_total, self.col_total, 2), dtype='float64')
        M = np.linalg.inv(self.invM)
        vecH, vecV = M[:,0], M[:,1]
        Hstep = (self.optiSystem.p_MLA()/self.optiSystem.pixel()) * vecH
        Vstep = (self.optiSystem.p_MLA()/self.optiSystem.pixel()) * vecV
        Hx, Hy = Hstep
        Vx, Vy = Vstep
        if Vy < 0: # ensure two vectors point at + direction
            Hx, Hy, Vx, Vy = Hx, Hy, -Vx, -Vy

        for row in range(self.row_total):
            for col in range(self.col_total):
                drow, dcol = row - self.row_center, col - self.col_center
                Y = y + drow*Vy + dcol*Hy
                X = x + dcol*Hx + drow*Vx
                center = (Y, X)
                centerArr[row, col, :] = center
        return self.invM, centerArr

    def generateGridModel(self, method, **kwargs):
        """Generate grid model

        Args:
            method (str):
                'P' or 'preliminary' for generating a preliminary grid model
                'A' or 'accurate' for generating an accurate grid model

        Returns:
            invM (numpy.ndarray): 2x2 transformation matrix. Define grid orientation (rotation & shear)
            peaks (numpy.ndarray): Center array, (row, col, 2)

        Self attribute created:
            invM
            peaks
        """
        assert method.upper() in ['P', 'preliminary'.upper(), 'A', 'accurate'.upper()]
        switch = {
            'P': self._generate_preliminary_grid_model,
            'A': self._generate_accurate_grid_model
        }
        keys = list(kwargs.keys())
        execute_string = 'invM, peaks = switch[method[0].upper()]'
        execute_string += '('
        for k in keys:
            value = kwargs[k]
            execute_string += (k + f'={value},')
        execute_string = execute_string[:-1]
        execute_string += ')'
        exec(execute_string)
        self.invM, self.peaks = locals()['invM'], locals()['peaks']

        return self.invM, self.peaks

    def extractFeatPoints(self, **kwargs):
        """Extract feature points

        Args:
            kwargs
                initDepth (float, optional): Depth range from.
                lastDepth (float, optional): Depth range to.

        Self attribute created:
            featPoints (dict): Feature points in each depth. {depth[m]: {'AA': class RayTraceModule}}
        """
        initDepth = kwargs['initDepth'] if 'initDepth' in kwargs else self.cbParams.initDepth()
        lastDepth = kwargs['lastDepth'] if 'lastDepth' in kwargs else self.cbParams.lastDepth()
        margin = self.cbParams.margin
        self.featPoints = {}

        imgStack = self.paraCheckerStack.imageStack
        depth_keys = list(imgStack.keys())

        # lookup list, whether the depth has been processed
        files = os.listdir(self.rawFeatPointFolder) # folders named as depth[m]
        curDepth_strings = []
        for curDepth_str in files:
            if not curDepth_str in curDepth_strings:
                curDepth_strings.append(curDepth_str)

        for curDepth in depth_keys:

            if (curDepth >= initDepth) and (curDepth <= lastDepth):

                # if not detect before
                # save when all views in one depth are completed
                if not str(curDepth) in curDepth_strings:

                    # init feat points at each depth
                    FPs = {}
                    for ii in range(26):
                        rName = chr(65+ii)
                        for jj in range(26):
                            cName = chr(65+jj)
                            fp = FeatPoint(rName, cName)
                            FPs[rName+cName] = RayTraceModule(
                                self.optiSystem.M_MLA('nominal'), self.optiSystem.f_MLA('pixel'),
                                self.peaks, fp, row_center=self.row_center, col_center=self.col_center
                                )

                    curLFImage = imgStack[curDepth]
                    subimages, anchors = segmentImage(curLFImage.image, self.invM, self.peaks)

                    assert len(subimages) == self.row_total
                    assert len(subimages[0]) == self.col_total
                    for row in range(self.row_total):
                        for col in range(self.col_total):

                            while True: # if corners not detected, select a smaller area and try again
                                if self.PoorRes:
                                    view = subimages[row][col]
                                    print(curDepth, row, col)
                                    print(anchors[row][col]['upperLeft'], anchors[row][col]['lowerRight']-anchors[row][col]['upperLeft'])
                                    low_light_corners = find_corners_manual(view) # (x, y) !!
                                    low_light_corners = np.flip(low_light_corners)
                                    low_light_corners += np.array(anchors[row][col]['upperLeft'])
                                else:
                                    # crop region of interest
                                    view = subimages[row][col]
                                    mouseUp = findROI(view, name=str(curDepth)+' '+str(row)+' '+str(col))
                                    top, bottom, left, right = getRect(mouseUp)
                                    roi = view[max(0,top-margin):min(bottom+margin,view.shape[0]), max(0,left-margin):min(right+margin,view.shape[1])]
                                    # corner detection function not compatible with filename containing '.'
                                    fname = 'view_' + curLFImage.name.split('.')[-1] + '_' + str(row).zfill(2) + '_' + str(col).zfill(2)
                                    cv.imwrite(os.path.join(self.ImgLog, fname+'.thumbnail.jpg'), roi.astype('uint8'))
                                    np.save(os.path.join(self.ImgLog, fname+'.npy'), roi.astype('float64'))

                                # shape & corner names
                                while True:
                                    corner1 = input('Name of upper-left corner: ')
                                    corner2 = input('Name of lower-right corner: ')
                                    try:
                                        assert not corner1 == ''
                                        assert not corner2 == ''
                                        assert len(corner1) == 2
                                        assert len(corner2) == 2
                                        break
                                    except AssertionError:
                                        print('Input Error, try again...')
                                r1, c1 = corner1.upper()
                                r2, c2 = corner2.upper()
                                shape = (ord(r2)+1-ord(r1)+1, ord(c2)+1-ord(c1)+1) # # of inner corners, not squares

                                if self.PoorRes:
                                    imagePoints = arrangePointCloud(low_light_corners.reshape(-1,2), shape, self.cbParams.number_of_pixels_per_checker)
                                    break
                                else:
                                    # detect corners
                                    match = getCheckerboardCorners(os.path.join(self.ImgLog, fname+'.npy'), shape, extension='npy', visualize=True)
                                    try:
                                        imagePoints = match['imagePoints'][0]
                                        imagePoints = arrangePointCloud(np.flip(imagePoints), shape, self.cbParams.number_of_pixels_per_checker)
                                        imagePoints += (np.array(anchors[row][col]['upperLeft']) + np.array([top-margin,left-margin])) # global
                                        # if success, imagePoints would be (row, col, 2)
                                        break
                                    except IndexError:
                                        print(fname, 'CORNER DETECTION FAIL\nTry again!')

                            # input corners
                            for r in range(ord(r1), ord(r2)+2):
                                for c in range(ord(c1), ord(c2)+2):
                                    p = imagePoints[r-ord(r1), c-ord(c1)]
                                    FPs[chr(r)+chr(c)].fp.add(p, (row,col))

                    self.featPoints[curDepth] = FPs

                    if not os.path.exists(os.path.join(self.rawFeatPointFolder, str(curDepth))):
                        prepareFolder(os.path.join(self.rawFeatPointFolder, str(curDepth)))
                    for ii in range(26):
                        rName = chr(65+ii)
                        for jj in range(26):
                            cName = chr(65+jj)
                            FPs[rName+cName].fp.save(os.path.join(self.rawFeatPointFolder, str(curDepth), rName+cName+'.json'))

                else:
                    print(curDepth, '...', end=' ')
                    FPs = {}
                    files = os.listdir(os.path.join(self.rawFeatPointFolder, str(curDepth)))
                    files = [f for f in files if f.endswith('json')]
                    for file in files:
                        data = loadDict(os.path.join(self.rawFeatPointFolder, str(curDepth), file))
                        assert data['row'] == file[0] # file == 'AR.json'
                        assert data['col'] == file[1]
                        rawPoints = data['rawPoints'] # [[p, view],...]
                        fp = FeatPoint(data['row'], data['col'], rawPoints=rawPoints)
                        FPs[data['row']+data['col']] = RayTraceModule(
                            self.optiSystem.M_MLA('nominal'), self.optiSystem.f_MLA('pixel'),
                            self.peaks, fp, row_center=self.row_center, col_center=self.col_center
                            )
                    self.featPoints[curDepth] = FPs
                    print('data loaded.')

    def featPoints_inView(self, targetView):
        """
        Args:
            targetView (tuple): One of the view. (row, col), index starts from 0.

        Returns:
            fp_in_view (dict): All feature points in this view by depth. {depth: points(row,col,2)}
            rcNames (dict): Row & col range in each depth. {depth: {'row':[minRowName, maxRowName], 'col':[minColName, maxColName]}}
        """
        assert len(targetView) == 2
        assert (targetView[0] >= 0) and (targetView[0] < self.row_total)
        assert (targetView[1] >= 0) and (targetView[1] < self.col_total)

        fp_in_view, rcNames = {}, {}
        # for each depth, record each point and row&col range
        for curDepth in self.featPoints:
            minRowName, maxRowName = 'Z', 'A'
            minColName, maxColName = 'Z', 'A'
            fp_in_view[curDepth] = []
            rcNames[curDepth] = {'row':[], 'col':[]}
            # check each feature point
            for fpName in self.featPoints[curDepth]:
                # `rawPoints` in `FeatPoint`
                for point_view in self.featPoints[curDepth][fpName].fp.rawPoints:
                    point, view = point_view
                    # if it is center view
                    if list(view) == list(targetView):
                        rowName, colName = fpName
                        if ord(rowName) < ord(minRowName):
                            minRowName = rowName
                        if ord(rowName) > ord(maxRowName):
                            maxRowName = rowName
                        if ord(colName) < ord(minColName):
                            minColName = colName
                        if ord(colName) > ord(maxColName):
                            maxColName = colName
                        # record feature point
                        fp_in_view[curDepth].append(point[:2])
            # convert to (row, col, 2) array
            fp_in_view[curDepth] = arrangePointCloud(
                np.array(fp_in_view[curDepth]), 
                (ord(maxRowName)-ord(minRowName)+1, ord(maxColName)-ord(minColName)+1),
                self.cbParams.number_of_pixels_per_checker
                )
            # row & col range
            rcNames[curDepth]['row'] = [minRowName, maxRowName]
            rcNames[curDepth]['col'] = [minColName, maxColName]

        return fp_in_view, rcNames

    def initPinholeModel(self, **kwargs):
        """Provide initial focal length estimation for camera calibration

        Returns:
            f (float): Focal length.
            z_obj (float): Object distance (optical conjugate).
            z_img (float): Image distance.

        Self attribute created:
            optiSystem.data['z_focus_bias']
        """
        show = kwargs.get('show', True)

        fp_in_center, _ = self.featPoints_inView((self.row_center, self.col_center))

        his = []
        # for each depth
        for curDepth in fp_in_center:
            # average checker size, h_i
            mu, sigma = averageDistanceFit(fp_in_center[curDepth])
            hi = mu
            his.append(hi) # in pixel

        N = len(his)
        z0s = []
        for i in range(N-1):
            z0 = -self.cbParams.depthInterval()*((i+1)*his[i+1]-i*his[i])/(his[i+1]-his[i])
            z0s.append(z0)
        z0_mu, z0_std = np.array(z0s).mean(), np.array(z0s).std()
        # update self.optiSystem
        self.optiSystem.data['z_focus_bias'] = z0_mu - self.cbParams.initDepth()

        # at conjugate depth
        n = int((self.optiSystem.z_focus() - self.cbParams.initDepth())/self.cbParams.depthInterval())
        z_obj = z0_mu + n*self.cbParams.depthInterval()
        z_img = -z_obj*(his[n]*self.optiSystem.pixel())/self.cbParams.CheckerSize()
        f = 1/(1/z_img-1/z_obj)

        if show:
            print('Rough estimation:')
            print('init depth', float(np.array([z0_mu*1e3]).round(3)), '±', float(np.array([z0_std*1e3]).round(3)), 'mm')
            print('z conjugate', float(np.array([z_obj*1e3]).round(3)), '±', float(np.array([z0_std*1e3]).round(3)), 'mm')
            print('z image', float(np.array([z_img*1e3]).round(3)), 'mm')
            print('focal length', float(np.array([f*1e3]).round(3)), 'mm')
        return f, z_obj, z_img

    def PinholeModel(self, **kwargs):
        """Pinhole model of the center view of this light field camera
        camera calibration with constraints

        Returns:
            mtx (numpy.ndarray): Camera matrix. (3x3)
            dist (numpy.ndarray): Camera distortion coefficients. (1x5)

        Self attribute created:
            pinhole_camera_matrix (numpy.ndarray): Camera matrix. (main lens + MLA)
            pinhole_camera_dist (numpy.ndarray): Distortion coefficients. (main lens + MLA)
        """
        fix_dist = kwargs.get('fix_dist', True)

        f0, _, _ = self.initPinholeModel(show=False)
        f0 /= self.optiSystem.pixel('m')

        winSize = kwargs.get('winSize', (9, 9))
        zeroZone = kwargs.get('zeroZone', (-1, -1))
        criteria = kwargs.get('criteria', (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-3))

        _, anchors = segmentImage(self.calibImages[0], self.invM, self.peaks)
        # Create a rectangular mask
        mask = np.zeros(self.calibImages[0].shape[:2], dtype=np.uint8)
        start_point = anchors[self.row_center][self.col_center]['upperLeft'].astype(int)  # Top-left corner of the rectangle
        end_point = anchors[self.row_center][self.col_center]['lowerRight'].astype(int)  # Bottom-right corner of the rectangle
        # Draw a white rectangle on the mask to cover the outer part
        cv.rectangle(mask, start_point[::-1], end_point[::-1], 255, -1)

        self.calibImages_mask = []
        for img in self.calibImages:
            masked_image = cv.bitwise_and(img, img, mask=mask)
            self.calibImages_mask.append(masked_image)
            # cv.imwrite('./ImageLog/mask'+str(len(self.calibImages_mask)).zfill(2)+'.jpg', masked_image.astype('uint8'))

        imageSize = self.calibImages_mask[0].shape

        objp = np.zeros((np.prod(self.cbParams.CalibCheckerShape[:2]), 3), dtype='float64')
        objp[:, :2] = np.indices(self.cbParams.CalibCheckerShape).T.reshape(-1, 2)
        objp *= self.cbParams.CalibCheckerSize('mm')

        objectPoints = []
        imagePoints = []
        for img in self.calibImages_mask:
            retval, corners = cv.findChessboardCorners(img.astype('uint8'), patternSize=self.cbParams.CalibCheckerShape)
            if retval:
                corners = cv.cornerSubPix(img.astype('uint8'), corners, winSize, zeroZone, criteria)
                imgp = corners.reshape(-1, 2) # (mxn, 1, 2) => (mxn, 2) # seems to be (x,y) points ! OpenCV mix (x,y) & (y,x) everywhere!!!
                imagePoints.append(imgp.astype('float32'))
                objectPoints.append(objp.astype('float32'))

        objectPoints = np.array(objectPoints)
        imagePoints = np.array(imagePoints)

        flags = cv.CALIB_USE_INTRINSIC_GUESS
        flags += cv.CALIB_FIX_PRINCIPAL_POINT
        flags += cv.CALIB_ZERO_TANGENT_DIST
        if fix_dist:
            flags += cv.CALIB_FIX_K1
            flags += cv.CALIB_FIX_K2
            flags += cv.CALIB_FIX_K3
        # flags += cv.CALIB_FIX_ASPECT_RATIO

        cy, cx = self.peaks[self.row_center, self.col_center, :]
        initCameraMatrix = np.array([[f0, 0, cx], [0, f0, cy], [0, 0, 1]])
        initDist = np.zeros(5)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectPoints, imagePoints, imageSize, initCameraMatrix, initDist, flags=flags)

        print('Reproj error: ', ret, ' px\nCamera matrix :\n', mtx, '\nDistortion coefficients: ', dist)
        self.pinhole_camera_matrix, self.pinhole_camera_dist = mtx, dist
        return mtx, dist

    def poseEstimate(self, featPoints, **kwargs): # it is not correct to use solvePnP with object point z=0; add images you want to estimate into calibration images, use tvec there
        """Estimate checkerboard orientation

        Args:
            featPoints (numpy.ndarray): Feature points on a board.

        Returns:
            pose (dict):
                world (numpy.ndarray): Feature points in world coordinate. Same shape as feature points.
                rx, ry, rz (float): Approx tilt angles, be calculated only when input points >= 4
        """
        try:
            assert len(featPoints.shape) == 3
        except AssertionError:
            print('Feature points must be in shape (row, col, 2).')
            raise

        try:
            assert featPoints.shape[0]*featPoints.shape[1] >= 4
        except AssertionError:
            print(
                "cv2.error: OpenCV("
                +cv.__version__
                +") ~/opencv/modules/calib3d/src/solvepnp.cpp:840: "
                +"error: (-215:Assertion failed) ( (npoints >= 4) || "
                +"(npoints == 3 && flags == SOLVEPNP_ITERATIVE && useExtrinsicGuess) || "
                +"(npoints >= 3 && flags == SOLVEPNP_SQPNP) ) && "
                +"npoints == std::max(ipoints.checkVector(2, CV_32F), "
                +"ipoints.checkVector(2, CV_64F)) in function 'solvePnPGeneric'"
                )
            raise

        pose = {}
        # CheckerSize = kwargs.get('CheckerSize', self.cbParams.CheckerSize('mm')) # consistent with the unit for calibration
        # imagePoints = featPoints.astype('float32').reshape(np.prod(featPoints.shape[:2]), 2)
        # objectPoints = np.zeros((np.prod(featPoints.shape[:2]), 3), dtype='float64')
        # objectPoints[:, :2] = np.indices(featPoints.shape[:2]).T.reshape(-1, 2)
        # objectPoints *= CheckerSize

        # ret, rvec, tvec = cv.solvePnP(objectPoints, imagePoints, self.pinhole_camera_matrix, self.pinhole_camera_dist)
        # rmat, _ = cv.Rodrigues(rvec)
        # rx, ry, rz = Rodrigue2Euler(rvec)
        # print('Approx tip & tilt: {:.2f}, {:.2f}, {:.2f} degrees.'.format(rx*180/np.pi, ry*180/np.pi, rz*180/np.pi))
        # pose['rx'], pose['ry'], pose['rz'] = rx, ry, rz

        # worldPoints = []
        # points = featPoints.reshape(np.prod(featPoints.shape[:2]), 2)
        # fx, fy = self.pinhole_camera_matrix[0][0], self.pinhole_camera_matrix[1][1]
        # cx, cy = self.pinhole_camera_matrix[0][2], self.pinhole_camera_matrix[1][2]
        # for point in points:
        #     x, y, z = point[1], point[0], (fx+fy)/2
        #     world_coordinate = np.matmul(np.linalg.inv(rmat), np.array([x,y,z]).reshape(3,1)-tvec.reshape(3,1))
        #     X, Y, Z = world_coordinate
        #     worldPoints.append([X, Y, Z])
        # worldPoints = np.array(worldPoints).reshape((featPoints.shape[0], featPoints.shape[1], 3))
        # pose['world'] = worldPoints
        return pose

    def depthCorrection(self): # to be done
        """Correct depth

        Self attribute created:
            optiSystem.data['z_focus_bias']
        """
        fp_in_center, rcNames = self.featPoints_inView((self.row_center, self.col_center))
        fp = fp_in_center[self.optiSystem.z_focus('m')]
        worldPoints = self.poseEstimate(fp)
        worldPoints *= self.optiSystem.pixel('m')
        z = worldPoints[-1] # unit ?
        self.optiSystem.data['z_focus_bias'] = z - self.optiSystem.z_focus('m')

    def optimize_local(self, **kwargs):
        """Optimize MLA magnification, MLA rotation, checkerboard tilt angle and MLA distortion

        Self attribute created:
            optiSystem.data['M_MLA_real']
            invM: update self.invM
            peaks: update self.peaks
            cbParams.dara['tilt']
            MLA_distortion: Distortion coefficients. (MLA only)
        """
        def cost_func(params, **kwargs):
            FPs = kwargs['FPs']
            centers = kwargs['centers']
            center_index = kwargs['center_index']
            normFactor = kwargs['normFactor']
            targetValue = kwargs.get('targetValue', 0)
            radii = calc_radii(params, FPs, centers, center_index, normFactor)
            return (radii/len(radii) - targetValue).flatten() # more points, larger circles

        def calc_radii(params, FPs, centers, center_index, normFactor):
            """
            Args:
                params (list): List of params to be optimized, [M, R (rad), alpha (rad), beta (rad), k1, k2, k3]
                FPs (dict): Feature points, {'AS': class RayTraceModule}
                centers (numpy.ndarray): Reprojection centers.
                center_index (tuple): Index of on-axis aperture. (self.row_center, self.col_center)
                normFactor (float): Distortion normalization factor.

            Returns:
                radii (numpy.ndarray): Radii of minEnclosingCircle of each feature points reprojected back to II plane.
            """
            if len(params) == 7:
                M, R, alpha, beta, k1, k2, k3 = params
            else:
                M, R, k1, k2, k3 = params
                alpha, beta = 0, 0

            rot = np.array([[np.cos(R), -np.sin(R)],[np.sin(R), np.cos(R)]])
            reproj_centers = centers - centers[center_index[0], center_index[1], :]
            reproj_centers = np.matmul(reproj_centers, rot)# (r,c,2) x (2,2); it rotates anyway!
            reproj_centers += centers[center_index[0], center_index[1], :]

            radii = []
            for fpName in FPs:
                rtm = FPs[fpName]
                fp = rtm.fp
                if fp.isValid:
                    RX = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
                    RY = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
                    II_norm_vec = np.matmul(np.matmul(RX, RY), np.array([[0],[0],[1]])).reshape(3)
                    rtm.project(M_MLA=M, centers=reproj_centers, II_norm_vec=II_norm_vec, dist=[k1, k2, k3], norm=normFactor)
                    minCircleParams = rtm.minCircle
                    radii.append(minCircleParams['radius'])
            radii = np.array(radii)

            return radii.flatten()

        # least_squares params
        method = kwargs.get('method', 'trf')
        ftol = kwargs.get('ftol', 1e-12)
        xtol = kwargs.get('xtol', 1e-12)
        gtol = kwargs.get('gtol', 1e-12)
        x_scale = kwargs.get('x_scale', np.array([1, 5e-2, 5e-2, 5e-2, 1e-2, 1e-2, 1e-2]))
        loss = kwargs.get('loss', 'linear')

        rz = kwargs.get('MLA_rotation', 0)
        rx, ry = kwargs.get('Checkerboard_tilt', [0, 0])
        k1, k2, k3 = kwargs.get('dist', [0, 0, 0])
        initGuess = [self.optiSystem.M_MLA('nominal'), rz, rx, ry, k1, k2, k3]
        inputData = {
            'FPs': self.featPoints[self.optiSystem.z_focus('m')],
            'centers': self.peaks,
            'center_index': (self.row_center, self.col_center),
            'normFactor': self.optiSystem.p_MLA('pixel'),
            }
        if self.PoorRes:
            x_scale = np.array([1, 5e-2, 1e-2, 1e-2, 1e-2]) # delete rx & ry
            initGuess = [self.optiSystem.M_MLA('nominal'), rz, k1, k2, k3]
        optimizeResult = least_squares(cost_func, initGuess, kwargs=inputData, method=method, ftol=ftol, xtol=xtol, gtol=gtol, x_scale=x_scale, loss=loss)
        print('Local optimization result using scipy.optimize.least_squares')
        if kwargs.get('detail', False):
            print(optimizeResult, '\n')

        M, rz, rx, ry, k1, k2, k3 = optimizeResult.x
        print('Magnification of MLA: {:.3f}'.format(M))
        print('MLA rotation: {:.3f} degrees'.format(rz*180/np.pi))
        print('Checkerboard tilt angle: {:.3f}, {:.3f} degrees'.format(rx*180/np.pi, ry*180/np.pi))
        print('Local distortion coefficients: k1 = {:.3f}, k2 = {:.3f}, k3 = {:.3f}\n'.format(k1, k2, k3))

        self.optiSystem.data['M_MLA_real'] = M
        peaks = self.peaks - self.peaks[self.row_center, self.col_center, :]
        peaks = np.matmul(peaks, np.array([[np.cos(rz), -np.sin(rz)],[np.sin(rz), np.cos(rz)]]))
        peaks += self.peaks[self.row_center, self.col_center, :]
        self.peaks = peaks
        self.invM = np.linalg.inv(np.matmul(np.linalg.inv(self.invM), np.array([[np.cos(rz), -np.sin(rz)],[np.sin(rz), np.cos(rz)]])))
        self.cbParams.data['tilt'] = [rx, ry]
        self.MLA_distortion = np.array([k1, k2, k3])

        return optimizeResult

    def optimize_global(self, **kwargs):
        """Optimize main lens distortion

        Self attribute created:
            main_lens_distortion: Distortion coefficients. (main lens only)
        """
        def cost_func(params, **kwargs):
            iipoints = kwargs['IIPoints']
            OpticalAxis = kwargs['OpticalAxis']
            normFactor = kwargs['normFactor']
            ds, mu, sigma = func(params, iipoints, OpticalAxis, normFactor)
            targetValue = kwargs.get('targetValue', mu)
            return (ds - targetValue).flatten()

        def func(params, iipoints, OpticalAxis, normFactor):
            """
            Args:
                params (list): Parameters to be optimized. Distortion coefficients, [k1, k2, k3]
                iipoints (list): Points on II plane. [{'center': center, 'row': 'R', 'col': 'C'}, ...]
                OpticalAxis (list): x & y coordinate of optical axis. Global coordinate.
                normFactor (float): Distortion normalization factor.

            Returns:
                ds (numpy.ndarray): All distances between adjacent points.
                mu (float): Average distance between adjacent points.
                sigma (float): Standard deviation of distances between adjacent points.
            """
            dist = params
            iipoints_undist = []
            for p in iipoints:
                local_p_vec = p['center'][:-1] - OpticalAxis # center @ optical axis
                local_norm_p_vec = local_p_vec / normFactor # normalize
                local_norm_r_2 = np.sum(local_norm_p_vec**2) # r^2
                factor = 1
                for i in range(len(dist)):
                    factor += dist[i]*local_norm_r_2**(i+1) # 1 + k1*r^2 + k2*r^4 + ...
                local_norm_p_vec_dist = local_norm_p_vec * factor
                p_vec_dist = local_norm_p_vec_dist * normFactor + OpticalAxis
                iipoints_undist.append({'center': p_vec_dist, 'row': p['row'], 'col': p['col']})
            ds, mu, sigma = averageDistanceFit_advance(iipoints_undist)
            return ds, mu, sigma

        # least_squares params
        method = kwargs.get('method', 'trf')
        ftol = kwargs.get('ftol', 1e-12)
        xtol = kwargs.get('xtol', 1e-12)
        gtol = kwargs.get('gtol', 1e-12)
        x_scale = kwargs.get('x_scale', np.array([1e-2, 1e-2, 1e-2]))
        loss = kwargs.get('loss', 'linear')

        IIPoints = []
        FPs = self.featPoints[self.optiSystem.z_focus('m')]
        for fpName in FPs:
            rtm = FPs[fpName]
            fp = rtm.fp
            if fp.isValid:
                alpha, beta = self.cbParams.tilt
                RX = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
                RY = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
                II_norm_vec = np.matmul(np.matmul(RX, RY), np.array([[0],[0],[1]])).reshape(3)
                rtm.project(
                    M_MLA=self.optiSystem.M_MLA('real'), centers=self.peaks, 
                    II_norm_vec=II_norm_vec, dist=self.MLA_distortion, norm=self.optiSystem.p_MLA('pixel')
                    )
                circle = rtm.minCircle
                IIPoints.append({'center': circle['center'], 'row': fp.row, 'col': fp.col})

        estimated_IIplane_image = np.zeros(
            (
                int((self.row_total-1+int(np.ceil(self.optiSystem.M_MLA('real'))))*self.optiSystem.p_MLA('pixel')),
                int((self.col_total-1+int(np.ceil(self.optiSystem.M_MLA('real'))))*self.optiSystem.p_MLA('pixel'))
                ), dtype='float64'
            )
        norm1 = np.sqrt(np.sum((np.array([0,0])-self.peaks[self.row_center, self.col_center])**2))
        norm2 = np.sqrt(np.sum((np.array([estimated_IIplane_image.shape[0],0])-self.peaks[self.row_center, self.col_center])**2))
        norm3 = np.sqrt(np.sum((np.array([0,estimated_IIplane_image.shape[1]])-self.peaks[self.row_center, self.col_center])**2))
        norm4 = np.sqrt(np.sum((np.array(estimated_IIplane_image.shape)-self.peaks[self.row_center, self.col_center])**2))
        normFactor = max(norm1, norm2, norm3, norm4)

        k1, k2, k3 = kwargs.get('dist', [0, 0, 0])
        initGuess = [k1, k2, k3]
        inputData = {
            'IIPoints': IIPoints,
            'OpticalAxis': self.peaks[self.row_center, self.col_center],
            'normFactor': normFactor,
            }
        optimizeResult = least_squares(cost_func, initGuess, kwargs=inputData, method=method, ftol=ftol, xtol=xtol, gtol=gtol, x_scale=x_scale, loss=loss)
        k1, k2, k3 = optimizeResult.x

        print('Global optimization result using scipy.optimize.least_squares')
        if kwargs.get('detail', False):
            print(optimizeResult, '\n')
        print('Global distortion coefficients: k1 = {:.3f}, k2 = {:.3f}, k3 = {:.3f}\n'.format(k1, k2, k3))

        self.main_lens_distortion = np.array([k1, k2, k3])

        return optimizeResult
