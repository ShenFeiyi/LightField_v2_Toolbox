# -*- coding:utf-8 -*-
import os
import math
import json
import cv2 as cv
import numpy as np
from queue import PriorityQueue as PQ

class ProgramSTOP(Exception):
    def __init__(self, message='Stop here for debugging'):
        self.message = message
        super().__init__(self.message)

def prepareFolder(folder):
    """Clean up a folder for storage

    Args:
        folder (str): Folder name.

    Returns:
        None
    """
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        files_or_dirs = os.listdir(folder)
        for file_or_dir in files_or_dirs:
            if os.path.isdir(os.path.join(folder, file_or_dir)):
                prepareFolder(os.path.join(folder, file_or_dir))
                os.rmdir(os.path.join(folder, file_or_dir))
            else:
                os.remove(os.path.join(folder, file_or_dir))

def saveDict(filename, data):
    """Save a dict file to json file

    Args:
        filename (str): File name.
        data (dict): Dict data to be stored.
    """
    assert type(data) == dict
    if not filename.endswith('json'):
        filename += '.json'
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def loadDict(filename):
    """Load a dict file from json

    Args:
        filename (str): Name of the file to load.

    Returns:
        data (dict): Loaded data.
    """
    def _convert_str_key_2_int(data):
        newdata = {}
        for key in data:
            content = data[key]
            try: # try to convert str to int
                key = int(key)
            except ValueError:
                pass
            try: # try to go through dict file
                assert type(content) == dict
                newdata[key] = _convert_str_key_2_int(content)
            except AssertionError:
                newdata[key] = content
        return newdata

    with open(filename, 'r') as file:
        data = json.load(file)
    
    data = _convert_str_key_2_int(data)
    return data

def findROI(background, **kwarg):
    """Find ROI in a cv window
    Press 'enter/return' to finish
    Press 'space' to delete last point

    Args:
        background (numpy.ndarray): Image to be cropped.
        windowName (str, default='click_here'): CV window name.
        origin (tuple, default=(0,0)): Origin of CV window.

    Returns:
        mouseUp (list): A list of points containing four corners of the ROI.
    """
    def _click(event, x, y, flags, param):
        """Click event in opencv window
        """
        nonlocal mouseDown, mouseUp, click_img
        if event == cv.EVENT_LBUTTONDOWN:
            mouseDown.append((x,y))

        elif event == cv.EVENT_LBUTTONUP:
            mouseUp.append((x,y))
            try:
                if not mouseUp[-1] == mouseDown[-1]:
                    raise ValueError
                mouse = mouseUp
                cv.circle(click_img, (mouse[-1][0], mouse[-1][1]), 2, (0,0,255), 2)
                if len(mouse) > 1:
                    for i in range(len(mouse)-1):
                        cv.line(click_img, (mouse[i][0], mouse[i][1]), (mouse[i+1][0], mouse[i+1][1]), (0, 255, 0), 1)
                if len(mouse) >= 4:
                    cv.line(click_img, (mouse[-1][0], mouse[-1][1]), (mouse[0][0], mouse[0][1]), (0, 255, 0), 1)
            except ValueError:
                print('Mouse was moved when clicking.')
                mouseUp.pop(-1)
                mouseDown.pop(-1)

    windowName = kwarg['name'] if 'name' in kwarg else 'click_here'
    origin = kwarg['origin'] if 'origin' in kwarg else (0,0)

    cv.namedWindow(windowName)
    cv.moveWindow(windowName, origin[0], origin[1])
    cv.setMouseCallback(windowName, _click)

    mouseDown, mouseUp = [], []
    click_img = background.copy().astype('uint8')
    if len(click_img.shape) == 2:
        click_img = np.repeat(click_img, 3).reshape((click_img.shape[0],click_img.shape[1],3)) # gray 2 rgb
        #click_img = cv.cvtColor(click_img, cv.COLOR_GRAY2BGR)
    while True:
        cv.imshow(windowName, click_img)
        if cv.waitKey(1) == 13: # key: enter/return
            break
        elif cv.waitKey(1) == 32: # key: space
            # will NOT refresh drawings in window
            try:
                mouseUp.pop(-1)
                p = mouseDown.pop(-1)
                print(p, ' deleted.')
            except IndexError:
                print('No Points Found')
        else:
            pass
    cv.destroyAllWindows()

    return mouseUp

def findSubPixPoint(background, **kwarg):
    """Find point at sub-pixel accuracy in a cv window
    Press 'enter/return' to finish
    Press 'space' to delete last point

    Args:
        background (numpy.ndarray): Image to be cropped.
        windowName (str, default='click_here'): CV window name.
        origin (tuple, default=(0,0)): Origin of CV window.

    Returns:
        subpixPoints (list): A list of points containing four corners of the ROI.
    """
    def _click(event, x, y, flags, param):
        """Click event in opencv window
        """
        nonlocal mouseDown, mouseUp, subpixPoints, click_img, background
        margin = 10

        if event == cv.EVENT_LBUTTONDOWN:
            mouseDown.append((x,y))

        elif event == cv.EVENT_LBUTTONUP:
            mouseUp.append((x,y))
            try:
                if not mouseUp[-1] == mouseDown[-1]:
                    raise ValueError
                point = mouseUp[-1]

                if len(background.shape) == 3: # BGR
                    background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
                localImage = background[point[1]-margin:point[1]+margin, point[0]-margin:point[0]+margin]
                maxCorners, qualityLevel, minDistance = 1000, 0.05, 1
                corners = cv.goodFeaturesToTrack(localImage.astype('uint8'), maxCorners, qualityLevel, minDistance)
                corners = corners.reshape(-1, 2).astype('float32')

                winSize, zeroZone, criteria = (5, 5), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)
                corners = cv.cornerSubPix(localImage.astype('uint8'), corners, winSize, zeroZone, criteria)
                corners += (np.array(point)-margin*np.ones(2))

                minD, P = np.sqrt((corners[0][0]-point[0])**2+(corners[0][1]-point[1])**2), corners[0]
                for c in corners:
                    D = np.sqrt((c[0]-point[0])**2+(c[1]-point[1])**2)
                    if D < minD:
                        minD = D
                        P = c
                subpixPoints.append(P)
                print(mouseUp[-1], P, np.array(mouseUp[-1])-P)

                cv.circle(click_img, (mouseUp[-1][0], mouseUp[-1][1]), 2, (0,0,255), 2)
                cv.circle(click_img, P.astype('uint8'), 1, (0,255,255), 1)
                if len(mouseUp) > 1:
                    for i in range(1, len(mouseUp)):
                        cv.line(click_img, (mouseUp[i][0], mouseUp[i][1]), (mouseUp[0][0], mouseUp[0][1]), (0, 255, 0), 1)
            except ValueError:
                print('Mouse was moved when clicking.')
                mouseUp.pop(-1)
                mouseDown.pop(-1)

    windowName = kwarg['name'] if 'name' in kwarg else 'click_here'
    origin = kwarg['origin'] if 'origin' in kwarg else (0,0)

    cv.namedWindow(windowName)
    cv.moveWindow(windowName, origin[0], origin[1])
    cv.setMouseCallback(windowName, _click)
    print('Click in this order: 1. origin, 2. x axis, 3. y axis')

    mouseDown, mouseUp = [], []
    subpixPoints = []
    click_img = background.copy().astype('uint8')
    if len(click_img.shape) == 2:
        click_img = np.repeat(click_img, 3).reshape((click_img.shape[0],click_img.shape[1],3)) # gray 2 rgb
        #click_img = cv.cvtColor(click_img, cv.COLOR_GRAY2BGR)
    while True:
        cv.imshow(windowName, click_img)
        if 'center' in kwarg:
            center = kwarg['center']
            cv.circle(click_img, center, 2, (255,0,255), 2)
        if cv.waitKey(1) == 13: # key: enter/return
            break
        elif cv.waitKey(1) == 32: # key: space
            # will NOT refresh drawings in window
            try:
                mouseUp.pop(-1)
                mouseDown.pop(-1)
                p = subpixPoints.pop(-1)
                print(p, ' deleted.')
            except IndexError:
                print('No Points Found')
        else:
            pass
    cv.destroyAllWindows()

    return subpixPoints

def sortRect(points):
    """Sort four boundaries of a rectangle.

    Args:
        points (list or tuple): Contain points (>=4).

    Returns:
        rect (tuple): Four boundaries in order, (top, bottom, left, right). 
    """
    rowq = PQ()
    colq = PQ()
    for p in points:
        rowq.put(p[1])
        colq.put(p[0])
    top = rowq.get()
    while not rowq.empty():
        bottom = rowq.get()
    left = colq.get()
    while not colq.empty():
        right = colq.get()
    rect = (top, bottom, left, right)
    return rect

def sortPointCloud(points, out_shape, squareSize, **kwarg):
    """Organize a list of point cloud into a rectangular shape

    Notes:
        Based on (y,x) points
        m, n = row, col
        y1 = y of the first point in row a
        y2 = y of the last point in row a
        y3 = y of the first point in row a+1
        Assuming y3-y1 > 2(y3-y2)

    Args:
        points (numpy.ndarray): A list of points. shape = N x 2
        out_shape (tuple): Output rectangular shape. (m, n), N = m x n
        squareSize (float): Side length of a small grid. 

    Returns:
        out_arr (numpy.ndarray): An array of points with shape (m, n, 2)
    """
    try:
        assert len(points) == out_shape[0]*out_shape[1]
    except AssertionError:
        print(len(points), out_shape)
        raise AssertionError

    a = int(squareSize/2)

    # sort by row
    rowq = PQ()
    for p in points:
        rowq.put((p[0],p))

    _, p0 = rowq.get()
    arr = [[p0]]
    row = 0
    while not rowq.empty():
        _, p = rowq.get()

        if p[0] - p0[0] > a:
            row += 1
            arr.append([])

        arr[row].append(p)
        p0 = p

    # sort each row
    for r in range(len(arr)):
        colq = PQ()
        for p in arr[r]:
            colq.put((p[1],p))

        arr[r] = []
        while not colq.empty():
            _, p = colq.get()
            arr[r].append(p)

    # convert to array
    out_arr = np.zeros((out_shape[0],out_shape[1],2))
    try:
        assert len(arr) == out_shape[0]
        for r in range(len(arr)):
            try:
                assert len(arr[r]) == out_shape[1]
                for c in range(len(arr[r])):
                    out_arr[r, c, :] = arr[r][c]
            except AssertionError:
                print(len(arr[r]), out_shape, 'col # not agree', points, '*** end ***')
                raise AssertionError
    except AssertionError:
        print(len(arr), out_shape, 'row # not agree', points, '*** end ***')
        raise AssertionError

    return out_arr

def hsv2rgb(h, s, v):
    """HSV color gumat to RGB

    Arguments:
        h (float): Hue, 0-360 degree.
        s (float): Saturation, 0-1.
        v (float): Intensity, 0-1.

    Returns:
        r (int): Red, 0-255.
        g (int): Green, 0-255.
        b (int): Blue, 0-255.
    """
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

def rgb2hsv(r, g, b):
    """RGB color gumat to HSV

    Arguments:
        r (int): Red, 0-255.
        g (int): Green, 0-255.
        b (int): Blue, 0-255.

    Returns:
        h (float): Hue, 0-360 degree.
        s (float): Saturation, 0-1.
        v (float): Intensity, 0-1.
    """
    r, g, b = r/255, g/255, b/255
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v

class KeyFinder:
    def __init__(self, data):
        self.data = data

    def find(self, key, **kwarg):
        """Find `key` in data

        Args:
            key (): Key in data dict.
            data (dict): Dict to find.

        Returns:
            value (): data[key].
        """
        if 'data' in kwarg:
            d = kwarg['data']
        else:
            d = self.data
        k0 = key
        if k0 in d:
            return d[k0]
        else:
            for k in d:
                if isinstance(d[k], dict):
                    try:
                        value = self.find(k0, data=d[k])
                        return value
                    except KeyError:
                        continue
            raise KeyError

class SystemParamReader:
    def __init__(self, filename='SystemParams.json'):
        self.filename = filename

    def read(self):
        filename = self.filename
        assert os.path.exists(filename)
        with open(filename, 'r') as file:
            data = json.load(file)

        data['RawImageFiles']['RawFileExtension'] =  '.' + data['RawImageFiles']['RawFileExtension']
        ext = data['RawImageFiles']['RawFileExtension']
        root = data['RawImageFiles']['root']
        data['ImagePaths'] = {}
        data['ImagePaths']['example01_path'] = os.path.join(root, str(data['RawImageFiles']['example01']).zfill(6) + ext)
        data['ImagePaths']['example02_path'] = os.path.join(root, str(data['RawImageFiles']['example02']).zfill(6) + ext)
        data['ImagePaths']['whiteImage_path'] = os.path.join(root, str(data['RawImageFiles']['whiteImage']).zfill(6) + ext)
        data['ImagePaths']['stopImage_path'] = os.path.join(root, str(data['RawImageFiles']['stopImage']).zfill(6) + ext)
        data['ImagePaths']['paraTarget_path'] = os.path.join(root, str(data['RawImageFiles']['paraTarget']).zfill(6) + ext)

        data['ImagePaths']['calibImagePath'] = []
        for item in data['RawImageFiles']['calibRange']:
            for i in range(item[0], item[1]+1):
                path = os.path.join(root, str(i).zfill(6)+ext)
                data['ImagePaths']['calibImagePath'].append(path)

        data['ImagePaths']['otherTargetPath'] = {}
        for key in data['RawImageFiles']['otherTargetRange']:
            data['ImagePaths']['otherTargetPath'][key] = []
            for item in data['RawImageFiles']['otherTargetRange'][key]:
                for i in range(item[0], item[1]+1):
                    path = os.path.join(root, str(i).zfill(6) + ext)
                    data['ImagePaths']['otherTargetPath'][key].append(path)

        data['ImagePaths']['stereoCalibPath'] = {}
        for key1 in data['RawImageFiles']['stereoCalibRange']:
            data['ImagePaths']['stereoCalibPath'][key1] = {}
            for key2 in data['RawImageFiles']['stereoCalibRange'][key1]:
                data['ImagePaths']['stereoCalibPath'][key1][key2] = []
                for item in data['RawImageFiles']['stereoCalibRange'][key1][key2]:
                    for i in range(item[0], item[1]+1):
                        path = os.path.join(root, str(i).zfill(6) + ext)
                        data['ImagePaths']['stereoCalibPath'][key1][key2].append(path)

        data['Checkerboard']['margin'] = int(data['Checkerboard']['number_of_pixels_per_block']/2)

        folder = data['Folder'].copy()
        Log = folder['Log']
        for key in folder:
            if not key in ['Log', 'txtLog']:
                data['Folder'][key+'Folder'] = os.path.join(Log, folder[key])

        return data

class Disparity:
    def __init__(self, rc1, rc2, RC, mask, VI1, VI2, **kwarg):
        self.R, self.C = RC
        self.row1, self.col1 = rc1
        self.row2, self.col2 = rc2
        self.n_pitch = np.sqrt((row1-row2)**2+(col1-col2)**2)
        self.mask = mask
        self.VI1 = VI1
        self.VI2 = VI2

        self.upperLeft, self.lowerRight = self._findBoundary()

        self.shape = kwarg['shape'] if 'shape' in kwarg else None
        self.margin = kwarg['margin'] if 'margin' in kwarg else 0
        self.path = kwarg['path'] if 'path' in kwarg else ''

        if self.shape is None:
            self._get_a_shape()

    def _findBoundary(self):
        true_points = np.argwhere(self.mask)
        upperLeft, lowerRight = true_points.min(axis=0), true_points.max(axis=0)
        return upperLeft, lowerRight

    @property
    def VI1_without_zero_padding(self):
        return self.VI1[int(self.upperLeft[0]):int(self.lowerRight[0]), int(self.upperLeft[1]):int(self.lowerRight[1])]

    @property
    def VI2_without_zero_padding(self):
        return self.VI2[int(self.upperLeft[0]):int(self.lowerRight[0]), int(self.upperLeft[1]):int(self.lowerRight[1])]

    @property
    def accurateDisparityValues(self):
        FIND = False

        if not self.margin == 0:
            img1 = self.VI1_without_zero_padding[self.margin:-self.margin, self.margin:-self.margin]
            img2 = self.VI2_without_zero_padding[self.margin:-self.margin, self.margin:-self.margin]
        else:
            img1 = self.VI1_without_zero_padding
            img2 = self.VI2_without_zero_padding
        filename = 'R'+str(self.row1).zfill(2)+'_C'+str(self.col1).zfill(2)+'_R'+str(self.row2).zfill(2)+'_C'+str(self.col2).zfill(2)
        if not os.path.exists(os.path.join(self.path, filename+'img1.npy')):
            np.save(os.path.join(self.path, filename+'img1.npy'), img1.astype('float64'))
            np.save(os.path.join(self.path, filename+'img2.npy'), img2.astype('float64'))
        shape = self.shape
        match1 = getCheckerboardCorners(os.path.join(self.path, filename+'img1.npy'),shape,visualize=True,extension='npy')
        match2 = getCheckerboardCorners(os.path.join(self.path, filename+'img2.npy'),shape,visualize=True,extension='npy')

        try:
            self.imgPoints1 = match1['imagePoints'][0]
            self.imgPoints2 = match2['imagePoints'][0]
            FIND = True
        except IndexError:
            shape = (self.shape[0]-1, self.shape[1]-1)
            match1 = getCheckerboardCorners(os.path.join(self.path, filename+'img1.npy'),shape,visualize=True,extension='npy')
            match2 = getCheckerboardCorners(os.path.join(self.path, filename+'img2.npy'),shape,visualize=True,extension='npy')
            try:
                self.imgPoints1 = match1['imagePoints'][0]
                self.imgPoints2 = match2['imagePoints'][0]
                FIND = True
            except IndexError:
                FIND = False
                print('Features in **', self, '** not detected.')

        if FIND:
            # self.imgPoints1 = np.flip(self.imgPoints1)
            # self.imgPoints2 = np.flip(self.imgPoints2)
            self._corners = self.imgPoints1
            return (self.imgPoints2 - self.imgPoints1)
        else:
            return None

    def disparityBase(self):
        if self.accurateDisparityValues is None:
            print('Features in **', self, '** not detected.')
            return None
        else:
            valuesY = self.accurateDisparityValues[:, 0] # float
            valuesX = self.accurateDisparityValues[:, 1] # float
            points = self._corners # float

            localUpperLeft, localLowerRight = points.min(axis=0), points.max(axis=0)
            upper, lower, left, right = np.ceil(localUpperLeft[0]), np.floor(localLowerRight[0]), np.ceil(localUpperLeft[1]), np.floor(localLowerRight[1])
            self.upperLeft, self.lowerRight = (left, upper), (right, lower) #(upper, left), (lower, right)

            ys = np.arange(upper, lower)
            xs = np.arange(left, right)
            Y, X = np.meshgrid(ys, xs)

            gdY = griddata(points, valuesY, (Y, X), method='linear', fill_value=0)
            gdX = griddata(points, valuesX, (Y, X), method='linear', fill_value=0)

            return gdX, gdY

    @property
    def disparityX(self):
        gdX, _ = self.disparityBase()
        return gdX

    @property
    def disparityY(self):
        _, gdY = self.disparityBase()
        return gdY

    @property
    def disparityABS(self):
        gdX, gdY = self.disparityBase()
        return np.sqrt(gdX**2 + gdY**2)

    @property
    def disparityX_with_zero_padding(self):
        disparity = np.zeros(self.mask.shape, dtype='float64')
        disparity[int(self.upperLeft[0]):int(self.lowerRight[0]), int(self.upperLeft[1]):int(self.lowerRight[1])] = self.disparityX
        return disparity

    @property
    def disparityY_with_zero_padding(self):
        disparity = np.zeros(self.mask.shape, dtype='float64')
        disparity[int(self.upperLeft[0]):int(self.lowerRight[0]), int(self.upperLeft[1]):int(self.lowerRight[1])] = self.disparityY
        return disparity

    @property
    def disparity_with_zero_padding(self):
        disparity = np.zeros(self.mask.shape, dtype='float64')
        disparity[int(self.upperLeft[0]):int(self.lowerRight[0]), int(self.upperLeft[1]):int(self.lowerRight[1])] = self.disparityABS
        return disparity

    def __repr__(self):
        name = 'Disparity between '
        name += '(' + str(self.row1).zfill(2) + ', ' + str(self.col1).zfill(2) + ')'
        name += '; (' + str(self.row2).zfill(2) + ', ' + str(self.col2).zfill(2) + ')'
        return name

    def __lt__(self, other):
        return self.mask.sum() < other.mask.sum()

    def __gt__(self, other):
        return self.mask.sum() > other.mask.sum()

    def __le__(self, other):
        return self.mask.sum() <= other.mask.sum()

    def __ge__(self, other):
        return self.mask.sum() >= other.mask.sum()

def GO(startPoint, targetPoint, info, **kwarg):
    """Determine how to go from start to target

    Args:
        startPoint (tuple): Start position, (row, col)
        targetPoint (tuple): Target position, (row, col)
        info (dict): See `info` defined in section 11.
        a_star (float): Total cost. This is a path search algorithm. https://en.wikipedia.org/wiki/A*_search_algorithm
        steps (list): Previous steps. 
        ban (list): Banned direction.

    Returns:
        steps (list): A list of steps. e.g. ['right', 'right', 'lower', ...]
    """
    a_star = kwarg['a_star'] if 'a_star' in kwarg else np.nan
    steps = kwarg['steps'] if 'steps' in kwarg else []
    ban = kwarg['ban'] if 'ban' in kwarg else []

    r0, c0 = startPoint
    r1, c1 = targetPoint

    if (r0==r1) and (c0==c1):
        return steps

    dir_list = [('upper',(-1,0)), ('lower',(1,0)), ('left',(0,-1)), ('right',(0,1))]
    opposite = {'upper':(1,0), 'lower':(-1,0), 'left':(0,1), 'right':(0,-1)}

    a = np.zeros(4,dtype='uint16')
    for i, direction in enumerate(dir_list):
        try:
            assert direction[0] in info[r0][c0]
            assert 'shift' in info[r0][c0][direction[0]]
            assert not direction[0] in ban
            a[i] = int(np.abs(r1-(r0+direction[1][0]))+np.abs(c1-(c0+direction[1][1])))
            if not a_star is np.nan: # == np.nan => False ; is np.nan => True
                a[i] += len(steps)
            else:
                a[i] += 1 # first step
        except (AssertionError, KeyError):
            a[i] = np.iinfo(a.dtype).max

    try:
        amin_index = list(a).index(min(min(a),a_star))
    except ValueError:
        last = steps.pop()
        r0 += opposite[last][0]
        c0 += opposite[last][1]
        ban.append(last)
    else:
        direction = dir_list[amin_index][0]
        steps.append(direction)
        ban = []
        r0 += dir_list[amin_index][1][0]
        c0 += dir_list[amin_index][1][1]

    return GO((r0,c0), (r1,c1), info, a_star=int(min(min(a),a_star)), steps=steps, ban=ban)

def intersect(img1, img2, **kwarg):
    """Compute intersection of two images

    Args:
        img1 (numpy.ndarray): Image 1.
        img2 (numpy.ndarray): Image 2.
        margin (int): Margin of image.
        min_area (int): Minimum area valid.
        max_ratio (float): Maximum ratio of width and height

    Returns:
        retval (bool): Whether intersection is valid.
        mask (numpy.ndarray): Intersection mask with the pixels in common set to 1, others set to 0.
    """
    assert img1.shape == img2.shape

    margin = kwarg['margin'] if 'margin' in kwarg else 0
    min_area = kwarg['min_area'] if 'min_area' in kwarg else 0
    max_ratio = kwarg['max_ratio'] if 'max_ratio' in kwarg else max(img1.shape)
    if max_ratio < 1:
        max_ratio = 1/max_ratio

    bg = np.zeros(img1.shape, dtype='uint8')
    if not margin == 0:
        img1 = img1[margin:-margin, margin:-margin]
        img2 = img2[margin:-margin, margin:-margin]

    arr1 = np.where(img1.astype('uint8')>0.5, 1, 0) # pixel>0 = 1; pixel==0 = 0
    arr2 = np.where(img2.astype('uint8')>0.5, 1, 0)
    mask = arr1.astype('uint8') + arr2.astype('uint8')
    mask = np.where(mask.astype('uint8')>1.5, 1, 0) # pixel==2 = 1; pixel<2 = 0

    area = mask.sum()
    if area > min_area:
        true_points = np.argwhere(mask)
        upperLeft = true_points.min(axis=0)
        bottomRight = true_points.max(axis=0)
        ratio = (upperLeft[0]-bottomRight[0])/(upperLeft[1]-bottomRight[1])
        if ratio < 1:
            ratio = 1/ratio
        if ratio <= max_ratio:
            retval = True
            if not margin == 0:
                bg[margin:-margin, margin:-margin] = mask
            else:
                bg = mask
            mask = bg
        else:
            retval = False
    else:
        retval = False

    return retval, mask
