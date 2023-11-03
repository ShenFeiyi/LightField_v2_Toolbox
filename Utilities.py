# -*- coding:utf-8 -*-
import os
import time
import math
import json
import imageio
import cv2 as cv
import numpy as np
from scipy import stats
from queue import PriorityQueue as PQ
from scipy.interpolate import griddata

class ProgramSTOP(Exception):
    def __init__(self, message='Stop here for debugging'):
        self.message = message
        super().__init__(self.message)

def StraightLine(i, k=1, b=0):
    return k*i+b

def prepareFolder(folder):
    """Clean up a folder for storage

    Args:
        folder (str): Folder name.
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

def create_gif(image_list, gif_name, duration=1.0):
    """Generate gif file using a list of images

    Args:
        image_list (list): A list of all image names.
        gif_name (str): GIF file name. (xxx.gif) 
        duration (float): Duration of each image. 
    """
    import warnings
    warnings.filterwarnings('ignore')
    print('Collecting images for GIF...')
    frames = [imageio.imread(image_name) for image_name in image_list]
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)

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
                print('Mouse moved when clicking.')
                mouseUp.pop(-1)
                mouseDown.pop(-1)

    windowName = kwarg['name'] if 'name' in kwarg else 'click_here'
    origin = kwarg['origin'] if 'origin' in kwarg else (0,0)

    cv.namedWindow(windowName)
    cv.moveWindow(windowName, origin[0], origin[1])
    cv.setMouseCallback(windowName, _click)

    mouseDown, mouseUp = [], []
    click_img = background.copy()
    click_img = click_img.astype('float64')
    click_img = (255*click_img/click_img.max()).astype('uint8')
    if len(click_img.shape) == 2:
        click_img = np.repeat(click_img, 3).reshape((click_img.shape[0],click_img.shape[1],3)) # gray 2 rgb
        #click_img = cv.cvtColor(click_img, cv.COLOR_GRAY2BGR)
    try:
        while True:
            cv.imshow(windowName, click_img)
            key = cv.waitKey(1) & 0xFF
            if key == 13: # key: enter/return
                if not mouseUp == []:
                    break
            elif key == 32: # key: space
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
    except KeyboardInterrupt:
        cv.destroyAllWindows()
        raise ProgramSTOP(message='KeyboardInterrupt!')
    cv.destroyAllWindows() # make sure cv window close

    return mouseUp

def findSubPixPoint(background, **kwarg):
    """Find point at sub-pixel accuracy in a cv window
    Press 'enter/return' to finish
    Press 'space' to delete last point

    Args:
        background (numpy.ndarray): Image to be cropped.
        windowName (str, default='click_here'): CV window name.
        origin (tuple, default=(0,0)): Origin of CV window.
        margin (int, default=10): Half length of ROI square side.

    Returns:
        subpixPoints (list): A list of points containing four corners of the ROI.
    """
    def _click(event, x, y, flags, param):
        """Click event in opencv window
        """
        nonlocal mouseDown, mouseUp, subpixPoints, click_img, background, margin

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
                print('Mouse moved when clicking.')
                mouseUp.pop(-1)
                mouseDown.pop(-1)

    windowName = kwarg['name'] if 'name' in kwarg else 'click_here'
    origin = kwarg['origin'] if 'origin' in kwarg else (0,0)
    margin = kwarg['margin'] if 'margin' in kwarg else 10

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
    try:
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
    except KeyboardInterrupt:
        cv.destroyAllWindows()
        raise ProgramSTOP(message='KeyboardInterrupt!')
    cv.destroyAllWindows() # make sure cv window close

    return subpixPoints

def find_corners_manual(background, **kwargs):
    """Find checkerboard corners manually in a cv window
    Press 'q' to quit
    Press 'w', + rows
    Press 'a', - cols
    Press 's', - rows
    Press 'd', + cols
    Press 'u', - row offset
    Press 'i', + row offset
    Press 'j', - col offset
    Press 'k', + col offset
    Press up arrow, corners go up
    Press down arrow, corners go down
    Press left arrow, corners go left
    Press right arrow, corners go right
    Press 'h', hide or unhide corners

    Args:
        background (numpy.ndarray): Image to detect.
        windowName (str, default='find_corners_manual'): CV window name.
        origin (tuple, default=(0,0)): Origin of CV window.

    Returns:
        corners (numpy.ndarray): An array of points containing corners.
    """
    image = background
    if len(image.shape) == 2: # gray image
        image = cv.cvtColor(image.astype('uint8'), cv.COLOR_GRAY2BGR)

    clean_image = image.copy()

    nrows, ncols = kwargs.get('nrows', 4), kwargs.get('ncols', 5)
    roff1, coff1 = kwargs.get('roff', 10), kwargs.get('coff', 10)
    roff2, coff2 = roff1, coff1

    dragging = False
    corner_to_drag = None

    def genCorners(nrs, ncs):
        nonlocal roff1, coff1, roff2, coff2, image
        pts = np.zeros((nrs, ncs, 2)) # each point (col, row)
        for row in range(nrs):
            for col in range(ncs):
                cr = (image.shape[0]-roff1-roff2)/(nrs-1)
                cc = (image.shape[1]-coff1-coff2)/(ncs-1)
                pts[row, col, :] = np.array([coff1+col*cc, roff1+row*cr])
        return pts

    def draw_grid():
        nonlocal image, corners
        for row in range(corners.shape[0]):
            for col in range(corners.shape[1]):
                cv.circle(image, corners[row, col].astype(int), 3, (0,255,255), 1)
                try:
                    cv.line(image, corners[row, col].astype(int), corners[row+1, col].astype(int), (0,255,0), 1)
                except IndexError:
                    pass
                try:
                    cv.line(image, corners[row, col].astype(int), corners[row, col+1].astype(int), (0,255,0), 1)
                except IndexError:
                    pass
                try:
                    cv.line(image, corners[row+1, col].astype(int), corners[row+1, col+1].astype(int), (0,255,0), 1)
                except IndexError:
                    pass
                try:
                    cv.line(image, corners[row, col+1].astype(int), corners[row+1, col+1].astype(int), (0,255,0), 1)
                except IndexError:
                    pass

    def mouse_callback(event, x, y, flags, param):
        nonlocal dragging, corner_to_drag, corners
        local_corners = corners.reshape(-1,2).copy()

        if event == cv.EVENT_LBUTTONDOWN:
            for ii, corner in enumerate(local_corners):
                corner_x, corner_y = corner
                if abs(corner_x - x) < 10 and abs(corner_y - y) < 10:
                    dragging = True
                    corner_to_drag = ii
                    break

        elif event == cv.EVENT_MOUSEMOVE:
            if dragging:
                new_corner_x = min(max(x, 0), image.shape[1])
                new_corner_y = min(max(y, 0), image.shape[0])
                corners[corner_to_drag//corners.shape[1], np.mod(corner_to_drag, corners.shape[1]), :] = np.array([new_corner_x, new_corner_y])

        elif event == cv.EVENT_LBUTTONUP:
            dragging = False
            corner_to_drag = None
            del local_corners

    corners = genCorners(nrows, ncols)
    origin = kwargs.get('origin', (0,0))

    windowName = kwargs.get('name', 'find_corners_manual')
    cv.namedWindow(windowName)
    cv.moveWindow(windowName, origin[0], origin[1])
    cv.setMouseCallback(windowName, mouse_callback)

    try:
        hide = False
        while True:
            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('w'):
                nrows += 1
                corners = genCorners(nrows, ncols)
            elif key == ord('a'):
                ncols = max(ncols-1, 2)
                corners = genCorners(nrows, ncols)
            elif key == ord('s'):
                nrows = max(nrows-1, 2)
                corners = genCorners(nrows, ncols)
            elif key == ord('d'):
                ncols += 1
                corners = genCorners(nrows, ncols)
            elif key == ord('u'):
                roff1 -= 0.5
                roff2 -= 0.5
                corners = genCorners(nrows, ncols)
            elif key == ord('i'):
                roff1 += 0.5
                roff2 += 0.5
                corners = genCorners(nrows, ncols)
            elif key == ord('j'):
                coff1 -= 0.5
                coff2 -= 0.5
                corners = genCorners(nrows, ncols)
            elif key == ord('k'):
                coff1 += 0.5
                coff2 += 0.5
                corners = genCorners(nrows, ncols)
            elif key == ord('h'):
                hide = not hide
            elif key == 0: # up
                roff1 -= 0.5
                roff2 += 0.5
                corners = genCorners(nrows, ncols)
            elif key == 1: # down
                roff1 += 0.5
                roff2 -= 0.5
                corners = genCorners(nrows, ncols)
            elif key == 2: # left
                coff1 -= 0.5
                coff2 += 0.5
                corners = genCorners(nrows, ncols)
            elif key == 3: # right
                coff1 += 0.5
                coff2 -= 0.5
                corners = genCorners(nrows, ncols)

            draw_grid()
            if hide:
                cv.imshow(windowName, clean_image)
            else:
                cv.imshow(windowName, image)
            del image
            image = clean_image.copy()

        cv.destroyAllWindows()
    except KeyboardInterrupt:
        cv.destroyAllWindows()
        raise ProgramSTOP(message='KeyboardInterrupt!')
    cv.destroyAllWindows() # make sure cv window close
    return corners

def getRect(points):
    """Get four boundaries of a rectangle.

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

def arrangePointCloud(points, out_shape, squareSize, **kwarg):
    """Rearrange a list of point cloud into a rectangular shape

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
        raise

    a = int(squareSize/2)

    # sort by row
    rowq = PQ()
    for p in points:
        rowq.put((p[0], p[1], p))

    _, _, p0 = rowq.get()
    arr = [[p0]]
    row = 0
    while not rowq.empty():
        _, _, p = rowq.get()

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
                raise
    except AssertionError:
        print(len(arr), out_shape, 'row # not agree', points, '*** end ***')
        raise

    return out_arr

def averageDistanceFit(group):
    """Find the average distance in a group of points and fit it to a Gaussion distribution

    Args:
        group (list): [[p11,p12,...],[p21,...],[prc,...],...] # 2d array may work fine, must be a rectangle

    Returns:
        mu (float): Average
        sigma (float): Standard deviation
    """
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

def averageDistanceFit_advance(points):
    """Find the average distance in a set of points (with labels) and fit it to a Gaussion distribution

    Args:
        points (list): A list in this structure, [{'center':(y,x), 'row':'R', 'col':'C'}, ...]

    Note:
        'row' & 'col' in range ('A' ~ 'Z')

    Returns:
        ds (numpy.ndarray): All distance values
        mu (float): Average
        sigma (float): Standard deviation
    """
    valid_labels = {}
    for p in points:
        valid_labels[p['row']+p['col']] = p['center']

    ds = []
    # try all possible points
    for row in range(26):
        for col in range(26):
            # convert to letters
            rName, cName = chr(row+65), chr(col+65)
            # if this point is valid
            if rName+cName in valid_labels:
                p = valid_labels[rName+cName]
                # try 2 points next to it
                newRowName, newColName = chr(row+66), chr(col+66)
                try:
                    newP1 = valid_labels[newRowName+cName]
                    d = np.sqrt((newP1[0]-p[0])**2+(newP1[1]-p[1])**2)
                    ds.append(d)
                except KeyError:
                    pass
                try:
                    newP2 = valid_labels[rName+newColName]
                    d = np.sqrt((newP2[0]-p[0])**2+(newP2[1]-p[1])**2)
                    ds.append(d)
                except KeyError:
                    pass
    ds = np.array(ds)
    mu, sigma = ds.mean(), ds.std()
    return ds, mu, sigma

def Rodrigue2Euler(R):
    """Transform Rodrigue's rotation matrix to Euler angles

    Args:
        R (numpy.ndarray): Rotation vector (3x1) or rotation matrix (3x3)

    Returns:
        a (float): Rotation angle about x axis [radian].
        b (float): Rotation angle about y axis [radian].
        c (float): Rotation angle about z axis [radian].
    """
    """
    R_{Euler} = np.array(
            [
                [cos(b)cos(c),  sin(a)sin(b)cos(c)-cos(a)sin(c),    cos(a)sin(b)cos(c)+sin(a)sin(c) ],
                [cos(b)sin(c),  sin(a)sin(b)sin(c)+cos(a)cos(c),    cos(a)sin(b)sin(c)-sin(a)cos(c) ],
                [-sin(b),       sin(a)cos(b),                       cos(a)cos(b)                    ]
            ]
        )
    """
    assert R.shape in [(3,3), (3,1)]
    if R.shape == (3,1):
        R, _ = cv.Rodrigues(R)

    # sinb = - R[2,0]
    # b = np.arcsin(sinb)
    # sinacosb = R[2,1]
    # sina = sinacosb / np.cos(b)
    # a = np.arcsin(sina)
    # cosbsinc = R[1,0]
    # sinc = cosbsinc / np.cos(b)
    # c = np.arcsin(sinc)

    ### just realized there's an easy way...
    angles = cv.RQDecomp3x3(R)
    a, b, c = angles[0]

    return a, b, c

def myUndistort(imgArr, undist, center, **kwarg):
    """Undistort an image, same size as original image

    Args:
        imgArr (numpy.ndarray): Image to be undistorted.
        undist (list): Undistortion coefficients, [k1, k2, k3].
        center (tuple): Undistortion center.
        method (str, default='linear'): Interpolation method, {'nearest', 'linear', 'cubic'}.

    Returns:
        u (numpy.ndarray): Undistorted image.
    """
    method = kwarg.get('method', 'linear')
    notification = kwarg.get('notification', False)
    if notification:
        t0 = time.time()
        print('Start undistorting... (may take a while)...', end=' ')

    y0, x0 = center
    shape = imgArr.shape

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
    RN2 = YNS**2 + XNS**2

    points = np.zeros((imgArr.shape[0]*imgArr.shape[1],2))
    values = np.zeros(imgArr.shape[0]*imgArr.shape[1]) if len(imgArr.shape) == 2 else np.zeros((imgArr.shape[0]*imgArr.shape[1],3))

    factor = 1
    for i in range(len(undist)):
        factor += undist[i]*RN2**(i+1)

    XDN = XNS * factor
    YDN = YNS * factor

    XD = XDN * normFactor + x0
    YD = YDN * normFactor + y0

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            xd, yd = XD[i,j], YD[i,j]
            points[i*len(ys)+j, 0] = yd
            points[i*len(ys)+j, 1] = xd
            values[i*len(ys)+j] = imgArr[y,x]

    X, Y = np.meshgrid(xs, ys)
    gd = griddata(points, values, (Y, X), method=method, fill_value=0)
    u = gd #.T

    if notification:
        print(time.time()-t0, 'seconds.')

    return u

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

        # ImageProfile
        if not data['ImageProfile']['RawFileExtension'][0] == '.':
            data['ImageProfile']['RawFileExtension'] =  '.' + data['ImageProfile']['RawFileExtension']
        ext = data['ImageProfile']['RawFileExtension']
        root = data['ImageProfile']['root']

        # RawImageFiles
        data['ImagePaths'] = {}
        for key in data['RawImageFiles']:
            if isinstance(data['RawImageFiles'][key], list):
                data['ImagePaths'][key] = []
                for i in range(data['RawImageFiles'][key][0], data['RawImageFiles'][key][1]+1):
                    path = os.path.join(root, str(i).zfill(6) + ext)
                    data['ImagePaths'][key].append(path)
            else:
                data['ImagePaths'][key] = os.path.join(root, str(data['RawImageFiles'][key]).zfill(6) + ext)

        self.data = data
        return data

    def showData(self, filename='SystemParams_FullEdition.json'):
        try:
            saveDict(filename, self.data)
        except AttributeError:
            saveDict(filename, self.read())
        finally:
            print('Data saved to', filename)
