# -*- coding:utf-8 -*-
import os
import cv2 as cv
import numpy as np

def unvignet(image, white, **kwarg):
    """Unvignetting an image

    Args:
        image (numpy.ndarray): Image to be unvignetted.
        white (numpy.ndarray or float): White background. Same size as image, or 1.0
        mean (float, default=100): Mean brightness. [0,255]

    Returns:
        u (numpy.ndarray): Unvignetted image.
    """
    mean = kwarg['mean'] if 'mean' in kwarg else 100
    i = image.astype('float64')
    i = i / (white+1e-3) # prevent x/0
    i = 255 * i / i.max()
    if i.mean() < mean:
        i = i*mean/i.mean()
    u = np.clip(i, 0, 255)
    return u

def imreadFunc(extension):
    """Read image based on its extension

    Args:
        extension (str): Extension of file.

    Returns:
        some_image_loading_function (function)
    """
    def imread_gray(path):
        return cv.imread(path, 0)

    readFunc = {'npy':np.load}
    for ext in ['jpg', 'jpeg', 'png', 'tif', 'pgm']:
        readFunc[ext] = imread_gray

    return readFunc[extension]

def readCheckerboardImages(root, **kwarg):
    """Iterator, return image path and image array

    Args:
        root (str): Image path root.
        extension (str, default='pgm'): Image file extension.

    Yields:
        path (str): Path to each image file.
        image_arr (numpy.ndarray): Image array.
    """
    extension = kwarg['extension'] if 'extension' in kwarg else 'pgm'

    try:
        imagenames = os.listdir(root)
        imagenames = [name for name in imagenames if name.endswith(extension)]
        imagenames.sort()
        for name in imagenames:
            path = os.path.join(root, name)
            image_arr = imreadFunc(extension)(path)
            yield path, image_arr
    except NotADirectoryError:
        # only one image file
        yield (root, imreadFunc(extension)(root))

def getCheckerboardCorners(root, patternSize, **kwarg):
    """Find checkerboard corners

    Args:
        # for `readCheckerboardImages`
        root (str): Image path root.
        extension (str, default='pgm'): Image file extension.

        # for this function
        patternSize (tuple): Number of inner corners per a chessboard, row and column.
        squareSize (float, default=1.0): Block side length. (mm)
        visualize (bool, default=False): Whether to draw and display corners detected.
        whiteImage (numpy.ndarray, default=1.0): Unvignetting images. Range [0,1], dtype=float64, shape=image.shape

        # for corner refine
        refine (bool, default=True): Whether to refine corners detected to sub-pixel level.
        winSize (tuple, default=(9,9)): Half of the side length of the search window. e.g. if winSize=(5,5), (11,11) search window is used.
        zeroZone (tuple, default=(-1,-1)): To avoid possible singularities of the autocorrelation matrix. (-1,-1) indicates there is no such a size.
        criteria (tuple, default see code): Criteria for termination of the iterative process of corner refinement.

    Returns:
        match (dict): Corresponding points and its raw image path. {'imagePoints':[],'objectPoints':[],'imagePath':[]}
    """
    extension = kwarg['extension'] if 'extension' in kwarg else 'pgm'
    squareSize = kwarg['squareSize'] if 'squareSize' in kwarg else 1.0
    visualize = kwarg['visualize'] if 'visualize' in kwarg else False
    whiteImage = kwarg['whiteImage'] if 'whiteImage' in kwarg else 1.0

    refine = kwarg['refine'] if 'refine' in kwarg else True
    if refine:
        winSize = kwarg['winSize'] if 'winSize' in kwarg else (9, 9)
        zeroZone = kwarg['zeroZone'] if 'zeroZone' in kwarg else (-1, -1)
        criteria = kwarg['criteria'] if 'criteria' in kwarg else (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-3)

    objectPoints = np.zeros((patternSize[0]*patternSize[1], 3), dtype='float64')
    objectPoints[:, :2] = np.indices(patternSize).T.reshape(-1, 2)
    #pS,patternSize; array([[0,0,0],...,[pS[0]-1,0,0],[0,1,0],...,[pS[0]-1,1,0],...,[pS[0]-1,pS[1]-1,0])
    objectPoints *= squareSize

    match = {'imagePoints':[],'objectPoints':[],'imagePath':[]}
    for path, image_arr in readCheckerboardImages(root, extension=extension):
        print('Processing image : ', path)
        # unvignetting
        if isinstance(whiteImage, type(np.array([1]))):
            image_arr = unvignet(image_arr, whiteImage) # 8-bit image for `findChessboardCorners`
        retval, corners = cv.findChessboardCorners(image_arr.astype('uint8'), patternSize=patternSize)
        if retval:
            print('Checkerboard detected ')
            if refine:
                # find sub-pixel accurate location of corners
                corners = cv.cornerSubPix(image_arr.astype('uint8'), corners, winSize, zeroZone, criteria)
            # match points
            imagePoints = corners.reshape(-1, 2) # (mxn, 1, 2) => (mxn, 2)
            match['imagePoints'].append(imagePoints.astype('float32'))
            match['objectPoints'].append(objectPoints.astype('float32'))
            match['imagePath'].append(path)
            if visualize:
                image_bgr = cv.cvtColor(image_arr.astype('uint8'), cv.COLOR_GRAY2BGR)
                cv.drawChessboardCorners(image_bgr, patternSize, corners, retval)
                # compatible with single file
                if len(root.split('.')) == 1:
                    fileroot = root
                    filenumber = path[len(fileroot)+1:].split('.')[0]
                    filename = 'visualize_' + filenumber + '.jpg'
                else:
                    r = root.split('/')[:-1]
                    fileroot = ''
                    for part in r:
                        fileroot = os.path.join(fileroot, part)
                    filename = 'visualize_' + root.split('/')[-1].split('.')[0] + '.jpg'
                cv.imwrite(os.path.join(fileroot, filename), image_bgr)
        else:
            print(f'Error in checkerboard detection. {path[len(root)+1:]}')

    return match

def calibrate(match):
    """ Calibrate camera

    Args:
        match (dict): Corresponding points and its raw image path. {'imagePoints':[],'objectPoints':[],'imagePath':[]}

    Returns:
        ret (float): Root mean square (RMS), re-projection error.
        mtx (numpy.ndarray): 3x3 camera Matrix.
        dist (numpy.ndarray): 1x5 distortion coefficients.
        rvecs (tuple): Rotation vectors, each image.
        tvecs (tuple): Translation vectors, each image.
    """
    objectPoints = np.array(match['objectPoints'])
    imagePoints = np.array(match['imagePoints'])
    path = match['imagePath']
    oneImage = imreadFunc(path[0].split('.')[-1])(path[0])

    print('Calibration start...')
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectPoints, imagePoints, oneImage.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

def calibOptimize(mtx, dist, imageShape):
    """Refine camera matrix

    Args:
        mtx (numpy.ndarray): Camera intrinsic matrix.
        dist (numpy.ndarray): Distortion coefficients.
        imageShape (tuple): Image shape.

    Returns:
        newMatrix (numpy.ndarray): New camera intrinsic matrix.
        roi (tuple): validPixROI, (x,y,w,h), rectangle that outlines all-good-pixels region in the undistorted image.
    """
    print('Optimizing camera matrix...')
    newMatrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, imageShape[::-1], 1, imageShape[::-1])
    return newMatrix, roi

def undistort(imagename, mtx, dist, newMatrix, **kwarg):
    """Undistort an image

    Args:
        imagename (str): Path to Image to be undistorted.
        mtx (numpy.ndarray): Camera intrinsic matrix.
        dist (numpy.ndarray): Distortion coefficients.
        newMatrix (numpy.ndarray): New camera intrinsic matrix.
        roi (tuple): validPixROI, (x,y,w,h).
        crop (bool, default=False): Whether to crop image.

    Returns:
        udst (numpy.ndarray): Undistorted image array.
    """
    image = imreadFunc(imagename.split('.')[-1])(imagename)
    roi = kwarg['roi'] if 'roi' in kwarg else (0, 0, image.shape[1], image.shape[0])
    crop = kwarg['crop'] if 'crop' in kwarg else False
    # undistort
    print(f'Undistorting image : {imagename}')
    udst = cv.undistort(image, mtx, dist, None, newMatrix)

    if crop:
        # crop image
        x, y, w, h = roi
        udst = udst[y:y+h, x:x+w]

    return udst

################################################################
if __name__ == '__main__':

    # specify image range for calibration before run this demo
    root = '/Volumes/GoogleDrive/Other computers/Lab/Feiyi_GoogleDrive/Feiyi_images'

    shape = (11,7)
    match = getCheckerboardCorners(root, shape, visualize=False) # visualize will write images to root folder

    RMS, cameraMatrix, dist, rvecs, tvecs = calibrate(match)

    imagename = match['imagePath'][9]
    oneImage = cv.cvtColor(cv.imread(imagename), cv.COLOR_BGR2GRAY)
    imageShape = oneImage.shape
    newCameraMatrix, roi = calibOptimize(cameraMatrix, dist, imageShape)

##    match['imagePath'].sort()
##    for i in [1, 5, 8, ]:
##        imagename = match['imagePath'][i]
##        oneImage = cv.cvtColor(cv.imread(imagename), cv.COLOR_BGR2GRAY)
##        udst = undistort(imagename, cameraMatrix, dist, newCameraMatrix)
##        cv.imwrite('test_orig_'+str(i)+'.png', oneImage)
##        cv.imwrite('test_udst_'+str(i)+'.png', udst)
