# LFToolbox

[https://github.com/SDaydreamer/LightField_v2_Toolbox](https://github.com/SDaydreamer/LightField_v2_Toolbox)

[toc]



## 0. Problem

to be written



## 1. Environment

1.   python 3.7.3
2.   numpy 1.21.2
3.   opencv-python 4.5.4-dev
4.   scipy 1.7.3
5.   matplotlib 3.4.3



## 2. Codes

### 2.0. Coordinate Standard

Since I always mix x and y axis, I'll write down the standard here. 

**Standard**: In an image, the $(0,0)$ point is the upper left corner of the image. X axis starts from the origin and goes from left to right. Y axis starts from the origin and goes from top to bottom. The coordinate of an arbitrary pixel in the image is $(x,y)$. 

$(0,0)$ -----------------------------------> $x$ 

|										|

|										|

|----------------------------  $(x_0,y_0)$ 

V

$y$ 

I find that whatever module you are using, when it is related to display, like showing an image or clicking on an image, it will always require/return a (x,y) coordinates, while when you are processing images within codes, it always requires a (y,x) coordinates, or (row, col). 



### 2.1. `GridModel.py`

#### 2.1.1. `BuildGridModel`

```python
def BuildGridModel(stp_arr, **kwarg):
    """Build grid model from stop image

    NOTE:
        May get warning:
            OptimizeWarning: Covariance of the parameters could not be estimated
        This is because when estimating lines, there are only two points feeded.
        To ignore ALL warnings, set ignoreWarning = True

    Args:
        stp_arr (numpy.ndarray): A numpy array image with small stops. (other image file that is able to convert to numpy.ndarray is alse acceptable)
        approxSpacing (int, default=200): Approximate spacing between stops.
        filterDiskMult (float, default=1/8): Filter disk radius, relative to spacing of stops.
        imageBoundary (int, default=50): Image boundary width, help remove local maximums that are close to boundary (not real centers).
        debugDisplay (bool, default=False): Debug display, to show whether centers are estimated correctly.
        ignoreWarning (bool, default=False): Whether to ignore warnings.

    Returns:
        invM (numpy.ndarray): 2x2 transformation matrix. (rotation & shear)
        allPeaks (list): A list of peaks detected. (exclude the ones on the edge)
        peakGroups (dict): Two groups, 'H' & 'V'
            peakGroups['H'] (list): Group peaks in horizontal direction, [[p11,p12,...],[],...]
            peakGroups['V'] (list): Group peaks in vertical direction, [[p11,p21,...],[],...]
    """
    return invM, allPeaks, peakGroups

# coordinates of peaks consist with the input `stp_arr` coordinate
# e.g. stp_arr (y,x) => peak (y,x)
# since there's y=kx+b line fitting in the function
# it is assmued input coordinate is (y,x)
# I'm not sure whether input coordinate (x,y) will affect the result
```

#### 2.1.2. `subimageFourCorners`

```python
def subimageFourCorners(invM, peakGroups):
    """Convert image centers into subimage corners

    Args:
        invM (numpy.ndarray): 2x2 transformation matrix. (rotation & shear)
        peakGroups (dict): Two groups, 'H' & 'V'
            peakGroups['H'] (list): Group peaks in horizontal direction, [[p11,p12,...],[],...]
            peakGroups['V'] (list): Group peaks in vertical direction, [[p11,p21,...],[],...]

    Returns:
        cornerGroups (dict): Similar structure to `peakGroups`
    """
    return cornerGroups
# coordinates of corners consist with the input peak's coordinate
# e.g. peak (y,x) => corners (y,x)
```

#### 2.1.3. `segmentImage`

```python
def segmentImage(image, invM, peakGroups):
    """Crop raw image into subimages

    Args:
        image (numpy.ndarray): Image to be cropped. (Height, Width, Color)
        invM (numpy.ndarray): 2x2 transformation matrix. (rotation & shear)
        peakGroups (dict): Two groups, 'H' & 'V'
            peakGroups['H'] (list): Group peaks in horizontal direction, [[p11,p12,...],[],...]
            peakGroups['V'] (list): Group peaks in vertical direction, [[p11,p21,...],[],...]

    Returns:
        subimages (dict): Subimages. {#row:{#col:image (Width,Height,Color)}}
        anchors (dict): Five points in each subimage. {#row:{#col:{points}}}
            points = {'center':p0, 'upperLeft':p1, 'upperRight':p2, 'lowerRight':p3, 'lowerLeft':p4}
    """
    return subimages, anchors
# coordinates of the output consists with the input coordinate
# e.g. stp_arr (y,x) => peak (y,x)
```



### 2.2. `Checkerboard.py`

#### 2.2.1. `readCheckerboardImages`

```python
def readCheckerboardImages(root, **kwarg):
    """Iterator, return image path and image array

    Args:
        root (str): Image path root.
        extension (str, default='pgm'): Image file extension.

    Yields:
        path (str): Path to each image file.
        image_arr (numpy.ndarray): Image array.
    """
    yield path, image_arr
# using cv.imread to read images
# in its convention, image (y,x)
# you can imagine it is (row, col) instead
```

#### 2.2.2. `getCheckerboardCorners`

```python
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

        # for corner refine
        refine (bool, default=True): Whether to refine corners detected to sub-pixel level.
        winSize (tuple, default=(9,9)): Half of the side length of the search window. e.g. if winSize=(5,5), (11,11) search window is used.
        zeroZone (tuple, default=(-1,-1)): To avoid possible singularities of the autocorrelation matrix. (-1,-1) indicates there is no such a size.
        criteria (tuple, default see code): Criteria for termination of the iterative process of corner refinement.

    Returns:
        match (dict): Corresponding points and its raw image path. {'imagePoints':[],'objectPoints':[],'imagePath':[]}
    """
    return match
# here object and image points are (x,y)
# though it is the result from cv
# it doesn't follow cv's convention
```

#### 2.2.3. `calibrate`

```python
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
    oneImage = cv.cvtColor(cv.imread(path[0]), cv.COLOR_BGR2GRAY)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectPoints, imagePoints, oneImage.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs
```

#### 2.2.4. `calibOptimize`

```python
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
    newMatrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, imageShape[::-1], 1, imageShape[::-1])
    return newMatrix, roi
# roi's coordinate is not tested
# since I'll not use it
```

#### 2.2.5. `undistort`

```python
def undistort(image_arr, mtx, dist, newMatrix, roi, **kwarg):
    """Undistort an image

    Args:
        image_arr (numpy.ndarray): Image to be undistorted.
        mtx (numpy.ndarray): Camera intrinsic matrix.
        dist (numpy.ndarray): Distortion coefficients.
        newMatrix (numpy.ndarray): New camera intrinsic matrix.
        roi (tuple): validPixROI, (x,y,w,h).
        crop (bool, default=False): Whether to crop image.

    Returns:
        udst (numpy.ndarray): Undistorted image array.
    """
    crop = kwarg['crop'] if 'crop' in kwarg else False
    # undistort
    udst = cv.undistort(image_arr, mtx, dist, None, newMatrix)

    if crop:
        # crop image
        x, y, w, h = roi
        udst = udst[y:y+h, x:x+w]

    return udst
# it is from cv
# therefore (y,x)
```

#### 2.2.6. `unvignet`

```python
def unvignet(image, white):
    """Unvignetting an image

    Args:
        image (numpy.ndarray): Image to be unvignetted.
        white (numpy.ndarray or float): White background. Same size as image, or 1.0

    Returns:
        u (numpy.ndarray): Unvignetted image.
    """
    mean = kwarg['mean'] if 'mean' in kwarg else 180
    i = image.astype('float64')
    i = i / (white+1e-3) # prevent x/0
    i = 255 * i / i.max()
    if i.mean() < mean:
        i = i*mean/i.mean()
    u = np.clip(i, 0, 255)
    return u.astype('uint8')
# coordinate consists with input
```



### 2.3. `LFCalib.py` 

#### Steps

```python
############################
""" 0. image preparation """
############################
```

```python
########################################
""" 1. check objective lens movement """
########################################
```

>   Because the iris is tight when its diameter is small, there may be some movement after adjusting its size. 

```python
#####################
""" 2. grid model """
#####################
# rectangular shape is expected (each row/col has the same number of points)
#
# input:    stopImage_path
#
# output:   invM, allPeaks, peakGroups
#
```

```python
#############################
""" 3. prepare white image"""
#############################
# white image, type = float64, [0,1]
#
# input:    whiteImage_path, invM, peakGroups
#
# output:   whiteImage, segWhiteImages, anchors
#
```

```python
#######################
""" 4. segment demo """
#######################
```

```python
#######################################
""" 5. calibrate center subaperture """
#######################################
#
# input:    calibImagePath, whiteImage, invM, peakGroups, R, C
#
# output:   RMS, cameraMatrix, K, dist
#
```

```python
############################
""" 6. check parallelism """
############################
#
# input:    margin, paraTarget_path, whiteImage, invM, peakGroups, anchors
#
# output:   paraCheckerboard, subParas
#
```

```python
##########################
""" 7. local undistort """
##########################
#
# input:    paraCheckerboard, subParas, cameraMatrix, dist, K, anchors
#
# output:   paraCheckerboard_LU
#
```

```python
###########################
""" 8. global undistort """
###########################
#
# input:    peakGroups, R, C, paraCheckerboard_LU, undist
#
# output:   paraCheckerboard_GU, subGUs
#
```

```python
#########################################
""" 9. common area in adjacent views """
########################################
#
# input:    paraTarget_path, margin, subGUs
#
# output:   do_not_input_again
#
```

```python
#####################################
""" 10. find corresponding points """
#####################################
#
# input:    do_not_input_again, margin, anchors, ROWS, COLS
#
# output:   info
#
```

```python
# skip 11
```

```python
########################################
""" 12. generate virtual image plane """
########################################
#
# input:    paraCheckerboard, subGUs, segWhiteImages, anchors, ROWS, COLS, info
#
# output:   subVIs
#
```

```python
###################################
""" 13. save calibration result """
###################################
```



### 2.4. `Undistort.py` 

#### 2.4.1. `_move` 

```python
def _move(x, y, shape, undist, center):
    ...
    return xd, yd
```

>   move pixel according to distortion equations

#### 2.4.2. `myUndistort` 

```python
def myUndistort(imgArr, undist, center, **kwarg):
    """Undistort an image, same size as original image

    Args:
        imgArr (numpy.ndarray): Image to be undistorted.
        undist (list): Undistortion coefficients, [k1, k2, p1, p2, k3].
        center (tuple): Undistortion center.
        method (str, default='linear'): Interpolation method, {'nearest', 'linear', 'cubic'}.

    Returns:
        u (numpy.ndarray): Undistorted image.
    """
    return u
# while u, imgArr are (y,x)
# center is (x,y)
```

>   Use `scipy.interpolate.griddata` to interpolate pixel values
>
>   pretty SLOW since it moves pixel by pixel. 



### 2.5. `Utilities.py` 

#### 2.5.1. `ProgromSTOP` 

```python
class ProgramSTOP(Exception):
    def __init__(self, message='Stop here for debugging'):
        self.message = message
        super().__init__(self.message)
```

#### 2.5.2. `prepareFolder` 

```python
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
            else:
                os.remove(os.path.join(folder, file_or_dir))
```

#### 2.5.3. `saveDict` 

```python
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
```

#### 2.5.4. `loadDict` 

```python
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
```

#### 2.5.5. `findROI` 

```python
def findROI(background, **kwarg):
    """Find ROI in a cv window
    Press 'enter/return' to finish
    Press 'space' to delete last point

    Args:
        background (numpy.ndarray): Image to be cropped.
        windowName (str, default='click_here'): CV window name.

    Returns:
        mouseUp (list): A list of points containing four corners of the ROI.
    """
    return mouseUp
```

>   Find ROI in a cv window by clicking. There's **no constraints** in length of points. Basically, it records every point you clicked. 

#### 2.5.6. `sortRect` 

```python
def sortRect(points):
    """Sort four boundaries of a rectangle.

    Args:
        points (list or tuple): Contain points.

    Returns:
        rect (tuple): Four boundaries in order, (top, bottom, left, right). 
    """
    return rect
```

#### 2.5.7. `sortPointCloud` 

```python
def sortPointCloud(points, out_shape, **kwarg):
    # reshape a point array
    # Nx2 -> mxnx2, N=mxn
    # based on (y,x) points
    return out_arr
```

#### 2.5.8. `findSubPixPoint` 

```python
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
    return subpixPoints
```

#### 2.5.9. `hsv2rgb` 

```python
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
```

#### 2.5.10. `rgb2hsv` 

```python
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
```

#### 2.5.11. `SystemParamReader` 

```python
class SystemParamReader:
    """Read system profile
    """
    def __init__(self, filename='SystemParams.json'):
        self.filename = filename

    def read(self):
        """Read json profile
        """
        return data
```

#### 2.5.12. `GO` 

```python
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
```



### 2.6. Estimate Main Lens Distortion

`EstimateMainLensDistortionII.py` estimate manually

`EstimateMainLensDistortionIII.py` automatically (SLOW)



### 2.7. On-axis point disparity

`onAxisPart1.py` 

1.   Calibrate and undistort local images
2.   Undistort global image
3.   Move subimages to `virtual image plane` 

`onAxisPart2.py` 

1.   Estimate real on-axis point in center view (it is different with the stop image center in `GridModel.py`)

`onAxisPart3.py` 

1.   Find local x-axis and y-axis in each view
2.   Determine real on-axis point coordinate under local basis in center view
3.   Determine real on-axis point coordinate in other views under local basis
4.   Compute disparity



## 3. Log

### version 0.6.0

**2022/10/28** 1. rename `LFDepth.py` as `LFCalib.py`. 2. `LFCalib.py` will calibrate the local distortion and provide an example of how to use functions. 3. `EstimateMainLensDistortionII.py` and `EstimateMainLensDistortionIII.py` will calibrate the global distortion. 4. These codes are right now serve for experiments, not an APP. 5. Add on-axis disparity calculation. 6. Add a profile file `SystemParams.json` to manage input parameters. 

### version 0.5.0

**2022/10/04** Re-organize files. Delete 1. `LFCalib.py` (replaced by `LFDepth.py`), 2. `EstimateMainLensDistortion.py` (replaced by `EstimateMainLensDistortionII.py`), 3. `checkerboard_test` and `images` folder (initially for testing images, now images are directly cited from images dataset `Feiyi_images`). Add `Utilities.py`, collect some common supporting functions

### version 0.4.1

**2022/09/29** update `LFDepth.py` 

### version 0.4.0

**2022/09/23** add a return value `anchors` in function `segmentImage`; add file `LFDepth.py`; bug fix in `unvignet`; upload `EstimateMainLensDistortion.py`; 

### version 0.3.2

**2022/09/08** bug fix (add `unvignet` to `Checkerboard.py`); 

### version 0.3.1

**2022/09/06** update `GridModel.py`. 

**2022/09/01** upload `LFCalib.py`. 

**2022/08/31** bug fix in `readCheckerboardImages` and `getCheckerboardCorners` in `Checkerboard.py`. 

### version 0.3.0

**2022/08/30** Add two functions to `GridModel.py` (`subimageFourCorners` and `segmentImage`). 

### version 0.2.0

**2022/08/04** Update `GridModel.py` (`BuildGridModel.py`), replace `Pillow` with `OpenCV`; Upload `CheckerBoard.py` and test checkerboard images

### version 0.1.0

**2022/07/28** Upload `README.md`, `BuildGridModel.py` and test images