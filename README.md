# LFToolbox

[https://github.com/SDaydreamer/LightField_v2_Toolbox](https://github.com/SDaydreamer/LightField_v2_Toolbox)

[toc]



## 0. Problem

Calibrate light field 2.0 camera and process light field images. 



## 1. Environment

-   python 3.10.10
-   imageio 2.27.0
-   matplotlib 3.7.1
-   numpy 1.24.2
-   opencv-contrib-python 4.7.0.72
-   scipy 1.10.1



## 2. Codes

### 2.0. Coordinate Standard

Since I always mix x and y axis, I'll write down the standard here. It is important to check the coordinates for each function ( or even step). 

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
        stp_arr (numpy.ndarray): A numpy array image. (other data format that is able to convert to numpy.ndarray is alse acceptable)
        approxSpacing (int, default=200): Approximate spacing between stops.
        filterDiskMult (float, default=1/6): Filter disk radius, relative to spacing of stops.
        imageBoundary (int, default=50): Image boundary width, help remove local maximums that are close to boundary (not real centers).
        debugDisplay (bool, default=False): Debug display, to show whether centers are estimated correctly.
        ignoreWarning (bool, default=False): Whether to ignore warnings.

    Returns:
        invM (numpy.ndarray): 2x2 transformation matrix. (rotation & shear)
        allPeaks (list): A list of peaks detected. (exclude the ones on the edge)
        peakArr (numpy.ndarray): Peak array, (row, col, 2)
    """
```

#### 2.1.2. `subimageFourCorners`

```python
def subimageFourCorners(invM, peakArr):
    """Convert image centers into subimage corners

    Args:
        invM (numpy.ndarray): 2x2 transformation matrix. (rotation & shear)
        peakArr (numpy.ndarray): Peak array, (row, col, 2)

    Returns:
        cornerGroups (numpy.ndarray): Similar structure to `peakArr`
    """
```

#### 2.1.3. `segmentImage`

```python
def segmentImage(image, invM, peakArr):
    """Crop raw image into subimages

    Args:
        image (numpy.ndarray): Image to be cropped. (Height, Width, Color)
        invM (numpy.ndarray): 2x2 transformation matrix. (rotation & shear)
        peakArr (numpy.ndarray): Peak array, (row, col, 2)

    Returns:
        subimages (dict): Subimages. {#row:{#col:image (Width,Height,Color)}}
        anchors (dict): Five points in each subimage. {#row:{#col:{points}}}
            points = {'center':p0, 'upperLeft':p1, 'upperRight':p2, 'lowerRight':p3, 'lowerLeft':p4}
    """
```



### 2.2. `Checkerboard.py`

#### 2.2.1. `getCheckerboardCorners`

```python
def getCheckerboardCorners(root, patternSize, **kwarg):
    """Find checkerboard corners

    Args:
        # for `readCheckerboardImages`
        root (str): Image path root.
        extension (str, default='pgm'): Image file extension.

        # for this function
        patternSize (tuple): Number of inner corners per a chessboard, row and column.
        squareSize (float, default=1.0): Checker size. (mm)
        visualize (bool, default=False): Whether to draw and display corners detected.

        # for corner refine
        refine (bool, default=True): Whether to refine corners detected to sub-pixel level.
        winSize (tuple, default=(9,9)): Half of the side length of the search window. e.g. if winSize=(5,5), (11,11) search window is used.
        zeroZone (tuple, default=(-1,-1)): To avoid possible singularities of the autocorrelation matrix. (-1,-1) indicates there is no such a size.
        criteria (tuple, default see code): Criteria for termination of the iterative process of corner refinement.

    Returns:
        match (dict): Corresponding points and its raw image path. {'imagePoints':[],'objectPoints':[],'imagePath':[]}
    """
```



### 2.3. `CheckerboardClass.py` 

```python
class CheckerboardParams:
    def __init__(self, data, **kwargs):
        """ data structure
        data = {
            'CalibCheckerSize': ,
            'CalibCheckerShape' ,
            'CheckerSize': ,
            'number_of_pixels_per_checker': ,
            'initDepth': ,
            'lastDepth': ,
            'depthInterval': ,
            'depthRepeat': ,
            'tilt': [rx, ry], [rad], (optional)
        }
        """
        self.data = data
        self.coeffi = kwargs['coeffi'] if 'coeffi' in kwargs else {'M':1, 'MM':1e-3}

    def CalibCheckerSize(self, unit='M'):
        return self.data['CalibCheckerSize'] / self.coeffi[unit.upper()]

    @property
    def CalibCheckerShape(self):
        return tuple(self.data['CalibCheckerShape'])

    def CheckerSize(self, unit='M'):
        return self.data['CheckerSize'] / self.coeffi[unit.upper()]

    @property
    def number_of_pixels_per_checker(self):
        return self.data['number_of_pixels_per_checker']

    @property
    def margin(self):
        return round(self.number_of_pixels_per_checker/2)

    def initDepth(self, unit='M'):
        if unit.upper() in ['M', 'PIXEL']:
            return self.data['initDepth'] / self.coeffi[unit.upper()]
        if unit.upper() in ['MM']:
            return round(self.data['initDepth']/self.coeffi[unit.upper()])

    def lastDepth(self, unit='M'):
        if unit.upper() in ['M', 'PIXEL']:
            return self.data['lastDepth'] / self.coeffi[unit.upper()]
        if unit.upper() in ['MM']:
            return round(self.data['lastDepth']/self.coeffi[unit.upper()])

    def depthInterval(self, unit='M'):
        if unit.upper() in ['M', 'PIXEL']:
            return self.data['depthInterval'] / self.coeffi[unit.upper()]
        if unit.upper() in ['MM']:
            return round(self.data['depthInterval']/self.coeffi[unit.upper()])

    @property
    def depthRepeat(self):
        return self.data['depthRepeat']

    @property
    def tilt(self):
        return self.data.get('tilt', [0, 0])

    def save(self, **kwargs):
        filename = kwargs['filename'] if filename in kwargs else 'CheckerboardParams.json'
        data = {
            'CalibCheckerSize': self.CalibCheckerSize(),
            'CalibCheckerShape': self.CalibCheckerShape,
            'CheckerSize': self.CheckerSize(),
            'number_of_pixels_per_checker': self.number_of_pixels_per_checker,
            'margin': self.margin,
            'initDepth': self.initDepth(),
            'lastDepth': self.lastDepth(),
            'depthInterval': self.depthInterval(),
            'depthRepeat': self.depthRepeat(),
            'tilt': self.tilt
        }
        saveDict(filename, data)
```



### 2.4. `OptiClass.py`

```python
class OptiParams:
    def __init__(self, data):
        """ data structure
        data = {
            'M_MLA': ,
            'f_MLA': ,
            'pixel': ,
            'p_MLA': ,
            'z_focus': ,
            'z_focus_bias': 0 (optional)
        }
        """
        self.data = data
        self.coeffi = {'M':1, 'MM':1e-3, 'PIXEL':self.data['pixel']}

    def M_MLA(self, title='nominal'):
        # title = 'nominal' or 'real'
        if title == 'nominal':
            M = self.data['M_MLA']
        if title == 'real':
            try:
                M = self.data['M_MLA_real']
            except KeyError:
                print("Real M_MLA haven't decided! ")
                raise
        return M

    def f_MLA(self, unit='M'):
        return self.data['f_MLA'] / self.coeffi[unit.upper()]

    def pixel(self, unit='M'):
        return self.data['pixel'] / self.coeffi[unit.upper()]

    def p_MLA(self, unit='M'):
        return self.data['p_MLA'] / self.coeffi[unit.upper()]

    def z_focus(self, unit='M'):
        return self.data['z_focus'] / self.coeffi[unit.upper()]

    def z_focus_bias(self, unit='M'):
        zfb = self.data['z_focus_bias'] if 'z_focus_bias' in self.data else 0
        return zfb / self.coeffi[unit.upper()]

    def save(self, **kwargs):
        filename = kwargs['filename'] if filename in kwargs else 'OptiParams.json'
        data = {
            'M_MLA': self.data['M_MLA'],
            'f_MLA': self.data['f_MLA'],
            'pixel': self.data['pixel'],
            'p_MLA': self.data['p_MLA'],
            'z_focus': self.data['z_focus'],
            'z_focus_bias': self.data['z_focus_bias'] if 'z_focus_bias' in self.data else 0
        }
        try:
            M = self.data['M_MLA_real']
            data['M_MLA_real'] = M
        except KeyError:
            pass
        saveDict(filename, data)
```



### 2.5. `NewImageClass.py`

#### 2.5.1. `LFImage`

```python
class LFImage:
    """Raw Light Field image class (Basic class)
    """
    def __init__(self, name, image):
        """
        Args:
            name (str): Image name.
            image (numpy.ndarray): Image array.
        """
        self.name = str(name)
        self.image = image

    def __repr__(self):
        return self.name
```

#### 2.5.2. `WhiteImage`

```python
class WhiteImage(LFImage):
    """White Image
    """
    def __init__(self, image):
        super().__init__('WhiteImage', image)
        self.image = image.astype('float64')/image.max()

    def unvignet(self, imgStack, **kwargs):
        """Unvignet one or more images

        Args:
            imgStack (numpy.ndarray): An image (w x h) or several images (n x w x h)
            kwargs
                'mean' (float, default=1): Mean brightness. [0,255]

        Returns:
            newImageStack (numpy.ndarray): Unvignetted image(s)
        """
```

#### 2.5.3. `DepthStack`

```python
class DepthStack:
    """Parallel Checkerboard Stack
    """
    def __init__(self, cbParams, images):
        """
        Args:
            cbParams (Class CheckerboardParams)
                initDepth (float): Initial depth, [m]
                lastDepth (float): Last depth, [m]
                depthInterval (float): Depth step, [m]
            images (numpy.ndarray): Images, (n x w x h) # gray

        Attrib:
            self.imageStack (dict): Stack of images. imageStack = {depth: LFImage(class)}
        """
```



### 2.6. `FeaturePoint.py`

#### 2.6.1. `FeatPoint`

```python
class FeatPoint:
	"""One feature point, top-left corner of the checker
	"""
	def __init__(self, row, col, **kwargs):
		"""
		Args:
			row (str or int): Label of the row. 'A' = 1, 'Z' = 26
			col (str or int): Label of the column.
			kwargs (dict):
				'rawPoints' (list): List of rawPoints, for loading data

		Note:
			rawPoints = [[point, view], ...]
			# feature point coordinate on sensor plane and its corresponding view
			# turn into 3D coordinates
			# on sensor plane, z = 0
		"""
		self.row = row.upper() if type(row) == str else chr(64+row)
		self.col = col.upper() if type(col) == str else chr(64+col)
		self.rawPoints = kwargs['rawPoints'] if 'rawPoints' in kwargs else []
		self._raw3D()

	def __repr__(self):
		return 'Feature point ' + self.row + self.col

	def __len__(self):
		return len(self.rawPoints)

	def _raw3D(self):
		"""Turn rawPoints into 3D coordinates, on sensor plane, z = 0
		"""

	def add(self, point, view):
		"""Add a point from a view.
		Args:
			point (tuple): Raw global coordinate (on sensor plane) of a feature point. (y, x)
			view (tuple): The view where the point is. (row, col), NOT the same as row/col in __init__
		"""

	def delete(self, point):
		"""Delete a point from a view
		Args:
			point (tuple): A point close to the one to be deleted.
		"""

	@property
	def isEmpty(self):
		return len(self.rawPoints) == 0

	@property
	def isValid(self):
		# at least 3 points to estimate min circle
		return len(self.rawPoints) > 2

	def save(self, filename):
		"""Save self.row, self.col, self.rawPoints
		Args:
			filename (str): filename, json file
		"""
```

#### 2.6.2. `RayTraceModule`

```python
class RayTraceModule:
	"""First order ray tracing (MLA)
	"""
	def __init__(self, M_MLA, f_MLA, centers, fp, **kwargs):
		"""
		Args:
			M_MLA (float): Magnification of MLA. (=z_obj/z_img)
			f_MLA (float): Focal length of MLA, [pixel]
			centers (numpy.ndarray): Centers of each view. Should be a (#row x #col x 2) array.
			fp (class FeatPoint): Feature point.
			kwargs:
				row_center (int): Index of center row
				col_center (int): Index of center col
				II_norm_vec (numpy.ndarray): Normal vector of II plane.
		"""

	def project(self, **kwargs):
		"""Project feature points to II plane
		Args:
			kwargs:
				1. update self variables
					points (list): Points to project, [[p(1x3), view(1x2)], ...]
					M_MLA (float): Custom M_MLA.
					II_norm_vec (numpy.ndarray): II plane norm vector, (1x3)
					centers (numpy.ndarray): Projection centers, (#row x #col x 2)
				2. new variables
					dist (list): Distortion coefficients, [k1, k2, k3]
					norm (float): Normalization factor.
		"""

	@property
	def minCircle(self):
		"""Error function, rotate II plane normal to z-axis, then calculate min circle
		Returns:
			minCircle (dict): Center and radius of the smallest circle enclosing all reprojected points. {'r':r,'c':(y,x)}
		"""
```



### 2.7. `LFCam.py`

```python
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

            pinhole_camera_matrix (numpy.ndarray): Camera matrix. (main lens + MLA)
            pinhole_camera_dist (numpy.ndarray): Distortion coefficients. (main lens + MLA) 

            MLA_distortion: Distortion coefficients. (MLA only)
            main_lens_distortion: Distortion coefficients. (main lens only)

    """
```

#### 2.7.1. `showExample`

Show example images

#### 2.7.2. `getRowColInfo`

Input row & col info

#### 2.7.3. `generateGridModel`

```python
    def generateGridModel(self, method, **kwargs):
        """Generate grid model

        Args:
            method (str):
                'P' or 'preliminary' for generating a preliminary grid model
                'A' or 'accurate' for generating an accurate grid model

        Returns:
            invM (numpy.ndarray): 2x2 transformation matrix. Define grid orientation (rotation & shear)
            peaks (numpy.ndarray): Center array, (row, col, 2)
        """
```

#### 2.7.4. `extractFeatPoint`

Extract feature points

#### 2.7.5. `PinholeModel`

```python
def PinholeModel(self, **kwargs):
    """Pinhole model of the center view of this light field camera
    camera calibration with constraints

    Returns:
        mtx (numpy.ndarray): Camera matrix. (3x3)
        dist (numpy.ndarray): Camera distortion coefficients. (1x5)
    """
```

#### 2.7.6. `poseEstimate`

```python
def poseEstimate(self, featPoints, **kwargs): # havent tested, no calibration image
    """Estimate checkerboard orientation

    Args:
        featPoints (numpy.ndarray): Feature points on a board.

    Returns:
        pose (dict):
            world (numpy.ndarray): Feature points in world coordinate. Same shape as feature points.
            rx, ry, rz (float): Approx tilt angles, be calculated only when input points >= 4
    """
```

#### 2.7.7. `depthCorrection`

Correct depth based on center view calibration result. 

#### 2.7.8. `optimize_local`

Optimize MLA magnification, MLA rotation, checkerboard tilt angle and MLA distortion. 

#### 2.7.9. `optimize_global`

Optimize main lens distortion. 



### 2.8. `Utilities.py` 

See details in file content. 



### 2.9. `GenCheckerboard/GenBoard.py`

Generate checkerboard for calibration. 



## 3. Log

### version 1.0.2

**2023/08/10** Final update. Add support to poor resolution images. 

### version 1.0.1

**2023/08/08** Final update. now it is able to refocus on different depth (a feature of light field camera). But as for depth estimation, currently I don't know how to do that with OpenCV. 

### version 1.0.0

**2023/05/22** 1. Introducing `LFCam` class to store and do calculations. 2. Introducing `CheckerboardClass`, `OptiClass`, `NewImageClass` and `FeaturePoint` class to help calculating light field information. 3. Update working flowchart. 

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