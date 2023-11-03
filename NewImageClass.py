# -*- coding:utf-8 -*-
import numpy as np

class LFImage:
    """Raw Light Field image class
    """
    def __init__(self, name, image):
        """
        Args:
            name (str): Image name.
            image (numpy.ndarray): Image array.
        """
        self.name = 'LFImage_'+str(name)
        self.image = image

    def __repr__(self):
        return self.name

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
                'clip' (float, default=255): Value to clip. [0,255]

        Returns:
            newImageStack (numpy.ndarray): Unvignetted image(s)
        """
        clip = kwargs['clip'] if 'clip' in kwargs else 255
        imgShape = imgStack.shape

        OneGrayImage = len(imgStack.shape) == 2 # one gray image
        OneRGBImage = len(imgStack.shape) == 3 and imgStack.shape[-1] == 3 # one RGB image

        if OneGrayImage: # one gray image
            newImageStack = imgStack.astype('float64') / (self.image+1e-3)
            newImageStack = 255*(newImageStack-newImageStack.min())/(newImageStack.max()-newImageStack.min())
            newImageStack = np.clip(newImageStack, 0, clip)*255/clip

        elif OneRGBImage: # one RGB image
            newImageStack = np.zeros(imgStack.shape, dtype='float64')
            for c in range(3):
                newImageStack[:,:,c] = imgStack[:,:,c].astype('float64') / (self.image+1e-3)
            newImageStack = 255*(newImageStack-newImageStack.min())/(newImageStack.max()-newImageStack.min())
            newImageStack = np.clip(newImageStack, 0, clip)*255/clip

        elif len(imgStack.shape) == 3: # n gray images
            newImageStack = np.zeros(imgStack.shape, dtype='float64')
            imgStack = imgStack.astype('float64')
            for i in range(imgStack.shape[0]):
                newImageStack[i,:,:] = imgStack[i,:,:] / (self.image+1e-3) # prevent x/0
                newImageStack[i,:,:] = 255*(newImageStack[i,:,:]-newImageStack[i,:,:].min())/(newImageStack[i,:,:].max()-newImageStack[i,:,:].min())
                newImageStack[i,:,:] = np.clip(newImageStack[i,:,:], 0, clip)*255/clip

        elif len(imgStack.shape) == 4: # n RGB images
            newImageStack = np.zeros(imgStack.shape, dtype='float64')
            imgStack = imgStack.astype('float64')
            for i in range(imgStack.shape[0]):
                for c in range(3): # RGB channel
                    newImageStack[i,:,:,c] = imgStack[i,:,:,c] / (self.image+1e-3) # prevent x/0
                newImageStack[i] = 255*(newImageStack[i]-newImageStack[i].min())/(newImageStack[i].max()-newImageStack[i].min())
                newImageStack[i] = np.clip(newImageStack[i], 0, clip)*255/clip

        else:
            print('Image Shape Does NOT Match')
            raise

        return newImageStack

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
        self.initDepth = round(cbParams.initDepth(), 3)
        self.lastDepth = round(cbParams.lastDepth(), 3)
        self.depthInterval = round(cbParams.depthInterval(), 3)

        N_images = np.round((self.lastDepth+self.depthInterval-self.initDepth)/self.depthInterval).astype(int)
        try:
            assert N_images == images.shape[0]
        except AssertionError:
            print('N_images', N_images, 'input images', images.shape[0])
            raise

        self.imageStack = {}
        ii, d = 0, self.initDepth
        while True:
            self.imageStack[d] = LFImage(d, images[ii])
            d = round(self.depthInterval+d, 3)
            ii += 1
            if self.initDepth < self.lastDepth and d > self.lastDepth:
                break
            if self.initDepth > self.lastDepth and d < self.lastDepth:
                break

    def __repr__(self):
        return 'Depth Stack from ' + str(self.initDepth) + ' to ' + str(self.lastDepth)
