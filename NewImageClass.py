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
        self.name = str(name)
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
                'mean' (float, default=1): Mean brightness. [0,255]

        Returns:
            newImageStack (numpy.ndarray): Unvignetted image(s)
        """
        mean = kwargs['mean'] if 'mean' in kwargs else 1
        imgShape = imgStack.shape[-2:]
        assert imgShape == self.image.shape

        if len(imgStack.shape) == 3: # n images
            newImageStack = np.zeros(imgStack.shape, dtype='float64')
            imgStack = imgStack.astype('float64')
            for i in range(imgStack.shape[0]):
                newImageStack[i,:,:] = imgStack[i,:,:] / (self.image+1e-3) # prevent x/0
                if newImageStack[i,:,:].mean() < mean:
                    newImageStack[i,:,:] = newImageStack[i,:,:]*mean/newImageStack[i,:,:].mean()
        else: # one image
            newImageStack = imgStack.astype('float64') / (self.image+1e-3)
            if newImageStack.mean() < mean:
                newImageStack = newImageStack*mean/newImageStack.mean()

        newImageStack = np.clip(newImageStack, 0, 255)
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
        self.initDepth = cbParams.initDepth()
        self.lastDepth = cbParams.lastDepth()
        self.depthInterval = cbParams.depthInterval()

        N_images = np.round((self.lastDepth+self.depthInterval-self.initDepth)/self.depthInterval).astype(int)
        try:
            assert N_images == images.shape[0]
        except AssertionError:
            print('N_images', N_images, 'input images', images.shape[0])
            raise

        self.imageStack = {}
        for ii, d in enumerate(np.arange(self.initDepth, self.lastDepth+self.depthInterval, self.depthInterval).round(3)): # precision 0.001
            self.imageStack[d] = LFImage(d, images[ii, :, :])

    def __repr__(self):
        return 'Depth Stack from ' + str(self.initDepth) + ' to ' + str(self.lastDepth)
