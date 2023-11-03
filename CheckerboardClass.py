# -*- coding:utf-8 -*-
import numpy as np
from Utilities import saveDict

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
        if data['initDepth'] > data['lastDepth']:
            data['depthInterval'] *= -1
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

    def __repr__(self):
        data = 'Checkerboard Parameters\n'
        data += 'Checkerboard calibrating center view: ' + str(self.CalibCheckerSize('mm')) + 'mm with shape ' + str(self.CalibCheckerShape) + '\n'
        data += 'Near-Parallel Checkerboard Size = ' + str(self.CheckerSize('mm')) + 'mm\n'
        data += 'Number of Pixels per Checker = ' + str(self.number_of_pixels_per_checker) + '\n'
        data += 'Depth Range\n'
        data += str(self.initDepth('mm')) + ', ' + str(self.lastDepth('mm')) + ', ' + str(self.depthInterval('mm')) + 'mm\n'
        data += 'Near-parallel checkerboard tilting angle: {:.3f}, {:.3f} degrees\n'.format(self.tilt[0]*180/np.pi, self.tilt[1]*180/np.pi)
        return data
