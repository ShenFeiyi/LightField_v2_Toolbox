#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
import json
import cv2 as cv
import numpy as np
from PIL import Image

def _genBoard(paperSize, dpi, data):
    info = {}

    res = {'inch': dpi} # dots / inch
    imgSize = {'mm': np.array(paperSize)*25.4}
    res['mm'] = res['inch']/25.4 # dots / mm
    imgSize['pixel'] = res['mm'] * imgSize['mm']

    CheckerSize = data['CheckerSize']
    xoffset = data.get('xoffset', 0)
    yoffset = data.get('yoffset', 0)
    nrows = data['nrows']
    ncols = data['ncols']

    img = 255*np.ones(imgSize['pixel'].astype(int), dtype='uint8')

    checkerSize = {'mm': CheckerSize}
    checkerSize['pixel'] = int(checkerSize['mm']*res['mm'])

    top = int((imgSize['pixel'][0] - checkerSize['pixel'] * nrows)/2)
    left = int((imgSize['pixel'][1] - checkerSize['pixel'] * ncols)/2)
    bottom = imgSize['pixel'][0].astype(int) - top
    right = imgSize['pixel'][1].astype(int) - left

    for i, row in enumerate(range(top, bottom, checkerSize['pixel'])):
        black = bool(np.mod(i, 2))
        for j, col in enumerate(range(left, right, checkerSize['pixel'])):
            # fill checkerboard
            if black:
                img[row:row+checkerSize['pixel'], col:col+checkerSize['pixel']] = 0
            black = not black

    img = cv.putText(
        img, 'Checkerboard | '+str(nrows)+'x'+str(ncols)+' | Checker Size '+str(checkerSize['mm'])+' mm',
        (500, 500), cv.FONT_HERSHEY_SIMPLEX, 10, (0,0,0), 10, cv.LINE_AA
        )
    img = cv.putText(
        img, 'Width: 150 px = '+str(150/res['mm'])+' mm',
        (500, 1000), cv.FONT_HERSHEY_SIMPLEX, 10, (0,0,0), 10, cv.LINE_AA
        )
    linewidth = [i+1 for i in range(150)]
    black = True
    for ii, lw in enumerate(linewidth):
        if black:
            img[1200:1800, 500+sum(linewidth[:ii]):500+sum(linewidth[:ii+1])] = 0
        black = not black

    info['img'] = img
    info['top'], info['bottom'], info['left'], info['right'] = top, bottom, left, right
    info['checkerSize'] = checkerSize
    info['xoffset'], info['yoffset'] = xoffset, yoffset
    return info

def genBlankBoard(paperSize, dpi, data):
    info = _genBoard(paperSize, dpi, data)
    img = info['img']
    image = Image.fromarray(img)
    filename = f"Checkerboard_{data['nrows']}x{data['ncols']}_checker_{data['CheckerSize']}mm.tif"
    image.save(filename, dpi=(dpi, dpi))

def genTextBoard(paperSize, dpi, data):
    info = _genBoard(paperSize, dpi, data)
    img = info['img']
    top, bottom, left, right = info['top'], info['bottom'], info['left'], info['right']
    checkerSize = info['checkerSize']
    xoffset, yoffset = info['xoffset'], info['yoffset']

    font = cv.FONT_HERSHEY_SIMPLEX
    for i, row in enumerate(range(top, bottom, checkerSize['pixel'])):
        for j, col in enumerate(range(left, right, checkerSize['pixel'])):
            # add text
            img = cv.putText(
                    img, chr(65+i)+chr(65+j), (col+xoffset, row-yoffset+checkerSize['pixel']),
                    font, data['fontscale'], (0,0,0), data['fontthickness'], cv.LINE_AA
                    )
    image = Image.fromarray(img)
    filename = f"Checkerboard_{data['nrows']}x{data['ncols']}_checker_{checkerSize['mm']}mm_text.tif"
    image.save(filename, dpi=(dpi, dpi))

if __name__ == '__main__':
    for filename in '234':
        with open(filename+'.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        genTextBoard([8.5,11], 1200, data)

    genBlankBoard([8.5,11], 1200, {'CheckerSize':1.5/0.994,'nrows':12,'ncols':8})
