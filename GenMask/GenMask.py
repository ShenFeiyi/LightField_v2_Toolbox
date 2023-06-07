#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
import json
import cv2 as cv
import numpy as np
from PIL import Image

def _mask(pMLA, r1, r2, innerF):
    img = np.zeros(np.array([pMLA['pixel'], pMLA['pixel']]).astype(int), dtype='uint8')

    center = (round(pMLA['pixel']/2), round(pMLA['pixel']/2))
    color = (0, 0, 0) if innerF else (255, 255, 255)
    thickness = -1
    img = cv.circle(img, center, int(r2), color, thickness)
    color = (255, 255, 255) if innerF else (0, 0, 0)
    img = cv.circle(img, center, int(r1), color, thickness)

    return img

def genMask(paperSize, dpi, data):
    res = {'inch': dpi} # dots / inch
    imgSize = {'mm': np.array(paperSize)*25.4}

    res['mm'] = res['inch']/25.4 # dots / mm
    imgSize['pixel'] = res['mm'] * imgSize['mm']

    innerF = data.get('inner', True)
    pmla = data.get('p_MLA', 1) # mm
    xoffset = data.get('xoffset', 0)
    yoffset = data.get('yoffset', 0)
    nrows = data['nrows']
    ncols = data['ncols']
    r1 = data.get('r1', 0.31)*res['mm']
    r2 = data.get('r2', 0.5)*res['mm']

    img = 255*np.ones(imgSize['pixel'].astype(int), dtype='uint8')

    pMLA = {'mm': pmla}
    pMLA['pixel'] = int(pMLA['mm']*res['mm'])

    # ref lines
    o1 = np.array([xoffset, yoffset], dtype=int)
    o2 = np.array([xoffset, yoffset+(5+5+17+5)*res['mm']], dtype=int)
    o3 = np.array([xoffset, yoffset+(5+5+17+5+5)*res['mm']], dtype=int)
    o4 = np.array([xoffset, yoffset+(2*(5+5+17+5)+5)*res['mm']], dtype=int)
    color = (0, 0, 0)
    thickness = 5

    # ref 1
    start_point = o1 + (np.array([0,5])*res['mm']).astype(int)
    end_point = o1 + (np.array([23,5])*res['mm']).astype(int)
    img = cv.line(img, start_point, end_point, color, thickness)  
    for i in range(5):
        start_point = o1 + (np.array([(i+1)*23/6,0])*res['mm']).astype(int)
        end_point = o1 + (np.array([(i+1)*23/6,5])*res['mm']).astype(int)
        img = cv.arrowedLine(img, start_point, end_point, color, thickness)

    # ref 2
    start_point = o2 + (np.array([0,0])*res['mm']).astype(int)
    end_point = o2 + (np.array([23,0])*res['mm']).astype(int)
    img = cv.line(img, start_point, end_point, color, thickness)  
    for i in range(5):
        start_point = o2 + (np.array([(i+1)*23/6,5])*res['mm']).astype(int)
        end_point = o2 + (np.array([(i+1)*23/6,0])*res['mm']).astype(int)
        img = cv.arrowedLine(img, start_point, end_point, color, thickness)

    # ref 3
    start_point = o3 + (np.array([0,5])*res['mm']).astype(int)
    end_point = o3 + (np.array([23,5])*res['mm']).astype(int)
    img = cv.line(img, start_point, end_point, color, thickness)  
    for i in range(5):
        start_point = o3 + (np.array([(i+1)*23/6,0])*res['mm']).astype(int)
        end_point = o3 + (np.array([(i+1)*23/6,5])*res['mm']).astype(int)
        img = cv.arrowedLine(img, start_point, end_point, color, thickness)

    # ref 4
    start_point = o4 + (np.array([0,0])*res['mm']).astype(int)
    end_point = o4 + (np.array([23,0])*res['mm']).astype(int)
    img = cv.line(img, start_point, end_point, color, thickness)  
    for i in range(5):
        start_point = o4 + (np.array([(i+1)*23/6,5])*res['mm']).astype(int)
        end_point = o4 + (np.array([(i+1)*23/6,0])*res['mm']).astype(int)
        img = cv.arrowedLine(img, start_point, end_point, color, thickness)

    # array
    o1 = np.array([yoffset+(5+5)*res['mm'], xoffset+1*res['mm']], dtype=int)
    o2 = np.array([yoffset+(5+5+17+5+5*3)*res['mm'], xoffset+1*res['mm']], dtype=int)
    lenslet = _mask(pMLA, r1, r2, innerF)
    for row in range(nrows):
        for col in range(ncols):
            img[
                o1[0]+row*pMLA['pixel']:o1[0]+(row+1)*pMLA['pixel'],
                o1[1]+col*pMLA['pixel']:o1[1]+(col+1)*pMLA['pixel']
                ] = lenslet
    lenslet = _mask(pMLA, r1, r2, not innerF)
    for row in range(nrows):
        for col in range(ncols):
            img[
                o2[0]+row*pMLA['pixel']:o2[0]+(row+1)*pMLA['pixel'],
                o2[1]+col*pMLA['pixel']:o2[1]+(col+1)*pMLA['pixel']
                ] = lenslet

    image = Image.fromarray(img)
    filename = f"MASK_{data['nrows']}x{data['ncols']}.tif"
    image.save(filename, dpi=(dpi, dpi))

if __name__ == '__main__':
    genMask([8.5,11], 1200, {'nrows':17,'ncols':21,'xoffset':1000,'yoffset':5000})
