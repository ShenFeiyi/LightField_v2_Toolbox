# -*- coding:utf-8 -*-
import os
import imageio
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from queue import PriorityQueue as PQ

def create_gif(image_list, gif_name, duration=1.0):
    import warnings
    warnings.filterwarnings('ignore')
    print('Collecting images for GIF...')
    frames = [imageio.imread(image_name) for image_name in image_list]
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)

from GridModel import segmentImage
from Utilities import ProgramSTOP, loadDict, Rodrigue2Euler, myUndistort

from LFCam import LFCam

cam = LFCam('SystemParams.json')

##cam.initFolder()
##cam.showExample()

if not os.path.exists(cam.RowColInfoFilename):
    cam.getRowColInfo()
else:
    data = loadDict(cam.RowColInfoFilename)
    cam.row_center, cam.row_total = data['R'], data['ROWS']
    cam.col_center, cam.col_total = data['C'], data['COLS']

initDepth, lastDepth, depthInterval = cam.cbParams.initDepth(), cam.cbParams.lastDepth(), cam.cbParams.depthInterval()
##initDepth, lastDepth, depthInterval = -129e-3, -100e-3, 1e-3

invM1, peaks1 = cam.generateGridModel('p', filterDiskMult=1/2.1, imageBoundary=75, debugDisplay=False)
##invM1, peaks1 = cam.generateGridModel('p', filterDiskMult=1/8, imageBoundary=75, debugDisplay=False)
cam.extractFeatPoints(initDepth=initDepth, lastDepth=lastDepth)
invM2, peaks2 = cam.generateGridModel('a', showStep=False)

mtx, dist = cam.PinholeModel()

##cam.depthCorrection()

ret1 = cam.optimize_local(detail=True)
ret2 = cam.optimize_global(detail=True)



def compare_3_sets_of_peaks():
    # result: compare 3 sets of peaks
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(255*cam.whiteImage.image, cmap='gray')
    s1 = ax.scatter(peaks1[:,:,1], peaks1[:,:,0], color='r')
    s2 = ax.scatter(peaks2[:,:,1], peaks2[:,:,0], color='g')
    s3 = ax.scatter(cam.peaks[:,:,1], cam.peaks[:,:,0], color='b', marker='x')
    plt.legend((s1, s2, s3), ('convolution', 'line fitting', 'local optimization'))
    plt.show()

def present_II_plane_reprojection_and_number_vs_radius():
    # result: present II plane reprojection
    depth_ref = cam.optiSystem.z_focus('m')

    fig1 = plt.figure()
    fig2 = plt.figure()
    sensor_plane = fig1.add_subplot(1,1,1)
    II_plane = fig2.add_subplot(1,1,1)
    sensor_plane.imshow(cam.paraCheckerStack.imageStack[depth_ref].image, cmap='gray')
    sensor_plane.set_title('Sensor Plane, Depth = ' + str(depth_ref) + ' m')
    II_plane.set_title('II Plane, Depth = ' + str(depth_ref) + ' m')

    nr, newnr = [], []
    oldRadii, newRadii = [], []
    FPs = cam.featPoints[depth_ref]
    for fpName in FPs:
        rtm = FPs[fpName]
        # present rawPoints on sensor plane
        for p, v in rtm.fp.rawPoints:
            sensor_plane.scatter(p[1], p[0], s=3)
            sensor_plane.text(p[1], p[0], rtm.fp.row+rtm.fp.col, fontsize=8)
        # present II points on II plane
        if rtm.fp.isValid: # more than 2 points
            # old
            IIPoints = rtm.project(M_MLA=cam.optiSystem.M_MLA('nominal'), centers=peaks2)
            II_plane.scatter(IIPoints[:,1], IIPoints[:,0], s=3, color='r')
            circle_params = rtm.minCircle
            oldRadii.append(circle_params['radius'])
            nr.append([circle_params['radius'], len(IIPoints)])
            circle = plt.Circle((circle_params['center'][1], circle_params['center'][0]), circle_params['radius'], fill=False)
            II_plane.add_patch(circle)
            II_plane.text(circle_params['center'][1], circle_params['center'][0], rtm.fp.row+rtm.fp.col, fontsize=8)
            II_plane.set_aspect('equal')
            # new
            alpha, beta = cam.cbParams.tilt
            RX = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
            RY = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
            II_norm_vec = np.matmul(np.matmul(RX, RY), np.array([[0],[0],[1]])).reshape(3)
            IIPoints = rtm.project(
                M_MLA=cam.optiSystem.M_MLA('real'), centers=cam.peaks, dist=cam.MLA_distortion,
                norm=cam.optiSystem.p_MLA('pixel'), II_norm_vec=II_norm_vec
                )
            II_plane.scatter(IIPoints[:,1], IIPoints[:,0], s=3, color='g')
            circle_params = rtm.minCircle
            newRadii.append(circle_params['radius'])
            newnr.append([circle_params['radius'], len(IIPoints)])
            circle = plt.Circle((circle_params['center'][1], circle_params['center'][0]), circle_params['radius'], fill=False)
            II_plane.add_patch(circle)
            II_plane.text(circle_params['center'][1], circle_params['center'][0], rtm.fp.row+rtm.fp.col, fontsize=8)
            II_plane.set_aspect('equal')

    oldRadii = np.array(oldRadii)
    newRadii = np.array(newRadii)
    print(oldRadii.mean(), oldRadii.std())
    print(newRadii.mean(), newRadii.std())

    plt.show()

    # number of points in circle VS circle radius
    def line(i, k=1, b=0):
        return k*i+b

    xs, ys = [], []
    for data in nr:
        y, x = data
        xs.append(x)
        ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    popt, pcov = curve_fit(line, xs, ys)
    k, b = popt
    print('number vs radius', k, b)
    plt.scatter(xs, ys, color='r')
    plt.plot([xs.min(), xs.max()], [line(xs.min(),k,b), line(xs.max(),k,b)], color='r', label='before optimization')

    xs, ys = [], []
    for data in newnr:
        y, x = data
        xs.append(x)
        ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    popt, pcov = curve_fit(line, xs, ys)
    k, b = popt
    print('number vs radius', k, b)
    plt.scatter(xs, ys, color='g')
    plt.plot([xs.min(), xs.max()], [line(xs.min(),k,b), line(xs.max(),k,b)], color='g', label='after optimization')

    plt.xlabel('number of points in circle', fontsize=18)
    plt.ylabel('circle radius / pixel', fontsize=18)
    plt.title('number of points in circle VS circle radius', fontsize=18)
    plt.legend()
    plt.show()

def reproject_2_depth():
    # result: present II plane reprojection @ different depth
    radii = {}
    length = [np.inf, -1]
    for depth in cam.paraCheckerStack.imageStack:
        try:
            FPs = cam.featPoints[depth]
            radii[depth] = {}
            for fpName in FPs:
                rtm = FPs[fpName]
                fp = rtm.fp
                if not len(fp) in radii[depth]:
                    radii[depth][len(fp)] = []
                # present II points on II plane
                if fp.isValid: # more than 2 points
                    alpha, beta = cam.cbParams.tilt
                    RX = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
                    RY = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
                    II_norm_vec = np.matmul(np.matmul(RX, RY), np.array([[0],[0],[1]])).reshape(3)
                    IIPoints = rtm.project(
                        M_MLA=cam.optiSystem.M_MLA('real'), centers=cam.peaks, dist=cam.MLA_distortion,
                        norm=cam.optiSystem.p_MLA('pixel'), II_norm_vec=II_norm_vec
                        )
                    circle_params = rtm.minCircle
                    radii[depth][len(fp)].append(circle_params['radius'])
                    if len(fp) < length[0]:
                        length[0] = len(fp)
                    if len(fp) > length[1]:
                        length[1] = len(fp)
        except KeyError:
            pass

    for i in range(length[0],length[1]+1):
        try:
            exec('fig'+str(i)+'=plt.figure(dpi=200)')
            plt.subplots_adjust(bottom=0.18)
            exec('ax'+str(i)+'=fig'+str(i)+'.add_subplot(1,1,1)')
            exec('ax'+str(i)+'.set_title(str(i)+" points", fontsize=16)')
            exec('ax'+str(i)+'.set_xlabel("depth / m", fontsize=16)')
            exec('ax'+str(i)+'.set_ylabel("radius / px", fontsize=16)')
            exec('ax'+str(i)+'.set_xticks(np.arange(initDepth, lastDepth, depthInterval).round(3), np.arange(initDepth, lastDepth, depthInterval).round(3), rotation=45)')
            for d in radii:
                exec('ax'+str(i)+'.scatter(d*np.ones(len(radii[d][i])), np.array(radii[d][i]))')
                mean = np.array(radii[d][i]).mean()
                std = np.array(radii[d][i]).std()
                exec('ax'+str(i)+'.plot([d-3e-4, d+3e-4], [mean+3*std, mean+3*std], color="r")')
                exec('ax'+str(i)+'.plot([d-3e-4, d+3e-4], [mean, mean], color="g")')
                exec('ax'+str(i)+'.plot([d-3e-4, d+3e-4], [mean-3*std, mean-3*std], color="b")')
        except KeyError:
            pass
        plt.savefig(os.path.join('ImageLog', 'points'+str(i)+'.jpg'))
        plt.close()
    fig = plt.figure(dpi=200)
    plt.subplots_adjust(bottom=0.18)
    ax = fig.add_subplot(1,1,1)
    ax.set_title("All Points", fontsize=16)
    ax.set_xlabel("depth / m", fontsize=16)
    ax.set_ylabel("radius / px", fontsize=16)
    ax.set_xticks(np.arange(initDepth, lastDepth, depthInterval).round(3), np.arange(initDepth, lastDepth, depthInterval).round(3), rotation=45)
    for d in radii:
        data = []
        for i in radii[d]:
            data += radii[d][i]
        ax.scatter(d*np.ones(len(data)), np.array(data))
        mean = np.array(data).mean()
        std = np.array(data).std()
        ax.plot([d-3e-4, d+3e-4], [mean+3*std, mean+3*std], color="r")
        ax.plot([d-3e-4, d+3e-4], [mean, mean], color="g")
        ax.plot([d-3e-4, d+3e-4], [mean-3*std, mean-3*std], color="b")
    plt.savefig(os.path.join('ImageLog', 'points_all.jpg'))
    plt.close()

def find_proper_M():
    depth = cam.optiSystem.z_focus('m')

    for M in np.arange(2.7, 2.8, 0.01).round(3):
        image_sensor_plane = cam.paraCheckerStack.imageStack[depth].image
        subimages, anchors = segmentImage(image_sensor_plane, cam.invM, cam.peaks)

        offset = np.array([200, 200]) # y,x
        IIimage = np.zeros(
            (
                int(7*cam.optiSystem.p_MLA('pixel')),
                int(8*cam.optiSystem.p_MLA('pixel')),
                ), dtype='float64'
            )

        II_norm_vec = np.array([0,0,1])

        z_obj = (M+1)*cam.optiSystem.f_MLA('pixel')
        z_img = z_obj/M
        xp, yp, zp = cam.peaks[cam.row_center, cam.col_center, 1], cam.peaks[cam.row_center, cam.col_center, 0], z_obj+z_img
        ap, bp, cp = II_norm_vec # represent a plane: a(x-x0)+b(y-y0)+c(z-z0)=0
        for row in range(cam.row_total):
            for col in range(cam.col_total):
                IIpoints, values = [], []
                img = subimages[row][col]

                center = np.array([cam.peaks[row,col,0], cam.peaks[row,col,1], z_img])
                for ix in range(img.shape[1]):
                    for iy in range(img.shape[0]):
                        p = np.array([iy+anchors[row][col]['upperLeft'][0], ix+anchors[row][col]['upperLeft'][1], 0])
                        dir_vec = center - p # direction vector (a, b, c); represent a line: (x-x0)/a = (y-y0)/b = (z-z0)/c
                        al, bl, cl = dir_vec
                        if not ((al==0) or (bl==0) or (cl==0)):
                            # solve intersection of line & plane
                            A = np.array([[ap, bp, cp], [1/al, -1/bl, 0], [0, 1/bl, -1/cl]])
                            B = np.array([[ap*xp+bp*yp+cp*zp], [p[0]/al-p[1]/bl], [p[1]/bl-p[2]/cl]])
                            xyz = np.matmul(np.linalg.inv(A), B).reshape(3)
                            IIpoints.append(xyz[:-1]+offset)
                            values.append(img[iy, ix])
                IIpoints = np.array(IIpoints)
                X, Y = np.meshgrid(np.arange(IIimage.shape[1]), np.arange(IIimage.shape[0]))
                gd = griddata(IIpoints, values, (Y,X), method='cubic', fill_value=0)
                IIimage += gd

        cv.imwrite(os.path.join('ImageLog','IIPlane_undist_'+str(M)+'.png'), (255*IIimage/IIimage.max()).astype('uint8'))

def image_undist(refocus=True):
    # image undistortion
    for depth in np.arange(initDepth, lastDepth+depthInterval, depthInterval).round(3):
        image_sensor_plane = cam.paraCheckerStack.imageStack[depth].image
        subimages, anchors = segmentImage(image_sensor_plane, cam.invM, cam.peaks)

        offset = np.array([200, 200]) # y,x
        IIimage = np.zeros(
            (
                int((cam.row_total-1+int(np.ceil(cam.optiSystem.M_MLA('real'))))*cam.optiSystem.p_MLA('pixel')),
                int((cam.col_total-1+int(np.ceil(cam.optiSystem.M_MLA('real'))))*cam.optiSystem.p_MLA('pixel')),
                ), dtype='float64'
            )

        alpha, beta = cam.cbParams.tilt
        RX = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
        RY = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
        II_norm_vec = np.matmul(np.matmul(RX, RY), np.array([[0],[0],[1]])).reshape(3)

        # refocus
        z_obj = (cam.optiSystem.M_MLA('real')+1)*cam.optiSystem.f_MLA('pixel')
        if refocus:
            zii1 = 1/(1/depth+1/20e-3)
            zii2 = 1/(1/cam.optiSystem.z_focus('m')+1/20e-3)
            z_obj += (zii2-zii1)/cam.optiSystem.pixel('m')
        z_img = (cam.optiSystem.M_MLA('real')+1)*cam.optiSystem.f_MLA('pixel')/cam.optiSystem.M_MLA('real')

        xp, yp, zp = cam.peaks[cam.row_center, cam.col_center, 0], cam.peaks[cam.row_center, cam.col_center, 1], z_obj+z_img
        ap, bp, cp = II_norm_vec # represent a plane: a(x-x0)+b(y-y0)+c(z-z0)=0
        for row in range(cam.row_total):
            for col in range(cam.col_total):
                IIpoints, values = [], []
                img = subimages[row][col]

                img = myUndistort(img, cam.MLA_distortion, anchors[row][col]['center']-anchors[row][col]['upperLeft'], method='cubic')

                center = np.array([cam.peaks[row,col,0], cam.peaks[row,col,1], z_img])
                for ix in range(img.shape[1]):
                    for iy in range(img.shape[0]):
                        p = np.array([iy+anchors[row][col]['upperLeft'][0], ix+anchors[row][col]['upperLeft'][1], 0])
                        dir_vec = center - p # direction vector (a, b, c); represent a line: (x-x0)/a = (y-y0)/b = (z-z0)/c
                        al, bl, cl = dir_vec
                        if not ((al==0) or (bl==0) or (cl==0)):
                            # solve intersection of line & plane
                            A = np.array([[ap, bp, cp], [1/al, -1/bl, 0], [0, 1/bl, -1/cl]])
                            B = np.array([[ap*xp+bp*yp+cp*zp], [p[0]/al-p[1]/bl], [p[1]/bl-p[2]/cl]])
                            xyz = np.matmul(np.linalg.inv(A), B).reshape(3)
                            IIpoints.append(xyz[:-1]+offset)
                            values.append(img[iy, ix])
                IIpoints = np.array(IIpoints)
                X, Y = np.meshgrid(np.arange(IIimage.shape[1]), np.arange(IIimage.shape[0]))
                gd = griddata(IIpoints, values, (Y,X), method='cubic', fill_value=0)
                IIimage += gd

        IIimage_u = myUndistort(IIimage, cam.main_lens_distortion, cam.peaks[cam.row_center, cam.col_center]+offset)
        filename = 'IIPlane_undist_'+str(depth).split('.')[-1]
        filename = filename + '_refocus.png' if refocus else filename + '.png'
        cv.imwrite(os.path.join('ImageLog',filename), (255*IIimage_u/IIimage_u.max()).astype('uint8'))

def image_refocus():
    # refocus example image in `cam`
    image_list = []
    for depth in np.arange(cam.optiSystem.z_focus('m')-25e-3, cam.optiSystem.z_focus('m')+26e-3, 1e-3).round(3):
        image_sensor_plane = cam.exampleImage.image
        subimages, anchors = segmentImage(image_sensor_plane, cam.invM, cam.peaks)

        offset = np.array([200, 200]) # y,x
        IIimage = np.zeros(
            (
                int((cam.row_total-1+int(np.ceil(cam.optiSystem.M_MLA('real'))))*cam.optiSystem.p_MLA('pixel')),
                int((cam.col_total-1+int(np.ceil(cam.optiSystem.M_MLA('real'))))*cam.optiSystem.p_MLA('pixel')),
                ), dtype='float64'
            )

        alpha, beta = cam.cbParams.tilt
        RX = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
        RY = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
        II_norm_vec = np.matmul(np.matmul(RX, RY), np.array([[0],[0],[1]])).reshape(3)

        # refocus
        zii1 = 1/(1/depth+1/20e-3)
        zii2 = 1/(1/cam.optiSystem.z_focus('m')+1/20e-3)
        z_obj = (cam.optiSystem.M_MLA('real')+1)*cam.optiSystem.f_MLA('pixel') + (zii2-zii1)/cam.optiSystem.pixel('m')
        z_img = (cam.optiSystem.M_MLA('real')+1)*cam.optiSystem.f_MLA('pixel')/cam.optiSystem.M_MLA('real')

        xp, yp, zp = cam.peaks[cam.row_center, cam.col_center, 0], cam.peaks[cam.row_center, cam.col_center, 1], z_obj+z_img
        ap, bp, cp = II_norm_vec # represent a plane: a(x-x0)+b(y-y0)+c(z-z0)=0
        for row in range(cam.row_total):
            for col in range(cam.col_total):
                IIpoints, values = [], []
                img = subimages[row][col]

                img = myUndistort(img, cam.MLA_distortion, anchors[row][col]['center']-anchors[row][col]['upperLeft'], method='cubic')

                center = np.array([cam.peaks[row,col,0], cam.peaks[row,col,1], z_img])
                for ix in range(img.shape[1]):
                    for iy in range(img.shape[0]):
                        p = np.array([iy+anchors[row][col]['upperLeft'][0], ix+anchors[row][col]['upperLeft'][1], 0])
                        dir_vec = center - p # direction vector (a, b, c); represent a line: (x-x0)/a = (y-y0)/b = (z-z0)/c
                        al, bl, cl = dir_vec
                        if not ((al==0) or (bl==0) or (cl==0)):
                            # solve intersection of line & plane
                            A = np.array([[ap, bp, cp], [1/al, -1/bl, 0], [0, 1/bl, -1/cl]])
                            B = np.array([[ap*xp+bp*yp+cp*zp], [p[0]/al-p[1]/bl], [p[1]/bl-p[2]/cl]])
                            xyz = np.matmul(np.linalg.inv(A), B).reshape(3)
                            IIpoints.append(xyz[:-1]+offset)
                            values.append(img[iy, ix])
                IIpoints = np.array(IIpoints)
                X, Y = np.meshgrid(np.arange(IIimage.shape[1]), np.arange(IIimage.shape[0]))
                gd = griddata(IIpoints, values, (Y,X), method='cubic', fill_value=0)
                IIimage += gd

        IIimage_u = myUndistort(IIimage, cam.main_lens_distortion, cam.peaks[cam.row_center, cam.col_center]+offset)
        filename = 'IIPlane_refocus_'+str(depth).split('.')[-1]+'.png'
        IIimage_u = 255*IIimage_u/IIimage_u.max()
        org, font, fontScale, color, thickness = (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2
        IIimage_u = cv.putText(IIimage_u, str(depth), org, font, fontScale, color, thickness, cv.LINE_AA)
        cv.imwrite(os.path.join('ImageLog',filename), IIimage_u.astype('uint8'))
        image_list.append(os.path.join('ImageLog',filename))
    re_image_list = image_list.copy()
    re_image_list.reverse()
    create_gif(image_list+re_image_list, os.path.join('ImageLog','refocus.gif'), 1/10)

def find_disparity():
    disparity_data = {i:{} for i in [1,2,4,5,8,9,10,13]}
    for depth in cam.featPoints:
        for fpName in cam.featPoints[depth]:
            rtm = cam.featPoints[depth][fpName]

            rawPoints = rtm.fp.rawPoints
            for ii in range(len(rawPoints)):
                p1, v1 = rawPoints[ii]
                for jj in range(ii+1, len(rawPoints)):
                    p2, v2 = rawPoints[jj]
                    value = int(np.sum((np.array(v1)-np.array(v2))**2))
                    if not depth in disparity_data[value]:
                        d_value = np.sqrt(np.sum((p1[:-1]-cam.peaks[v1[0],v1[1],:])**2)+np.sum((p2[:-1]-cam.peaks[v2[0],v2[1],:])**2))
                        disparity_data[value][depth] = [[p1, v1, p2, v2, d_value]]
                    else:
                        d_value = np.sqrt(np.sum((p1[:-1]-cam.peaks[v1[0],v1[1],:])**2)+np.sum((p2[:-1]-cam.peaks[v2[0],v2[1],:])**2))
                        disparity_data[value][depth].append([p1, v1, p2, v2, d_value])

    for value in disparity_data:
        exec("fig" + str(value) + " = plt.figure(dpi=200)")
        exec("ax = fig" + str(value) + ".add_subplot(1,1,1)")
        title_str = 'value = ' + str(value)
        exec("ax.set_title('" + title_str + "', fontsize=18)")
        xticks = [d for d in disparity_data[value]]
        try:
            h0 = np.array([v[4] for v in disparity_data[value][cam.optiSystem.z_focus('m')]]).mean()
            heights = [np.array([v[4] for v in disparity_data[value][d]]).mean()-h0 for d in disparity_data[value]]
            error = [np.array([v[4] for v in disparity_data[value][d]]).std() for d in disparity_data[value]]
            exec("ax.bar(xticks, heights, 0.015*0.618, yerr=error, align='center', capsize=10)")
            exec("ax.plot(xticks, heights, color='r')")
            for i in range(len(xticks)):
                d = xticks[i]
                h = float(np.array(heights[i]).round(2))
                exec("ax.text(d, h, str(h), fontsize=12)")
            exec("ax.set_xticks(np.arange(initDepth, lastDepth+depthInterval, depthInterval).round(3), np.arange(initDepth, lastDepth+depthInterval, depthInterval).round(3), rotation=45)")
            exec("ax.set_xlabel('depth / m', fontsize=16)")
            exec("ax.set_ylabel('mean disparity value / pixel', fontsize=16)")
            path = os.path.join('.', 'ImageLog', str(value).zfill(2)+'.jpg')
            exec("plt.savefig('" + path + "')")
        except KeyError: # feat points pairs not in all depth
            pass

def sensor_plane_images():
    for depth in cam.paraCheckerStack.imageStack:
        img = cam.paraCheckerStack.imageStack[depth].image
        cv.imwrite(os.path.join('ImageLog', str(depth)+'.jpg'), img.astype('uint8'))

def est_pose():
    # LOCAL Variable
    _, anchors = segmentImage(np.zeros(cam.calibImages[0].shape), cam.invM, cam.peaks)
    for depth in cam.featPoints:

        fp_this_depth_center_view = []
        rmin, rmax = ord('Z'), ord('A')
        cmin, cmax = ord('Z'), ord('A')

        for fpName in cam.featPoints[depth]:
            rtm = cam.featPoints[depth][fpName]

            rawPoints = rtm.fp.rawPoints
            for p, v in rawPoints:
                if v[0] == cam.row_center and v[1] == cam.col_center:
                    fp_this_depth_center_view.append(p[:2]-anchors[cam.row_center][cam.col_center]['upperLeft'])
                    row, col = fpName
                    row = ord(row)
                    col = ord(col)
                    if row < rmin:
                        rmin = row
                    if row > rmax:
                        rmax = row
                    if col < cmin:
                        cmin = col
                    if col > cmax:
                        cmax = col

        rs = rmax - rmin + 1
        cs = cmax - cmin + 1
        fps = np.array(fp_this_depth_center_view).reshape(rs, cs, 2)
        pose = cam.poseEstimate(fps)
        return pose

if __name__ == '__main__':
##    compare_3_sets_of_peaks()
##    present_II_plane_reprojection_and_number_vs_radius()
##    reproject_2_depth()
##    image_undist(refocus=False)
##    image_undist()
##    image_refocus()
    find_disparity()
##    sensor_plane_images()

##    pose = est_pose()
##    plt.figure('01')
##    plt.scatter(pose['world'][:,:,0], pose['world'][:,:,1])
##    plt.figure('12')
##    plt.scatter(pose['world'][:,:,1], pose['world'][:,:,2])
##    plt.figure('02')
##    plt.scatter(pose['world'][:,:,0], pose['world'][:,:,2])
##    plt.show()
