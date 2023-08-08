#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
import colour
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider

def hsv2str(h,s,v):
    rgb = colour.HSV_to_RGB([h,s,v])
    rgb *= 255
    RGB = [hex(int(i))[2:] for i in rgb]
    RGB = [i.zfill(2) for i in RGB]
    return '#' + RGB[0] + RGB[1] + RGB[2]

zobj = 200
fobj = 20
fmla = 4
pmla = 1
dmla = 1
dep = 5
zap = 1
m = 3

NUMBER_OF_VIEWS = 7

layout = plt.figure(dpi=200)
spec = layout.add_gridspec(24, 24)
layout_ax = layout.add_subplot(spec[:20, :])

axzobj = layout.add_subplot(spec[-4, :])
axdep = layout.add_subplot(spec[-3, :])
axzap = layout.add_subplot(spec[-2, :])
axm = layout.add_subplot(spec[-1, :])

slider_zobj = Slider(axzobj, 'zobj', 50, 500, valinit=zobj, valstep=1)
slider_dep = Slider(axdep, 'Dep', 1, 30, valinit=dep, valstep=0.01)
slider_zap = Slider(axzap, 'Zap', 0, 30, valinit=zap, valstep=1)
slider_m = Slider(axm, 'M', 2, 4, valinit=m, valstep=0.01)

def update(val):
    zobj = slider_zobj.val
    dep = slider_dep.val
    zap = slider_zap.val
    m = slider_m.val

    layout_ax.clear()
    calc_and_draw(zobj, dep, zap, m)

    layout.canvas.draw_idle()

slider_zobj.on_changed(update)
slider_dep.on_changed(update)
slider_zap.on_changed(update)
slider_m.on_changed(update)

def calc_and_draw(zobj, dep, zap, m, fobj=20, fmla=4, pmla=1, dmla=1):
    zii = 1/(1/fobj-1/zobj)
    zvc = -zap + 1/(1/fobj-1/(zii+(m+1)*fmla))

    mvc = (zii+(m+1)*fmla)/(zvc+zap)
    dvc = dmla/mvc
    pvc = pmla/mvc

    # use on-axis virtual camera as origin
    # calculate coordinates in mm, then convert to pixel
    vc_centers = []
    mla_centers = []
    y1s, ycs, y2s = [], [], []
    t1s, tcs, t2s = [], [], []
    t1ps, tcps, t2ps = [], [], []
    ei_centers, ei_tops, ei_bots = [], [], []

    ep_inner = [(zvc, dep/2), (zvc, -dep/2)]
    ep_outer = [(zvc, dep), (zvc, -dep)]
    ep_width = 1
    lens = [(zvc+zap, 3*dep), (zvc+zap, -3*dep)]

    for i in range(NUMBER_OF_VIEWS):
        n = i - NUMBER_OF_VIEWS//2

        vc_centers.append((0, -n*pvc))
        mla_centers.append((zvc+zap+zii+(m+1)*fmla, n*pmla))

        t1 = np.arctan((dep/2+n*pvc)/zvc)
        t2 = np.arctan((-dep/2+n*pvc)/zvc)
        tc = np.arctan(n*pvc/zvc)
        t1s.append(t1)
        t2s.append(t2)
        tcs.append(tc)

        y1 = np.tan(t1)*(zvc+zap)-n*pvc
        y2 = np.tan(t2)*(zvc+zap)-n*pvc
        yc = np.tan(tc)*(zvc+zap)-n*pvc
        y1s.append(y1)
        y2s.append(y2)
        ycs.append(yc)

        # t1p = t1 - y1/fobj
        # t2p = t2 - y2/fobj
        # tcp = tc - yc/fobj
        t1p = np.arctan((n*pmla-y1)/(zii+(m+1)*fmla))
        t2p = np.arctan((n*pmla-y2)/(zii+(m+1)*fmla))
        tcp = np.arctan((n*pmla-yc)/(zii+(m+1)*fmla))
        t1ps.append(t1p)
        t2ps.append(t2p)
        tcps.append(tcp)

        ei_centers.append((zvc+zap+zii+(m+1)*fmla*(1+1/m), yc+np.tan(tcp)*(zii+(m+1)*fmla*(1+1/m))))
        ei_tops.append((zvc+zap+zii+(m+1)*fmla*(1+1/m), y2+np.tan(t2p)*(zii+(m+1)*fmla*(1+1/m))))
        ei_bots.append((zvc+zap+zii+(m+1)*fmla*(1+1/m), y1+np.tan(t1p)*(zii+(m+1)*fmla*(1+1/m))))

    # optical axis
    layout_ax.plot([-(zobj-zvc-zap), zvc+zap+zii+(m+1)*fmla*(1+1/m)], [0, 0], color='k', linestyle='-.')
    # entrance pupil
    layout_ax.plot([ep_inner[0][0], ep_outer[0][0]], [ep_inner[0][1], ep_outer[0][1]], color='k')
    layout_ax.plot([ep_inner[1][0], ep_outer[1][0]], [ep_inner[1][1], ep_outer[1][1]], color='k')
    layout_ax.plot([ep_inner[0][0]-ep_width, ep_inner[0][0]+ep_width], [ep_inner[0][1], ep_inner[0][1]], color='k')
    layout_ax.plot([ep_inner[1][0]-ep_width, ep_inner[1][0]+ep_width], [ep_inner[1][1], ep_inner[1][1]], color='k')
    # obj lens
    layout_ax.plot([lens[0][0], lens[1][0]], [lens[0][1], lens[1][1]], color='k')

    for i in range(NUMBER_OF_VIEWS):
        n = i - NUMBER_OF_VIEWS//2
        c = hsv2str(i/NUMBER_OF_VIEWS,1,1)

        layout_ax.plot([-(zobj-zvc-zap), zvc+zap], [-n*pvc-np.tan(t1s[i])*(zobj-zvc-zap), y1s[i]], color=c, linestyle=':', linewidth=1)
        layout_ax.plot([-(zobj-zvc-zap), zvc+zap], [-n*pvc-np.tan(t2s[i])*(zobj-zvc-zap), y2s[i]], color=c, linestyle='--', linewidth=1)
        layout_ax.plot([-(zobj-zvc-zap), zvc+zap], [-n*pvc-np.tan(tcs[i])*(zobj-zvc-zap), ycs[i]], color=c, linewidth=1)

        layout_ax.plot([zvc+zap, zvc+zap+zii+(m+1)*fmla], [y1s[i], n*pmla], color=c, linestyle=':', linewidth=1)
        layout_ax.plot([zvc+zap, zvc+zap+zii+(m+1)*fmla], [y2s[i], n*pmla], color=c, linestyle='--', linewidth=1)
        layout_ax.plot([zvc+zap, zvc+zap+zii+(m+1)*fmla], [ycs[i], n*pmla], color=c, linewidth=1)

        ellipse = Ellipse(xy=vc_centers[i], width=pvc/5, height=pvc, edgecolor=c, fc='None', lw=1)
        layout_ax.add_patch(ellipse)

        ellipse = Ellipse(xy=mla_centers[i], width=pmla/5, height=pmla, edgecolor=c, fc='None', lw=1)
        layout_ax.add_patch(ellipse)

        layout_ax.plot([zvc+zap+zii+(m+1)*fmla, zvc+zap+zii+(m+1)*fmla*(1+1/m)], [n*pmla, ei_bots[i][1]], color=c, linestyle=':', linewidth=1)
        layout_ax.plot([zvc+zap+zii+(m+1)*fmla, zvc+zap+zii+(m+1)*fmla*(1+1/m)], [n*pmla, ei_tops[i][1]], color=c, linestyle='--', linewidth=1)
        layout_ax.plot([zvc+zap+zii+(m+1)*fmla, zvc+zap+zii+(m+1)*fmla*(1+1/m)], [n*pmla, ei_centers[i][1]], color=c, linewidth=1)

        layout_ax.plot([ei_bots[i][0], ei_tops[i][0]], [ei_bots[i][1], ei_tops[i][1]], color=c, linewidth=1)

calc_and_draw(zobj, dep, zap, m)
plt.show()
