# coding: utf-8

import torch
import numpy as np
from .spherical_harmonics_any_order import Y0, Y1, Y2, Y3, Y4, Y5, Yl


"""
This returns a list of all spherical harmonics (up to order L)
derived from the input unit vectors u. For up to L=5, this 
function is optimized to reduce the number of floating point
operations. When L>5 is requested, a general formula is used,
which is not as efficient.
NOTE: All spherical harmonics lack the constant 1/sqrt(4*Pi) 
for efficiency and work only for unit vectors (to
remove unnecessary terms involving radius r)
The m values are stored from -L (index 0), -L+1 (index 1), ..., L
Condon-Shortley phase is included!

input:
    L: integer that specifies order (0:s, 1:p, 2:d, 3:f, 4:g, 5:h, ...)
    u: Cartesian unit vectors of shape [...,3] (last dimension must be 3, the rest of the shape is arbitrary)
output:
    Y: list of length L+1 containing the spherical harmonics of shape [...,2*L+1]
"""


def spherical_harmonics(L, u):
    Y = []

    if L >= 0:
        Y.append(Y0(u))
    if L >= 1:
        shape = (*u.shape[:-1], 1)
        x = torch.gather(u, -1, u.new_full(shape, 0, dtype=torch.long))
        y = torch.gather(u, -1, u.new_full(shape, 1, dtype=torch.long))
        z = torch.gather(u, -1, u.new_full(shape, 2, dtype=torch.long))
        Y.append(Y1(x, y, z))
    if L >= 2:
        x2 = x * x
        y2 = y * y
        z2 = z * z
        xy = x * y
        yz = y * z
        xz = x * z
        _x2my2 = x2 - y2
        _3z2m1 = 3 * z2 - 1
        Y.append(Y2(xy, yz, xz, _3z2m1, _x2my2))
    if L >= 3:
        xyz = xy * z
        _3x2my2 = 3 * x2 - y2
        _x2m3y2 = x2 - 3 * y2
        Y.append(Y3(x, y, z, z2, xyz, _x2my2, _3x2my2, _x2m3y2))
    if L >= 4:
        x4 = x2 * x2
        y4 = y2 * y2
        x2y2 = x2 * y2
        Y.append(Y4(x2, y2, z2, xy, yz, xz, x4, y4, x2y2, _x2my2, _x2m3y2))
    if L >= 5:
        Y.append(Y5(x, y, z, z2, xyz, x4, y4, x2y2, _x2my2, _3z2m1, _3x2my2, _x2m3y2))
    if L >= 6:
        for l in range(6, L + 1):
            Y.append(Yl(l, x, y, z))
    return Y
