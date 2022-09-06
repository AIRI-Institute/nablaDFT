# coding: utf-8

import math
import torch
import numpy as np
from scipy.special import binom

# spherical harmonics of order 0
def Y0(u):
    return u.new_ones((*u.shape[:-1],1))

# spherical harmonics of order 1
sqrt3 = np.sqrt(3)


def Y1(x, y, z):
    return sqrt3*torch.cat((y, z, x), dim=-1)

# spherical harmonics of order 2
sqrt15      = np.sqrt(15)
sqrt5over2  = np.sqrt(5)/2
sqrt15over2 = sqrt15/2


def Y2(xy, yz, xz, _3z2m1, _x2my2):
    return torch.cat((
        sqrt15      * xy,
        sqrt15      * yz,
        sqrt5over2  * _3z2m1,
        sqrt15      * xz,
        sqrt15over2 * _x2my2
    ), dim=-1)

# spherical harmonics of order 3
sqrt70over4  = np.sqrt(70)/4
sqrt105      = np.sqrt(105)
sqrt42over4  = np.sqrt(42)/4
sqrt7over2   = np.sqrt(7)/2
sqrt105over2 = sqrt105/2


def Y3(x, y, z, z2, xyz, _x2my2, _3x2my2, _x2m3y2):
    _5z2 = 5*z2
    _5z2m1 = (_5z2 - 1)
    return torch.cat((
        sqrt70over4  * y*_3x2my2,
        sqrt105      * xyz,
        sqrt42over4  * y*_5z2m1,
        sqrt7over2   * z*(_5z2 - 3),
        sqrt42over4  * x*_5z2m1,
        sqrt105over2 * z*_x2my2,
        sqrt70over4  * x*_x2m3y2
    ), dim=-1)

# spherical harmonics of order 4
sqrt35_3over2 = np.sqrt(35)*3/2
sqrt70_9over4 = np.sqrt(70)*9/4
sqrt45over2   = np.sqrt(45)/2
sqrt10_3over4 = np.sqrt(10)*3/4
oneover8      = 1/8
sqrt45over4   = sqrt45over2/2
sqrt70_3over4 = sqrt70_9over4/3
sqrt35_3over8 = sqrt35_3over2/4


def Y4(x2, y2, z2, xy, yz, xz, x4, y4, x2y2, _x2my2, _x2m3y2):
    _7z2 = 7*z2
    _7z2m1 = _7z2 - 1
    _7z2m3 = _7z2 - 3

    return torch.cat((
        sqrt35_3over2 * xy*_x2my2,
        sqrt70_9over4 * yz*(x2 - y2/3),
        sqrt45over2   * xy*_7z2m1,
        sqrt10_3over4 * yz*_7z2m3,
        oneover8      * (z2*(105*z2-90) + 9),
        sqrt10_3over4 * xz*_7z2m3,
        sqrt45over4   * _7z2m1*_x2my2 ,
        sqrt70_3over4 * xz*_x2m3y2,
        sqrt35_3over8 * (x4 - 6*x2y2 + y4)
    ), dim=-1)


# spherical harmonics of order 5
sqrt154_3over16 = np.sqrt(154)*3/16
sqrt385_3over2  = np.sqrt(385)*3/2
sqrt770over16   = np.sqrt(770)/16
sqrt1155over2   = np.sqrt(1155)/2
sqrt165over8    = np.sqrt(165)/8
sqrt11over8     = np.sqrt(11)/8
sqrt1155over4   = sqrt1155over2/2
sqrt385_3over8  = sqrt385_3over2/4


def Y5(x, y, z, z2, xyz, x4, y4, x2y2, _x2my2, _3z2m1, _3x2my2, _x2m3y2):
    z4 = z2*z2
    _9z2m1 = 9*z2 - 1
    _21z4m14z2p1 = 21*z4 - 14*z2 + 1
    return torch.cat((
        sqrt154_3over16 * y*(5*x4 - 10*x2y2 + y4),
        sqrt385_3over2 * xyz*_x2my2,
        sqrt770over16 * y*_3x2my2*_9z2m1,
        sqrt1155over2 * xyz*_3z2m1,
        sqrt165over8 * y*_21z4m14z2p1,
        sqrt11over8 * z*(63*z4 - 70*z2 + 15),
        sqrt165over8 * x*_21z4m14z2p1,
        sqrt1155over4 * z*_3z2m1*_x2my2,
        sqrt770over16 * x*_x2m3y2*_9z2m1,
        sqrt385_3over8 * z*(x4 - 6*x2y2 + y4), 
        sqrt154_3over16 * x*(x4 - 10*x2y2 + 5*y4)
    ), dim=-1)


# utility functions to generate higher order spherical harmonics
def _A(m, x, y):
    A = torch.zeros_like(x)
    for p in range(m+1):
        A += binom(m,p)* x**p * y**(m-p) * math.cos((m-p)*math.pi/2)
    return A


def _B(m, x, y):
    B = torch.zeros_like(x)
    for p in range(m+1):
        B += binom(m,p)* x**p * y**(m-p) * math.sin((m-p)*math.pi/2)
    return B


def _Pi(l, m, z):
    Pi = torch.zeros_like(z)
    for k in range((l-m)//2+1):
        Pi += ((-1)**k * 2**(-l) * binom(l,k) * binom(2*l-2*k,l) 
            * math.factorial(l-2*k)/math.factorial(l-2*k-m) 
            * z**(l-2*k-m))
    return math.sqrt(math.factorial(l-m)/math.factorial(l+m)) * Pi


# Herglotz generating function for Y(l,m)
def _Y(l, m, x, y, z):
    if m > 0:
        return math.sqrt(4*l+2)*_Pi(l,m,z)*_A(m,x,y)
    elif m < 0:
        return math.sqrt(4*l+2)*_Pi(l,-m,z)*_B(-m,x,y)
    else:
        return math.sqrt(2*l+1)*_Pi(l,m,z)


# spherical harmonics of order l (works for any order)
def Yl(l, x, y, z):
    Yl = []
    for m in range(-l,l+1):
        Yl.append(_Y(l,m,x,y,z))
    return torch.cat(Yl, dim=-1)
