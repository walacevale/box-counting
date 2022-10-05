import numpy as np
import matplotlib.pyplot as plt
import cv2  # sudo apt install python3-opencv
import os
import glob  # sudo pip3 install glob3
from numba import jit
import time
from skimage.filters import threshold_otsu
from scipy.stats import chisquare


def th(image):
    ret, thresh1 = cv2.threshold(image, 1, image.max(), cv2.THRESH_BINARY)

    return thresh1

#-----------------------------------------------------------------------------------------------------#


def gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray


def th_otsu(image):
    thresh1 = threshold_otsu(image)
    binary = image > thresh1
    thresh1 = (np.array(binary, dtype=int))*254

    return thresh1

#-----------------------------------------------------------------------------------------------------#


def Fractal(imagem):

    pixels = []
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            if imagem[i, j] > 0:
                pixels.append((i, j))



    Lx = imagem.shape[1]
    Ly = imagem.shape[0]

    pixels = np.array(pixels)

    # computing the fractal dimension
    # considering only scales in a logarithmic list
    #scales = np.logspace(0.01, , num=100, endpoint=False, base=2)
    scales = [8,12,16,32,64]
    Ns = []
    # looping over several scales
    for scale in scales:
        #print ("======= Scale :",scale)
        # computing the histogram
        H, edges = np.histogramdd(pixels, bins=(np.arange(0, Lx, scale), np.arange(0, Ly, scale)))
        Ns.append(np.sum(H > 0))

    coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)
    plt.plot(np.log(scales),np.log(Ns), 'o', mfc='none')
    plt.plot(np.log(scales), np.polyval(coeffs,np.log(scales)))
    plt.xlabel('log $\epsilon$')
    plt.ylabel('log N')
    plt.savefig('sierpinski_dimension.pdf')

    F = -coeffs[0]
    return F
#-----------------------------------------------------------------------------------------------------#
