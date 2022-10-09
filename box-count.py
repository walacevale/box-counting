import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from Func import th, gray
##____________________________________________________#

imagem = os.path.join('Dados', '3.png')
img_th = th(gray(cv2.imread(imagem)))

def fractal(img):
    log_eta = []
    log_n_box = []
    screen_size = np.shape(img)[0]
    scale = int(np.log(screen_size)/np.log(2))

    for p in range(scale):
        scale = int(2**p)
        num_box = 0
        y = 0
        num_steps = 0
        while y <= screen_size:
            x = 0
            while x <= screen_size:
                if np.count_nonzero(img[x:x+scale, y:y+scale]) > 0:
                    num_box += 1

                x += scale
            y += scale
        num_steps += 1
        log_eta.append(np.log(scale))
        log_n_box.append(np.log(1/num_box))

    coeffs = np.polyfit(log_eta, log_n_box, 1)

    return coeffs[0]

print(fractal(img_th))
