import matplotlib.pyplot as plt
import cv2
import os
#from scipy.signal import convolve2d
from tqdm import tqdm
from Func import *


# abrir imagem direto da pasta Dados
img = os.path.join('Dados', '3.png')
# transforma a imagem em escala de cinza e dps em preto e branco
# utilizei a biblioteca cv2 para a imagem abrir em uma escala de 0 a 255.
# pelo matplot a imagem fica em uma escala de 0 a 1
img_th = th(gray(cv2.imread(img)))

LogEta = []
LogNBox = []
ScreenSize = np.shape(img_th)[0]
Side = int(np.log(ScreenSize)/np.log(2))

NumSteps = 0

for p in range(Side):
    Side = int(2**p)

    NumBox = 0
    y = 0
    while y <= ScreenSize:
        x = 0
        while x <= ScreenSize:
            if np.count_nonzero(img_th[x:x+Side, y:y+Side]) > 0:
                NumBox += 1

            x += Side
        y += Side
    NumSteps += 1
    LogEta.append(np.log(Side))
    LogNBox.append(np.log(NumBox))


coeffs = np.polyfit(LogEta, LogNBox, 1)

plt.plot(LogEta, LogNBox, 'o', mfc='none')
plt.plot(LogEta, np.polyval(coeffs, LogEta),  label=str(coeffs))
plt.xlabel('log $\epsilon$')
plt.ylabel('log N')
plt.legend()
plt.show()
