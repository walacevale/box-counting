import matplotlib.pyplot as plt
import cv2
import os

from numpy import zeros
from Func import *

# abrir imagem direto da pasta Dados
img = os.path.join('Dados', '3.png')

# transforma a imagem em escala de cinza e dps em preto e branco
# utilizei a biblioteca cv2 para a imagem abrir em uma escala de 0 a 255.
# pelo matplot a imagem fica em uma escala de 0 a 1
img_th = th(gray(cv2.imread(img)))/255
print(np.count_nonzero(img_th[:512,512:]))
T = 0

width = np.shape(img_th)[1]
p = np.log(width)/np.log(2)

p = int(np.ceil(p))
width = 2**p
n = np.zeros(p+1)

for g in range(p-1, 0, -1):
    siz = 2**(p-g)
    siz2 = int(np.round(siz/2))
    for i in range(1, (width - siz ), siz):
        
        for j in range(1, (width - siz ), siz):
            #img[i,j] = (img[i,j] or img[i+siz2,j] or img[i,j+siz2] or img[i+siz2,j+siz2] )
            #img[i,j] = np.sum(img[1: , :])
            #print(str(siz) + str((width-siz+1)))
            n[g] = np.sum( img_th[1:(width-siz+1),1:(width-siz+1)])
    
r = 2**(np.arange(0,p+1))
print(r)
plt.plot(np.log(n),np.log(r), 'o', mfc='none')

#coeffs = np.polyfit(np.log(n), np.log(r), 1)
#plt.plot(np.log(n), np.polyval(coeffs,np.log(r)))
#print(coeffs[0])
plt.show()
