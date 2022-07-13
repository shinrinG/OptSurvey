import numpy as np
import matplotlib.pyplot as plt 
from scipy.fftpack import dct, idct
import pywt

#Get Squared Error
def get_se(x, x2):
    return np.sum((x - x2) ** 2)

#WaveletH
def horizontal(im):
    im2 = np.roll(im, 1, axis=1)
    return (im + im2) / 2., (im - im2) / 2.
#WaveletV
def vertical(im):
    im2 = np.roll(im, 1, axis=0)
    return (im + im2) / 2., (im - im2) / 2.
#Wavelet Lv1
def wavelet(im):
    l, h = horizontal(im)
    hl, hh = vertical(h)
    ll, lh = vertical(l)
    return ll, lh, hl, hh
#Wavelet Lv2
def wavelet2(im):
    ll, lh, hl, hh = wavelet(im)
    ll2, lh2, hl2, hh2 = wavelet(ll)
    return np.array([ll2, lh2, hl2, hh2, lh, hl, hh])

#Get Kernel for blur
def get_blur_kernel():
    x = np.arange(-7, 8)
    y = np.arange(-7, 8)
    xx, yy = np.meshgrid(x, y)
    return 1. / (xx ** 2 + yy ** 2 + 1)

#Get Normalized Kernel for blur
def get_blur_kernel_norm():
    h = get_blur_kernel()
    h /=h.sum()
    return h

#Show Dict
def show_dictionary(A, name=None):
    n = int(np.sqrt(A.shape[0]))
    m = int(np.sqrt(A.shape[1]))
    A_show = A.reshape((n, n, m, m))
    fig, ax = plt.subplots(m, m, figsize=(4, 4))
    for row in range(m):
        for col in range(m):
            ax[row, col].imshow(A_show[:, :, col, row], cmap='gray', interpolation='Nearest')
            ax[row, col].axis('off')
    if name is not None:
        plt.savefig(name, dpi=220)

#get psnr
def get_psnr(im, recon):
    return 10. * np.log(im.max() / np.sqrt(np.mean((im - recon) ** 2)))

#get 2D dct
def get_2D_dct(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

#get 2D idct
def get_2d_idct(coeffs):
    return idct(idct(coeffs.T, norm='ortho').T, norm='ortho')
