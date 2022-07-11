import numpy as np

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