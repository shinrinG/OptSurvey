import numpy as np
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio
from scipy.signal import convolve2d
import matplotlib.pyplot as plt 
import numpy as np
from utils.func import get_blur_kernel_norm,wavelet2
from proc.IterativeShrinkage import IterativeShrinkage

OUTROOT="D:\\test\\SMPy500"

if __name__=="__main__":
    #amimage
    im = camera().astype(np.float)[::2, ::2]
    sname = OUTROOT + "\\00_origin.png"
    plt.imsave(sname,im)

    #sample of wavelet
    coeffs = wavelet2(im)
    titles = ['$LL_{2}$', '$LH_{2}$', '$HL_{2}$', '$HH_{2}$', '$LH_{1}$', '$HL_{1}$', '$HH_{1}$']
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    ax = ax.flatten()
    for i, (coef, title) in enumerate(zip(coeffs, titles)):
        ax[i].imshow(coef, cmap='gray', interpolation='Nearest')
        ax[i].axis('off')
        ax[i].set_title(title)
    ax[-1].axis('off')
    sname = OUTROOT + "\\01_wavelet.png"
    plt.savefig(sname, dpi=220)

    #sample of blur
    kernel = get_blur_kernel_norm()
    y_obs = convolve2d(im, kernel, mode='same', boundary='symm') + np.random.randn(256, 256) * np.sqrt(2.)
    sname = OUTROOT + "\\02_observed.png"    
    plt.imsave(sname,y_obs)

    #SSF
    isc1 = IterativeShrinkage(y_obs,0.01, 0.075)
    x1, fs1, errs1 = isc1.SSF(niter=100, x0=im)
    isc2 = IterativeShrinkage(y_obs, 0.01, 0.075)
    x2, fs2, errs2 = isc2.SSF(niter=60, ls=True, x0=im)
    isc3 = IterativeShrinkage(y_obs, 0.01, 0.075)
    x3, fs3, errs3 = isc3.SSF(niter=25, sesop=True, x0=im)

    #Results Show
    y_pred1 = np.sum(x1, axis=0)
    y_pred2 = np.sum(x2, axis=0)
    y_pred3 = np.sum(x3, axis=0)

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax = ax.flatten()
    ax[0].imshow(im, cmap='gray', interpolation='Nearest')
    ax[1].imshow(y_obs, cmap='gray', interpolation='Nearest')
    ax[2].imshow(y_pred1, cmap='gray', interpolation='Nearest')
    ax[3].imshow(y_pred2, cmap='gray', interpolation='Nearest')
    ax[4].imshow(y_pred3, cmap='gray', interpolation='Nearest')
    ax[5].axis('off')

    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[3].axis('off')
    ax[4].axis('off')
    ax[0].set_title('原画像')
    ax[1].set_title('ボケ画像 {:.3f}'.format(peak_signal_noise_ratio(im / 255, y_obs / 255)))
    ax[2].set_title('SSF（100反復）{:.3f}'.format(peak_signal_noise_ratio(im / 255, y_pred1 / 255)))
    ax[3].set_title('SSF-LS（60反復）{:.3f}'.format(peak_signal_noise_ratio(im / 255, y_pred2 / 255)))
    ax[4].set_title('SSF-SESOP-5（25反復）{:.3f}'.format(peak_signal_noise_ratio(im / 255, y_pred3 / 255)))
    sname = OUTROOT + "\\03_predicted.png"    
    plt.savefig(sname, dpi=220)






