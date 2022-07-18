from utils.func import show_dictionary,get_psnr,denoise_with_learned_dictionary
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

OUTROOT="E:\\test\\SMPy542"

if __name__=="__main__":
    rname = OUTROOT + "\\barbara.png"
    im = imread(rname).astype(np.float)
    rname = OUTROOT + "\\barbara_sig20"
    Y = np.fromfile(rname).reshape(im.shape)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(im, cmap='gray', interpolation='Nearest', vmin=0, vmax=255)
    ax[1].imshow(Y, cmap='gray', interpolation='Nearest', vmin=0, vmax=255)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_title('ノイズなし')
    ax[1].set_title('ノイズあり\n{:.3f}'.format(get_psnr(im, Y)))
    sname = OUTROOT + "\\00_barbara.png"
    plt.savefig(sname,dpi=220)
    plt.close()

    #create dct dict
    patch_size = 8
    dict_size = 16
    A_1D = np.zeros((patch_size, dict_size))
    for k in np.arange(dict_size):
        for i in np.arange(patch_size):
            A_1D[i, k] = np.cos(i * k * np.pi / float(dict_size))
        if k != 0:
            A_1D[:, k] -= A_1D[:, k].mean()

    A_DCT = np.kron(A_1D, A_1D)
    sname = OUTROOT + "\\01_DCT_DICT.png"
    show_dictionary(A_DCT,name=sname)
    sname = OUTROOT + "\\02_A_DCT"
    A_DCT.tofile(sname)

    eps = (patch_size ** 2) * (20. ** 2) * 1.15
    recon_dct_dictionary = denoise_with_learned_dictionary(Y, A_DCT, 4, eps, im0=im)
    sname = OUTROOT + "\\03_recon_dct_dictionary"
    recon_dct_dictionary.tofile(sname)
    recon_dct_dictionary = np.fromfile(sname).reshape(im.shape)
    plt.close()
    plt.imshow(recon_dct_dictionary, cmap='gray', interpolation='Nearest')
    plt.axis('off')
    plt.title('冗長DCT辞書によるOMPノイズ除去\n{:.3f}'.format(get_psnr(im, recon_dct_dictionary)))
    plt.colorbar()
    plt.tight_layout()
    sname = OUTROOT + "\\04_recon_dct_dictionary.png"
    plt.savefig(sname, dpi=220)





