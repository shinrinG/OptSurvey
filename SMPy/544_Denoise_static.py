from utils.func import show_dictionary,get_psnr
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means
from bm3d import bm3d
OUTROOT="E:\\test\\SMPy544"

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

    #nlmeans
    recon_nlm = denoise_nl_means(Y, h=20., multichannel=False)    
    sname = OUTROOT + "\\11_recon_nlm"
    recon_nlm.tofile(sname)
    recon_nlm = np.fromfile(sname, dtype=np.float64).reshape(im.shape)
    plt.close()
    plt.imshow(recon_nlm, cmap='gray', interpolation='Nearest', vmin=0, vmax=255)
    plt.axis('off')
    plt.colorbar()
    plt.title('NL-means\n{:.3f}'.format(get_psnr(im, recon_nlm)))
    plt.tight_layout()
    sname = OUTROOT + "\\12_recon_nlm.png"
    plt.savefig(sname, dpi=220)
    plt.close()


    #BM3D
    recon_bm3d = np.array(bm3d(Y[:, :, np.newaxis].astype(np.float32), 20.)).reshape(Y.shape)
    sname = OUTROOT + "\\21_recon_bm3d"
    recon_bm3d.tofile(sname)
    recon_bm3d = np.fromfile(sname, dtype=np.float64).reshape(im.shape)
    plt.close()
    plt.imshow(recon_bm3d, cmap='gray', interpolation='Nearest', vmin=0, vmax=255)
    plt.axis('off')
    plt.colorbar()
    plt.title('BM3D\n{:.3f}'.format(get_psnr(im, recon_bm3d)))
    plt.tight_layout()
    sname = OUTROOT + "\\22_recon_bm3d.png"
    plt.savefig(sname, dpi=220)
