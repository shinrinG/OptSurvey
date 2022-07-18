from skimage.io import imread
import matplotlib.pyplot as plt
from utils.func import global_mca
import numpy as np

OUTROOT="E:\\test\\SMPy551"
if __name__=="__main__":
    sname = OUTROOT + "\\barbara.png"
    im = imread(sname)[::2, ::2]

    Y_c_global_mca, Y_t_global_mca = global_mca(im)
    sname = OUTROOT + "\\01_Y_c_global_mca"
    Y_c_global_mca.tofile(sname)
    Y_c_global_mca = np.fromfile(sname).reshape(im.shape)
    sname = OUTROOT + "\\02_Y_t_global_mca"
    Y_t_global_mca.tofile(sname)
    Y_t_global_mca = np.fromfile(sname).reshape(im.shape)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(Y_c_global_mca, cmap='gray', interpolation='Nearest')
    ax[1].imshow(Y_t_global_mca, cmap='gray', interpolation='Nearest')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_title('content')
    ax[1].set_title('texture')
    sname = OUTROOT + "\\03_global_mca.png"
    plt.savefig(sname, dpi=440)