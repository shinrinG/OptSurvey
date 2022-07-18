from skimage.io import imread
import matplotlib.pyplot as plt
from utils.func import get_psnr,show_dictionary,sparse_coding,recon_image
from sklearn.feature_extraction.image import extract_patches_2d
from proc.DictionaryLearning import DictionaryLearning
import numpy as np

OUTROOT="E:\\test\\SMPy552"
if __name__=="__main__":

    #Load
    sname = OUTROOT + "\\barbara.png"
    im = imread(sname)
    sname = OUTROOT + "\\barbara_sig10"
    Y = np.fromfile(sname).reshape(im.shape)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(im, cmap='gray', interpolation='Nearest', vmin=0, vmax=255)
    ax[1].imshow(Y, cmap='gray', interpolation='Nearest', vmin=0, vmax=255)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_title('ノイズなし')
    ax[1].set_title('ノイズあり\n{:.3f}'.format(get_psnr(im, Y)))
    sname = OUTROOT + "\\01_barbara_sig10.png"
    plt.savefig(sname, dpi=220)
    plt.close()
    #dictionary init
    patch_size = 8
    dict_size = 16
    A_1D = np.zeros((patch_size, dict_size))
    for k in np.arange(dict_size):
        for i in np.arange(patch_size):
            A_1D[i, k] = np.cos(i * k * np.pi / float(dict_size))
        if k != 0:
            A_1D[:, k] -= A_1D[:, k].mean()
    A_DCT = np.kron(A_1D, A_1D)
    sname = OUTROOT + "\\02_A_DCT"
    A_DCT.tofile(sname)
    A_DCT = np.fromfile(sname).reshape((patch_size ** 2, dict_size ** 2))
    sname = OUTROOT + "\\03_A_DCT.png"
    show_dictionary(A_DCT,name=sname)

    #dictionary learning
    dl = DictionaryLearning()
    patch_size = 8
    patches = extract_patches_2d(Y, (patch_size, patch_size)).reshape((-1, patch_size ** 2))
    M = len(patches)
    A_KSVD = A_DCT.copy()
    for _ in range(25):
        ndx = np.random.permutation(M)[:M // 10]
        A_KSVD, _ = dl.KSVD(patches[ndx].T, 20., 256, 4, n_iter=1, initial_dictionary=A_KSVD)
    sname = OUTROOT + "\\04_A_KSVD_sig10"
    A_KSVD.tofile(sname)
    A_KSVD = np.fromfile(sname).reshape((patch_size ** 2, dict_size ** 2))
    sname = OUTROOT + "\\05_A_KSVD_sig10.png"
    show_dictionary(A_KSVD, name=sname)

    #get TV-like activity
    activity = np.zeros_like(A_KSVD)
    for i, atom in enumerate(A_KSVD.T):
        atom = atom.reshape((patch_size, patch_size))
        act = np.abs(atom - np.roll(atom, 1, axis=1)).sum()
        act += np.abs(atom - np.roll(atom, 1, axis=0)).sum()
        activity[:, i] = act
    activity /= activity.max()    
    sname = OUTROOT + "\\06_activity.png"
    show_dictionary(activity, vmin=0, vmax=1, name=sname)
    t = 0.4
    sname = OUTROOT + "\\07_activity_th.png"
    show_dictionary((activity > t) * 0.4 + 0.5, vmin=0, vmax=1, name=sname)

    #sparse coding
    q = sparse_coding(Y, A_KSVD, 4, (patch_size * 10.) ** 2)

    #Reconstruction
    component = activity.mean(axis=0) <= t
    texture = activity.mean(axis=0) > t
    Y_c_local = recon_image(im, q[:, component], A_KSVD[:, component], lam=0)
    Y_t_local = recon_image(im, q[:, texture], A_KSVD[:, texture], lam=0)
    sname = OUTROOT + "\\08_Y_c_local"
    Y_c_local.tofile(sname)
    Y_c_local = np.fromfile(sname).reshape((512, 512))
    sname = OUTROOT + "\\09_Y_t_local"
    Y_t_local.tofile(sname)
    Y_t_local = np.fromfile(sname).reshape((512, 512))

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(Y_c_local, cmap='gray', interpolation='Nearest')
    ax[1].imshow(Y_t_local, cmap='gray', interpolation='Nearest')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_title('線画')
    ax[1].set_title('テクスチャ')
    sname = OUTROOT + "\\10_local_mca.png"
    plt.savefig(sname, dpi=440)



