import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt 
import numpy as np
from utils.func import show_dictionary
from proc.DictionaryLearning import DictionaryLearning

OUTROOT="D:\\test\\SMPy520"

if __name__=="__main__":
    # create initial dictionary
    A0 = np.random.randn(30, 60)
    A0 = np.dot(A0, np.diag(1. / np.sqrt(np.diag(np.dot(A0.T, A0)))))
    sname = OUTROOT + "\\00_dict_init.png"
    plt.imsave(sname,A0)

    #generate signal
    Y = np.zeros((30, 4000))
    sig = 0.1
    k0 = 4
    for i in range(4000):
        Y[:, i] = np.dot(A0[:, np.random.permutation(range(60))[:k0]], np.random.randn(4)) + np.random.randn(30) * sig
    plt.bar(range(30), Y[:, 0])
    plt.xlim(0, 29)
    sname = OUTROOT + "\\01_signal.png"
    plt.savefig(sname, dpi=220)

    #d-learning
    dl = DictionaryLearning()
    A_MOD, log_MOD = dl.MOD(Y, sig, A0.shape[1], k0, A0=A0, n_iter=50)
    A_KSVD, log_KSVD = dl.KSVD(Y, sig, A0.shape[1], k0, A0=A0, n_iter=50)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(log_MOD[:, 0], label='MOD')
    ax[0].plot(log_KSVD[:, 0], ls='--', label='K-SVD')
    ax[0].set_ylabel('平均表現誤差')
    ax[0].set_xlabel('反復回数')
    ax[0].legend(loc='best')
    ax[0].grid()
    ax[1].plot(log_MOD[:, 1], label='MOD')
    ax[1].plot(log_KSVD[:, 1], ls='--', label='K-SVD')
    ax[1].set_ylabel('正しく復元されたアトムの割合')
    ax[1].set_xlabel('反復回数')
    ax[1].legend(loc='best')
    ax[1].grid()
    sname =  OUTROOT + "\\02_LearningCurve.png"
    plt.savefig(sname, dpi=220)

    #image test
    rname = OUTROOT + "\\barbara.png"
    im = imread(rname).astype(np.float)

    #get patch
    patch_size = 8
    patches = []
    for row in range(im.shape[0] - patch_size + 1):
        for col in range(im.shape[1] - patch_size + 1):
            patches.append(im[row:row + patch_size, col:col + patch_size])
    patches = np.array(patches)
    n = 8
    sample = patches[np.random.permutation(patches.shape[0])[:n ** 2]]
    sample = sample.reshape((n, n, patch_size, patch_size))
    patch_work = np.swapaxes(sample, 1, 2).reshape((n * patch_size, n * patch_size))
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(im, cmap='gray', interpolation='Nearest')
    ax[0].axis('off')
    ax[0].set_title('辞書学習を評価する原画像Barbara')
    ax[1].imshow(patch_work, cmap='gray', interpolation='Nearest')
    ax[1].axis('off')
    ax[1].set_title('抽出したパッチの一部')
    plt.tight_layout()
    sname = OUTROOT + "\\11_barbara_patches.png"
    plt.savefig(sname, dpi=220)

    #create 2DDCT-Dictionary
    A_1D = np.zeros((8, 11))
    for k in np.arange(11):
        for i in np.arange(8):
            A_1D[i, k] = np.cos(i * k * np.pi / 11.)
        if k != 0:
            A_1D[:, k] -= A_1D[:, k].mean()
    A_2D = np.kron(A_1D, A_1D)
    sname = OUTROOT + "\\12_dct_dict_init.png"
    show_dictionary(A_2D,name=sname)

    #d-learning
    Y = patches[::10].reshape((-1, 64)).swapaxes(0, 1)
    sig = 0
    k0 = 4
    dl = DictionaryLearning()
    A_KSVD_barbara, log_KSVD_barbara = dl.KSVD(Y, sig, A_2D.shape[1], k0, n_iter=50, initial_dictionary=A_2D.copy())
    sname = OUTROOT + "\\13_learned_dict.png"
    show_dictionary(A_KSVD_barbara, name=sname)

    #show down
    plt.plot(log_KSVD_barbara, label='K-SVD')
    plt.ylabel('平均表現誤差')
    plt.xlabel('反復回数')
    plt.legend(loc='best')
    plt.grid()
    sname = OUTROOT + "\\14_learning_curve.png"
    plt.savefig(sname, dpi=220)











