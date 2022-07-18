import numpy as np
import matplotlib.pyplot as plt 
from scipy.fftpack import dct, idct
import pywt
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from utils.omp import OMP

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
def show_dictionary(A, name=None, figsize=(4, 4), vmin=None, vmax=None):
    n = int(np.sqrt(A.shape[0]))
    m = int(np.sqrt(A.shape[1]))
    A_show = A.reshape((n, n, m, m))
    fig, ax = plt.subplots(m, m, figsize=figsize)
    for row in range(m):
        for col in range(m):
            ax[row, col].imshow(A_show[:, :, col, row], cmap='gray', interpolation='Nearest', vmin=vmin, vmax=vmax)
            ax[row, col].axis('off')
    if name is not None:
        plt.savefig(name, dpi=220)
        plt.close()

#get psnr
def get_psnr(im, recon):
    return 10. * np.log(im.max() / np.sqrt(np.mean((im - recon) ** 2)))

#get 2D dct
def get_2D_dct(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

#get 2D idct
def get_2d_idct(coeffs):
    return idct(idct(coeffs.T, norm='ortho').T, norm='ortho')

#Shrinkage Function by wavelet
def wavelet_shrinkage(im, t, level, hard=True):
    Ds = []
    L = im.copy()
    for _ in range(level):
        L, D = pywt.dwt2(L, 'haar')
        if hard:
            D = np.where(np.abs(D) > t, D, 0)
        else:
            D = np.where(np.abs(D) > t, np.sign(D) * (np.abs(D) - t), 0)
        Ds.append(D)
    for l in range(level):
        L = pywt.idwt2((L, Ds[-(l + 1)]), 'haar')
    return L

#Shrinkage Function by dct
def dct_shrinkage(im, t, patch_size=8, hard=True):
    patches = extract_patches_2d(im, (patch_size, patch_size))
    coeffs = get_2D_dct(patches)
    if hard:
        coeffs = np.where(np.abs(coeffs) > t, coeffs, 0)
    else:
        coeffs = np.where(np.abs(coeffs) > t, np.sign(coeffs) * (np.abs(coeffs) - t), 0)
    patches = get_2d_idct(coeffs)        
    return reconstruct_from_patches_2d(patches, im.shape)

#get shrinkage using dct
def get_shrinkage_curve(target, noisy, J=6, patch_size=6):
    target_patches = extract_patches_2d(target, (patch_size, patch_size))
    noisy_patches = extract_patches_2d(noisy, (patch_size, patch_size))
    target_coeffs = get_2D_dct(target_patches).reshape((-1, patch_size ** 2))
    noisy_coeffs = get_2D_dct(noisy_patches).reshape((-1, patch_size ** 2))
    b = np.zeros((len(noisy_coeffs), J))
    c = np.zeros((patch_size ** 2, J))
    for m in range(patch_size ** 2):
        for j in range(J):
            b[:, j] = noisy_coeffs[:, m] ** j
        bb = np.dot(b.T, b)
        bt = np.dot(b.T, target_coeffs[:, m])
        c[m] = np.dot(np.linalg.pinv(bb), bt)
    return c, noisy_patches.min(), noisy_patches.max()

#show shrinkage
def plot_shrinkage_curve(c, J=6, patch_size=6, sname = None):
    fig, ax = plt.subplots(6, 6, figsize=(18, 18))
    ax = ax.flatten()
    x = np.linspace(-1, 1, 21)
    for i in range(patch_size ** 2):
        y = np.zeros_like(x)
        for j in range(J):
            y += (x ** j) * c[i * J + j]
        ax[i].plot(x, y)
        ax[i].set_ylim(-1, 1)
        ax[i].set_yticks([-0.5, 0.5])
        ax[i].set_xticks([-0.5, 0.5])
    if sname is not None:
        plt.savefig(sname, dpi=220)
        plt.close()

#Learn DCT Shrinkage function (Local Patch model) 
def dct_with_shrinkage_curve(im, c, sc_min, sc_max, patch_size=6):
    patches = extract_patches_2d(im, (patch_size, patch_size))
    patches = (patches - 127.) / 128.
    coeffs = get_2D_dct(patches).reshape((-1, patch_size ** 2))
    for m in range(patch_size ** 2):
        x = coeffs[:, m]
        y = np.zeros_like(x)
        for j in range(c.shape[1]):
            y += (x ** j) * c[m, j]
        roi = (sc_min <= x) * (x <= sc_max)
        coeffs[roi, m] = y[roi]
    patches = get_2d_idct(coeffs.reshape((-1, patch_size, patch_size)))        
    patches = patches * 128 + 127
    return reconstruct_from_patches_2d(patches, im.shape)

#Learn DCT Shrinkage function (Global model) 
def get_shrinkage_curve_with_global_loss_function(target, noisy,A, J=6, patch_size=6):
    noisy_patches = extract_patches_2d(noisy, (patch_size, patch_size))
    noisy_coeffs = get_2D_dct(noisy_patches).reshape((-1, patch_size ** 2))

    n = patch_size ** 2
    m = A.shape[1]
    M = noisy.shape[0] * noisy.shape[1]
    R = np.zeros((n, M))
    U = np.zeros((m, m * J))

    mat = np.zeros((m * J, M))
    mat2 = np.zeros((M, m * J))
    r = np.zeros(M)
    for k in range(0, len(noisy_patches), 1):
        if k % 400 == 0:
            print(k)
        # set R
        for row in range(patch_size):
            for col in range(patch_size):
                k1 = k + row * noisy.shape[1] + col
                r[:] = 0
                r[k1] = 1
                R[row * patch_size + col] = r
        # set U    
        for i in range(m):
            b = noisy_coeffs[k, i] 
            for j in range(J):
                U[i, i * J + j] = b ** j
        mat += np.dot(U.T, np.dot(A.T, R))
        mat2 += np.dot(R.T, np.dot(A, U))

    inv = np.linalg.pinv(np.dot(mat, mat2))
    c = np.dot(inv, np.dot(mat, target.reshape((target.shape[0] * target.shape[1], 1))))
    
    return c, noisy_patches.min(), noisy_patches.max()

#infer dct (global model)
def rdct_with_global_shrinkage_curve(im, c, sc_min, sc_max, A, patch_size=6):
    patches = extract_patches_2d(im, (patch_size, patch_size))
    patches = (patches - 127.) / 128.
    coeffs = np.dot(patches.reshape((-1, patch_size ** 2)), A)
    for m in range(patch_size ** 2):
        x = coeffs[:, m]
        y = np.zeros_like(x)
        for j in range(c.shape[1]):
            y += (x ** j) * c[m, j]
        roi = (sc_min <= x) * (x <= sc_max)
        coeffs[roi, m] = y[roi]
    patches = np.dot(coeffs, A.T).reshape((-1, patch_size, patch_size))
    patches = patches * 128 + 127
    return reconstruct_from_patches_2d(patches, im.shape)

# denoize with omp&dct
def denoise_with_learned_dictionary(im, A, k0, eps, patch_size=8, lam=0.5, n_iter=1, im0=None):
    recon = im.copy()
    for h in range(n_iter):
        patches = extract_patches_2d(recon, (patch_size, patch_size))
        if h == 0:
            q = np.zeros((len(patches), A.shape[1]))
        for i, patch in enumerate(patches):
            if i % 1000 == 0:
                print(i)
            q[i], _ = OMP(A, patch.flatten(), k0, eps=eps)
        recon_patches = (np.dot(A, q.T).T).reshape((-1, patch_size, patch_size))
        recon = reconstruct_from_patches_2d(recon_patches, im.shape)
        recon = (im * lam + recon) / (lam + 1.)
        if im0 is not None:
            print(h, get_psnr(im0, recon))
    return recon

#Global MCA
def global_mca(im, c=2., t=20., level=2, patch_size=8, n_iter=600):
    Y = im.copy()
    Y_c = np.zeros_like(Y) # content
    Y_t = np.zeros_like(Y) # texture
    for i in range(n_iter):
        Y_c_new = wavelet_shrinkage((Y - Y_c - Y_t) / c + Y_c, t, level,hard=False)
        Y_t_new = dct_shrinkage((Y - Y_c - Y_t) / c + Y_t, t, patch_size=patch_size,hard=False)
        opt = np.linalg.norm(Y_c_new - Y_c)
        opt2 = np.linalg.norm(Y_t_new - Y_t)
        Y_c = Y_c_new
        Y_t = Y_t_new
        if i % 10 == 0:
            print(i, opt, opt2) 
    return Y_c, Y_t

#sparse coding
def sparse_coding(im, A, k0, eps, patch_size=8):      
    patches = extract_patches_2d(im, (patch_size, patch_size))
    q = np.zeros((len(patches), A.shape[1]))
    for i, patch in enumerate(patches):
        if i % 1000 == 0:
            print(i)
        q[i], _ = OMP(A, patch.flatten(), k0, eps=eps)
    return q

#reconstruct image
def recon_image(im, q, A, lam=0.5, patch_size=8):
    recon_patches = (np.dot(A, q.T).T).reshape((-1, patch_size, patch_size))
    recon = reconstruct_from_patches_2d(recon_patches, im.shape)
    return (im * lam + recon) / (lam + 1.)