import numpy as np
from multiprocessing.managers import SyncManager
from skimage.io import imread
import pywt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from utils.func import wavelet_shrinkage,dct_shrinkage,get_psnr,get_shrinkage_curve,plot_shrinkage_curve
from utils.func import dct_with_shrinkage_curve,show_dictionary,get_shrinkage_curve_with_global_loss_function,rdct_with_global_shrinkage_curve
OUTROOT="E:\\test\\SMPy541"

if __name__=="__main__":
    # font_path = u'/Library/Fonts/ヒラギノ角ゴ Pro W3.otf'
    # font_prop = FontProperties(fname=font_path)
    # plt.rc('font',family=font_prop.get_name())
    src = OUTROOT + "//barbara.png"
    src_n = OUTROOT + "//barbara_sig20"

    #Load Data
    im = imread(src).astype(np.float)
    Y = np.fromfile(src_n).reshape(im.shape) #Observed Image

    #show compared
    sname = OUTROOT + "//01_src_vs_noised.png"
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(im, cmap='gray', interpolation='Nearest', vmin=0, vmax=255)
    ax[1].imshow(Y, cmap='gray', interpolation='Nearest', vmin=0, vmax=255)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_title('ノイズなし')
    ax[1].set_title('ノイズあり\n{:.3f}'.format(get_psnr(im, Y)))
    plt.savefig(sname, dpi=220)
    plt.close()

    #WaveletShirink
    log = []
    opt_max = 0
    for t in np.linspace(0, 200, 21):
        recon = wavelet_shrinkage(Y, t, 2)  
        opt = get_psnr(im, recon)
        log.append(opt)
        if opt > opt_max:
            recon_max = recon.copy()
            opt_max = opt
    recon_ws = recon_max.copy()
    snamep = OUTROOT + "//11_recon_ws"
    recon_ws.tofile(snamep)
    plt.plot(np.linspace(0, 200, 21), log)
    plt.ylabel('PSNR [db]')
    plt.xlabel('閾値')
    plt.title('ウェーブレット縮小')
    sname = OUTROOT + "//12_wavelet_shrinkage_threshold.png"
    plt.savefig(sname, dpi=220)
    plt.close()
    recon_ws = np.fromfile(snamep).reshape(im.shape)
    plt.imshow(recon_ws, cmap='gray', interpolation='Nearest', vmin=0, vmax=255)
    plt.axis('off')
    plt.title('ウェーブレット縮小\n{:.3f}'.format(get_psnr(im, recon_ws)))
    plt.colorbar()
    plt.tight_layout()
    sname = OUTROOT + "//19_wavelet_shrinkage.png"
    plt.savefig(sname, dpi=220)
    plt.close()

    #DCTShirink
    log = []
    opt_max = 0
    for t in np.linspace(0, 200, 21):
        recon = dct_shrinkage(Y, t, 2)  
        opt = get_psnr(im, recon)
        print(t, opt)
        log.append(opt)
        if opt > opt_max:
            recon_max = recon.copy()
            opt_max = opt
            
    recon_dct_shrink = recon_max.copy()
    snamep = OUTROOT + "//21_recon_dct_shrink"
    recon_dct_shrink.tofile(snamep)
    plt.plot(np.linspace(0, 200, 21), log)
    plt.ylabel('PSNR [db]')
    plt.xlabel('閾値')
    plt.title('重複したパッチベースのDCT縮小')
    sname = OUTROOT + "//22_dct_shrinkage_threshold.png"
    plt.savefig(sname, dpi=220)
    plt.close()
    recon_dct_shrink = np.fromfile(snamep).reshape(im.shape)
    plt.imshow(recon_dct_shrink, cmap='gray', interpolation='Nearest', vmin=0, vmax=255)
    plt.axis('off')
    plt.title('重複したパッチベースのDCT縮小\n{:.3f}'.format(get_psnr(im, recon_dct_shrink)))
    plt.colorbar()
    plt.tight_layout()
    sname = OUTROOT + "//29_dct_shrinkage.png"
    plt.savefig(sname, dpi=220)
    plt.close()

    #Src2 Load
    sname = OUTROOT + "\\lena.jpg"
    lena = imread(sname, as_gray=True) * 255
    plt.imshow(lena, cmap='gray', interpolation='Nearest')
    plt.axis('off')
    sname = OUTROOT + "\\31_lena.png"
    plt.savefig(sname, dpi=220)
    plt.close()
    #Get Training Image
    target = lena[100:300, 100:300]
    noisy = target + np.random.randn(200, 200) * 20
    target = (target - 127.) / 128.
    noisy = (noisy - 127.) / 128.
    plt.imshow(target, cmap='gray', interpolation='Nearest')
    plt.axis('off')
    plt.colorbar()
    sname = OUTROOT + "\\32_lena_200_200_train.png"
    plt.savefig(sname, dpi=220)
    plt.close()

    #Train sh func (local model)
    c_local, sc_min, sc_max = get_shrinkage_curve(target, noisy)
    sname = OUTROOT + "\\33_c_local"
    c_local.tofile(sname)
    sname = OUTROOT + "\\34_c_local.png"
    plot_shrinkage_curve(c_local.flatten(),sname= sname)

    #Train sh func (Local Patch model) 
    recon_dct_shrinkage_curve = dct_with_shrinkage_curve(Y, c_local, sc_min, sc_max)
    snamep = OUTROOT + "\\41_recon_dct_shrinkage_curve"
    recon_dct_shrinkage_curve.tofile(snamep)
    recon_dct_shrinkage_curve = np.fromfile(snamep).reshape(im.shape)
    plt.imshow(recon_dct_shrinkage_curve, cmap='gray', interpolation='Nearest', vmin=0, vmax=255)
    plt.axis('off')
    plt.title('重複したパッチベースのDCT学習縮小曲線\n{:.3f}'.format(get_psnr(im, recon_dct_shrinkage_curve)))
    plt.colorbar()
    plt.tight_layout()
    sname = OUTROOT + "\\42_recon_dct_shrinkage_curve.png"
    plt.savefig(sname, dpi=220)
    plt.close()

    #Train sh func (Global model)
    patch_size = 6
    dict_size = 6
    A_1D = np.zeros((patch_size, dict_size))
    for k in np.arange(dict_size):
        for i in np.arange(patch_size):
            A_1D[i, k] = np.cos(i * k * np.pi / float(dict_size))
        if k != 0:
            A_1D[:, k] -= A_1D[:, k].mean()
    A = np.kron(A_1D, A_1D)
    sname = OUTROOT + "\\51_dct_dict.png"
    show_dictionary(A,name=sname)
    c_global, sc_min, sc_max = get_shrinkage_curve_with_global_loss_function(target, noisy,A)
    snamep = OUTROOT + "\\52_c_global"
    c_global.tofile(snamep)
    sname = OUTROOT + "\\53_recon_dct_shrinkage_curve.png"
    plot_shrinkage_curve(c_global,sname=sname)
    recon_dct_global_shrinkage_curve = rdct_with_global_shrinkage_curve(Y, c_global, sc_min, sc_max, A)
    recon_dct_global_shrinkage_curve -= recon_dct_global_shrinkage_curve.min()# スケールがあわない…
    recon_dct_global_shrinkage_curve *= im.max() / recon_dct_global_shrinkage_curve.max()
    snamep = OUTROOT + "\\54_recon_dct_shrinkage_curve"
    recon_dct_global_shrinkage_curve.tofile(snamep)
    recon_dct_global_shrinkage_curve = np.fromfile(snamep).reshape(im.shape)
    #plt.imshow(recon_dct_global_shrinkage_curve, cmap='gray', interpolation='Nearest')
    plt.axis('off')
    plt.title('重複したパッチベースのDCT大域的学習縮小曲線\n{:.3f}'.format(get_psnr(im, recon_dct_global_shrinkage_curve)))
    plt.colorbar()
    plt.tight_layout()
    sname = OUTROOT + "\\54_recon_dct_global_shrinkage_curve.png"
    plt.savefig(sname, dpi=220)
    plt.close()





















