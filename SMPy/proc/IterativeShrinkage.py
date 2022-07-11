import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.signal import convolve2d
from utils.func import get_blur_kernel_norm

#Algorithm of Iterative Shrinkage
class IterativeShrinkage(object):
    #Construction
    def __init__(self, b, s, lam):
        self.b = b
        self.eps = 0
        self.s = s
        self.lam = lam
        self.c = 1.
        self.kernel = get_blur_kernel_norm()
    
    #wavelet H
    def horizontal(self, im):
        im2 = np.roll(im, 1, axis=1)
        return (im + im2) / 2., (im - im2) / 2.

    #wavelet V
    def vertical(self, im):
        im2 = np.roll(im, 1, axis=0)
        return (im + im2) / 2., (im - im2) / 2.
    
    #wavelet Lv1
    def wavelet(self, im):
        l, h = self.horizontal(im)
        hl, hh = self.vertical(h)
        ll, lh = self.vertical(l)
        return ll, lh, hl, hh
    
    #wavelet Lv2
    def wavelet2(self, im):
        ll, lh, hl, hh = self.wavelet(im)
        ll2, lh2, hl2, hh2 = self.wavelet(ll)
        return np.array([ll2, lh2, hl2, hh2, lh, hl, hh])

    def forward(self, x):
        """ 順投影 """
        return convolve2d(np.sum(x, axis=0), self.kernel, mode='same', boundary='symm')  

    def backward(self, y):
        """ 逆投影 """
        return self.wavelet2(convolve2d(y, self.kernel, mode='same', boundary='symm'))

    #Shrink
    def shrink(self, x):
        """ 縮小 """
        s, lam = self.s, self.lam
        ndx = x < 0
        x[ndx] *= -1
        x = (x - s - lam + np.sqrt((s + lam - x) ** 2 + 4. * s * x)) / 2.
        x[ndx] *= -1
        return x

    #
    def rho(self, x):
        """ 関数ρ(x) """
        return np.abs(x) + self.s * np.log(1. + np.abs(x) / self.s)
    
    #Objective function
    def f(self, x, y):
        """ 目標関数 """
        return self.lam * np.sum(self.rho(x)) + 0.5 * np.sum((self.b - y) ** 2)

    def line(self, x, gx, y, gy):
        """ 直線探索 """
        
        def f_ls(mu, *args):
            """ 目標関数 """
            x, gx, y, gy = args
            x_new = x + mu * gx
            e = self.b - (y + mu * gy)
            return self.lam * np.sum(self.rho(x_new)) + 0.5 * np.sum(e ** 2)

        res = minimize_scalar(f_ls, args=(x, gx, y, gy))
        return res.x                  
    
    def sesop(self, x, ses_gx, y, ses_gy, q):
        """ 逐次的部分空間最適化 """
    
        def f_sesop(mu, *args):
            """ 目標関数 """
            x, ses_gx, y, ses_gy, q = args
            gx = ses_gx[0] * mu[0]
            gy = ses_gy[0] * mu[0]
            for i in range(1, q + 1):
                gx += ses_gx[i] * mu[i]
                gy += ses_gy[i] * mu[i]
            x_new = x + gx
            e = self.b - (y + gy)
            return self.lam * np.sum(self.rho(x_new)) + 0.5 * np.sum(e ** 2)

        res = minimize(f_sesop, np.ones(q + 1), args=(x, ses_gx, y, ses_gy, q))
        return res.x
    
    def SSF(self, niter=50, ls=False, sesop=False, q=5, x0=None):
        """ 分割可能代理汎関数（separable surrogate functional; SSF） """
        y = np.zeros_like(self.b)
        x = self.backward(y)
        r = self.b.copy()      
        ses_gx = np.zeros((q + 1, x.shape[0], x.shape[1], x.shape[2]))
        ses_gy = np.zeros((q + 1, y.shape[0], y.shape[1]))
            
        fs = []
        errs = []
        for k in range(niter):
            fs.append(self.f(x, y))
            if x0 is not None:
                x_hat = np.sum(x, axis=0)
                errs.append(np.sum((x_hat - x0) ** 2) / np.sum(x0 ** 2))
                print(k, errs[-1])
            e = self.backward(r)
            e_s = self.shrink(x + e / self.c)
            gx = e_s - x
            if sesop:
                ses_gx[k % (q + 1)] = gx
                ses_gy[k % (q + 1)] = self.forward(gx)
                q2 = min(q, k)
                res = self.sesop(x, ses_gx, y, ses_gy, q2)
                u = ses_gx[0] * res[0]
                for i in range(1, q2 + 1):
                    u += ses_gx[i] * res[i]
            elif ls:
                u = self.line(x, gx, y, self.forward(gx)) * gx
            else:
                u = gx
            x += u
            y = self.forward(x)
            r = self.b - y
            if np.sum(u ** 2) < self.eps:
                return x, fs, errs
        return x, fs, errs