import numpy as np

class NumericalExample(object):
    """ 数値例 """

    def __init__(self, n=20, m=30, k=3, sig_x=1., sig_e=0.1):
        
        # パラメータ 
        self.n = n
        self.m = m
        self.k = k
        self.sig_x = sig_x
        self.sig_e = sig_e
        
        # 辞書A
        self.A = np.random.randn(n, m)
        self.A = np.dot(self.A, np.diag(1. / np.sqrt(np.diag(np.dot(self.A.T, self.A)))))

        # スパースベクトルx
        self.x = np.zeros(m)
        self.ndx = np.random.permutation(range(m))[:k]
        self.x[self.ndx] = np.random.randn(k) * sig_x
    
        # 信号z
        self.z = np.dot(self.A, self.x)

        # ノイズベクトルe
        self.e = np.random.randn(n) * sig_e
    
        # 観測されるベクトルy
        self.y = self.z + self.e 