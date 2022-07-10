from itertools import combinations
import numpy as np

class MAP_MMSE(object):
    """ MAP推定 """
    def __init__(self, nr):
        # 数値例を引き継ぎ
        self.nr = nr
    
    def get_Q(self, A):
        """ Q^-1を得る """
#         return np.linalg.inv(np.dot(A.T, A)/ (self.nr.sig_e ** 2) + 1. / (self.nr.sig_x ** 2))
        return np.linalg.pinv(np.dot(A.T, A)/ (self.nr.sig_e ** 2) + 1. / (self.nr.sig_x ** 2))

    def oracle(self, s):
        """ オラクル推定 """
        A_s = self.nr.A[:, s]
        a = self.get_Q(A_s)
        b = np.dot(A_s.T, self.nr.y) / (self.nr.sig_e ** 2)
        x_hat = np.zeros_like(self.nr.x)
        x_hat[s] = np.dot(a, b)
        return x_hat, s
     
    def exhaustive_search_MAP_support(self):
        """ MAPサポートを全探索 """
        p_max = 0
        for s in combinations(range(self.nr.m), self.nr.k):
            A_s = self.nr.A[:, s]
            a = self.get_Q(A_s)
            b = np.dot(A_s.T, self.nr.y)
            p = np.dot(b.T, np.dot(a, b))
            if p > p_max:
                p_max = p
                s_max = s
        return np.array(s_max)

    def exact_MAP(self):
        """ 厳密なMAP推定を得る """
        return self.oracle(self.exhaustive_search_MAP_support())

    def exact_MMSE(self):
        """ 厳密なMMSE推定を得る """
        nor = 0.
        x_hat = np.zeros_like(self.nr.x)
        for s in combinations(range(self.nr.m), self.nr.k):
            s = np.array(s)
            A_s = self.nr.A[:, s]
            a = self.get_Q(A_s)
            b = np.dot(A_s.T, self.nr.y) / (self.nr.sig_e ** 2)
            x_s = np.dot(a, b)
#             q_s = np.exp(np.dot(b.T, x_s) / 2 + np.log(np.linalg.det(a)) / 2)
            q_s = np.exp(np.dot(b.T, x_s) / 2)
            x_hat[s] += x_s * q_s
            nor += q_s
        return x_hat / nor
    
    def random_OMP(self, J=25):
        """ 近似MMSE推定を得るためのランダムOMP """
        # 定数
        sig_e2 = self.nr.sig_e ** 2
        sig_x2 = self.nr.sig_x ** 2
        C = 1 / (2 * sig_e2 * (1 + sig_e2 / sig_x2))
        C2 = np.log(1 / sig_e2 + 1 / sig_x2) / 2
        
        x_hat = np.zeros_like(self.nr.x)
        for _ in range(J):
            # 初期化
            x = np.zeros(self.nr.A.shape[1])
            S = np.zeros(self.nr.A.shape[1], dtype=np.uint8)
            r = self.nr.y.copy()

            for _ in range(self.nr.k):
                # サポート更新
                q = np.exp((np.dot(self.nr.A.T, r) ** 2) * C + C2)
                q /= q.sum()
                q_sum = np.cumsum(q)
                
                while True:
                    p = np.random.rand()
                    ndx = np.where(q_sum > p)[0]
                    if S[ndx[0]] == 0:
                        S[ndx[0]] = 1
                        break

                # 解更新
                As = self.nr.A[:, S == 1]
                pinv = np.linalg.pinv(np.dot(As, As.T))
                x[S == 1] = np.dot(As.T, np.dot(pinv, self.nr.y))
        
                # 残差更新
                r = self.nr.y - np.dot(self.nr.A, x)
            
            ndx = np.where(S == 1)[0]
            x, S = self.oracle(ndx)
            x_hat += x

        return x_hat / J       