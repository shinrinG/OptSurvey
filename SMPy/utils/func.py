import numpy as np

#Get Squared Error
def get_se(x, x2):
    return np.sum((x - x2) ** 2)