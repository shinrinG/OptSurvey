from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from skimage.io import imread
import pywt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from utils.func import *
OUTROOT="D:\\test\\SMPy520"

if __name__=="__main__":
    font_path = u'/Library/Fonts/ヒラギノ角ゴ Pro W3.otf'
    font_prop = FontProperties(fname=font_path)
    plt.rc('font',family=font_prop.get_name())
    src = OUTROOT + "//barbara.png"
    src_n = OUTROOT + "//barbara_sig20"

    #Load Data
    im = imread('barbara.png').astype(np.float)
    Y = np.fromfile('barbara_sig20').reshape(im.shape) #Observed Image
