from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

"""
from PIL import Image
img = Image.open("cat.jpg")
img = img.convert("L")
img = np.asarray(img, dtype="int32" )
"""

from matplotlib.image import imread

img = imread("cat.jpg")
# Questa funzione permette di convertire l'immagine in bianco e nero
img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
img.shape

imshow(img)

start = (0,0)
offset = (img.shape[0],int(img.shape[1]/2))
img_crop = crop_with_numpy(img,start,offset)

imshow(img_crop)

img_small = resize_with_sk(img_crop,(48,48))
#img_small = resize_with_pil(img_crop,(48,48))
#img_small = resize_with_scipy(img_crop,(48,48))
#img_small = resize_with_cv2(img,(48,48))

imshow(img_small)

def crop_with_numpy(img,start,offset):
    return img[start[0]:start[0]+offset[0],start[1]:start[1]+offset[1]]

def resize_with_cv2(img,size):
    from cv2 import resize
    return resize(img, dsize=size)

def resize_with_sk(img,size):
    from skimage.transform import resize
    return resize(img, size)

def resize_with_pil(img,size):
    from PIL import Image
    img = Image.fromarray(img)
    img = img.resize(size, Image.ANTIALIAS)
    img = np.asarray(img, dtype="int32")
    return img

def resize_with_scipy(img,size):
    # Questa funzione è deprecata, meglio non utilizzarla
    from scipy.misc import imresize
    return imresize(img,size)

def imshow(img):
    plt.axis("off")
    plt.imshow(img,cmap="gray")
