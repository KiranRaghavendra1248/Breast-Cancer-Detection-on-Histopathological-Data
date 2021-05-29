# Imports
import numpy as np
import cv2 as cv


# Preprocessing functions

# Contrast Limited Adaptive Histogram Equilization
def clahe(img):
    # Takes a PIL instance input, and returns nd.array
    # Here we recieve PIL instance as input shape [H,W,C]
    img=np.array(img,dtype=np.uint8)

    lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)

    lab_planes = cv.split(lab)

    clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv.merge(lab_planes)

    rgb = cv.cvtColor(lab, cv.COLOR_LAB2RGB)

    return rgb

# Medium filter to remove the noisiness of an image
def median_filter(img):
    img=np.pad(img,[(1,1),(1,1),(0,0)],'constant',constant_values=(0))
    print(img.shape)
    median = cv.medianBlur(np.float32(img),5)
    return median