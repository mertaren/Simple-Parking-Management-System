import cv2 as cv
import numpy as np

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    
    return cv.LUT(image, table)

def apply_clahe(image, clip_limit=2, tile_gridSize=(8,8)):

    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)

    l, a , b = cv.split(lab)

    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_gridSize)
    clh = clahe.apply(l)

    limg = cv.merge((clh, a, b))
    return cv.cvtColor(limg, cv.COLOR_LAB2BGR)

def increase_contrast(img, alpha=1.5, beta=0):
    return cv.convertScaleAbs(img, alpha=alpha, beta=beta)