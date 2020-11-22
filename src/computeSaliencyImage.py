# Author: Trevor Sherrard
# Course: RBE526 - Human Robot Interaction
# Assignment: Individual Algorithm Implementation
# Since: November 22, 2020

import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt

def computeSaliencyImg(img):
    # compute dft and shift
    fImage = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    fImageShifted = np.fft.fftshift(fImage)

    # extract real and imaginary image components in polar form
    real,imaginary = cv2.cartToPolar(fImageShifted[:,:,0], fImageShifted[:,:,1])

    # compute log of real image
    logImage = np.log(real)

    # compute blurred version of real image
    blurKernel = np.ones((7,7), np.float32)/(7*7)
    logImageBlured = cv2.filter2D(logImage, -1, blurKernel)

    # compute spectral residue image
    spectralResidualImage = logImage - logImageBlured

    # move back to cartesian complex form
    fImageShifted[:,:,0], fImageShifted[:,:,1] = cv2.polarToCart(spectralResidualImage,imaginary)

    # compute neccessary shift for invese DFT
    fInvShift = np.fft.ifftshift(fImageShifted)

    # perform inverse DFT
    saliencyMapComplex = cv2.idft(fInvShift)**2

    # find real part of saliency map from obtained DFT results 
    saliencyMap = cv2.magnitude(saliencyMapComplex[:,:,0], saliencyMapComplex[:,:,1])

    # normalize around the maximum value (result between 0-1)
    maxVal = np.max(saliencyMap)
    saliencyMap = saliencyMap / maxVal

    # scale image up to 8-bit range (0-255)
    saliencyMap = saliencyMap*255

    return saliencyMap

if(__name__ == "__main__"):
    # grab test image filenames
    testImages = glob.glob("/home/sherrardtr/Desktop/RBE526-Individual-Alg/testImages/*")

    # compute saliency image and display it
    for imgFile in testImages:
        img = cv2.imread(imgFile, 0)
        saliencyImg = computeSaliencyImg(img)
        plt.imshow(saliencyImg, cmap="gray")
        plt.show()
