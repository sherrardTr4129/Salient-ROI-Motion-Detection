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

    # compute spectral residue image on log scale
    spectralResidualImageLog = logImage - logImageBlured

    # undo log scaling to real image component matches non-log scaling
    # of phase/imaginary image component
    spectralResidualImage = np.exp(spectralResidualImageLog)

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

def computeObjectMap(colorImage, saliencyMap, threshVal):

    # create dilation kernel
    dilateKernel = np.ones((10,10),np.uint8)

    # threshold saliency map image
    ret, thresh = cv2.threshold(saliencyMap, threshVal, 255, cv2.THRESH_BINARY)

    # convert to uint8 image type using numpy
    thresh = np.uint8(thresh)

    # dilate image using dilateKernel. This should hopefully blow up the
    # detected saliency points to the point that they can be grouped together 
    dilation = cv2.dilate(thresh, dilateKernel, iterations = 14)

    # find contours in dilated image
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # filter contours by size
    goodContours = []
    for cont in contours:
        area = cv2.contourArea(cont)
        if(area > 7000):
            goodContours.append(cont)

    # draw contours on color image
    cv2.drawContours(colorImage, goodContours, -1, (0, 255, 0), 3)

    # draw bounding boxes around detected contours
    for cont in goodContours:
        x,y,w,h = cv2.boundingRect(cont)
        cv2.rectangle(colorImage,(x,y),(x+w,y+h),(0,0,255),2)

    return colorImage

if(__name__ == "__main__"):
    capObj = cv2.VideoCapture(0)
    threshVal = 180

    while(True):
        # read frame from camera
        ret, frame = capObj.read()

        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # compute saliency image from grayscale image
        saliencyMap = computeSaliencyImg(gray)

        # compute object map
        objectMap = computeObjectMap(frame, saliencyMap, threshVal)

        # display object map image
        cv2.imshow("object map", objectMap)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cv2.destroyAllWindows()
    capObj.release()

