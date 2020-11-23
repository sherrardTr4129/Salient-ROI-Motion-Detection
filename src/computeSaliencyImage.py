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

def computeObjectMap(colorImage, saliencyMap, sizeThreshVal):

    # create dilation kernel
    dilateKernel = np.ones((3,3),np.uint8)

    # convert to uint8 image type using numpy
    saliencyMap = np.uint8(saliencyMap)

    # threshold saliency map image
    ret, thresh = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # dilate image using dilateKernel. This should hopefully blow up the
    # detected saliency points to the point that they can be grouped together 
    dilation = cv2.dilate(thresh, dilateKernel, iterations = 2)

    # find contours in dilated image
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # filter contours by size and create
    # bounding rectangles from good contours
    goodContours = []
    goodRects = []
    for cont in contours:
        area = cv2.contourArea(cont)
        if(area > sizeThreshVal):
            goodContours.append(cont)
            x,y,w,h = cv2.boundingRect(cont)
            goodRects.append([x,y,w,h])

    # combine redundant overlapping rectangles
    groupedRects, weights = cv2.groupRectangles(goodRects, 0, 0)

    # draw rectangles on image
    for rect in groupedRects:
        x,y,w,h = rect
        cv2.rectangle(colorImage,(x,y),(x+w,y+h),(0,0,255),-1)

    return colorImage

def isSalientROIMoving(curSaliencyFrame, prevSaliencyFrame):
    pass
if(__name__ == "__main__"):
    # initalize capture object
    capObj = cv2.VideoCapture(0)

    # initialize constants
    threshVal = 200
    sizeThreshVal = 4000
    lastFrame = None
    diffFrame = None
    firstIteration = True

    while(True):
        # read frame from camera
        ret, frame = capObj.read()

        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # compute saliency image from grayscale image
        saliencyMap = computeSaliencyImg(gray)

        # compute object map
        objectMap = computeObjectMap(frame, saliencyMap, sizeThreshVal)

        # determine if salient object is moving
        if(not firstIteration):
            isMoving = isSalientROIMoving(objectMap, lastFrame)

        # update last frame
        lastFrame = objectMap

        # indicate we have been through the loop once and 
        # skip the display step for this iteration
        if(firstIteration):
            firstIteration = False
            continue

        # display object map image
        cv2.imshow("object map", objectMap)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cv2.destroyAllWindows()
    capObj.release()

