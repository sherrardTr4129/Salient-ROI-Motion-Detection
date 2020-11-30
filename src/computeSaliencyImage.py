# Author: Trevor Sherrard
# Course: RBE526 - Human Robot Interaction
# Assignment: Individual Algorithm Implementation
# Since: November 22, 2020

import cv2
import glob
import csv
import numpy as np
from matplotlib import pyplot as plt

def writeCSV(iterPoints):
    """
    This function writes the numWhite values out to a CSV file.

    params:
        iterPoints (int[]): The list of numWhite points to write out
    
    returns:
        None
    """
    with open("points.csv", "w") as csvFile:
        writerObj = csv.writer(csvFile, delimiter = ",")
        for i in range(len(iterPoints)):
            writerObj.writerow([str(i), str(iterPoints[i])])

def computeSaliencyImg(img):
    """
    This function computes the saliency map of a given image per the algorithm
    put forth in the work of Xiaodi Hou et. al. 

    params:
        img (int8 image): the image to compute the saliency map for.
    
    returns:
        saliencyMap (int8 image): the computed saliency map
    """
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
    """
    This function uses standard image processing techniques to create an object
    map for highly salient ROIs within a given saliency map.

    params:
        colorImage (int32 image): the original color image taken from the webcam
        saliencyMap (int8 image): the computed saliency map for the current color image
        sizeThreshVal (int): the area in pixels for an ROI to be considered in the object map

    returns:
        colorImage (int32 image): the original color image with detected saliency ROIs drawn over.
    """

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

    # draw white rectangles on image
    for rect in groupedRects:
        x,y,w,h = rect
        cv2.rectangle(colorImage,(x,y),(x+w,y+h),(255,255,255),-1)

    return colorImage

def isSalientROIMoving(curSaliencyFrame, prevSaliencyFrame, numForMoving):
    """
    This function uses the current saliency frame and the previous saliency frame to detect if
    the ROI is moving. The algorithm to do this was adapted from the work of Jong-Hann Jean et. al

    params:
        curSaliencyFrame (int8 image): the object map computed from the current frame
        prevSaliencyFrame (int8 image): the object map computed from the previous frame
        numFroMoving (int): the threshold of white pixels within the subtracted edge image for the
                            ROI to be considered moving.

    returns:
        movingEdges (binary image): the binary image representing moving salient edges in the video stream
        diffImage (binary image): the result of the subtraction of the two object map images
        isMoving (boolean): a boolean indicating whether or not the detected ROI is moving. returns None
                            if nothing is detected in the image.
        countWhite (int): the number of white edge pixels in the difference image between the two object maps.
    """
    # compute binary image issolating drawn white rectangles
    ret, threshCurFrame = cv2.threshold(curSaliencyFrame, 254, 255, cv2.THRESH_BINARY)
    ret, threshLastFrame = cv2.threshold(prevSaliencyFrame, 254, 255, cv2.THRESH_BINARY)

    # find edges in current image
    currentFrameEdge = cv2.Canny(curSaliencyFrame, 120, 255)
    
    # compute difference image between two frames
    diffImage = threshCurFrame - threshLastFrame

    # compute number of white pixels in diffImage
    countWhite = np.sum(diffImage == 255)

    # determine if saliency ROI is moving
    isMoving = False
    if(countWhite > numForMoving):
        isMoving = True
    elif(countWhite < numForMoving and countWhite != 0):
        isMoving = False
    elif(countWhite == 0):
        isMoving = None

    # compute bitwise and between edge map and diffimage
    movingEdges = cv2.bitwise_and(diffImage, currentFrameEdge)

    return movingEdges, diffImage, isMoving, countWhite

if(__name__ == "__main__"):
    # initalize capture object
    capObj = cv2.VideoCapture(0)

    # initialize threshold constants
    numForMoving = 3500
    sizeThreshVal = 4000

    # initialize sequential frame place holders
    lastFrame = None
    movingEdges = None
    diffImage = None

    # initialize control boolean variables
    firstIteration = True
    isMoving = False

    # declare array to hold white count number for later processing
    countWhiteArr = []

    while(True):
        # read frame from camera
        ret, frame = capObj.read()

        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # compute saliency image from grayscale image
        saliencyMap = computeSaliencyImg(gray)

        # compute object map
        objectMap = computeObjectMap(frame, saliencyMap, sizeThreshVal)
        objectMapGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # determine if salient object is moving
        if(not firstIteration):
            movingEdges, diffImage, isMoving, countWhite = isSalientROIMoving(objectMapGray, lastFrame, numForMoving)
            countWhiteArr.append(countWhite)
        
        if(isMoving == True):
            cv2.putText(movingEdges, 'Moving!', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        elif(isMoving == False):
            cv2.putText(movingEdges, 'Not Moving!', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # update last frame
        lastFrame = objectMapGray

        # indicate we have been through the loop once and 
        # skip the display step for this iteration
        if(firstIteration):
            firstIteration = False
            continue

        # display saliency map
        cv2.imshow("saliency map", saliencyMap)

        # display object map with ROIs drawn over
        cv2.imshow("object map", objectMap)
        
        # display difference image
        cv2.imshow("difference image", diffImage)

        # display moving edges image
        cv2.imshow("moving edges image", movingEdges)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            # write out points to csv before exiting
            writeCSV(countWhiteArr)
            break

    cv2.destroyAllWindows()
    capObj.release()

