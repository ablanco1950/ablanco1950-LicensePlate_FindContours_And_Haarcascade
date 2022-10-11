# -*- coding: utf-8 -*-
"""
Created on October 2022

@author: Alfonso Blanco García
"""
######################################################################
# PARAMETERS
######################################################################
#
dir=""
dirname= dir +"test6Training\\images"

dirname_codfilters=dir + "test6Training\\codfilters"

######################################################################

import pytesseract

import numpy as np

import cv2

#https://github.com/spmallick/mallick_cascades/tree/master/haarcascades
plat_detector =  cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

X_resize=220
Y_resize=70

import os
import re

import imutils
#####################################################################
"""
Copied from https://gist.github.com/endolith/334196bac1cac45a4893#

other source:
    https://stackoverflow.com/questions/46084476/radon-transformation-in-python
"""

from skimage.transform import radon

import numpy
from numpy import  mean, array, blackman, sqrt, square
from numpy.fft import rfft

try:
    # More accurate peak finding from
    # https://gist.github.com/endolith/255291#file-parabolic-py
    from parabolic import parabolic

    def argmax(x):
        return parabolic(x, numpy.argmax(x))[0]
except ImportError:
    from numpy import argmax


def GetRotationImage(image):

   
    I=image
    I = I - mean(I)  # Demean; make the brightness extend above and below zero
    
    
    # Do the radon transform and display the result
    sinogram = radon(I)
   
    
    # Find the RMS value of each row and find "busiest" rotation,
    # where the transform is lined up perfectly with the alternating dark
    # text and white lines
      
    # rms_flat does no exist in recent versions
    #r = array([mlab.rms_flat(line) for line in sinogram.transpose()])
    r = array([sqrt(mean(square(line))) for line in sinogram.transpose()])
    rotation = argmax(r)
    #print('Rotation: {:.2f} degrees'.format(90 - rotation))
    #plt.axhline(rotation, color='r')
    
    # Plot the busy row
    row = sinogram[:, rotation]
    N = len(row)
    
    # Take spectrum of busy row and find line spacing
    window = blackman(N)
    spectrum = rfft(row * window)
    
    frequency = argmax(abs(spectrum))
   
    return rotation, spectrum, frequency
#####################################################################
def ThresholdStable(image):
    # -*- coding: utf-8 -*-
    """
    Created on Fri Aug 12 21:04:48 2022
    Author: Alfonso Blanco García
    
    Looks for the threshold whose variations keep the image STABLE
    (there are only small variations with the image of the previous 
     threshold).
    Similar to the method followed in cv2.MSER
    https://datasmarts.net/es/como-usar-el-detector-de-puntos-clave-mser-en-opencv/https://felipemeganha.medium.com/detecting-handwriting-regions-with-opencv-and-python-ff0b1050aa4e
    """
  
    thresholds=[]
    Repes=[]
    Difes=[]
    
    gray=image 
    grayAnt=gray

    ContRepe=0
    threshold=0
    for i in range (255):
        
        ret, gray1=cv2.threshold(gray,i,255,  cv2.THRESH_BINARY)
        Dife1 = grayAnt - gray1
        Dife2=np.sum(Dife1)
        if Dife2 < 0: Dife2=Dife2*-1
        Difes.append(Dife2)
        if Dife2<22000: # Case only image of license plate
        #if Dife2<60000:    
            ContRepe=ContRepe+1
            
            threshold=i
            grayAnt=gray1
            continue
        if ContRepe > 0:
            
            thresholds.append(threshold) 
            Repes.append(ContRepe)  
        ContRepe=0
        grayAnt=gray1
    thresholdMax=0
    RepesMax=0    
    for i in range(len(thresholds)):
        #print ("Threshold = " + str(thresholds[i])+ " Repeticiones = " +str(Repes[i]))
        if Repes[i] > RepesMax:
            RepesMax=Repes[i]
            thresholdMax=thresholds[i]
            
    #print(min(Difes))
    #print ("Threshold Resultado= " + str(thresholdMax)+ " Repeticiones = " +str(RepesMax))
    return thresholdMax

 
# Copied from https://learnopencv.com/otsu-thresholding-with-opencv/ 
def OTSU_Threshold(image):
# Set total number of bins in the histogram

    bins_num = 256
    
    # Get the image histogram
    
    hist, bin_edges = np.histogram(image, bins=bins_num)
   
    # Get normalized histogram if it is required
    
    #if is_normalized:
    
    hist = np.divide(hist.ravel(), hist.max())
    
     
    
    # Calculate centers of bins
    
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    
    
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    
    weight1 = np.cumsum(hist)
    
    weight2 = np.cumsum(hist[::-1])[::-1]
   
    # Get the class means mu0(t)
    
    mean1 = np.cumsum(hist * bin_mids) / weight1
    
    # Get the class means mu1(t)
    
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    # Maximize the inter_class_variance function val
    
    index_of_max_val = np.argmax(inter_class_variance)
    
    threshold = bin_mids[:-1][index_of_max_val]
    
    #print("Otsu's algorithm implementation thresholding result: ", threshold)
    return threshold

#########################################################################
def ApplyCLAHE(gray):
#https://towardsdatascience.com/image-enhancement-techniques-using-opencv-and-python-9191d5c30d45
    #hist=cv2.calcHist(gray,[0],None,[256],[0,256])
    gray_img_eqhist=cv2.equalizeHist(gray)
    hist=cv2.calcHist(gray_img_eqhist,[0],None,[256],[0,256])
    clahe=cv2.createCLAHE(clipLimit=200,tileGridSize=(3,3))
    gray_img_clahe=clahe.apply(gray_img_eqhist)
    return gray_img_clahe

def Detect_Spanish_LicensePlate(Text):
    
    if len(Text) != 7: return -1
    if (Text[0] < "0" or Text[0] > "9" ) : return -1 
    if (Text[1] < "0" or Text[2] > "9" ) : return -1   
    if (Text[2] < "0" or Text[2] > "9" ) : return -1   
    if (Text[3] < "0" or Text[3] > "9" ) : return -1     
    if (Text[4] < "A" or Text[4] > "Z" ) : return -1 
    if (Text[5] < "A" or Text[5] > "Z" ) : return -1 
    if (Text[6] < "A" or Text[6] > "Z" ) : return -1 
    return 1

######################################################
# Detection by Haarcascade
########################################################
def DetectLicenseInCar2(gray):
    TabCordsLicensesDetected=[]
   
    img=gray
    
    plates = plat_detector.detectMultiScale(img,scaleFactor=1.25,
        minNeighbors = 3, minSize=(3,3))   
      
    
    for (x,y,w,h) in plates:
        
        lCords=[]
        lCords.append(x)
        lCords.append(y)
        lCords.append(w)
        lCords.append(h)
       
        TabCordsLicensesDetected.append(lCords)
       
              
    return TabCordsLicensesDetected 
    
#############################################################
#https://stackoverflow.com/questions/64530229/how-do-i-get-tesseract-to-read-the-license-plate-in-the-this-python-opencv-proje
##############################################################
def DetectLicenseInCar1(gray_image, thresh):
    
    TabCordsLicensesDetected=[]
   
    # Removes Noise
    gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
    
    # Canny Edge Detection
    if thresh==-1:
        canny_edge = cv2.Canny(gray_image, 100, 200)
        
    else:     
    #https://stackoverflow.com/questions/21324950/how-can-i-select-the-best-set-of-parameters-in-the-canny-edge-detection-algorith
    #high_thresh, thresh_im = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_thresh=thresh
        lowThresh = 0.5*high_thresh
        canny_edge = cv2.Canny(gray_image, lowThresh, high_thresh)
  
    
    # Find contours based on Edges
    # The code below needs an - or else you'll get a ValueError: too many values to unpack (expected 2) or a numpy error
    contours, new = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    
    # # Initialize license Plate contour and x,y coordinates
    contour_with_license_plate = None
    license_plate = None
    x = None
    y = None
    w = None
    h = None
    
    # Find the contour with 4 potential corners and create a Region of Interest around it
    for contour in contours:
        # Find Perimeter of contour and it should be a closed contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        # This checks if it's a rectangle
        if len(approx) == 4:
            contour_with_license_plate = approx
            x, y, w, h = cv2.boundingRect(contour)
            #license_plate = gray_image[y:y + h, x:x + w]
            #Area=cv2.minAreaRect(gray)
            Area=h*w
            
            if Area < 500 or Area > 70000:
                continue
           
            if w/h < 1 or w/h > 7:
                continue
            
            lCords=[]
            lCords.append(x)
            lCords.append(y)
            lCords.append(w)
            lCords.append(h)
           
            TabCordsLicensesDetected.append(lCords)
            

    return TabCordsLicensesDetected 


################################################################
def DetectLicenseInCar(gray):
    TabCordsLicensesDetectedTotal=[]
    #gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    TabCordsLicensesDetected=DetectLicenseInCar1(gray, -1)
    if len(TabCordsLicensesDetected) > 0:
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
    TabCordsLicensesDetected=DetectLicenseInCar2(gray)
    if len(TabCordsLicensesDetected) > 0:
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
   
   
    ret, gray1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    TabCordsLicensesDetected=DetectLicenseInCar1(gray1,ret)
    
    if len(TabCordsLicensesDetected) > 0:
       
        print(License + " FindContours detected with Otsu's thresholding of cv2 and THRESH_BINARY ")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])
          
    TabCordsLicensesDetected=DetectLicenseInCar2(gray1)
    if len(TabCordsLicensesDetected) > 0:
        print(License + " Haarcode detected with Otsu's thresholding of cv2 and THRESH_BINARY ")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
   
    ret, gray1 = cv2.threshold(gray,0,255,cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
    TabCordsLicensesDetected=DetectLicenseInCar1(gray1,ret)
    
    if len(TabCordsLicensesDetected) > 0:
       
        print(License + "findcontours detected with Otsu's thresholding of cv2 and THRESH_TRUNC ")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])
    TabCordsLicensesDetected=DetectLicenseInCar2(gray)
    if len(TabCordsLicensesDetected) > 0:
        print(License + " haarcode detected with Otsu's thresholding of cv2 and THRESH_TRUNC ")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
    
    ret, gray1 = cv2.threshold(gray,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    TabCordsLicensesDetected=DetectLicenseInCar1(gray1,ret)
    
    if len(TabCordsLicensesDetected) > 0:
       
        print(License + " findcontours detected with Otsu's thresholding of cv2 and THRESH_TOZERO ")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])
    TabCordsLicensesDetected=DetectLicenseInCar2(gray1)
    if len(TabCordsLicensesDetected) > 0:
        print(License + " haarcode detected with Otsu's thresholding of cv2 and THRESH_TOZERO ")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
  
    
   
    ####################################################
    # experimental formula based on the brightness
    # of the whole image 
    ####################################################
    
    SumBrightness=np.sum(gray)  
    threshold=(SumBrightness/177600.00) 
    
    #####################################################
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TRUNC)
    TabCordsLicensesDetected=DetectLicenseInCar1(gray1,ret)
    
    if len(TabCordsLicensesDetected) > 0:
       
        print(License + " findcontours detected with Brightness and THRESH_TRUNC ")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])
   
    TabCordsLicensesDetected=DetectLicenseInCar2(gray1)
    print(License + " haarcode detected with Brightness and THRESH_TRUNC ")    
    if len(TabCordsLicensesDetected) > 0:
       for i in range(len(TabCordsLicensesDetected)):
         TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
    
    gray1 = cv2.medianBlur(gray,3)  
    ret, gray1=cv2.threshold(gray1,threshold,255,  cv2.THRESH_TRUNC)
    TabCordsLicensesDetected=DetectLicenseInCar1(gray1,ret)
    
    if len(TabCordsLicensesDetected) > 0:
       
        print(License + " findcontours detected with Brightness and THRESH_TRUNC and medianBlur")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i]) 
    TabCordsLicensesDetected=DetectLicenseInCar2(gray1)
    if len(TabCordsLicensesDetected) > 0:
        for i in range(len(TabCordsLicensesDetected)):
          print(License + " haarcode detected with Brightness and THRESH_TRUNC and medianBlur")  
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
   
    gray1 = cv2.GaussianBlur(gray,(3,3), sigmaX=0, sigmaY=0)  
    ret, gray1=cv2.threshold(gray1,threshold,255,  cv2.THRESH_TRUNC)
    TabCordsLicensesDetected=DetectLicenseInCar1(gray1,ret)
    
    if len(TabCordsLicensesDetected) > 0:
       
        print(License + " findcontours detected with Brightness and THRESH_TRUNC and GaussianBlur")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i]) 
    TabCordsLicensesDetected=DetectLicenseInCar2(gray1)
    if len(TabCordsLicensesDetected) > 0:
        print(License + " haarcode detected with Brightness and THRESH_TRUNC and GaussianBlur")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])   
   
    gray1= cv2.bilateralFilter(gray,3, 75, 75)  
    ret, gray1=cv2.threshold(gray1,threshold,255,  cv2.THRESH_BINARY)
    TabCordsLicensesDetected=DetectLicenseInCar1(gray1,ret)
    if len(TabCordsLicensesDetected) > 0:
       
        print(License + " findcontours  detected with Brightness and THRESH_BINARY and BilateralFilter")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i]) 
    TabCordsLicensesDetected=DetectLicenseInCar2(gray1)
    if len(TabCordsLicensesDetected) > 0:
        print(License + " haarcode  detected with Brightness and THRESH_BINARY and BilateralFilter")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
   
    gray1= cv2.bilateralFilter(gray,3, 75, 75)
    ret, gray1=cv2.threshold(gray1,threshold,255,  cv2.THRESH_TOZERO) 
    TabCordsLicensesDetected=DetectLicenseInCar1(gray1,ret)
    
    if len(TabCordsLicensesDetected) > 0:
       
        print(License + " findcontours detected with Brightness and THRESH_TOZERO and BilateralFilter")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i]) 
    TabCordsLicensesDetected=DetectLicenseInCar2(gray1)
    if len(TabCordsLicensesDetected) > 0:
        print(License + " haarcode detected with Brightness and THRESH_TOZERO and BilateralFilter")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
    
    
    threshold=ThresholdStable(gray)
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TRUNC) 
    TabCordsLicensesDetected=DetectLicenseInCar1(gray1,ret)
    if len(TabCordsLicensesDetected) > 0:
       
        print(License + " findcontours detected with Stable and THRESH_TRUNC")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i]) 
    TabCordsLicensesDetected=DetectLicenseInCar2(gray1)
    if len(TabCordsLicensesDetected) > 0:
        print(License + " haarcode detected with Stable and THRESH_TRUNC")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
    
    gray1= cv2.bilateralFilter(gray,3, 75, 75) 
    ret, gray1=cv2.threshold(gray1,threshold,255,  cv2.THRESH_BINARY) 
    TabCordsLicensesDetected=DetectLicenseInCar1(gray1,ret)
    if len(TabCordsLicensesDetected) > 0:
       
        print(License + " findcontours detected with Stable and THRESH_BINARY and BilateralFilter")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])
    TabCordsLicensesDetected=DetectLicenseInCar2(gray1)
    if len(TabCordsLicensesDetected) > 0:
        print(License + " haarcode detected with Stable and THRESH_BINARY and BilateralFilter")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
    
  
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO) 
    TabCordsLicensesDetected=DetectLicenseInCar1(gray1,ret)   
    if len(TabCordsLicensesDetected) > 0:
       
        print(License + "  findcontours detected with Stable and THRESH_TOZERO ")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])   
    TabCordsLicensesDetected=DetectLicenseInCar2(gray1)
    if len(TabCordsLicensesDetected) > 0:
        print(License + "  haarcode detected with Stable and THRESH_TOZERO ")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
  
    #https://en.wikipedia.org/wiki/Kernel_(image_processing)
    #https://stackoverflow.com/questions/4993082/how-can-i-sharpen-an-image-in-opencv, respuesta 66
   
    for z in range(4,8):
    
       kernel = np.array([[0,-1,0], [-1,z,-1], [0,-1,0]])
       gray1 = cv2.filter2D(gray, -1, kernel)
       TabCordsLicensesDetected=DetectLicenseInCar1(gray1,-1)   
       if len(TabCordsLicensesDetected) > 0:
          
           #print(License + "  detected with Sharpen filter ")
           for i in range(len(TabCordsLicensesDetected)):
             print(License + " findcontours detected with Sharpen filter z=" + str(z))
             TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])       
       TabCordsLicensesDetected=DetectLicenseInCar2(gray1)
       if len(TabCordsLicensesDetected) > 0:
        print(License + " haarcode detected with Sharpen filter z=" + str(z))
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
    #https://en.wikipedia.org/wiki/Kernel_(image_processing)
    #https://en.wikipedia.org/wiki/Kernel_(image_processing)
    
    gray2= cv2.bilateralFilter(gray,3, 75, 75)
    for z in range(5,11):
       kernel = np.array([[-1,-1,-1], [-1,z,-1], [-1,-1,-1]])
       gray1 = cv2.filter2D(gray2, -1, kernel)
       TabCordsLicensesDetected=DetectLicenseInCar1(gray1,-1)   
       if len(TabCordsLicensesDetected) > 0:
          
           #print(License + "  detected with Sharpen filter modified")
           for i in range(len(TabCordsLicensesDetected)):
             print(License + " findcontours detected with Sharpen filter modified z="+str(z))
             TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i]) 
       TabCordsLicensesDetected=DetectLicenseInCar2(gray)
       if len(TabCordsLicensesDetected) > 0:
        print(License + " haarcode detected with Sharpen filter modified z="+str(z))
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
      
    gray1= cv2.bilateralFilter(gray,3, 75, 75) 
    gray1 = cv2.Canny(gray1,200,255)    
    #ret, gray1 = cv2.threshold(gray, 240, 255, 1)
    gray1= 255 - gray1
    TabCordsLicensesDetected=DetectLicenseInCar1(gray1,-1)   
    if len(TabCordsLicensesDetected) > 0:
       
        #print(License + "  detected with canny filter ")
        for i in range(len(TabCordsLicensesDetected)):
          print(License + " findcontours  detected with canny filter ")
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i]) 
    TabCordsLicensesDetected=DetectLicenseInCar2(gray1)
    if len(TabCordsLicensesDetected) > 0:
        
        for i in range(len(TabCordsLicensesDetected)):
          print(License + " haarcode  detected with canny filter ")
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
    
    #https://www.sicara.fr/blog-technique/2019-03-12-edge-detection-in-opencv
    gray1 = cv2.Canny(gray,60,120)    
   
    
    TabCordsLicensesDetected=DetectLicenseInCar1(gray1,-1)   
    if len(TabCordsLicensesDetected) > 0:
       
        #print(License + "  detected with canny filter with high thresolds")
        for i in range(len(TabCordsLicensesDetected)):
          print(License + " findcontours  detected with canny filter with high thresolds")
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])
    TabCordsLicensesDetected=DetectLicenseInCar2(gray1)
    if len(TabCordsLicensesDetected) > 0:
        print(License + " haarcode  detected with canny filter with high thresolds")
        for i in range(len(TabCordsLicensesDetected)):
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
    
    # Smoothing without removing edges.
    gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)
    gray1 = cv2.Canny(gray_filtered,60,120)    
   
    TabCordsLicensesDetected=DetectLicenseInCar1(gray1,-1)   
    if len(TabCordsLicensesDetected) > 0:
       
        #print(License + "  detected with canny filter with high thresolds and removing edges")
        for i in range(len(TabCordsLicensesDetected)):
          print(License + " findcontours  detected with canny filter with high thresolds and removing edges")
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])
    TabCordsLicensesDetected=DetectLicenseInCar2(gray1)
    if len(TabCordsLicensesDetected) > 0:
        for i in range(len(TabCordsLicensesDetected)):
          print(License + " haarcode  detected with canny filter with high thresolds and removing edges")
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
    
   
    gray1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    cv2.THRESH_BINARY,11,2) 
    TabCordsLicensesDetected=DetectLicenseInCar1(gray1, -1)
    
    if len(TabCordsLicensesDetected) > 0:
       
        #print(License + "  detected with adaptive Threshold Mean and THRESH_BINARY" )
        for i in range(len(TabCordsLicensesDetected)):
          print(License + " findcontours  detected with adaptive Threshold Mean and THRESH_BINARY" )  
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])
    TabCordsLicensesDetected=DetectLicenseInCar2(gray1)
    if len(TabCordsLicensesDetected) > 0:
        for i in range(len(TabCordsLicensesDetected)):
          print(License + " haarcode  detected with adaptive Threshold Mean and THRESH_BINARY" )   
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
   
    
   
    gray_img_clahe=ApplyCLAHE(gray)
    
          
    th=OTSU_Threshold(gray_img_clahe)
    
    max_val=255
    ret, o1 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_BINARY)
    TabCordsLicensesDetected=DetectLicenseInCar1(o1, ret)
    if len(TabCordsLicensesDetected) > 0:
        
        for i in range(len(TabCordsLicensesDetected)):
          print(License + " findcontours detected with CLAHE and THRESH_BINARY" )  
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])
    TabCordsLicensesDetected=DetectLicenseInCar2(o1)
    if len(TabCordsLicensesDetected) > 0:
        for i in range(len(TabCordsLicensesDetected)):
          print(License + " haarcode detected with CLAHE and THRESH_BINARY" )   
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
    
    
    ret, o3 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO)
    TabCordsLicensesDetected=DetectLicenseInCar1(o3, ret)
    if len(TabCordsLicensesDetected) > 0:
      #print(License[i] + "  detected with CLAHE and THRESH_TOZERO" )
      for i in range(len(TabCordsLicensesDetected)):
        print(License + " findcontours detected with CLAHE and THRESH_TOZERO" )  
        TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i]) 
    TabCordsLicensesDetected=DetectLicenseInCar2(o3)
    if len(TabCordsLicensesDetected) > 0:
        for i in range(len(TabCordsLicensesDetected)):
          print(License + " haarcode detected with CLAHE and THRESH_TOZERO" )
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
        
    
    ret, o5 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TRUNC)
    TabCordsLicensesDetected=DetectLicenseInCar1(o5, ret)
    if len(TabCordsLicensesDetected) > 0:
     #print(License[i] + "  detected with CLAHE and THRESH_TRUNC" )
     for i in range(len(TabCordsLicensesDetected)):
       print(License + " findcontours detected with CLAHE and THRESH_TRUNC" )
       TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i]) 
    TabCordsLicensesDetected=DetectLicenseInCar2(o5)
    if len(TabCordsLicensesDetected) > 0:
        for i in range(len(TabCordsLicensesDetected)):
          print(License + " haarcode detected with CLAHE and THRESH_TRUNC" )  
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
    
    ret, o6 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_OTSU)
    TabCordsLicensesDetected=DetectLicenseInCar1(o6, ret)
    if len(TabCordsLicensesDetected) > 0:
     #print(License[i] + "  detected with CLAHE and THRESH_OTSU" )
     for i in range(len(TabCordsLicensesDetected)):
       print(License + " findcontours detected with CLAHE and THRESH_OTSU" )  
       TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i]) 
    TabCordsLicensesDetected=DetectLicenseInCar2(o6)
    if len(TabCordsLicensesDetected) > 0:
        for i in range(len(TabCordsLicensesDetected)):
          print(License + " haarcode detected with CLAHE and THRESH_OTSU" )  
          TabCordsLicensesDetectedTotal.append(TabCordsLicensesDetected[i])  
   
    return TabCordsLicensesDetectedTotal

def ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text):
    
    SwFounded=0
    for i in range( len(TabLicensesFounded)):
        if text==TabLicensesFounded[i]:
            ContLicensesFounded[i]=ContLicensesFounded[i]+1
            SwFounded=1
            break
    if SwFounded==0:
       TabLicensesFounded.append(text) 
       ContLicensesFounded.append(1)
    return TabLicensesFounded, ContLicensesFounded
#########################################################################
def FindLicenseNumber (gray_input, x_offset, y_offset,  License, x_resize, y_resize, \
                       Resize_xfactor, Resize_yfactor,  TabLicensesDetected,TabLicensesFounded, ContLicensesFounded):
#########################################################################
# adapted from:
#  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
# by Alfonso Blanco García
####
 
    #print ("LLEGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    Cont=0
    for w in range(len(TabLicensesDetected)):
       
      
        lCords=TabLicensesDetected[w]
        x=lCords[0]
        y=lCords[1]
        w=lCords[2]
        h=lCords[3]
        
        #print(gray_input.shape)
        #print("x="+ str(x))
        #print("y="+ str(y))
        #print("w="+str(w))
        #print("h="+ str(h))
        #print("x_offset"+ str(x_offset))
        #print("y_offset"+ str(y_offset))
       
        gray=gray_input[y:y+h,x+ y_offset:x+w+ x_offset]
        
       
        Cont=Cont+1
        if Cont > 10: 
           print (" The process takes too long, it ends")
           return TabLicensesFounded, ContLicensesFounded
        #print("crop : " + str(y) + " : " + str(y+h) + " , " + str(x+ y_offset) + " : "+ str(x+w+ x_offset))
        
       
        if len(gray)==0:
        #   print(Licenses[i]+ "REMAINS BLANK WHEN BOXING REGISTRATION")
           continue
        if len(gray[0])==0:
        #   print(Licenses[i]+ "REMAINS BLANK WHEN BOXING REGISTRATION")
           continue
        if gray is None:
            print(License[i]+ "Es NONE")
            continue
        
        print(Licenses[i] + " in process")
        
        
        TotHits=0
        
        X_resize=x_resize
        Y_resize=y_resize
        
        #cv2.imshow('gray',gray)
        #cv2.waitKey()  
        gray=cv2.resize(gray,None,fx=Resize_xfactor,fy=Resize_yfactor,interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.resize(gray, (X_resize,Y_resize), interpolation = cv2.INTER_AREA)
        
        rotation, spectrum, frquency =GetRotationImage(gray)
        rotation=90 - rotation
       
        if (rotation > 0 and rotation < 30)  or (rotation < 0 and rotation > -30):
          
            gray=imutils.rotate(gray,angle=rotation)
       
        
       
        #   Otsu's thresholding
        ret2,gray1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(gray1, lang='eng',  \
        config='--psm 6 --oem 3')
       
        text = ''.join(char for char in text if char.isalnum())
        if Detect_Spanish_LicensePlate(text)== 1:
            ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)
            if text==Licenses[i]:
                print(text + "  Hit with Otsu's thresholding of cv2 and THRESH_BINARY" )
                TotHits=TotHits+1
            else:
                print(Licenses[i] + " detected as "+ text)
             
        # INV options are not considered as Spanish licenses plates are not INV
       
        #   Otsu's thresholding
        ret2,gray1 = cv2.threshold(gray,0,255,cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(gray1, lang='eng',  \
            config='--psm 6 --oem 3')
           
        text = ''.join(char for char in text if char.isalnum())
        
        if Detect_Spanish_LicensePlate(text)== 1:
            ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)
            if text==Licenses[i]:
                print(text + "  Hit with Otsu's thresholding of cv2 and THRESH_TRUNC" )
                TotHits=TotHits+1
            else:
                print(Licenses[i] + " detected as "+ text)
             
        #   Otsu's thresholding
        
        ret2,gray1 = cv2.threshold(gray,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(gray1, lang='eng',  \
            config='--psm 6 --oem 3')
       
        text = ''.join(char for char in text if char.isalnum())
        
        if Detect_Spanish_LicensePlate(text)== 1:
            ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            if text==Licenses[i]:
                print(text + "  Hit with Otsu's thresholding of cv2 and THRESH_TOZERO" )
                TotHits=TotHits+1
            else:
                print(Licenses[i] + " detected as "+ text)                    
        
      
       
        threshold=ThresholdStable(gray)
        ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TRUNC) 
        text = pytesseract.image_to_string(gray1, lang='eng',  \
        config='--psm 6 --oem 3')
        text = ''.join(char for char in text if char.isalnum())
       
        if Detect_Spanish_LicensePlate(text)== 1:
            ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)    
            if text==Licenses[i]:
                print(text + "  Hit with Stable and THRESH_TRUNC" )
                TotHits=TotHits+1
            else:
                print(Licenses[i] + " detected as "+ text)         
           
        gray1= cv2.bilateralFilter(gray,3, 75, 75) 
        ret, gray1=cv2.threshold(gray1,threshold,255,  cv2.THRESH_BINARY) 
        text = pytesseract.image_to_string(gray1, lang='eng',  \
        config='--psm 6 --oem 3')
        text = ''.join(char for char in text if char.isalnum())
       
        if Detect_Spanish_LicensePlate(text)== 1:
            ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)    
            if text==Licenses[i]:
                print(text + "  Hit with Stable and THRESH_BINARY" )
                TotHits=TotHits+1
            else:
                print(Licenses[i] + " detected as "+ text)    
           
                   
        ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO) 
        text = pytesseract.image_to_string(gray1, lang='eng',  \
        config='--psm 6 --oem 3')
        text = ''.join(char for char in text if char.isalnum())
        
        if Detect_Spanish_LicensePlate(text)== 1:
            ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            if text==Licenses[i]:
               print(text + "  Hit with Stable and THRESH_TOZERO" )
               TotHits=TotHits+1
            else:
                print(Licenses[i] + " detected as "+ text) 
      
        ####################################################
        # experimental formula based on the brightness
        # of the whole image 
        ####################################################
        
        SumBrightness=np.sum(gray)  
        threshold=(SumBrightness/177600.00) 
        
        #####################################################
        ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TRUNC)
        text = pytesseract.image_to_string(gray1, lang='eng',  \
        config='--psm 6 --oem 3')
        text = ''.join(char for char in text if char.isalnum())
        
        if Detect_Spanish_LicensePlate(text)== 1:
            ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            if text==Licenses[i]:
               print(text + "  Hit with Brightness and THRESH_TRUNC" )
               TotHits=TotHits+1
            else:
                print(Licenses[i] + " detected as "+ text) 
                
        ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO)
        text = pytesseract.image_to_string(gray1, lang='eng',  \
        config='--psm 6 --oem 3')
        text = ''.join(char for char in text if char.isalnum())
        
        if Detect_Spanish_LicensePlate(text)== 1:
            ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            if text==Licenses[i]:
               print(text + "  Hit with Brightness and THRESH_TOZERO" )
               TotHits=TotHits+1
            else:
                print(Licenses[i] + " detected as "+ text)
                
        ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(gray1, lang='eng',  \
        config='--psm 6 --oem 3')
        text = ''.join(char for char in text if char.isalnum())
        
        if Detect_Spanish_LicensePlate(text)== 1:
            ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            if text==Licenses[i]:
               print(text + "  Hit with Brightness and THRESH_BINARY" )
               TotHits=TotHits+1
            else:
                print(Licenses[i] + " detected as "+ text) 
                
        ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(gray1, lang='eng',  \
        config='--psm 6 --oem 3')
        text = ''.join(char for char in text if char.isalnum())
        
        if Detect_Spanish_LicensePlate(text)== 1:
            ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            if text==Licenses[i]:
               print(text + "  Hit with Brightness and THRESH_OTSU" )
               TotHits=TotHits+1
            else:
                print(Licenses[i] + " detected as "+ text) 
       
        ################################################################
        # Filters eliminated to reduce time processing
        ################################################################
       
        """
        gray_img_clahe=ApplyCLAHE(gray)    
        th=OTSU_Threshold(gray_img_clahe)
        max_val=255
        ret, o1 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(o1, lang='eng',  \
        config='--psm 6 --oem 3')
        text = ''.join(char for char in text if char.isalnum())
        
        if Detect_Spanish_LicensePlate(text)== 1:
            ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            if text==Licenses[i]:
               print(text + "  Hit with CLAHE and THRESH_BINARY" )
               TotHits=TotHits+1
            else:
                print(Licenses[i] + " detected as "+ text)  
      
        ret, o2 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO)
        text = pytesseract.image_to_string(o2, lang='eng',  \
        config='--psm 6 --oem 3')
        text = ''.join(char for char in text if char.isalnum())
        
        if Detect_Spanish_LicensePlate(text)== 1:
            ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            if text==Licenses[i]:
               print(text + "  Hit with  CLAHE and TRESH_TOZERO " )
               TotHits=TotHits+1
            else:
                print(Licenses[i] + " detected as "+ text)  
        
        ret, o3 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TRUNC)
        text = pytesseract.image_to_string(o3, lang='eng',  \
        config='--psm 6 --oem 3')
        text = ''.join(char for char in text if char.isalnum())
        
        if Detect_Spanish_LicensePlate(text)== 1:
            ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            if text==Licenses[i]:
               print(text + "  Hit with  CLAHE and TRESH_TRUNC " )
               TotHits=TotHits+1
            else:
                print(Licenses[i] + " detected as "+ text)  
        
        ret, o4 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(o3, lang='eng',  \
        config='--psm 6 --oem 3')
        text = ''.join(char for char in text if char.isalnum())
        
        if Detect_Spanish_LicensePlate(text)== 1:
            ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            if text==Licenses[i]:
               print(text + "  Hit with  CLAHE and TRESH_OTSU " )
               TotHits=TotHits+1
            else:
                print(Licenses[i] + " detected as "+ text)  
        
        
        for z in range(4,8):
        
           kernel = np.array([[0,-1,0], [-1,z,-1], [0,-1,0]])
           gray1 = cv2.filter2D(gray, -1, kernel)
                  
           text = pytesseract.image_to_string(gray1, lang='eng',  \
           config='--psm 6 --oem 3 -c tessedit_char_whitelist= ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 ')
          
           text = ''.join(char for char in text if char.isalnum())
           if Detect_Spanish_LicensePlate(text)== 1:
               ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
               if text==Licenses[i]:
                  print(text +  "  Hit with Sharpen filter"  )
                  TotHits=TotHits+1
               else:
                   print(Licenses[i] + " detected as "+ text) 
        gray2= cv2.bilateralFilter(gray,3, 75, 75)
        for z in range(5,11):
           kernel = np.array([[-1,-1,-1], [-1,z,-1], [-1,-1,-1]])
           gray1 = cv2.filter2D(gray2, -1, kernel)
           text = pytesseract.image_to_string(gray1, lang='eng',  \
           config='--psm 6 --oem 3 -c tessedit_char_whitelist= ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 ')
          
           text = ''.join(char for char in text if char.isalnum())
           if Detect_Spanish_LicensePlate(text)== 1:
               ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
               if text==Licenses[i]:
                  print(text +  "  Hit with Sharpen filter modified"  )
                  TotHits=TotHits+1
               else:
                   print(Licenses[i] + " detected as "+ text) 
              
    """     
       
    return TabLicensesFounded, ContLicensesFounded

 ########################################################################
def loadimagesRoboflow (dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     
     images = []
     Licenses=[]
    
     
     print("Reading imagenes from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
         
         
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                 License=filename[:len(filename)-4]
                 
                 # Spanish license plate is NNNNAAA
                 if Detect_Spanish_LicensePlate(License)== -1: continue
                 image = cv2.imread(filepath)
                 
               
                 #Color Balance
                #https://blog.katastros.com/a?ID=01800-4bf623a1-3917-4d54-9b6a-775331ebaf05
                
                 img =  image
                    
                 r, g, b = cv2.split(img)
                
                 r_avg = cv2.mean(r)[0]
                
                 g_avg = cv2.mean(g)[0]
                
                 b_avg = cv2.mean(b)[0]
                
                 
                 # Find the gain occupied by each channel
                
                 k = (r_avg + g_avg + b_avg)/3
                
                 kr = k/r_avg
                
                 kg = k/g_avg
                
                 kb = k/b_avg
                
                 
                 r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
                
                 g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
                
                 b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
                
                 
                 balance_img = cv2.merge([b, g, r])
                 
                 image=balance_img
                
                 images.append(image)
                 Licenses.append(License)
                 
                 
                
                 Cont+=1
     
     return images, Licenses

###########################################################
# MAIN
##########################################################

imagesComplete, Licenses=loadimagesRoboflow(dirname)

print("Number of imagenes : " + str(len(imagesComplete)))

print("Number of   licenses : " + str(len(Licenses)))

TotHits=0
TotFailures=0
TotNoDetect=0

NumberImageOrder=0

with open( "LicenseResults.txt" ,"w") as  w:

    for i in range (len(imagesComplete)):
        
            #if Licenses[i] < "LIMES":
            #  print ("Salta "+ Licenses[i]) 
            #   continue
           
            
            NumberImageOrder=NumberImageOrder+1
            
            gray=imagesComplete[i]
            
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                  
            License=Licenses[i]
            
            TabCordsLicencesDetected=DetectLicenseInCar(gray)
            
            
            if len(TabCordsLicencesDetected)==0:
                print(License + " not even detected any license plate")
                print("")
                TotNoDetect=TotNoDetect+1
                lineaw=[]
                lineaw.append(License) 
                lineaw.append("-------")
                lineaWrite =','.join(lineaw)
                lineaWrite=lineaWrite + "\n"
                w.write(lineaWrite)
                continue
           
           
          
            x_off=0
            y_off=0
            x_resize=220
            y_resize=70
            
            Resize_xfactor=1.78
            Resize_yfactor=1.78
            
            ContLoop=0
            
            TabLicensesFounded=[]
            ContLicensesFounded=[]
            
            while ( ContLoop <5):
                ContLoop+=1
                TabLicensesFounded, ContLicensesFounded= FindLicenseNumber (gray, x_off, y_off,  License, x_resize, y_resize, \
                                   Resize_xfactor, Resize_yfactor, TabCordsLicencesDetected,TabLicensesFounded, ContLicensesFounded )
               
               
                if ContLoop==1 :
                    x_off=15
                    y_off=15
                    #print("SECOND TRY")
                    
                if ContLoop==2 :
                     x_off=5
                     y_off=5
                     #print("THIRD TRY")
                     
                if ContLoop==3 :
                       x_off=10
                       y_off=10
                       #print("FOURTH TRY")
                if ContLoop==4:
                      x_off=-15
                      y_off=15
                      #Resize_xfactor=2.0
                      #Resize_yfactor=2.0
                      #print("FIFTH TRY")
                
                if ContLoop==5:
                       x_off=-10
                       y_off=10
                       Resize_xfactor=1.78
                       Resize_yfactor=1.78
                       BilateralOption=1
                       #print("SIXTH ATTEMPT")     
               
            ymax=-1
            contmax=0
            licensemax=""
          
            for y in range(len(TabLicensesFounded)):
                if ContLicensesFounded[y] > contmax:
                    contmax=ContLicensesFounded[y]
                    licensemax=TabLicensesFounded[y]
            
            if licensemax == License:
               print(License + " correctly recognized") 
               TotHits+=1
            else:
                print(License + " Detected but not correctly recognized")
                TotFailures +=1
            print ("")  
            lineaw=[]
            lineaw.append(License) 
            lineaw.append(licensemax)
            lineaWrite =','.join(lineaw)
            lineaWrite=lineaWrite + "\n"
            w.write(lineaWrite)
             
           
                
      
print("")           
print("Total Hits = " + str(TotHits ))
print("Total Failures = " + str(TotFailures ))
print("Total Not detected = " + str(TotNoDetect ))
      
                 
        