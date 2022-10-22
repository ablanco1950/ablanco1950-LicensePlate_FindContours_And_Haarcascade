# ablanco1950-LicensePlate_FindContours_And_Haarcascade
Using together cv2's findcontours and Haarcascade license plate detection together with the application of an extensive set of filters
applying on a restricted set of car registration plates of known format: the Spanish ones with NNNNAAA format, 14 are obtained
 hits in a sample of 21 photos (more than 66% hits). No need to label photos but the proccess takes a long  time.

Requirements:

pytesseract

numpy

cv2

re

imutils

skimage (is scikit-image)

In the download directory you should find the downloaded test6Training.zip and must unzip folder: test6Training with all its subfolders, containing the images for the test. This directory must be in the same directory where you program LicensePlateFindContours.py ( unziping may create two directories with you name test6Training and the images may not be founded when executing it, it would be necessary copy of inner directory test6Training in the same directory where is LicensePlateFindContours.py)

from the download directory,

 run:

LicensePlateFindContoursHaarcascade_SpanishLicense_WithMaxFilters.py

The types of messages presented are:

The name of the filter that has detected a possible license plate indicating whether it has been done by findcontours or haarcascade

Hit followed by the name of the filter that, applied, has resulted in pytesseract detecting the correct license plate number. That is posible because le name of each image is de number of license plate

Detected, pyteseract has decrypted a license plate number that does not match the true one, which is detected because
 the true registration number is part of the name of the jpg file that constitutes the photo
 
Messages indicated that the system is not dead, but in process, and the termination of processes due to excess time.

At the end of the processing of each license plate, the number plate that has been detected the most is assigned to the image

At the end of the whole process, the LicenseResults.txt file is obtained with the relationship between the true car license plate and the detected license plate for each car.

==========

Is also included:

LicensePlateFindContoursHaarcascade_InternationalLicense_WithMaxFilters.py

That deals with all types of license plates, not just Spanish ones. By not filtering car license plates by car license plate format,
the hit rate drops to 36%.

For example: the car license plate 2537JJD appears misrecognized as 2537JJ0 (confuses the final D with a 0) when it would not have been recognized
by the program that only deals with Spanish license plates by filtering the NNNNAAA format through it and verifying that in the last position there is a zero and not an alphabetic character.

Other projects presented on car license plate recognition:

https://github.com/ablanco1950/LicensePlate_FindContours

With labeled images:

https://github.com/ablanco1950/LicensePlate_CLAHE

https://github.com/ablanco1950/LicensePlateImage_ThresholdFiltered


References:

https://github.com/spmallick/mallick_cascades/tree/master/haarcascades, the well known haarcascade_russian_plate_number.xml

https://www.roboflow.com

https://blog.katastros.com/a?ID=01800-4bf623a1-3917-4d54-9b6a-775331ebaf05

https://github.com/ablanco1950/LicensePlate_CLAHE

https://gist.github.com/endolith/334196bac1cac45a4893#

https://stackoverflow.com/questions/46084476/radon-transformation-in-python

https://learnopencv.com/otsu-thresholding-with-opencv/

https://towardsdatascience.com/image-enhancement-techniques-using-opencv-and-python-9191d5c30d45

https://stackoverflow.com/questions/64530229/how-do-i-get-tesseract-to-read-the-license-plate-in-the-this-python-opencv-proje

https://stackoverflow.com/questions/21324950/how-can-i-select-the-best-set-of-parameters-in-the-canny-edge-detection-algorith

https://en.wikipedia.org/wiki/Kernel_(image_processing)

https://stackoverflow.com/questions/4993082/how-can-i-sharpen-an-image-in-opencv, answer 66

https://en.wikipedia.org/wiki/Kernel_(image_processing)

https://www.sicara.fr/blog-technique/2019-03-12-edge-detection-in-opencv

https://www.aprendemachinelearning.com/clasacion-de-images-en-python/
