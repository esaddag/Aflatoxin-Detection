import cv2
import numpy as np


#img=cv2.imread("images/uv9.jpeg", 0)

img=cv2.imread("images/uv7.jpeg", 1)



 
#convert 1D array to 3D, then convert it to HSV and take the first element 
# this will be same as shown in the above figure [65, 229, 158]
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
 
minHSV = np.array([0, 0, 0]	)
maxHSV = np.array([1000, 1000, 1000])
 
maskHSV = cv2.inRange(hsv, minHSV, maxHSV)
resultHSV = cv2.bitwise_and(hsv, hsv, mask = maskHSV)
 
 

cv2.namedWindow('Result BGR',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Result BGR', 600,600)

cv2.namedWindow('Result HSV',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Result HSV', 600,600)

cv2.namedWindow('Original Image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original Image', 600,600)

cv2.imshow("Original Image",hsv)
cv2.imshow("Result BGR", maskHSV)
cv2.imshow("Result HSV", resultHSV)


cv2.waitKey(0)
cv2.destroyAllWindows()