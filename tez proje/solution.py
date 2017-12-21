import cv2
import numpy as np

img=cv2.imread("images/uv8.jpeg",1)
img_to_show=img.copy()

#eliminating red channel
img[:,:,2] = 0

#upper and lower limits for mask
lower = np.array([0, 150, 0])
upper = np.array([255, 255, 0])

#creating the mask
mask = cv2.inRange(img, lower, upper)

#applying the mask
result = cv2.bitwise_and(img, img, mask = mask)

#applying Gaussian blur to get a clear boundaries 
GaussianMedianBlur = cv2.GaussianBlur(result,(5,5),0)

#converting to gray scale to apply threshold
gray=cv2.cvtColor(GaussianMedianBlur, cv2.COLOR_BGR2GRAY)

#applying threshold, used Otsu threshold to getting an adaptative threshold value 
ret,threshFromHist = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#applying closing to draw counters more accurate
kernel = np.ones((1,1), np.uint8)
closing = cv2.morphologyEx(threshFromHist, cv2.MORPH_CLOSE, kernel)

#finding and drawing counters
binary_filtered, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_to_show, contours, -1, (0,0,255), 3)

print (ret)
#creating windows and showing images
cv2.namedWindow('Original Image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original Image', 600,600)

cv2.namedWindow('Binary Image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Binary Image', 600,600)

cv2.namedWindow('bitwise',cv2.WINDOW_NORMAL)
cv2.resizeWindow('bitwise', 600,600)


cv2.imshow("Binary Image",threshFromHist)
cv2.imshow("Original Image",img_to_show)
cv2.imshow("bitwise",result)


cv2.waitKey(0)
cv2.destroyAllWindows()