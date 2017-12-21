import cv2
import numpy as np



img=cv2.imread("images/uv8.jpeg", 0)

img2=img

GaussianBlured = cv2.GaussianBlur(img,(5,5),0)
ret,threshFromHist = cv2.threshold(GaussianBlured,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#MedianBlured = cv2.medianBlur(img,5)
#adaptive_thresh=cv2.adaptiveThreshold(MedianBlured,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)

#closing
kernel = np.ones((1,1), np.uint8)
closing = cv2.morphologyEx(threshFromHist, cv2.MORPH_CLOSE, kernel)



binary_filtered, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0,0,255), 3)

cv2.namedWindow('image_with_contours',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image_with_contours', 600,600)

cv2.namedWindow('binary_filtered',cv2.WINDOW_NORMAL)
cv2.resizeWindow('binary_filtered', 600,600)

cv2.imshow('binary_filtered',binary_filtered)
cv2.imshow("image_with_contours",img)




cv2.waitKey(0)
cv2.destroyAllWindows()