import cv2
import numpy as np




img = cv2.imread('images/uv2.jpeg',0)

#MedianBlur= cv2.medianBlur(img,5)

GaussianMedianBlur = cv2.GaussianBlur(img,(5,5),0)


#thresholding
ret,thresh = cv2.threshold(GaussianMedianBlur,160,255,cv2.THRESH_BINARY)



#closing
kernel = np.ones((1,1), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)







binary_filtered, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (50,50,50), 1)

two=np.concatenate((img, binary_filtered), axis=1)
cv2.imshow('binary_filtered',binary_filtered)
cv2.imshow("image_with_contours",two)



cv2.waitKey(0)
cv2.destroyAllWindows()