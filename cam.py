#-*- coding:utf-8 -*-
import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt

chips = cv2.imread('E:\program\cont\sumple\chip.png')#画像読み込み
chips_gray = cv2.cvtColor(chips, cv2.COLOR_BGR2GRAY)#グレースケール化
chips_preprocessed = cv2.GaussianBlur(chips_gray, (5, 5), 0)#平滑化
_, chips_binarya = cv2.threshold(chips_preprocessed, 230, 255, cv2.THRESH_BINARY)#2値化
chips_binary = cv2.bitwise_not(chips_binarya)

_, chips_contours, _ = cv2.findContours(chips_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#輪郭
chips_and_contours = np.copy(chips)
min_chip_area = 60
large_contours = [cnt for cnt in chips_contours if cv2.contourArea(cnt) > min_chip_area]
#cv2.contourArea(cnt)でx個目の輪郭の面積を判断

bounding_img = np.copy(chips)
for contour in large_contours:
	rect = cv2.minAreaRect(contour)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	cgx = int(rect[0][0])
	cgy = int(rect[0][1])
	leftx = int(cgx - (rect[1][0]/2.0))
	lefty = int(cgy - (rect[1][1]/2.0))
	angle = round(rect[2],1)
	cv2.drawContours(bounding_img,[box],0,(0,0,255),2)
	cv2.circle(bounding_img,(cgx,cgy), 10, (255,0,0), -1)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(bounding_img,'Rot: '+str(angle)+'[deg]',(leftx,lefty), font, 0.7, (0,0,0),2,cv2.LINE_AA)
# plt.imshow(chips_preprocessed)
# plt.imshow(bounding_img)

print len(chips_contours)
cv2.imshow("Show BINARIZATION Image",chips_binarya)
plt.axis("off")
cv2.waitKey(0)
cv2.imshow("Show BINARIZATION Image",chips_binary)
cv2.waitKey(0)
plt.imshow(bounding_img)
plt.axis("off")
plt.show()

#
# large_contours = []
# for cnt in chips_contours
# 	if cv2.contourArea(cnt) > min_chip_area
# 		extension_2.append(cnt)
