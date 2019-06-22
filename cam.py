#-*- coding:utf-8 -*-
import cv2
import numpy as np
# thresh=250#閾値
# maxpix=255#全て白にする
imgs = cv2.imread("E:\program\cont\loccc.jpg")
imgss = cv2.cvtColor(imgs, cv2.COLOR_BGR2HSV)#元画像をHSVに変換
#imgs = cv2.adaptiveThreshold(imgs,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3)
# imgs = cv2.Canny(imgs, 10, 100)
kernel = np.ones((3,3),np.uint8)#輪郭をぼかす
imgs = cv2.morphologyEx(imgs, cv2.MORPH_OPEN, kernel)
nomal_color= np.uint8([[[0,255,255]]])#色指定
high_low_color=[10,150,255]#[Hの+-,SV値の濃,SV値の薄]
color_hsv = cv2.cvtColor(nomal_color,cv2.COLOR_BGR2HSV)
hsv_h = color_hsv[0][0][0]
hsv_h=int(hsv_h)
hsv_hp=hsv_h+high_low_color[0]
hsv_hm=hsv_h-high_low_color[0]

# 取得する色の範囲を指定する
lower_yellow = np.array([hsv_hm, high_low_color[1], high_low_color[1]])
upper_yellow = np.array([hsv_hp, high_low_color[2], high_low_color[2]])

# 指定した色に基づいたマスク画像の生成
img_mask = cv2.inRange(imgss, lower_yellow, upper_yellow)
img_color = cv2.bitwise_and(imgs, imgs, mask=img_mask)

img_color = cv2.Canny(img_color, 50, 200)
a,img_color =cv2.threshold(img_color, 127, 255,cv2.THRESH_BINARY_INV)
cv2.imshow("Show BINARIZATION Image",img_color)

#輪郭検知
image, contours, hierarchy = cv2.findContours(img_color,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
imax=len(contours)

cnt = contours[x]

area = cv2.contourArea(cnt)


img_colo = cv2.drawContours(imgs, contours, -1, (0,255,0), 1)
cv2.imshow("Show BINARIZATION Image",img_colo)

cv2.waitKey(0)
