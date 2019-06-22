#-*- coding:utf-8 -*-
import cv2
import numpy as np

#変数
nomal_color= np.uint8([[[0,255,255]]])#選択する色指定
high_low_color=[10,150,255]#色を抽出する際の[色相の誤差範囲,明彩度の最低値,明彩度の最高値]を設定
th_num=100#二値化する際の閾値
kernel=5#ガウシアンフィルタ用のカーネルを指定
min_img_area = 35823#対象物の領域の最小サイズ

#画像形式の変換
img = cv2.imread("E:\program\cont\sumple\loccc.jpg")#画像読み込み
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#元画像をHSVに変換

# 抽出する色の前処理
choice_color= cv2.cvtColor(nomal_color,cv2.COLOR_BGR2HSV)#選択した色をHSVに変換
choice_color_h =choice_color[0][0][0]#[[[xx,yy,zz]]]の形で出力されたHSV形式の色の色相のみを選択
choice_color_h =int(choice_color_h )#int型に変換
h_error_plus=choice_color_h+high_low_color[0]#色相の-側誤差を設定
h_error_minus=choice_color_h-high_low_color[0]#色相の-側誤差を設定
h_error_lower = np.array([h_error_minus, high_low_color[1], high_low_color[1]])#抽出する色の-側を設定
h_error_upper = np.array([h_error_plus, high_low_color[2], high_low_color[2]])#抽出する色の+側を設定


# 指定した色に基づいたマスク画像を適用
img_mask = cv2.inRange(img_hsv,h_error_lower, h_error_upper)#マスク画像を生成
img_color = cv2.bitwise_and(img_hsv, img_hsv, mask=img_mask)#マスク画像を適用
Gaussian_img = cv2.GaussianBlur(img_color, (kernel,kernel), 0)#ガウシアンフィルタによる平滑化
Gaussian_img = cv2.cvtColor(Gaussian_img, cv2.COLOR_BGR2GRAY)#グレースケール化
_,img_bw =cv2.threshold(Gaussian_img, th_num, 255,cv2.THRESH_BINARY)#二値化
img_contour,img_contours, _ = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#輪郭取得

#仕上げ
large_contours = [i for i in img_contours if cv2.contourArea(i) > min_img_area]#対象物かノイズかの判断

print cv2.contourArea(large_contours[0])
# square_list=[]
# apex_list=[]
# for contour in large_contours:
#     rect = cv2.minAreaRect(contour)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
#     for i1 in range(4):
#         for i2 in range(2):
#             boxcontor= box[i1][i2]
#             apex_list.append(boxcontor)
#     b1=(apex_list[6]-apex_list[2])*(apex_list[1]-apex_list[5])
#     s1=((apex_list[4]-apex_list[2])*(apex_list[3]-apex_list[5])/2)
#     s2=((apex_list[0]-apex_list[2])*(apex_list[1]-apex_list[3])/2)
#     s3=((apex_list[6]-apex_list[0])*(apex_list[1]-apex_list[7])/2)
#     s4=((apex_list[6]-apex_list[4])*(apex_list[7]-apex_list[5])/2)
#     squarex=b1-s1-s2-s3-s4
#     square_list.append(squarex)
#debug
# # print large_contours
# print square_list
# print len(large_contours)
# # print img_contours
# print len(img_contours)
# img = cv2.drawContours(img, large_contours, -1, (0,255,0), 3)
# imgs_hsv = cv2.cvtColor(img_color, cv2.COLOR_HSV2BGR)#元画像をBGRに変換
cv2.imshow("Show BINARIZATION Image",img_bw)
cv2.waitKey(0)
# # imgs_hsv = cv2.cvtColor(img_color, cv2.COLOR_HSV2BGR)#元画像をBGRに変換
cv2.imshow("Show BINARIZATION Image",img)
# # cv2.waitKey(0)
# # imgs_hsv = cv2.cvtColor(img_color, cv2.COLOR_HSV2BGR)#元画像をBGRに変換
# cv2.imshow("Show BINARIZATION Image",img_color)
cv2.waitKey(0)
