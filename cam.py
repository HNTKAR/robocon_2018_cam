
#-*- coding:utf-8 -*-
import cv2
import numpy as np
square=[]

#変数
nomal_color= np.uint8([[[0,255,255]]])#選択する色指定
high_low_color=[10,150,255]#色を抽出する際の[色相の誤差範囲,明彩度の最低値,明彩度の最高値]を設定
th_num=100#二値化する際の閾値
kernel=5#ガウシアンフィルタ用のカーネルを指定
min_img_area = 300#対象物の領域の最小サイズ

#画像形式の変換
img = cv2.imread("E:\program\cont\sumple\locc.jpg")#画像読み込み
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
gaussian_img = cv2.GaussianBlur(img_color, (kernel,kernel), 0)#ガウシアンフィルタによる平滑化
gray_img = cv2.cvtColor(gaussian_img, cv2.COLOR_BGR2GRAY)#グレースケール化
_,img_bw =cv2.threshold(gray_img, th_num, 255,cv2.THRESH_BINARY)#二値化
img_simple_outline,img_simple_contour, _ = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#輪郭の頂点を取得

#仕上げ
large_simple_contour = [i for i in img_simple_contour if cv2.contourArea(i) > min_img_area]#対象物かノイズかの判断
for i in large_simple_contour:#面積のリスト化
    square.append(cv2.contourArea(i))


#debug
img2 = img.copy()
print large_simple_contour
print square#出力される面積
cv2.imshow("debug",img)#入力画像を表示
cv2.waitKey(0)
cv2.imshow("debug",img_hsv)#入力画像のHSV画像を表示
cv2.waitKey(0)
cv2.imshow("debug",gaussian_img)#平滑化後の画像を表示
cv2.waitKey(0)
cv2.imshow("debug",gray_img)#グレースケール化した画像を表示
cv2.waitKey(0)
cv2.imshow("debug",img_bw)#二値化した画像を表示
cv2.waitKey(0)

img_outline,img_contour, _ = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#輪郭全域を取得
large_contour = [i for i in img_contour if cv2.contourArea(i) > min_img_area]#対象物かノイズかの判断
for i in large_contour:#画像の輪郭全域を抽出
    img_large_contour = cv2.drawContours(img, i, -1, (0,255,0), 3)
cv2.imshow("debug",img_large_contour)#輪郭全域を表示
cv2.waitKey(0)

for i in large_simple_contour:#画像の輪郭の全頂点を抽出
    img_large_simple_contour = cv2.drawContours(img2, i, -1, (0,255,0), 3)
cv2.imshow("debug",img_large_simple_contour)#画像の輪郭の全頂点を表示
cv2.waitKey(0)
