#-*- coding:utf-8 -*-
import cv2
import numpy as np
from operator import itemgetter

#変数

#調整必須
high_low_color=[10,115,105]#色を抽出する際の[色相の誤差範囲,彩度の最低値,明度の最低値]を設定
min_img_area = 1000#対象物の領域の最小サイズ
#調整しなくても割といける
nomal_color= np.uint8([[[0,255,255]]])#選択する色指定
bsize=13#閾値を決める際の領域設定(線の太さみたいな)(奇数)
c=1 #2値化の際のノイズを消す
sigmaxy=100000#バイラテラルフィルタ用のぼかし加減
#tips
#色部分が分離する場合:明彩度の最低値の変更
#そもそも面積が出力されない場合:対象物の領域の最小サイズの変更
#近い色で反応する場合:色相の誤差範囲の変更


def main():
    for i in range(1500, 4600,100):
        jpegfile=file_settings(i)
        h_error_lower,h_error_upper=mask_settings()
        img,img2,img_hsv=clor_to_hsv(jpegfile)
        bilateral_img,img_mask=usemask(img_hsv,h_error_lower, h_error_upper)
        img_simple_outline,img_simple_contour=decision_mask(img_mask)
        square,img3=list_area(bilateral_img,img_mask,img_simple_contour,img2)
        #debug_mask_images(img,img_hsv,bilateral_img,img_mask,img3)
        math_long(square[0])#出力される面積


def math_long(square):
    square=5.986*(10**-12)*(square**3)-1.484*(10**-6)*(square**2)+0.13*square+324.471
    print square


def file_settings(num):#ファイル名設定
    jpegfile_start="E:\program\cont\python2\c"
    jpegfile_end=".jpg"
    jpegfile_num=num
    jpegfile=jpegfile_start+str(jpegfile_num)+jpegfile_end
    return jpegfile


def clor_to_hsv(jpegfile):#画像形式の変換
    img = cv2.imread(jpegfile)#画像読み込み
    img2 = img.copy()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#元画像をHSVに変換
    return img,img2,img_hsv


def mask_settings():# マスク作成
    choice_color= cv2.cvtColor(nomal_color,cv2.COLOR_BGR2HSV)#選択した色をHSVに変換
    choice_color_h =choice_color[0][0][0]#[[[xx,yy,zz]]]の形で出力されたHSV形式の色の色相のみを抽出
    choice_color_h =int(choice_color_h )#int型に変換
    h_error_plus=choice_color_h+high_low_color[0]#色相の+側誤差を設定
    h_error_minus=choice_color_h-high_low_color[0]#色相の-側誤差を設定
    h_error_lower = np.array([h_error_minus, high_low_color[1],high_low_color[2]])#抽出する色の-側を設定
    h_error_upper = np.array([h_error_plus, 255, 255])#抽出する色の+側を設定
    return h_error_lower,h_error_upper


def usemask(img_hsv,h_error_lower, h_error_upper):# 指定した色に基づいたマスク画像を適用
    bilateral_img = cv2.bilateralFilter(img_hsv,bsize,sigmaxy,sigmaxy)#バイラテラルフィルタによる平滑化
    img_mask = cv2.inRange(bilateral_img,h_error_lower, h_error_upper)#マスク画像を生成
    return bilateral_img,img_mask


def decision_mask(img_mask):#マスク画像から判別する場合
    img_simple_outline,img_simple_contour, _ = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#輪郭の頂点を取得
    return img_simple_outline,img_simple_contour


def decision_img(bilateral_img,img_mask):#画像から判別する場合
    img_color = cv2.bitwise_and(bilateral_img, bilateral_img, mask=img_mask)#マスク画像を適用
    gray_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)#グレースケール化
    img_bw=cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,bsize,c)#二値化
    img_simple_outline,img_simple_contour, _ = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#輪郭の頂点を取得
    return img_color,gray_img,img_bw,img_simple_outline,img_simple_contour


def list_area(bilateral_img,img_mask,img_simple_contour,img2):#面積のリスト化
    large_simple_contour = [i for i in img_simple_contour if (cv2.contourArea(i) > min_img_area)and(cv2.contourArea(i)<782500)]#対象物かノイズかの判断
    square=[cv2.contourArea(i) for i in large_simple_contour]
    for i in large_simple_contour:
        img3 = cv2.drawContours(img2, i,-1, (255,0,0), 3)
    return square,img3
    #list内包型
    #large_simple_contour = []
    #for i in img_simple_contour:
    #   if cv2.contourArea(i) > min_img_area
    #   large_simple_contour.append(i)


def debug_mask_images(img,img_hsv,bilateral_img,img_mask,img3):# マスクから判別する場合のデバッグ
    cv2.imshow("debug",img)#入力画像
    cv2.waitKey(0)
    cv2.imshow("debug",img_hsv)#入力画像のhsv画像
    cv2.waitKey(0)
    cv2.imshow("debug",bilateral_img)#バイラテラルフィルタ処理後
    cv2.waitKey(0)
    cv2.imshow("debug",img_mask)#マスクの画像
    cv2.waitKey(0)
    cv2.imshow("debug",img3)#頂点の画像
    cv2.waitKey(0)


def debug_img_images(img,img_hsv,bilateral_img,img_mask,img3,img_color,gray_img,img_bw):# 画像から判別する場合のデバッグ
    cv2.imshow("debug",img)#入力画像
    cv2.waitKey(0)
    cv2.imshow("debug",img_hsv)#入力画像のhsv画像
    cv2.waitKey(0)
    cv2.imshow("debug",bilateral_img)#バイラテラルフィルタ処理後
    cv2.waitKey(0)
    cv2.imshow("debug",img_mask)#マスクの画像
    cv2.waitKey(0)
    cv2.imshow("debug",img_color)#マスクの画像を適用した画像
    cv2.waitKey(0)
    cv2.imshow("debug",gray_img)#グレースケール化した画像
    cv2.waitKey(0)
    cv2.imshow("debug",img_bw)#二値化した画像を表示
    cv2.waitKey(0)
    cv2.imshow("debug",img3)#マスクの画像
    cv2.waitKey(0)

main()
