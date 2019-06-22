#-*- coding:utf-8 -*-
import cv2
import picamera
import serial
import numpy as np
from operator import itemgetter
import RPi.GPIO as GPIO
import time

#変数

#調整必須
countsphoto=2#サンプル数(1以上)
awbx="fluorescent"#撮影パラメータ
meterx="matrix"#撮影パラメータ
exposurex="verylong"#撮影パラメータ
servoint=3/10000#サーボの変化量(0~5/10000(首振り９０度)まで)
cutset=[0,250,1000,400]#切り取りの[左上の位置y,左上の位置x,x座標長さ,y座標長さ]

#調整しなくても割といける
high_low_color=[10,115,105]#色を抽出する際の[色相の誤差範囲,彩度の最低値,明度の最低値]を設定
min_img_area = 1000#対象物の領域の最小サイズ
max_img_area = 782500#対象物の領域の最大サイズ
nomal_color= np.uint8([[[0,255,255]]])#選択する色指定
bsize=13#閾値を決める際の領域設定(線の太さみたいな)(奇数)
c=1 #2値化の際のノイズを消す
sigmaxy=100000#バイラテラルフィルタ用のぼかし加減
jpegfile_start="/home/pi/Desktop/robocon/debug/"
jpegfile_end=".jpg"
ser=serial.Serial('/dev/ttyUSB0',9600)
anses=[]
bigsquares=[]
#tips
#色部分が分離する場合:明彩度の最低値の変更
#そもそも面積が出力されない場合:対象物の領域の最小サイズの変更
#近い色で反応する場合:色相の誤差範囲の変更

def math_long(square,i):
    #直線上の場合
    if i==0:
        #####この下の行を変える#####
        square=(-9.958*(10**(-12))*(square**3))+1.956*(10**(-6))*(square**2)-0.135*square+5250.4548
    #斜めの場合
    else:
        #####この下の行を変える#####
        square=(-9.958*(10**(-12))*(square**3))+1.956*(10**(-6))*(square**2)-0.135*square+6250.4548
        
    anses.append(square)

def Image_processing():
    for i in range(3):
        gpiocon(servoint,i)
        campic(i)
        for ese in range(countsphoto):
            #print("count="+str(ese))
            jpegfile=file_settings(ese)
            h_error_lower,h_error_upper=mask_settings()
            img,img2,img_hsv=clor_to_hsv(jpegfile)
            bilateral_img,img_mask=usemask(img_hsv,h_error_lower, h_error_upper)
            img_simple_outline,img_simple_contour=decision_mask(img_mask)
            bigsquares,img3=list_area(bilateral_img,img_mask,img_simple_contour,img2)
            
            #####デバッグモード
            #debug_mask_images(img,img_hsv,bilateral_img,img_mask,img3)
            
        fixpix=avepix(bigsquares)
        math_long(fixpix,i)
        bigsquares.clear()
        
def campic(i):#カメラで撮影
    with picamera.PiCamera() as picameras:
        picameras.awb_mode=awbx
        picameras.meter_mode=meterx
        picameras.exposure_mode=exposurex
        for nums in range(countsphoto):
            jpegfile_name=jpegfile_start+str(nums)+jpegfile_end
            picameras.capture(jpegfile_name)
            #print(jpegfile_name)
            
            if i==0:
                img = cv2.imread(jpegfile_name)
                des=img[cutset[1]:cutset[1]+cutset[3],cutset[0]:cutset[0]+cutset[2]]
                cv2.imwrite(jpegfile_name,des)
                
            
def gpiocon(servoint,i):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(14, GPIO.OUT)
    GPIO.setup(15, GPIO.OUT)
    servoi=0
    if i==0:
        GPIO.output(14, False)
        GPIO.output(15, False)
    elif i==1:
        GPIO.output(14, True)
        GPIO.output(15, False)
    elif i==2:
        GPIO.output(14, False)
        GPIO.output(15, True)



def file_settings(num):#ファイル名設定
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


def list_area(bilateral_img,img_mask,img_simple_contour,img2):#面積のリスト化
    large_simple_contour = [i for i in img_simple_contour if (cv2.contourArea(i) > min_img_area)and(cv2.contourArea(i)<max_img_area)]#対象物かノイズかの判断
    square=[cv2.contourArea(i) for i in large_simple_contour]
    for i in large_simple_contour:
        img3 = cv2.drawContours(img2, i,-1, (255,0,0), 3)
    square.sort(reverse=True)
    #print(square)
    bigsquares.append(square[0])
    return bigsquares,img3
    #list内包型
    #large_simple_contour = []
    #for i in img_simple_contour:
    #   if cv2.contourArea(i) > min_img_area
    #   large_simple_contour.append(i)

def avepix(bigsquares):#ピクセルの平均を得る
    #print (bigsquares)
    fixpix=sum(bigsquares)/len(bigsquares)
    #print (fixpix)
    return fixpix

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
    
while True:
    GPIO.cleanup()
    readxbyte=ser.read()
    readx=readxbyte.strip().decode('utf-8')
    if readx=="A":
        Image_processing()
        anses[0]=anses[0]/20
        xa=int(anses[0])
        xb=xa.to_bytes(1,"little")
        anses[1]=anses[1]/20
        ya=int(anses[1])
        yb=ya.to_bytes(1,"little")
        anses[2]=anses[2]/20
        za=int(anses[2])
        zb=za.to_bytes(1,"little")
        ser.write(b"B")
        ser.write(xb)
        ser.write(yb)
        ser.write(zb)
        #print(sendcodeans)
        anses.clear()
        
