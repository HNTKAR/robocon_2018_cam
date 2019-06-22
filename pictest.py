#-*- coding:utf-8 -*-
import time
import picamera
import subprocess
parae=["off","auto","night","nightpreview","backlight","spotlight","sports","snow","beach","verylong","fixedfps","antishake","fireworks"]
param=["average","spot","backlit","matrix"]
paraa=["off","auto","sunlight","cloudy","shade","tungsten","fluorescent","incandescent","flash","horizon"]
with picamera.PiCamera() as picamera:
    num=0
    file_name=str(num)
    for e in parae:
        print(e)
        for m in param:
            for a in paraa:
                file_name=a+"-"+m+"-"+e
                picamera.awb_mode=a
                picamera.meter_mode=m
                picamera.exposure_mode=e
                picamera.capture("/home/pi/Desktop/robocon/test1/"+file_name+'.jpg')
                
    print ("\n")
    print("end")