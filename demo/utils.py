#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/2/16 14:50
# @File  : utils.py
# @Author: 
# @Desc  :
import os
from PIL import Image

def png_convert_jpg(image):
    """
    转换图片或者这个文件下的所有png图片到jpg
    xxx.png --> xxx.jpg
    或者/dira  --> /dira/xxx.jpg
    :param image:
    :return:
    """
    def convert(img):
        im = Image.open(img)
        im = im.convert("RGB")
        desname = '.'.join(img.split('.')[:-1]) + '.jpg'
        im.save(desname)
        print(f'{img}转换完成-->{desname}')
    #如果是文件夹，那么列出1级目录下的所有png图片，并转换
    if os.path.isdir(image):
        for img in os.listdir(image):
            if not img.split('.')[-1].lower() == 'png':
                continue
            #图片完整路径
            img = os.path.join(image,img)
            convert(img)
    else:
        if image.split('.')[-1].lower() == 'png':
            convert(image)
        else:
            print("不是png图片")

if __name__ == '__main__':
    png_convert_jpg(image="cosmetic/train/images")
    png_convert_jpg(image="cosmetic/dev/images")