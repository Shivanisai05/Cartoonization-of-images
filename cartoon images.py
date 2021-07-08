import os
import cv2
import numpy as np


input_fold=r'C:\Users\ASUS\Desktop\CartoonGan-tensorflow\dataset\realworld2cartoon\trainA'
output_fold=r'C:\Users\ASUS\Desktop\CartoonGan-tensorflow\dataset\realworld2cartoon\trainB'

for images in os.listdir(input_fold):
  image = os.path.join(input_fold, images)
  img = cv2.imread(image)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray = cv2.medianBlur(gray,5)
  edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 9, 9)
  color = cv2.bilateralFilter(img, 9, 250, 250)
  cartoon = cv2.bitwise_and(color,color, mask =edges)
  cartoon_img = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
  cv2.imwrite(os.path.join(output_fold,images),cartoon_img)
