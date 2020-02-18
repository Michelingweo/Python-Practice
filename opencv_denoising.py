# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as plot
import aircv as ac

imobj1=cv2.imread('F:\opencv_pic\\imodel_binary023.jpg')
# imobj2=cv2.imread('F:\opencv_pic\\imobj1_binary100.jpg')


dst1 = cv2.fastNlMeansDenoising(imobj1, None, 20,  7, 21)
# dst2 = cv2.fastNlMeansDenoising(imobj2, None, 10,  7, 21)
cv2.imwrite('F:\opencv_pic\\denoising023.jpg', dst1)
# cv2.imwrite('F:\opencv_pic\\imobj2_binary10010.jpg', dst2)
