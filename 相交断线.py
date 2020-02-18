# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import aircv as ac
from opencv_binarization import binarization


def take_one(elem):
    return elem[0]


def findposinlist(obj, lis):
    for i in range(len(lis)):
        if obj >= lis[i][0]:
            i = i+1
            continue
        else:
            return i


def reposition(obj_index, frame, dot_pos):
    reframe=[]
    obj_point = dot_pos[obj_index]
    avg_height = (dot_pos[1][1] + dot_pos[0][1]) / 2
    if obj_point[1] > avg_height:
        deltax = obj_point[0] - frame[1][0]
        deltay = obj_point[1] - frame[1][1]
        delta = (deltax, deltay)
    else:
        deltax = obj_point[0] - frame[0][0]
        deltay = obj_point[1] - frame[0][1]
        delta = (deltax, deltay)
    for j in range(len(frame)):
        rx = int(frame[j][0] + delta[0])
        ry = int(frame[j][1] + delta[1])
        r = (rx, ry)
        reframe.append(r)

    return reframe


def pointplot(pic, points,  point_size=1, point_color=(0, 0, 255), thickness=4):
    for point in points:
        cv2.circle(pic, point, point_size, point_color, thickness)

    cv2.namedWindow("frame mapping")
    cv2.imshow('image', pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#假设瑕疵样本像素坐标
defaut = (1000, 900)


#线路名称
wire = ['M1-1', 'M1-2', 'M1-3', 'M2-1', 'M2-2', 'M2-3', 'M2-4', 'M2-5']
num = [0, 1, 2, 3, 4, 5, 6, 7]
wire_mapping = dict(zip(num, wire))
print(wire_mapping)


#电路框架像素坐标（前两个为两个正圆白点中心坐标）
frame = [(115, 140), (245, 1200), (0, 0), (360, 0), (0, 220), (180, 220), (360, 220), (0, 795), (180, 795),\
       (360, 795), (0, 1100), (180, 1100), (360, 1100), (0, 1320), (360, 1320)]
#图像提取与二值化
file_path = "F:\opencv_pic\images\\test.jpg"
writw_path = 'F:\opencv_pic\\binary\\binary_test.jpg'
binarization(file_path, writw_path)
#瑕疵原图
imsrc = ac.imread(writw_path)
#白点目标图
imobj = ac.imread('F:\opencv_pic\\dot_detect2.jpg')


# find the match position
pos1 = ac.find_all_template(imsrc, imobj, 0.7)

# print('pos1:', pos1)

#提取各个白点的中心点坐标 左上角为（0，0） （横坐标，纵坐标）
dot_pos = []
for i in range(len(pos1)):
    cv2.rectangle(imsrc, pos1[i]['rectangle'][0], pos1[i]['rectangle'][3], (0, 255, 0), 5)
    dot_pos.append(((pos1[i]['rectangle'][0][0]+pos1[i]['rectangle'][3][0])/2,\
                    (pos1[i]['rectangle'][0][1]+pos1[i]['rectangle'][3][1])/2))
#根据白点对瑕疵进行定位
dot_pos.sort(key=take_one)
print(dot_pos)
obj_index = findposinlist(defaut[0], dot_pos)


#检查白点检测结果

plt.imshow(imsrc, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])
plt.show()



#跟据瑕疵坐标和白点坐标对框架进行定位
reframe=reposition(obj_index, frame, dot_pos)


#框架映射
orig=ac.imread("F:\opencv_pic\images\\M0.jpg")
pointplot(orig, reframe)