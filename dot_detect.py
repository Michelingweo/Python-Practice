import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as plot
import aircv as ac

if __name__ == "__main__":
    # img = cv2.imread('F:\opencv_pic\images\\M0_023.png')
    gray = cv2.imread('F:\opencv_pic\images\\M0_023.png', 0)
    # imobj1 = cv2.imread('F:\opencv_pic\\upper.jpg', 0)
    # imobj2 = cv2.imread('F:\opencv_pic\\lower.jpg', 0)
    #
    # cv2.imwrite('F:\opencv_pic\\imodel023_gray.png', imodel)
    # cv2.imwrite('F:\opencv_pic\\imobj1_gray.jpg', imobj1)
    # cv2.imwrite('F:\opencv_pic\\imobj2_gray.jpg', imobj2)
    #
    # ret, binary = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # 以色彩边界轮廓框出白点
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    #
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    #
    imsrc = ac.imread('F:\opencv_pic\imodel_binary023.jpg')
    # imobj11 = ac.imread('F:\opencv_pic\\test1_gray.jpg')
    imobj11 = ac.imread('F:\opencv_pic\\dot_detect2.jpg')
    # imobj22 = ac.imread('F:\opencv_pic\\imobj2_binary180.jpg')
    #
    # find the match position
    pos1 = ac.find_all_template(imsrc, imobj11,0.7)
    # pos2 = ac.find_template(imsrc, imobj2)
    # print('pos1:',pos1,'\npos2:',pos2)
    print('pos1:', pos1)
    for i in range(len(pos1)):
        cv2.rectangle(imsrc, pos1[i]['rectangle'][0], pos1[i]['rectangle'][3], (0, 255, 0), 5)
    plt.imshow(imsrc,cmap='gray',interpolation='bicubic')
    plt.xticks([]),plt.yticks([])
    plt.show()
    cv2.rectangle(imsrc, (0,0), (100,100), (0, 255, 0), 5)
    plt.imshow(imsrc, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()


    # pos 散点图

    frame=[(0,0),(360,0),(0,220),(180,220),(360,220),(0,795),(180,795),(360,795),(0,1100),(180,1100),(360,1100),(0,1320),(360,1320)]
    x,y=[],[]
    print(len(frame))
    for i in range(13):
        x.append(frame[i][0])
        y.append(frame[i][1])

    # for i in range(len(pos1)):
    #     for j in range(4):
    #         x.append(pos1[i]['rectangle'][j][0])
    #         y.append(-pos1[i]['rectangle'][j][1])
    plt.scatter(x,y)
    plt.show()


