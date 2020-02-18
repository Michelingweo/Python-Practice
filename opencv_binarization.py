import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as plot
import aircv as ac

def binarization(file_path,write_path,threshold=220):
    imodel = cv2.imread(file_path, 0)
    row, col = imodel.shape

    imodel_binary = np.zeros((row, col))


    for r in range(row):
        for l in range(col):
            if imodel[r, l] >= threshold:
                imodel_binary[r, l] = 255
            else:
                imodel_binary[r, l] = 0
    cv2.imwrite(write_path, imodel_binary)
if __name__ == '__main__':
    imodel = cv2.imread('F:\opencv_pic\\M0_023.png', 0)
    row, col = imodel.shape
    print(imodel, '\nshape:', imodel.shape, '\nsize', imodel.size)

    imodel_binary = np.zeros((row, col))
    threshold =220
    binarization(row, col, threshold, imodel, imodel_binary)

    cv2.imwrite('F:\opencv_pic\\imodel_binary023.jpg', imodel_binary)

    # # imsrc = imodel_binary
    # imobj1 = cv2.imread('F:\opencv_pic\\upper.jpg', 0)
    # imobj2 = cv2.imread('F:\opencv_pic\\lower.jpg', 0)
    #
    # row1, col1 = imobj1.shape
    # row2, col2 = imobj2.shape
    # imobj1_binary = np.zeros((row1, col1))
    # imobj2_binary = np.zeros((row2, col2))
    # binarization(row1, col1, threshold, imobj1, imobj1_binary)
    # binarization(row2, col2, threshold, imobj2, imobj2_binary)
    #
    # cv2.imwrite('F:\opencv_pic\\imobj1_binary100.jpg', imobj1_binary)
    # cv2.imwrite('F:\opencv_pic\\imobj2_binary100.jpg', imobj2_binary)

    #

    # cv2.imshow('binary',imodel_binary)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

