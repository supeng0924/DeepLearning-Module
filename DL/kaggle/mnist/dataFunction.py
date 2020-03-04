import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2


def get_data(path):
    x_train = pd.read_csv(path)
    y_train = x_train['label']
    x_train.drop(['label'], inplace=True, axis=1)
    x_train = x_train.values
    y_train = y_train.values
    return x_train,y_train


def data_normalization(x_train):
    x_train_normalization = x_train / 255.0
    return x_train_normalization


def converVecToImg(x_train):
    x_train_reshape = x_train.reshape(-1, 28, 28, 1)
    return x_train_reshape

def data_one_hot(label,num_class):
    # 将label转为one-hot格式
    index_row_st=np.arange(label.shape[0])*num_class
    label_one_hot=np.zeros((label.shape[0],num_class))
    label_one_hot.flat[index_row_st+label.ravel()]=1
    return label_one_hot


def showImg(x_train_img,index,y_train):
    img = x_train_img[index].reshape((28, 28))
    print("label: ",y_train[index])
    plt.figure("Image"+str(index))
    plt.imshow(img)
    plt.show()
    returnqq
def justShowImg(x_train_img,index):
    img = np.array(x_train_img[index].reshape((28, 28)), dtype=np.uint8)
    img_resize = cv2.resize(img, (400, 400))
    # cv2.imshow("img_"+str(index+1),img_resize)
    cv2.imshow("img",img_resize)


    # (h, w) = img_resize.shape[:2]
    # center = (w / 2, h / 2)
    # M = cv2.getRotationMatrix2D(center, 10, 1.25)  # 旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
    # rotated = cv2.warpAffine(img_resize, M, (w, h))
    # cv2.imshow("Rotated by 45 Degrees", rotated)
    return

def ImgArgumentation(x_train_vec):
    img = np.array(x_train_vec, dtype=np.uint8)
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)

    mat = np.array(img.reshape(1, 28 * 28))

    # 旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
    M = cv2.getRotationMatrix2D(center, 10, 1.25)  # 旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
    rotated = cv2.warpAffine(img, M, (w, h))
    mat = np.r_[mat, np.array(rotated.reshape(1, 28 * 28))]
    # mat=np.array(rotated.reshape(1,28*28))

    M = cv2.getRotationMatrix2D(center, -10, 1.25)
    rotated = cv2.warpAffine(img, M, (w, h))
    mat=np.r_[mat,np.array(rotated.reshape(1,28*28))]



    M = cv2.getRotationMatrix2D(center, 10, 0.8)
    rotated = cv2.warpAffine(img, M, (w, h))
    mat = np.r_[mat, np.array(rotated.reshape(1, 28 * 28))]

    M = cv2.getRotationMatrix2D(center, -10, 0.8)
    rotated = cv2.warpAffine(img, M, (w, h))
    mat = np.r_[mat, np.array(rotated.reshape(1, 28 * 28))]

    M = cv2.getRotationMatrix2D(center, 20, 0.9)
    rotated = cv2.warpAffine(img, M, (w, h))
    mat = np.r_[mat, np.array(rotated.reshape(1, 28 * 28))]

    M = cv2.getRotationMatrix2D(center, -20, 0.9)
    rotated = cv2.warpAffine(img, M, (w, h))
    mat = np.r_[mat, np.array(rotated.reshape(1, 28 * 28))]

    M = cv2.getRotationMatrix2D(center, 20, 1.1)
    rotated = cv2.warpAffine(img, M, (w, h))
    mat = np.r_[mat, np.array(rotated.reshape(1, 28 * 28))]

    M = cv2.getRotationMatrix2D(center, -20, 1.1)
    rotated = cv2.warpAffine(img, M, (w, h))
    mat = np.r_[mat, np.array(rotated.reshape(1, 28 * 28))]



    # print(mat.shape)


    return mat

