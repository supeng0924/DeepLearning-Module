from dataFunction import *
import pandas as pd
import numpy as np
import cv2
def main():
    input_path="data/train.csv"
    output_path="data/argumore.csv"

    x_train, y_train=get_data(input_path)
    x_train_img = converVecToImg(x_train)
    print(x_train_img.shape)


    for i in range(x_train_img.shape[0]):
        if i%1000==0:
            print(i)
        mat = ImgArgumentation(x_train_img[i])
        res = np.ones((9, 1), dtype=np.uint8) * y_train[i]
        ressss = np.c_[res, mat]
        results = pd.DataFrame(ressss)
        results.to_csv(output_path, index=False, header=False, mode='a')

    # # # 数据验证
    # x_train, y_train = get_data(output_path)
    # showImg(x_train, 10000, y_train)
    # showImg(x_train, 10001, y_train)
    # showImg(x_train, 10002, y_train)
    # showImg(x_train, 10003, y_train)

if __name__ == '__main__':
    main()

