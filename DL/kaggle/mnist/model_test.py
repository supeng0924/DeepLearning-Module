import tensorflow as tf
import pandas as pd
import numpy as np
import model_train
from dataFunction import *
def test(mnist):
    # 导入计算图
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        y = tf.placeholder(tf.float32, [None, 10])
        # 计算前向传播
        y = model_train.forward(x)

        # 计算标签值
        act_val=tf.argmax(y, 1)
        # 计算准确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(act_val, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver=tf.train.Saver()
        with tf.Session() as sess:
            # 载入模型参数
            ckpt = tf.train.get_checkpoint_state(model_train.MODEL_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 计算预测的标签
                actual_label = sess.run(act_val, feed_dict={x: mnist})
                print("deal success")
                return actual_label
            else:
                print('No checkpoint file found')
                return


def main():
    # 训练集csv文件路径
    input_path_csv='data/test.csv'
    result_path_csv = "data/result1105.csv"
    x_test = pd.read_csv(input_path_csv)
    x_test = x_test.values

    # # 显示图片
    # for i in range(5000):
    #     cv2.destroyAllWindows()
    #     justShowImg(x_test, i)
    #
    #     if cv2.waitKey(300) & 0xFF == ord('q'):
    #         break

    x_test = data_normalization(x_test)
    x_test_norml = converVecToImg(x_test)
    res=None
    for i in range(28):

        # 得到预测的标签
        actual_label=test(x_test_norml[i*1000:(i+1)*1000])
        if res is None:
            res=actual_label
        else:
            res=np.append(res,actual_label)

    results = pd.Series(res, name="Label")
    index = pd.Series(range(1,28001),name = "ImageId")
    submission = pd.concat([index, results], axis=1)
    submission.to_csv(result_path_csv, index=False)




if __name__ == '__main__':
    main()

