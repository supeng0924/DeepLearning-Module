import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import os


from dataFunction import *
def forward(train_data):
    net=slim.stack(train_data,slim.conv2d,[(64,[3,3]),(32,[3,3]),(32,[3,3])],weights_regularizer=slim.l2_regularizer(0.0005),scope='conv3')
    conv_out=tf.reshape(net,(-1,net.shape[1]*net.shape[2]*net.shape[3]))
    fc=slim.stack(conv_out,slim.fully_connected,[500,100,10],weights_regularizer=slim.l2_regularizer(0.0005),scope='fc')
    return fc

BATCH_SIZE=100
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
TRAININT_STEP=500000
MODEL_DIR="model1105"
MODEL_NAME="mnist"

def backward(X_train, X_val, Y_train, Y_val):
    # train_label_oh=tf.one_hot(train_label,10)
    x=tf.placeholder(tf.float32,[None,28,28,1])
    y=tf.placeholder(tf.float32,[None,10])
    y_pre=forward(x)
    global_step=tf.Variable(0,trainable=False)
    loss_temp=tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss=loss_temp+tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pre,labels=tf.arg_max(y,1)))
    # loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pre,labels=y))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        X_train.shape[0] / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_pre,1)),tf.float32))
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt=tf.train.get_checkpoint_state(MODEL_DIR)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)

        for i in range(TRAININT_STEP):
            index = np.random.randint(0, X_train.shape[0], [BATCH_SIZE])
            # X_train, X_val, Y_train, Y_val
            xs = X_train[index]
            ys = Y_train[index]
            loss_temp,_=sess.run([loss,train_step],feed_dict={x:xs,y:ys})
            if i%100==0:
                acc=sess.run([accuracy],feed_dict={x:X_val,y:Y_val})
                print(loss_temp,acc)
                saver.save(sess,os.path.join(MODEL_DIR,MODEL_NAME),global_step=global_step)



def main():
    x_train, y_train = get_data('data/argumore.csv')
    x_train = data_normalization(x_train)

    x_train_norml = converVecToImg(x_train)
    y_label = data_one_hot(y_train, 10)
    random_seed=2
    X_train, X_val, Y_train, Y_val = train_test_split(x_train_norml, y_label, test_size=0.01, random_state=random_seed)
    # backward(X_train, X_val, Y_train, Y_val)
    # print(y_label.shape)
    backward(X_train, X_val, Y_train, Y_val)



if __name__ == '__main__':
    main()