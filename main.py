from load_data import Loader
from tools import Tools
from model import Model
import preprocessing

from tqdm import tqdm
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf
import cv2
import argparse

import os
import random

parser = argparse.ArgumentParser(description='Fashion-Domain-Transfer-Network')
parser.add_argument('--mode', action="store", default='preprocessing')
result = parser.parse_args()

mode = result.mode

if mode != 'preprocessing' and mode != 'pretrain' and mode != 'pretrain-test' and mode != 'train' and mode != 'test':
    print('\nCan\'t start process. Invalid arguments.')
    print('Argument List : [preprocessing, pretrain, pretrain-test, train]\n')
    exit()

loader = Loader(mode=mode)
model = Model(mode=mode)
tools = Tools()

print('MODE :', mode)

if mode == 'preprocessing':

    lists = preprocessing.get_dir_lists(dataset_path='./Fashion-images')
    cropped = preprocessing.get_cropped_images(lists, dataset_path='./Fashion-images/')
    preprocessing.save_list_images(cropped, save_path='./crop-images/')

elif mode == 'pretrain':

    loader.load_fashion_image_data()

    model.build()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    batch_size = 128

    for i in range(5000):
        fImages, fLabels = loader.fashion_get_next_batch(batch_size=batch_size)
        _, loss, acc = sess.run([model.trainer, model.loss, model.accuracy], feed_dict={model.images: fImages, model.labels: fLabels})
        if i % 50 == 0:
            print('Step :', i, 'Acc :', acc, 'Loss :', loss)

    saver.save(sess, './model-save/fashion-pretrain.ckpt')

elif mode == 'pretrain-test':
    loader.load_fashion_image_data()

    model.build()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if os.path.exists('./model-save'):
        print("Pretrained Model exists!")
        restorer = tf.train.Saver(slim.get_model_variables('extractor'))
        restorer.restore(sess, './model-save/fashion-pretrain.ckpt')
        print('Restore complete.')

        batch_size = 128
        acc_average = 0.0
        num_iter = int(loader.fashion_num_examples / batch_size)

        for i in tqdm(range(num_iter), ascii=True, desc='Pretrained Model Test'):
            fImages, fLabels = loader.fashion_get_next_batch(batch_size=batch_size)
            acc = sess.run(model.accuracy, feed_dict={model.images: fImages, model.labels: fLabels})
            acc_average += acc

        print('Average Accuracy :', acc_average / float(num_iter))

elif mode == 'train':

    loader.load_mnist_data()
    #eval_data = loader.get_mnist_test_data()
    loader.load_fashion_image_data()

    model.build()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if os.path.exists('./model-save'):
        print("Pretrained Model exists!")
        restorer = tf.train.Saver(slim.get_model_variables('extractor'))
        restorer.restore(sess, './model-save/fashion-pretrain.ckpt')
        print('Restore complete.')
    else:
        print('Need to pretrained model. Exit.')
        exit()

    saver = tf.train.Saver()

    batch_size = 128

    for i in range(10001):
        fashion_image, fashion_label = loader.fashion_get_next_batch(batch_size)
        mnist_image, _ = loader.mnist_get_next_batch(batch_size)

        feed_dict = {model.source_images: fashion_image, model.target_images: mnist_image}
        sess.run(model.trainer_src_d, feed_dict=feed_dict)
        for n in range(5):
            sess.run(model.trainer_src_g, feed_dict=feed_dict)
        #sess.run(model.trainer_tv, feed_dict=feed_dict)

        sess.run(model.trainer_trg_d, feed_dict=feed_dict)
        for n in range(5):
            sess.run(model.trainer_trg_g, feed_dict=feed_dict)
        sess.run(model.trainer_tv, feed_dict=feed_dict)

        if i % 15 == 0:
            sess.run(model.trainer_src_const, feed_dict=feed_dict)

        if i % 10 == 0:
            test_image, _ = loader.fashion_get_next_batch(10)
            test = ((sess.run(model.fake, feed_dict={
                model.source_images: np.array([test_image[random.randrange(0, len(test_image))]])})) + 1) * 127.5 + 1
            cv2.imwrite('./Samples/Sample_'+str(i)+'.jpg', test[0])

        srcd, srcg, trgd, trgg, tv = sess.run([model.loss_src_g, model.loss_src_d, model.loss_trg_d, model.loss_trg_g, model.loss_tv], feed_dict=feed_dict)

        print('Step : %d Source D : %9f Source G : %9f Target D : %9f, Target G : %9f, TotalVariation : %9f' % (i, srcd, srcg, trgd, trgg, tv))

        if i % 1000 == 0 and i != 0:
            saver.save(sess, './train-save/train_'+str(i)+'.ckpt', global_step=1000)

elif mode == 'test':

    loader.load_mnist_data()
    eval_data = loader.get_mnist_test_data()

    model.build()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if os.path.exists('./train-save'):
        print('Trained Model exists!')
        restorer = tf.train.Saver()
        restorer.restore(sess, './train-save/train_10')
        print('Restore complete.')
    else:
        print('Need to trained model. Exit.')
        exit()

    test_image, _ = loader.mnist_get_next_batch(10)
    test = ((sess.run(model.fake, feed_dict={
        model.source_images: np.array([eval_data[random.randrange(0, len(eval_data))]])})) + 1) * 127.5 + 1
    cv2.imwrite('./Samples/Test.jpg', test[0])