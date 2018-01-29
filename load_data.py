from random import shuffle
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os

class Loader:
    def __init__(self, mode='pretrain'):
        self.fashion_path = './crop-images/'
        self.mnist_train_path = './Fashion-MNIST/fashion-mnist_train.csv'
        self.mnist_test_path = './Fashion-MNIST/fashion-mnist_test.csv'
        self.mnist_batch = 0
        self.fashion_batch = 0
        self.mode = mode

    def load_mnist_data(self):
        print('Loader mode :', self.mode)
        print('Read', self.mnist_train_path, '...')
        self.mnist_data = pd.read_csv(self.mnist_test_path).values.tolist()

        self.mnist_labels = [self.mnist_data[i][0] for i in range(len(self.mnist_data)) if self.mnist_data[i][0] < 5]
        self.mnist_images = [self.mnist_data[i][1:] for i in range(len(self.mnist_data)) if self.mnist_data[i][0] < 5]

        self.mnist_labels = np.array(self.mnist_labels)
        self.mnist_images = np.array([np.reshape(self.mnist_images[i], [28, 28]) for i in range(len(self.mnist_images))])

        assert len(self.mnist_labels) == len(self.mnist_labels)

        del self.mnist_data

        self.mnist_num_examples = len(self.mnist_images)

        print('Fashion-MNIST Load complete.')
        print('MNIST Images :', self.mnist_images.shape, 'MNIST Labels :', self.mnist_labels.shape)

    def load_fashion_image_data(self):
        self.fashion_images = []
        self.fashion_filenames = []

        print('Read image files in', self.fashion_path, '...')
        for root, dirs, files in tqdm(os.walk(self.fashion_path)):
            for filename in files:
                self.fashion_filenames.append(self.fashion_path + filename)

        self.fashion_filenames = sorted(self.fashion_filenames)

        # ---------------------------------------------------------------------- #
        temp = []
        fnames = []

        # TODO : Change to iterations
        temp.append([self.fashion_filenames[i] for i in range(len(self.fashion_filenames)) if 'Coat' in self.fashion_filenames[i]])
        temp.append([self.fashion_filenames[i] for i in range(len(self.fashion_filenames)) if 'Pullover' in self.fashion_filenames[i]])
        temp.append([self.fashion_filenames[i] for i in range(len(self.fashion_filenames)) if 'Trouser' in self.fashion_filenames[i]])
        temp.append([self.fashion_filenames[i] for i in range(len(self.fashion_filenames)) if 'Shirt' in self.fashion_filenames[i]])
        temp.append([self.fashion_filenames[i] for i in range(len(self.fashion_filenames)) if 'Dress' in self.fashion_filenames[i]])

        if self.mode == 'pretrain':
            for i in range(len(temp)):
                for k in range(len(temp[i]) if len(temp[i]) < 5000 else 5000):
                    fnames.append(temp[i][k])

            self.fashion_filenames = fnames

        elif self.mode == 'pretrain-test' or self.mode == 'train':
            for i in range(len(temp)):
                for k in range(len(temp[i])):
                    fnames.append(temp[i][k])
        # ---------------------------------------------------------------------- #

        print("Fashion Image(Target domain sample) number :", len(self.fashion_filenames))
        shuffle(self.fashion_filenames)

        # TODO : Batch loading

        self.fashion_num_examples = len(self.fashion_filenames)

        print('Fashion Images dataset Load complete.')


    def mnist_get_next_batch(self, batch_size=128):
        if self.mnist_batch + batch_size > self.mnist_num_examples:
            self.mnist_batch = 0

        images = np.array(self.mnist_images[self.mnist_batch:self.mnist_batch+batch_size])
        labels = np.array(self.mnist_labels[self.mnist_batch:self.mnist_batch+batch_size])

        images = np.array(self.resize_images(images)) / 127.5 - 1

        self.mnist_batch += batch_size

        return images, labels

    def fashion_get_next_batch(self, batch_size=128):
        if self.fashion_batch + batch_size > self.fashion_num_examples:
            self.fashion_batch = 0

        images = []
        labels = []

        for i in range(batch_size):
            images.append((cv2.imread(self.fashion_filenames[self.fashion_batch + i])/127.5 - 1))

            # TODO : Change to iterations
            if 'Coat' in self.fashion_filenames[self.fashion_batch + i]:
                labels.append(0)
            elif 'Pullover' in self.fashion_filenames[self.fashion_batch + i]:
                labels.append(1)
            elif 'Trouser' in self.fashion_filenames[self.fashion_batch + i]:
                labels.append(2)
            elif 'Shirt' in self.fashion_filenames[self.fashion_batch + i]:
                labels.append(3)
            elif 'Dress' in self.fashion_filenames[self.fashion_batch + i]:
                labels.append(4)

        self.fashion_batch += batch_size

        return images, labels


    # resize_images : Implemented by Yunjey Choi
    # (https://github.com/yunjey/domain-transfer-network)
    def resize_images(self, image_arrays, size=[64, 64]):
        # convert float type to integer
        image_arrays = (image_arrays * 255).astype('uint8')

        resized_image_arrays = np.zeros([image_arrays.shape[0]] + size)
        for i, image_array in enumerate(image_arrays):
            image = Image.fromarray(image_array)
            resized_image = image.resize(size=size, resample=Image.ANTIALIAS)

            resized_image_arrays[i] = np.asarray(resized_image)

        return np.expand_dims(resized_image_arrays, 3)