import tensorflow as tf
import tensorflow.contrib.slim as slim

def lrelu(x, alpha=0.2):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

class Model:
    def __init__(self, mode='train'):
        print('Model Mode :', mode)
        self.mode = mode
        self.size = 64
        self.pretrain_learning_rate = 0.0001
        self.train_learning_rate = 0.0003
        self.alpha = 15.0
        self.beta = 15.0
        self.gamma = 0.0

    def feature_extractor(self, image, reuse=False):
        if image.get_shape()[3] == 1:
            image = tf.image.grayscale_to_rgb(image)

        with tf.variable_scope('extractor', reuse=reuse):
            with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
                conv = slim.conv2d(image, num_outputs=128, kernel_size=2, stride=2)  # 32 x 32 x 128
                conv = slim.conv2d(conv, num_outputs=256, kernel_size=2, stride=2)  # 16 x 16 x 256
                conv = slim.conv2d(conv, num_outputs=256, kernel_size=2, stride=2)  # 8 x 8 x 256
                conv = slim.conv2d(conv, num_outputs=128, kernel_size=2, stride=2)  # 4 x 4 x 128
                conv = slim.conv2d(conv, num_outputs=128, kernel_size=4, stride=4)  # 1 x 1 x 128

            if self.mode == 'pretrain' or self.mode == 'pretrain-test':
                output = slim.conv2d(conv, num_outputs=10, kernel_size=1, stride=1)
                output = slim.flatten(output)

            elif self.mode == 'train' or self.mode == 'test':
                output = conv

        return output


    def generator(self, feature, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, \
                                weights_initializer=tf.contrib.layers.xavier_initializer()):
                conv = slim.conv2d_transpose(feature, num_outputs=128, kernel_size=4, stride=4)     # 4 x 4 x 128
                conv = slim.conv2d_transpose(conv, num_outputs=512, kernel_size=3, stride=2)        # 8 x 8 x 256
                conv = slim.conv2d_transpose(conv, num_outputs=256, kernel_size=3, stride=2)        # 16 x 16 x 512
                conv = slim.conv2d_transpose(conv, num_outputs=128, kernel_size=3, stride=2)        # 32 x 32 x 256
                #gen = slim.conv2d_transpose(conv, num_outputs=1, kernel_size=3, stride=2, activation_fn=tf.nn.tanh)     # 64 x 64 x 1
                gen = slim.conv2d_transpose(conv, num_outputs=3, kernel_size=3, stride=2, activation_fn=tf.nn.tanh)  # 64 x 64 x 1
                gen = tf.image.rgb_to_grayscale(gen)

        return gen


    def discriminator(self, image, reuse=False):
        input = slim.conv2d(image, num_outputs=3, kernel_size=1, stride=1)
        with tf.variable_scope('discriminator', reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=lrelu, normalizer_fn=slim.batch_norm, \
                                weights_initializer=tf.contrib.layers.xavier_initializer()):
                conv = slim.conv2d(input, num_outputs=128, kernel_size=3, stride=2)     # 32 x 32 x 128
                conv = slim.conv2d(conv, num_outputs=256, kernel_size=3, stride=2)      # 16 x 16 x 256
                conv = slim.conv2d(conv, num_outputs=512, kernel_size=3, stride=2)      # 8 x 8 x 512
                conv = slim.conv2d(conv, num_outputs=256, kernel_size=3, stride=2)      # 4 x 4 x 256
                conv = slim.conv2d(conv, num_outputs=1, kernel_size=4, stride=4)        # 1 x 1 x 1

            output = slim.flatten(conv)

        return output


    def build(self):
        if self.mode == 'pretrain' or self.mode == 'pretrain-test':
            # with tf.variable_scope('pretrain_variables', reuse=False):
            self.images = tf.placeholder(tf.float32, [None, 64, 64, 3])
            self.labels = tf.placeholder(tf.int64, [None])

            self.feature = self.feature_extractor(self.images)
            self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.feature)
            self.prediction = tf.argmax(self.feature, 1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.labels), tf.float32))

            with tf.variable_scope('pretrain_trainer', reuse=False):
                self.trainer = tf.train.AdamOptimizer(self.pretrain_learning_rate).minimize(self.loss)

        if self.mode == 'train' or self.mode == 'test':
            self.source_images = tf.placeholder(tf.float32, [None, 64, 64, 3])      # Fashion
            self.target_images = tf.placeholder(tf.float32, [None, 64, 64, 1])      # MNIST

            # source image
            self.src_feature = self.feature_extractor(self.source_images)                   # f(x)
            self.fake = self.generator(self.src_feature)                                    # g(f(x))
            self.logits_dsc1 = self.discriminator(self.fake)                                # D_1
            self.re_ext = self.feature_extractor(self.fake, reuse=True)                     # f(g(f(x)))

            self.loss_src_g = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.logits_dsc1), self.logits_dsc1)
            #self.loss_src_const = tf.reduce_mean(tf.square(self.src_feature - self.re_ext)) * self.alpha              # L2
            #self.loss_src_const = tf.reduce_mean(self.src_feature - self.re_ext) * 15.0                               # L1
            self.loss_src_const = tf.reduce_mean(tf.maximum(0.0, (1-self.src_feature * self.re_ext))) * self.alpha     # Hinge
            self.loss_src_d = tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.logits_dsc1), self.logits_dsc1)

            with tf.variable_scope('source_trainer', reuse=False):
                self.trainer_src_g = tf.train.AdamOptimizer(self.train_learning_rate).minimize(self.loss_src_g)
                self.trainer_src_d = tf.train.AdamOptimizer(self.train_learning_rate).minimize(self.loss_src_d)
                self.trainer_src_const = tf.train.AdamOptimizer(self.train_learning_rate).minimize(self.loss_src_const)

            # target image
            self.trg_feature = self.feature_extractor(self.target_images, reuse=True)       # f(x)
            self.remake = self.generator(self.trg_feature, reuse=True)                      # g(f(x))
            self.logits_dsc2 = self.discriminator(self.remake, reuse=True)                  # D_2
            self.logits_dsc3 = self.discriminator(self.target_images, reuse=True)           # D_3

            self.loss_trg_d1 = tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.logits_dsc2), self.logits_dsc2)
            self.loss_trg_d2 = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.logits_dsc3), self.logits_dsc3)
            self.loss_trg_g = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.logits_dsc2), self.logits_dsc2)

            #self.loss_trg_tid = tf.reduce_mean(tf.square(self.target_images - self.remake)) * self.beta            # L2
            #self.loss_trg_tid = tf.reduce_mean(self.target_images - self.remake) * 15.0                            # L1
            self.loss_trg_tid = tf.reduce_mean(tf.maximum(0.0, (1-self.target_images * self.remake))) * self.beta   # Hinge

            self.loss_trg_d = self.loss_trg_d1 + self.loss_trg_d2
            self.loss_trg_g = self.loss_trg_g + self.loss_trg_tid

            with tf.variable_scope('target_trainer', reuse=False):
                self.trainer_trg_d = tf.train.AdamOptimizer(self.train_learning_rate).minimize(self.loss_trg_d)
                self.trainer_trg_g = tf.train.AdamOptimizer(self.train_learning_rate).minimize(self.loss_trg_g)

            # Total Variation Loss
            self.loss_tv_src = tf.image.total_variation(self.fake)
            self.loss_tv_trg = tf.image.total_variation(self.remake)
            self.loss_tv = tf.reduce_mean(self.loss_tv_src + self.loss_tv_trg) * self.gamma

            with tf.variable_scope('tv_trainer', reuse=False):
                self.trainer_tv = tf.train.AdamOptimizer(self.train_learning_rate).minimize(self.loss_tv)
