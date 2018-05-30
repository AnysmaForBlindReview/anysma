# -*- coding: utf-8 -*-
"""Train a LVQ Capsule Network on the smallNORB dataset (resized to 48x48).

The network is inspired by the SimpleNet. See the paper for a description.

The network trains to an accuracy of >95% in ~200 epochs.
To train the network of the paper we applied the following schedule:
    0-50 epoch:     lr = 0.0005
    51-100 epoch:   lr = 0.0001
    101-150 epoch:  lr = 0.00005
    151-200 epoch:  lr = 0.00001
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

from keras import layers, models, optimizers
from keras.utils import to_categorical
from keras.layers import *
from keras.initializers import *
from keras.utils.data_utils import get_file
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks

from anysma import Capsule
from anysma.modules import InputModule
from anysma.modules.measuring import TangentDistance, RestrictedTangentDistance
from anysma.modules.routing import GibbsRouting
from anysma.modules.final import Classification
from anysma.losses import generalized_kullback_leibler_divergence
from anysma.regularizers import MaxDistance
from anysma.utils.caps_utils import list_to_dict
from anysma.callbacks import TensorBoard
from anysma.datasets import smallnorb

from PIL import ImageEnhance
from PIL import Image
from skimage import transform

import matplotlib
matplotlib.use('Agg')  # needed to avoid cloud errors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

K.set_image_data_format('channels_last')


class Mask(layers.Layer):
    """
    Mask all vectors except the best matching one.
    """
    def __init__(self, **kwargs):
        super(Mask, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        signals, prob = inputs
        mask = K.one_hot(indices=K.argmax(prob, 1), num_classes=prob.get_shape().as_list()[1])

        masked = K.batch_flatten(signals * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        return tuple([None, input_shape[0][1] * input_shape[0][2]])


def crop(dimension, start, end):
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)


def center_crop(images):
    images = images[:, 8:48-8, 8:48-8, :]
    return images


def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y + dy), x:(x + dx), :]


def random_brightness(images, range):
    images = (np.copy(images) * 255).astype('uint8')
    for i, img in enumerate(images):
        u = np.random.uniform(range[0], range[1])

        imgL = Image.fromarray(img[:, :, 0], 'L')
        imgL = imgenhancer_Brightness = ImageEnhance.Brightness(imgL)
        imgL = imgenhancer_Brightness.enhance(u)
        imgL = np.array(imgL.getdata())
        imgL = imgL.reshape((48, 48))
        images[i, :, :, 0] = imgL

        imgR = Image.fromarray(img[:, :, 1], 'L')
        imgR = imgenhancer_Brightness = ImageEnhance.Brightness(imgR)
        imgR = imgenhancer_Brightness.enhance(u)
        imgR = np.array(imgR.getdata())
        imgR = imgR.reshape((48, 48))
        images[i, :, :, 1] = imgR

    return images.astype('float32') / 255.


def random_contrast(images, range):
    images = (np.copy(images) * 255).astype('uint8')
    for i, img in enumerate(images):
        u = np.random.uniform(range[0], range[1])

        imgL = Image.fromarray(img[:, :, 0], 'L')
        imgL = imgenhancer_Contrast = ImageEnhance.Contrast(imgL)
        imgL = imgenhancer_Contrast.enhance(u)
        imgL = np.array(imgL.getdata())
        imgL = imgL.reshape((48, 48))
        images[i, :, :, 0] = imgL

        imgR = Image.fromarray(img[:, :, 1], 'L')
        imgR = imgenhancer_Contrast = ImageEnhance.Contrast(imgR)
        imgR = imgenhancer_Contrast.enhance(u)
        imgR = np.array(imgR.getdata())
        imgR = imgR.reshape((48, 48))
        images[i, :, :, 1] = imgR

    return images.astype('float32') / 255.


def LvqCapsNet(input_shape):
    input_img = layers.Input(shape=input_shape)

    # Block 1
    caps1 = Capsule()
    caps1.add(Conv2D(32+2, (3, 3), padding='same', kernel_initializer=glorot_normal(), input_shape=x_train.shape[1:]))
    caps1.add(BatchNormalization())
    caps1.add(Activation('relu'))
    caps1.add(Dropout(0.25))
    x = caps1(input_img)

    # Block 2
    caps2 = Capsule()
    caps2.add(Conv2D(64+2, (3, 3), padding='same', kernel_initializer=glorot_normal()))
    caps2.add(BatchNormalization())
    caps2.add(Activation('relu'))
    caps2.add(Dropout(0.25))
    x = caps2(x)

    # Block 3
    caps3 = Capsule()
    caps3.add(Conv2D(64+2, (3, 3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    caps3.add(BatchNormalization())
    caps3.add(Activation('relu'))
    caps3.add(Dropout(0.25))
    x = caps3(x)

    # Block 4
    caps4 = Capsule(prototype_distribution=32)
    caps4.add(Conv2D(64+2, (5, 5), strides=2, padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    caps4.add(BatchNormalization())
    caps4.add(Activation('relu'))
    caps4.add(Dropout(0.25))
    x = caps4(x)

    # Block 5
    caps5 = Capsule()
    caps5.add(Conv2D(32+2, (3, 3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    caps5.add(Dropout(0.25))
    x = caps5(x)

    # Block 6
    caps6 = Capsule()
    caps6.add(Conv2D(64+2, (3, 3), padding='same', kernel_initializer=glorot_normal()))
    caps6.add(Dropout(0.25))
    x = caps6(x)

    # Block 7
    x = Conv2D(64 + 2, (3, 3), padding='same', kernel_initializer=glorot_normal())(x)
    x = [crop(3, 0, 64)(x), crop(3, 64, 66)(x)]
    x[1] = Activation('relu')(x[1])
    x[1] = Flatten()(x[1])

    # Caps1
    caps7 = Capsule(prototype_distribution=(1, 8 * 8))
    caps7.add(InputModule(signal_shape=(-1, 32), dissimilarity_initializer=None, trainable=False))
    diss7 = TangentDistance(squared_dissimilarity=False, epsilon=1.e-12, linear_factor=0.66, projected_atom_shape=16)
    caps7.add(diss7)
    caps7.add(GibbsRouting(norm_axis='channels', trainable=False))
    x = caps7(list_to_dict(x))

    # Caps2
    caps8 = Capsule(prototype_distribution=(1, 4 * 4))
    caps8.add(Reshape((8, 8, 32)))
    caps8.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=glorot_normal()))
    caps8.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=glorot_normal()))
    caps8.add(InputModule(signal_shape=(8 * 8, 64), dissimilarity_initializer=None, trainable=False))
    diss8 = TangentDistance(projected_atom_shape=16, squared_dissimilarity=False,
                            epsilon=1.e-12, linear_factor=0.66, signal_output='signals')
    caps8.add(diss8)
    caps8.add(GibbsRouting(norm_axis='channels', trainable=False))
    x = caps8(x)

    # Caps3
    digit_caps = Capsule(prototype_distribution=(1, 5))
    digit_caps.add(Reshape((4, 4, 64)))
    digit_caps.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=glorot_normal()))
    digit_caps.add(InputModule(signal_shape=128, dissimilarity_initializer=None, trainable=False))
    diss = RestrictedTangentDistance(projected_atom_shape=16, epsilon=1.e-12, squared_dissimilarity=False,
                                     linear_factor=0.66, signal_output='signals')
    digit_caps.add(diss)
    digit_caps.add(GibbsRouting(norm_axis='channels', trainable=False,
                                diss_regularizer=MaxDistance(alpha=0.0001)))
    digit_caps.add(Classification(probability_transformation='neg_softmax', name='lvq_capsule'))

    digitcaps = digit_caps(x)

    model = models.Model(input_img, digitcaps[2])

    return model


def train(model, data, args):
    # unpacking the data
    (x_train, y_train), (x_val, y_val) = data

    def train_generator(x, y, batch_size):
        crop_length = 32

        train_datagen = ImageDataGenerator()

        # make a pseudo RGB image
        x = np.concatenate((x[:, :, :, :], x[:, :, :, 0:1]), -1)

        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while True:
            batch_x, batch_y = generator.next()

            batch_x = batch_x[:, :, :, 0:2]
            batch_x = random_brightness(batch_x, (0.2, 1.8))
            batch_x = random_contrast(batch_x, (0.3, 3))

            batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, batch_x.shape[3]))

            for i in range(batch_x.shape[0]):
                batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))

            yield (batch_crops, batch_y)

    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                     batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5',
                                           monitor='val_acc', save_best_only=True,
                                           save_weights_only=True, verbose=1)

    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=generalized_kullback_leibler_divergence,
                  metrics={'lvq_capsule': 'accuracy'})

    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[center_crop(x_val), y_val],
                        callbacks=[log, checkpoint, tb])

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    return model


def test(model, data, args):
    x_val, y_val = data
    y_pred = model.predict(center_crop(x_val), batch_size=args.batch_size)
    print('-' * 30 + 'Begin: Validation' + '-' * 30)

    print('Validation acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_val, 1)) / y_val.shape[0])

    print('-' * 30 + 'End: Validation' + '-' * 30)


def plot_heatmaps(model, data, args):
    print('-' * 30 + 'Begin: plot heatmaps' + '-' * 30)

    data = center_crop(data)

    for _ in range(10):
        idx = np.random.randint(0, data.shape[0])

        plt.clf()
        fig = plt.figure(1)
        gridspec.GridSpec(1, 10)

        plt.subplot2grid((1, 10), (0, 0), colspan=2)
        ax = plt.gca()
        img = plt.imshow(data[idx, :, :, 0])
        img.set_cmap('gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('input')

        plt.subplot2grid((1, 10), (0, 2), colspan=2)
        out = models.Model(model.input, model.get_layer('activation_5').output).predict(data[idx:idx+1])
        ax = plt.gca()
        img = ax.imshow(out[0, :, :, 0])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.03)
        plt.colorbar(img, cax=cax)
        ax.set_title('d_in Caps1')

        plt.subplot2grid((1, 10), (0, 4), colspan=2)
        out = np.reshape(models.Model(model.input,
                                      model.get_layer('gibbs_routing_1').output[1]).predict(data[idx:idx+1]),
                         (-1, 8, 8, 1))
        ax = plt.gca()
        img = ax.imshow(out[0, :, :, 0])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.03)
        plt.colorbar(img, cax=cax)
        ax.set_title('d_in Caps2')

        plt.subplot2grid((1, 10), (0, 6), colspan=2)
        out = np.reshape(models.Model(model.input,
                                      model.get_layer('gibbs_routing_2').output[1]).predict(data[idx:idx+1]),
                         (-1, 4, 4, 1))
        ax = plt.gca()
        img = ax.imshow(out[0, :, :, 0])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.03)
        plt.colorbar(img, cax=cax)
        ax.set_title('d_in Caps3')

        plt.subplot2grid((1,10), (0,8))
        out = np.reshape(models.Model(model.input,
                                      model.get_layer('gibbs_routing_3').output[1]).predict(data[idx:idx+1]),
                         (-1, 5, 1, 1))
        ax = plt.gca()
        img = ax.imshow(np.tile(out[0, :, :, 0], (1, 4)))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.03)
        plt.colorbar(img, cax=cax)
        ax.set_xticks([])
        ax.set_title('d_out')

        plt.subplot2grid((1, 10), (0, 9))
        out = np.reshape(models.Model(model.input,
                                      model.get_layer('lvq_capsule').output[2]).predict(data[idx:idx+1]),
                         (-1, 5, 1, 1))
        ax = plt.gca()
        img = ax.imshow(np.tile(out[0, :, :, 0], (1, 4)))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.03)
        plt.colorbar(img, cax=cax)
        ax.set_xticks([])
        ax.set_title('p_out')

        # we have to call it a couple of times to adjust the plot correctly
        for _ in range(5):
            plt.tight_layout(pad=0, w_pad=0.1, h_pad=0.0)
            fig.set_size_inches(w=9, h=1.5)
        plt.savefig(args.save_dir + '/heatmap_' + str(idx) + '.png')

    print('-' * 30 + 'End: plot heatmaps' + '-' * 30)


def plot_spectral_lines(model, data, args):
    print('-' * 30 + 'Begin: plot spectral lines' + '-' * 30)

    out = models.Model(model.input,
                       model.get_layer('gibbs_routing_2').output[1]).predict(center_crop(data[:1000]),
                                                                             batch_size=args.batch_size)

    plt.clf()
    fig = plt.figure(1)
    classes = ['animal', 'human', 'plane', 'truck', 'car']

    for idx in range(5):
        ax = plt.subplot(3, 2, idx + 1)
        tmp = out[y_train[:1000, idx].astype('bool'), :]
        ax.set_title('class ' + str(idx) + ': ' + classes[idx])
        plt.plot(np.arange(1, 17), tmp.transpose())

    # we have to call it a couple of times to adjust the plot correctly
    for _ in range(5):
        plt.tight_layout(pad=0, w_pad=0.1, h_pad=0.0)
        fig.set_size_inches(w=10, h=5)
    plt.savefig(args.save_dir + '/spectral_lines' + '.png')

    print('-' * 30 + 'End: plot spectral lines' + '-' * 30)


def load_smallnorb48(setting):
    # info translation:
    #   column 0: the instance in the category (0 to 9)
    #   column 1: the elevation (0 to 8, which mean cameras are 30, 35,40,45,50,55,60,65,70 degrees from the horizontal
    #             respectively)
    #   column 2: the azimuth (0,2,4,...,34, multiply by 10 to get the azimuth in degrees)
    #   column 3: the lighting condition (0 to 5)

    (x_train, y_train, info_train), (x_val, y_val, info_val) = smallnorb.load_data()

    x_train = x_train.reshape(-1, 96, 96, 2).astype('float32') / 255.
    x_val = x_val.reshape(-1, 96, 96, 2).astype('float32') / 255.

    # resize to 48x48
    def resize(data):
        data_ = np.zeros((data.shape[0], 48, 48, 2))
        for i, img in enumerate(data):
            for j in range(2):
                data_[i, :, :, j] = transform.resize(img[:, :, j], (48, 48), mode='reflect')
        return data_

    print('-' * 30 + 'resize input images' + '-' * 30)
    x_val = resize(x_val)
    x_train = resize(x_train)

    if setting != 'normal':
        if setting == 'elevation':
            idx_train = np.in1d(info_train[:, 1], (0, 1, 2))
            idx_val = np.in1d(info_val[:, 1], (0, 1, 2))
            idx_test = np.in1d(info_val[:, 1], list(range(3, 9, 1)))
        else:
            idx_train = np.in1d(info_train[:, 2], (0, 2, 4, 30, 32, 34))
            idx_val = np.in1d(info_val[:, 2], (0, 2, 4, 30, 32, 34))
            idx_test = np.in1d(info_val[:, 2], list(range(6, 30, 2)))

        x_train = x_train[idx_train, :]
        x_test = x_val[idx_test, :]
        x_val = x_val[idx_val, :]

        y_train = to_categorical(y_train[idx_train].astype('float32'))
        y_test = to_categorical(y_val[idx_test].astype('float32'))
        y_val = to_categorical(y_val[idx_val].astype('float32'))
    else:
        y_train = to_categorical(y_train.astype('float32'))
        y_val = to_categorical(y_val.astype('float32'))
        x_test = None
        y_test = None

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="LVQ Capsule network on smallNORB48.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.0005, type=float,
                        help="Initial learning rate")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help='The `weights` argument should be either `None` (random initialization), `smallnorb48` or '
                             'the path to the weights file to be loaded.')
    parser.add_argument('-s', '--setting', default='normal',
                        help="Experimental setting 'normal', 'elevation' or 'azimuth'.")
    args = parser.parse_args()
    print(args)

    # load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_smallnorb48(args.setting)

    # define model
    model = LvqCapsNet(input_shape=(32, 32, 2))
    model.summary(line_length=200, positions=[.33, .6, .67, 1.])

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.setting not in ('normal', 'elevation', 'azimuth'):
        raise ValueError('The `setting` argument should be either `normal` (train and test on the standard datasets), '
                         '`elevation` or `azimuth`.')

    if args.setting == 'elevation':
        NAME = 'smallnorb48_el.h5'
        HASH = '1e229b53096ff7df124b6c146869002f'
        URL = 'http://tiny.cc/anysma_models_snorb48_el'
    elif args.setting == 'azimuth':
        NAME = 'smallnorb48_az.h5'
        HASH = '2a44c034226a96a51000ca1c97f0bbb4'
        URL = 'http://tiny.cc/anysma_models_snorb48_az'
    else:
        NAME = 'smallnorb48.h5'
        HASH = '97a60d803b143f1386de538e4c1e3e92'
        URL = 'http://tiny.cc/anysma_models_snorb48'

    if not (args.weights in {'smallnorb48', None} or os.path.exists(args.weights)):
        raise ValueError('The `weights` argument should be either `None` (random initialization), `smallnorb48` '
                         'or the path to the weights file to be loaded.')

    if args.weights is not None:  # init the model weights with provided one
        if args.weights == 'smallnorb48':
            args.weights = get_file(NAME,
                                    URL,
                                    cache_subdir='models',
                                    file_hash=HASH)
        model.load_weights(args.weights)

    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_val, y_val)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        plot_spectral_lines(model, x_val, args)
        plot_heatmaps(model, x_val, args)
        test(model=model, data=(x_val, y_val), args=args)

    if x_test is not None:
        y_pred = model.predict(center_crop(x_test), batch_size=args.batch_size)
        print('-' * 30 + 'Begin: test test' + '-' * 30)
        print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])
