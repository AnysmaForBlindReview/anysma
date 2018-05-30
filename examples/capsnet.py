# -*- coding: utf-8 -*-
"""Train a CapsNet Network on the MNIST dataset.

See the corresponding paper for explanations of the network
@inproceedings{sabour2017dynamic,
  title={Dynamic routing between capsules},
  author={Sabour, Sara and Frosst, Nicholas and Hinton, Geoffrey E},
  booktitle={Advances in Neural Information Processing Systems},
  pages={3859--3869},
  year={2017}
}

The network trains to an accuracy of >99% in few epochs. The most epochs are needed to train the reconstruction network.

The implementation is based on the code of (thanks to the great and inspiring implementation!):
    Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
with changes to incorporate the anysma package.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import argparse

import numpy as np
from PIL import Image
from keras import backend as K
from keras import layers, models, optimizers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks

from anysma import Capsule
from anysma.modules import InputModule
from anysma.modules.final import DynamicRouting
from anysma.modules.transformations import LinearTransformation
from anysma.utils.normalization_funcs import dynamic_routing_squash as squash
from anysma.callbacks import TensorBoard
from anysma.losses import margin_loss
from anysma.datasets import mnist

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


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


def CapsNet(input_shape, n_class, routings):
    """ Initialize the CapsNet"""
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    digitcaps = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primary_caps = Capsule(name='PrimaryCaps')
    primary_caps.add(layers.Conv2D(filters=8 * 32, kernel_size=9, strides=2, padding='valid', name='primarycap_conv2d'))
    primary_caps.add(layers.Reshape(target_shape=[-1, 8], name='primarycap_reshape'))
    primary_caps.add(layers.Lambda(squash, name='primarycap_squash'))

    digitcaps = primary_caps(digitcaps)

    # Layer 3: Capsule layer. Routing algorithm works here.
    digit_caps = Capsule(name='digitcaps', prototype_distribution=(1, n_class))
    digit_caps.add(InputModule(signal_shape=None, dissimilarity_initializer='zeros', trainable=False))
    digit_caps.add(LinearTransformation(output_dim=16, scope='local'))
    digit_caps.add(DynamicRouting(iterations=routings, name='capsnet'))

    digitcaps = digit_caps(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps[0], y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()([digitcaps[0], digitcaps[2]])  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16 * n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [digitcaps[2], decoder(masked_by_y)])
    eval_model = models.Model(x, [digitcaps[2], decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps[0], noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def train(model, data, args):
    """
    Training
    :param model: the model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                     batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    def train_generator(x, y, batch_size):
        train_datagen = ImageDataGenerator(width_shift_range=2,
                                           height_shift_range=2)
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint])

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    return model


def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=args.batch_size)
    print('-' * 30 + 'Begin: test' + '-' * 30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)


def manipulate_latent(model, data, args):
    print('-' * 30 + 'Begin: manipulate' + '-' * 30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:, :, dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate.")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder.")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard.")
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset.")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate during test.")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing.")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings)
    model.summary(line_length=200, positions=[.33, .6, .67, 1.])

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        manipulate_latent(manipulate_model, (x_test, y_test), args)
        test(model=eval_model, data=(x_test, y_test), args=args)
