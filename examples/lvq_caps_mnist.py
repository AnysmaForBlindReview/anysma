# -*- coding: utf-8 -*-
"""Train a LVQ Capsule Network on the MNIST dataset.

The network is inspired by the SimpleNet. See the paper for a description.

The network trains to an accuracy of >99% in few epochs. The most epochs are needed to train the reconstruction network.

To train the network of the paper we applied the following schedule:
    0-50 epoch:     lr = 0.001
    51-100 epoch:   lr = 0.0005
    101-150 epoch:  lr = 0.0001
    151-200 epoch:  lr = 0.00005
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import argparse
from PIL import Image

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
from anysma.utils.caps_utils import list_to_dict
from anysma.regularizers import MaxDistance
from anysma.callbacks import TensorBoard
from anysma.datasets import mnist, affnist

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


def LvqCapsNet(input_shape):
    input_img = Input(shape=input_shape)

    # Block 1
    caps1 = Capsule()
    caps1.add(Conv2D(32 + 1, (3, 3), padding='same', kernel_initializer=glorot_normal()))
    caps1.add(BatchNormalization())
    caps1.add(Activation('relu'))
    caps1.add(Dropout(0.25))
    x = caps1(input_img)

    # Block 2
    caps2 = Capsule()
    caps2.add(Conv2D(64 + 1, (3, 3), padding='same', kernel_initializer=glorot_normal()))
    caps2.add(BatchNormalization())
    caps2.add(Activation('relu'))
    caps2.add(Dropout(0.25))
    x = caps2(x)

    # Block 3
    caps3 = Capsule()
    caps3.add(Conv2D(64 + 1, (3, 3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    caps3.add(BatchNormalization())
    caps3.add(Activation('relu'))
    caps3.add(Dropout(0.25))
    x = caps3(x)

    # Block 4
    caps4 = Capsule(prototype_distribution=32)
    caps4.add(Conv2D(64 + 1, (5, 5), strides=2, padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    caps4.add(BatchNormalization())
    caps4.add(Activation('relu'))
    caps4.add(Dropout(0.25))
    x = caps4(x)

    # Block 5
    caps5 = Capsule()
    caps5.add(Conv2D(32 + 1, (3, 3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    caps5.add(Dropout(0.25))
    x = caps5(x)

    # Block 6
    caps6 = Capsule()
    caps6.add(Conv2D(64 + 1, (3, 3), padding='same', kernel_initializer=glorot_normal()))
    caps6.add(Dropout(0.25))
    x = caps6(x)

    # Block 7
    x = Conv2D(64 + 1, (3, 3), padding='same', kernel_initializer=glorot_normal())(x)
    x = [crop(3, 0, 64)(x), crop(3, 64, 65)(x)]
    x[1] = Activation('relu')(x[1])
    x[1] = Flatten()(x[1])

    # Caps1
    caps7 = Capsule(prototype_distribution=(1, 8 * 8))
    caps7.add(InputModule(signal_shape=(-1, 64), dissimilarity_initializer=None, trainable=False))
    diss7 = TangentDistance(squared_dissimilarity=False, epsilon=1.e-12, linear_factor=0.66, projected_atom_shape=16)
    caps7.add(diss7)
    caps7.add(GibbsRouting(norm_axis='channels', trainable=False))
    x = caps7(list_to_dict(x))

    # Caps2
    caps8 = Capsule(prototype_distribution=(1, 4 * 4))
    caps8.add(Reshape((8, 8, 64)))
    caps8.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=glorot_normal()))
    caps8.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=glorot_normal()))
    caps8.add(InputModule(signal_shape=(8 * 8, 64), dissimilarity_initializer=None, trainable=False))
    diss8 = TangentDistance(projected_atom_shape=16, squared_dissimilarity=False,
                            epsilon=1.e-12, linear_factor=0.66, signal_output='signals')
    caps8.add(diss8)
    caps8.add(GibbsRouting(norm_axis='channels', trainable=False))
    x = caps8(x)

    # Caps3
    digit_caps = Capsule(prototype_distribution=(1, 10))
    digit_caps.add(Reshape((4, 4, 64)))
    digit_caps.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=glorot_normal()))
    digit_caps.add(InputModule(signal_shape=128, dissimilarity_initializer=None, trainable=False))
    diss = RestrictedTangentDistance(projected_atom_shape=16, epsilon=1.e-12, squared_dissimilarity=False,
                                     linear_factor=0.66, signal_output='signals')
    digit_caps.add(diss)
    digit_caps.add(GibbsRouting(norm_axis='channels', trainable=False,
                                diss_regularizer=MaxDistance(alpha=0.0001)))
    digit_caps.add(Classification(probability_transformation='neg_softmax', name='lvq_caps'))

    digitcaps = digit_caps(x)

    # intermediate model for Caps2; used for visualizations
    input_diss8 = [Input((4, 4, 64)), Input((16,))]
    model_vis_caps2 = models.Model(input_diss8, digit_caps(list_to_dict(input_diss8)))

    # Decoder network.
    y = layers.Input(shape=(10,))
    masked_by_y = Mask()([digitcaps[0], y])

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=128 * 10))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod((28, 28, 1)), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=(28, 28, 1), name='out_recon'))

    # Models for training and evaluation (prediction)
    model = models.Model([input_img, y], [digitcaps[2], decoder(masked_by_y)])

    return model, decoder,  model_vis_caps2


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    def train_generator(x, y, batch_size):
        train_datagen = ImageDataGenerator(width_shift_range=2,
                                           height_shift_range=2)
        generator = train_datagen.flow(x, y, batch_size=batch_size, seed=1)
        # generator without shift
        train_datagen2 = ImageDataGenerator(width_shift_range=0,
                                            height_shift_range=0)
        generator2 = train_datagen2.flow(x, y, batch_size=batch_size, seed=1)
        while 1:
            x_batch, y_batch = generator.next()
            x_batch2, y_batch2 = generator2.next()
            assert np.all(y_batch == y_batch2)
            yield ([x_batch, y_batch], [y_batch2, x_batch2])

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                     batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_lvq_caps_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[generalized_kullback_leibler_divergence, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'lvq_caps': 'accuracy'})

    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint])

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    return model


def test(model, data, args, text):
    x_test, y_test = data
    print('-' * 30 + 'Begin: ' + text + ' ' + '-' * 30)
    y_pred, _ = model.predict([x_test, y_test], batch_size=args.batch_size)

    print(text + ' acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])

    print('-' * 30 + 'End: ' + text + ' ' + '-' * 30)


def plot_prototypes(model, decoder, args):
    print('-' * 30 + 'Begin: plot ideal prototypes (center of r-orthotope)' + '-' * 30)
    proto_layer = model.get_layer('restricted_tangent_distance_1')
    protos = proto_layer.get_weights()

    protos_recons = []
    for dim in range(10):
        mask = np.zeros((10, 128))
        mask[dim, :] = 1
        protos_masked = protos * mask
        protos_masked = np.reshape(protos_masked, [1, -1])
        protos_recon = decoder.predict([protos_masked])
        protos_recons.append(protos_recon)

    protos_recons = np.concatenate(protos_recons)

    img = combine_images(protos_recons, height=16)
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/ideal_protos_reconstructed.png')

    print('-' * 30 + 'End: plot ideal prototypes (center of r-orthotope)' + '-' * 30)


def plot_heatmaps(model, data, args):
    print('-' * 30 + 'Begin: plot heatmaps' + '-' * 30)

    (x_test, y_test) = data

    for _ in range(10):
        idx = np.random.randint(0, x_test.shape[0])

        plt.clf()
        fig = plt.figure(1)
        gridspec.GridSpec(1, 10)

        plt.subplot2grid((1, 10), (0, 0), colspan=2)
        ax = plt.gca()
        img = plt.imshow(x_test[idx, :, :, 0])
        img.set_cmap('gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('input')

        plt.subplot2grid((1, 10), (0, 2), colspan=2)
        out = models.Model(model2.input[0], model2.get_layer('activation_10').output).predict(x_test[idx:idx+1])
        ax = plt.gca()
        img = ax.imshow(out[0, :, :, 0])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.03)
        plt.colorbar(img, cax=cax)
        ax.set_title('d_in Caps1')

        plt.subplot2grid((1, 10), (0, 4), colspan=2)
        out = np.reshape(models.Model(model2.input[0],
                                      model2.get_layer('gibbs_routing_4').output[1]).predict(x_test[idx:idx+1]),
                         (-1, 8, 8, 1))
        ax = plt.gca()
        img = ax.imshow(out[0, :, :, 0])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.03)
        plt.colorbar(img, cax=cax)
        ax.set_title('d_in Caps2')

        plt.subplot2grid((1, 10), (0, 6), colspan=2)
        out = np.reshape(models.Model(model2.input[0],
                                      model2.get_layer('gibbs_routing_5').output[1]).predict(x_test[idx:idx+1]),
                         (-1, 4, 4, 1))
        ax = plt.gca()
        img = ax.imshow(out[0, :, :, 0])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.03)
        plt.colorbar(img, cax=cax)
        ax.set_title('d_in Caps3')

        _, x_recon = model2.predict([x_test[idx:idx+1, :], y_test[idx:idx+1, :]])

        plt.subplot2grid((1, 10), (0, 8), colspan=2)
        ax = plt.gca()
        img = plt.imshow(x_recon[0, :, :, 0])
        img.set_cmap('gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('reconstructed')

        # we have to call it a couple of times to adjust the plot correctly
        for _ in range(5):
            plt.tight_layout(pad=0, w_pad=0.1, h_pad=0.0)
            fig.set_size_inches(w=7.3, h=1.5)
        plt.savefig(args.save_dir + '/heatmap_' + str(idx) + '.png')

    print('-' * 30 + 'End: plot heatmaps' + '-' * 30)


def plot_spectral_lines(model, data, args):
    print('-' * 30 + 'Begin: plot spectral lines' + '-' * 30)

    (x_test, y_test) = data

    out = models.Model(model.input[0],
                       model.get_layer('gibbs_routing_2').output[1]).predict(x_test[:1000],
                                                                             batch_size=args.batch_size)

    plt.clf()
    fig = plt.figure(1)

    for idx in range(10):
        plt.subplot(5, 2, idx + 1)
        tmp = out[y_test[:1000, idx].astype('bool'), :]

        plt.plot(np.arange(1, 17), tmp.transpose())
        ax = plt.gca()
        ax.set_title('class ' + str(idx) + ': digit ' + str(idx))

    # we have to call it a couple of times to adjust the plot correctly
    for _ in range(5):
        plt.tight_layout(pad=0, w_pad=0.1, h_pad=0.0)
        fig.set_size_inches(w=10, h=8)
    plt.savefig(args.save_dir + '/spectral_lines' + '.png')

    print('-' * 30 + 'End: plot spectral lines' + '-' * 30)


def plot_random_walk_caps3(model, decoder):
    print('-' * 30 + 'Begin: plot random walk Caps3' + '-' * 30)
    diss = model.get_layer('restricted_tangent_distance_1')
    protos, tangents, bounds = diss.get_weights()
    num = 20
    images = np.zeros((28 * 10, 28 * num))
    for i in range(num):
        for idx in range(10):
            params = np.tile(protos[idx, :] + np.matmul(tangents[idx, :],
                                                        np.clip((np.random.rand(16) - 0.5) * np.max(bounds[idx, :]) * 2,
                                                                -bounds[idx, :], bounds[idx, :])), [10, 1])

            mask = np.zeros((10, 128))
            mask[idx, :] = 1
            protos_masked = params * mask
            protos_masked = np.reshape(protos_masked, [1, -1])
            protos_recon = decoder.predict([protos_masked])

            images[idx * 28: (idx + 1) * 28, i * 28: (i + 1) * 28] = np.reshape(protos_recon, [28, 28])

    plt.clf()
    fig = plt.figure(1)

    img = plt.imshow(images)
    ax = plt.gca()
    img.set_cmap('gray')
    ax.set_xticks([])
    ax.set_yticks([])

    # we have to call it a couple of times to adjust the plot correctly
    for _ in range(5):
        plt.tight_layout(pad=0, w_pad=0.1, h_pad=0.0)
        fig.set_size_inches(w=10, h=5)
    plt.savefig(args.save_dir + '/random_walk_caps3' + '.png')

    print('-' * 30 + 'End: plot random walk Caps3' + '-' * 30)


def plot_random_walk_caps2(model, model_vis_caps2, decoder, proto_num):
    print('-' * 30 + 'Begin: plot random walk Caps2' + '-' * 30)

    proto_num = proto_num - 1
    diss = model.get_layer('restricted_tangent_distance_1')
    protos_final, _, _ = diss.get_weights()

    num = 20
    images = np.zeros((28 * 10, 28 * num))
    for idx in range(10):
        params = np.tile(protos_final[idx, :], [10, 1])

        mask = np.zeros((10, 128))
        mask[idx, :] = 1
        protos_masked = params * mask
        protos_masked = np.reshape(protos_masked, [1, -1])
        protos_recon = decoder.predict([protos_masked])

        ideal = np.reshape(protos_recon, [1, 28, 28, 1])

        protos, tangents = model.get_layer('tangent_distance_2').get_weights()

        out_ideal = models.Model(model.input[0], [model.get_layer('gibbs_routing_2').output[0],
                                                  model.get_layer('gibbs_routing_2').output[1]]).predict(ideal,
                                                                                                         batch_size=16)

        for j in range(num):
            out = out_ideal
            for _, i in enumerate((proto_num,)):
                out[0][0, i, :] = protos[i, :] + np.matmul(tangents[i, :], (np.random.rand(16) - 0.5) * 10)
                out[1][0, i] = 0

            out2 = model_vis_caps2.predict([np.reshape(out[0], (1, 4, 4, 64)), np.reshape(out[1], (1, 16))])[0]

            mask = np.zeros((10, 128))
            mask[idx, :] = 1
            protos_masked = out2 * mask
            protos_masked = np.reshape(protos_masked, [1, -1])
            img = decoder.predict([protos_masked])
            images[idx * 28: (idx + 1) * 28, j * 28: (j + 1) * 28] = np.reshape(img, [28, 28])

    plt.clf()
    fig = plt.figure(1)

    img = plt.imshow(images)
    ax = plt.gca()
    img.set_cmap('gray')
    ax.set_xticks([])
    ax.set_yticks([])

    # we have to call it a couple of times to adjust the plot correctly
    for _ in range(5):
        plt.tight_layout(pad=0, w_pad=0.1, h_pad=0.0)
        fig.set_size_inches(w=10, h=5)
    plt.savefig(args.save_dir + '/random_walk_caps2_proto' + str(proto_num+1) + '.png')

    print('-' * 30 + 'End: plot random walk Caps2' + '-' * 30)


def plot_decoder(model, data):
    print('-' * 30 + 'Begin: plot decoder' + '-' * 30)

    (x_test, y_test) = data

    num = 10
    images = np.zeros((28 * 10, 28 * num*2))
    for idx in range(10):
        digit_idx = np.where(np.argmax(y_test, 1) == idx)[0]
        for i in range(0, num*2, 2):
            j = np.random.randint(0, len(digit_idx))
            sample = x_test[digit_idx[j] : digit_idx[j]+1, :]
            _, x_recon = model.predict([sample,
                                        to_categorical(np.expand_dims(np.array([idx]), 0), 10)],
                                       batch_size=16)

            images[idx * 28: (idx + 1) * 28, i * 28: (i + 1) * 28] = np.reshape(sample, [28, 28])
            images[idx * 28: (idx + 1) * 28, (i + 1) * 28: (i + 2) * 28] = np.reshape(x_recon, [28, 28])

    plt.clf()
    fig = plt.figure(1)

    img = plt.imshow(images)
    ax = plt.gca()
    img.set_cmap('gray')
    ax.set_xticks([])
    ax.set_yticks([])

    # we have to call it a couple of times to adjust the plot correctly
    for _ in range(5):
        plt.tight_layout(pad=0, w_pad=0.1, h_pad=0.0)
        fig.set_size_inches(w=10, h=5)
    plt.savefig(args.save_dir + '/decoder' + '.png')

    print('-' * 30 + 'End: plot decoder' + '-' * 30)


def load_mnist_affnist():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_val, y_val) = mnist.load_data()
    _, (x_test, y_test) = affnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_val = x_val.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 40, 40, 1).astype('float32') / 255.

    y_train = to_categorical(y_train.astype('float32'))
    y_val = to_categorical(y_val.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="LVQ Capsule network on mnist and test on affnist.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help='The `weights` argument should be either `None` (random initialization), `mnist` or '
                             'the path to the weights file to be loaded.')
    args = parser.parse_args()
    print(args)

    # load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_affnist()

    # define model
    model, decoder, model_vis_caps2 = LvqCapsNet(input_shape=x_train.shape[1:])
    model.summary(line_length=200, positions=[.33, .6, .67, 1.])

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not (args.weights in {'mnist', None} or os.path.exists(args.weights)):
        raise ValueError('The `weights` argument should be either `None` (random initialization), `mnist` '
                         'or the path to the weights file to be loaded.')

    if args.weights is not None:  # init the model weights with provided one
        if args.weights == 'mnist':
            args.weights = get_file('mnist.h5',
                                    'http://tiny.cc/anysma_models_mnist',
                                    cache_subdir='models',
                                    file_hash='03f11d19f82fd31c1abedec22d37058a')
        model.load_weights(args.weights)

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_val, y_val)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')

        # clone model with new input shape
        model2, _, _ = LvqCapsNet(input_shape=x_test.shape[1:])
        # copy the weights
        for idx, layer in enumerate(model.layers):
            model2.get_layer(index=idx).set_weights(layer.get_weights())

        model2.compile(optimizer=optimizers.Adam(lr=args.lr),
                       loss=[generalized_kullback_leibler_divergence, 'mse'],
                       loss_weights=[1., args.lam_recon],
                       metrics={'lvq_caps': 'accuracy'})

        plot_spectral_lines(model, (x_val, y_val), args)
        plot_heatmaps(model2, (x_test, y_test), args)
        plot_random_walk_caps3(model, decoder)
        plot_random_walk_caps2(model, model_vis_caps2, decoder, 1)
        plot_random_walk_caps2(model, model_vis_caps2, decoder, 16)
        plot_decoder(model, (x_val, y_val))
        test(model, (x_val, y_val), args, 'test on validation dataset')
        test(model2, (x_test, y_test), args, 'test on test dataset')
