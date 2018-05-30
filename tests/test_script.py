'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras import initializers
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import TensorBoard
from anysma import Capsule
from anysma.modules import InputModule
from anysma.modules.transformations import LinearTransformation, AddBias
from anysma.utils.caps_utils import dict_to_list, list_to_dict
from anysma.modules.measuring import MinkowskiDistance, TangentDistance, OmegaDistance, RestrictedTangentDistance
from anysma.modules.competition import NearestCompetition, GibbsCompetition
import numpy as np
from keras.utils import to_categorical


# register tests here
def main():
    tests = [[test_basic_model,                 False],
             [test_capsule_and_module_wrapper,  False],
             [test_input_module,                True],
             [test_transformation_module,       True],
             [test_measuring_module,            True],
             [test_competition_module,          True],
             [test_omega_distance,              True],
             [test_tangent_distance,            True],
             [test_restricted_tangent_distance, True]]

    for test in tests:
        if test[1]:
            K.clear_session()
            test[0]()


def get_layers():
    num_classes = 10

    if K.image_data_format() == 'channels_first':
        input_shape = (1, 28, 28)
    else:
        input_shape = (28, 28, 1)

    inputs = Input(shape=input_shape)
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')
    conv2 = Conv2D(64, (3, 3), activation='relu')
    pool3 = MaxPooling2D(pool_size=(2, 2))
    drop4 = Dropout(0.25)
    flat5 = Flatten()
    dens6 = Dense(128, activation='relu')
    drop7 = Dropout(0.5)
    dens8 = Dense(num_classes, name='dens8', activation='relu', kernel_constraint=None)
    dens9 = Dense(num_classes, activation='relu')
    dens10 = Dense(num_classes, activation='softmax')

    return inputs, conv1, conv2, pool3, drop4, flat5, dens6, drop7, dens8, dens9, dens10


def train_model(inputs, outputs):
    model = Model(inputs, outputs)

    batch_size = 128
    num_classes = 10
    epochs = 1

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    tb = TensorBoard(log_dir='/home/sascha/keras_ckp/')

    print(model.summary())

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[tb])

    print(model.get_layer('dens8').get_weights())

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# define the tests here
def test_basic_model():
    layers = get_layers()

    for i in range(len(layers)):
        if i == 0:
            inputs = layers[i]
            outputs = inputs
        else:
            outputs = layers[i](outputs)

    train_model(inputs, outputs)


def test_capsule_and_module_wrapper():
    """
    Test that:
        * Capsule can call capsule
        * capsule accept input of capsule
        * capsule can be connected to layers
        * capsule can call layers
        * shared layers
        * shared capsules
    """
    # ToDo: Build tests like keras and all singlewise; interface to Travis CI

    layers = get_layers()
    inputs = layers[0]

    caps1 = Capsule()
    caps2 = Capsule(prototype_distribution=(2, 2))
    caps3 = Capsule(prototype_distribution=3)
    caps4 = Capsule(name='FinalCaps')

    outputs = caps1.add(layers[1:3])(inputs)
    outputs = layers[3](outputs)

    caps2.add(layers[4])
    caps2.add([layers[5], caps3.add(layers[6:7])])
    caps2.add(layers[7])

    outputs = caps2(outputs)

    # share dens layer with 10 neurons out and in
    caps4.add(layers[8:11])
    outputs = layers[10](caps4(outputs))
    # outputs = caps4(outputs)

    caps4.add(layers[10])

    for caps in [caps1, caps2, caps3, caps4]:
        caps.summary()

    train_model(inputs, outputs)


def test_input_module():
    inputs = Input(shape=(28, 28))

    caps = Capsule(prototype_distribution=[3, 2])

    caps.add(InputModule(signal_shape=(98, 8), dissimilarity_initializer='zeros'))

    outputs = caps(inputs)

    print(outputs)

    assert tuple(K.int_shape(outputs[0])) == (None, 2, 98, 8)
    assert tuple(K.int_shape(outputs[1])) == (None, 5, 98)

    # test module input
    signals = Input(shape=(12, 8))
    diss = Input(shape=(11,))
    inputs = {0: signals, 1: diss}

    caps = Capsule(prototype_distribution=(2, 5))

    inputModule = InputModule(signal_shape=(12, -1, 2), dissimilarity_initializer='zeros')

    caps.add(inputModule)

    outputs = caps(inputs)

    print(outputs)

    assert tuple(K.int_shape(outputs[0])) == (None, 5, 12, 4, 2)
    assert tuple(K.int_shape(outputs[1])) == (None, 10, 12)

    # test module input
    signals = Input(shape=(12, 8))
    diss = Input(shape=(12,))
    inputs = {0: signals, 1: diss}

    caps = Capsule(prototype_distribution=(2, 5))

    inputModule = InputModule()

    caps.add(inputModule)

    outputs = caps(inputs)

    print(outputs)

    assert tuple(K.int_shape(outputs[0])) == (None, 5, 12, 8)
    assert tuple(K.int_shape(outputs[1])) == (None, 10, 12)


def test_transformation_module():
    input_vector = np.array([[1, 2], [3, 4]])
    input_vector = np.expand_dims(input_vector, 0)
    # signals = np.tile(input_vector, [2, 1, 1])
    signals = np.concatenate([input_vector, -input_vector, np.ones((1, 2, 2))], axis=0)
    diss = np.ones((4, 3))

    inputs = [Input(signals.shape), Input(diss.shape)]

    # test local
    caps = Capsule(prototype_distribution=(2, 2))
    A = LinearTransformation(output_dim=3,
                             linear_map_initializer=initializers.Constant(2))
    caps.add(A)
    outputs = caps(list_to_dict(inputs))

    # two channels, two capsules
    A.set_weights([np.array([[[[1, 2, 3], [3, 4, 5]],
                              [[1, 1, 1], [1, 1, 1]]],
                             [[[0, 0, 0], [-1, -1, -1]],
                              [[0, 1, 0], [1, 0, 1]]],
                             [[[1, 1, 1], [1, 1, 1]],
                              [[1, 1, 1], [1, 1, 1]]]])])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    assert np.all(y[1] == diss)
    assert np.all(y[0] == np.array([[[[7, 10, 13], [7, 7, 7]],
                                     [[2, 2, 2], [-4,  -3,  -4]],
                                     [[2, 2, 2], [2,  2,  2]]]]))

    # test global
    caps = Capsule(prototype_distribution=(2, 2))
    A = LinearTransformation(output_dim=3,
                             linear_map_initializer=initializers.Constant(2),
                             scope='global')
    caps.add(A)
    outputs = caps(list_to_dict(inputs))

    A.set_weights([np.array([[1, 2, 3], [3, 4, 5]])])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    assert np.all(y[1] == diss)
    assert np.all(y[0] == np.array([[[[7, 10, 13], [15, 22, 29]],
                                     [[-7, -10, -13], [-15, -22, -29]],
                                     [[4, 6, 8], [4, 6, 8]]]]))

    # test capsule_wise
    caps = Capsule(prototype_distribution=(2, 2))
    A = LinearTransformation(output_dim=3,
                             linear_map_initializer=initializers.Constant(2),
                             scope='capsule_wise')
    caps.add(A)
    outputs = caps(list_to_dict(inputs))

    A.set_weights([np.array([[[1, 2, 3], [3, 4, 5]],
                             [[-1, 0, -3], [-3, 0, -5]],
                             [[1, 1, 1], [1, 1, 1]]])])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    assert np.all(y[1] == diss)
    assert np.all(y[0] == np.array([[[[7, 10, 13], [15, 22, 29]],
                                     [[7, 0, 13], [15, 0, 29]],
                                     [[2, 2, 2], [2, 2, 2]]]]))

    # ------------ test complex transformations
    def np_linear_transformation(signals, matrices, output_dim, axis):
        def inv_perm(perm):
            inverse = [0] * len(perm)
            for i, p in enumerate(perm):
                inverse[p] = i
            return inverse

        axis -= 1  # remove missing batch dimension
        # compute distances by numpy
        signals_shape = list(signals.shape)
        signals_shape[axis] = output_dim

        out_signals = np.zeros(signals_shape)

        for i in range(out_signals.shape[0]):  # caps
                for j in range(out_signals.shape[1]):  # channels
                    signal = signals[i, j, :]
                    matrix = matrices[i, j, :]

                    dims = list(range(len(signal.shape)))
                    dims.remove(axis-2)
                    dims.append(axis-2)

                    signal = np.transpose(signal, dims)

                    out_signals[i, j, :] = np.transpose(np.matmul(signal, matrix), inv_perm(dims))

        return out_signals

    def repeat_matrices(matrices, scope, capsules, channels):
        if scope == 'global':
            matrices = np.tile(np.expand_dims(np.expand_dims(matrices, 0), 0), [capsules, channels, 1, 1])
        elif scope == 'channel_wise':
            matrices = np.tile(np.expand_dims(matrices, 0), [capsules, 1, 1, 1])
        elif scope == 'capsule_wise':
            matrices = np.tile(np.expand_dims(matrices, 1), [1, channels, 1, 1])

        return matrices

    # ---------------- local, repeat test from above
    input_vector = np.array([[1, 2], [3, 4]])
    input_vector = np.expand_dims(input_vector, 0)
    # signals = np.tile(input_vector, [2, 1, 1])
    signals = np.concatenate([input_vector, -input_vector, np.ones((1, 2, 2))], axis=0)
    diss = np.ones((4, 3))

    inputs = [Input(signals.shape), Input(diss.shape)]

    # test local
    caps = Capsule(prototype_distribution=(2, 2))
    A = LinearTransformation(output_dim=3,
                             linear_map_initializer=initializers.Constant(2))
    caps.add(A)
    outputs = caps(list_to_dict(inputs))

    # two channels, two capsules
    matrices = np.array([[[[1, 2, 3], [3, 4, 5]],
                              [[1, 1, 1], [1, 1, 1]]],
                             [[[0, 0, 0], [-1, -1, -1]],
                              [[0, 1, 0], [1, 0, 1]]],
                             [[[1, 1, 1], [1, 1, 1]],
                              [[1, 1, 1], [1, 1, 1]]]])
    A.set_weights([matrices])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    np_signal = np_linear_transformation(signals, matrices, 3, 3)  # dim without batch no -1 support

    assert np.all(y[1] == diss)
    assert np.all(y[0] == np_signal)

    # ---------------- global, repeat test from above
    caps = Capsule(prototype_distribution=(2, 2))
    A = LinearTransformation(output_dim=3,
                             linear_map_initializer=initializers.Constant(2),
                             scope='global')
    caps.add(A)
    outputs = caps(list_to_dict(inputs))

    matrices = np.array([[1, 2, 3], [3, 4, 5]])

    A.set_weights([matrices])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    np_signal = np_linear_transformation(signals,
                                         np.tile(np.expand_dims(np.expand_dims(matrices, 0), 0), [3, 2, 1, 1]),
                                         3, 3)

    assert np.all(y[1] == diss)
    assert np.all(y[0] == np_signal)

    # --------------- test capsule_wise, repeat test from above
    caps = Capsule(prototype_distribution=(2, 2))
    A = LinearTransformation(output_dim=3,
                             linear_map_initializer=initializers.Constant(2),
                             scope='capsule_wise')
    caps.add(A)
    outputs = caps(list_to_dict(inputs))

    matrices = np.array([[[1, 2, 3], [3, 4, 5]],
                         [[-1, 0, -3], [-3, 0, -5]],
                         [[1, 1, 1], [1, 1, 1]]])

    A.set_weights([matrices])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    np_signal = np_linear_transformation(signals,
                                         np.tile(np.expand_dims(matrices, 1), [1, 2, 1, 1]),
                                         3, 3)

    assert np.all(y[1] == diss)
    assert np.all(y[0] == np_signal)

    # ---------------- local, repeat test from above --> with fake dimensions of one
    input_vector = np.array([[1, 2], [3, 4]])
    input_vector = np.expand_dims(input_vector, 0)
    # signals = np.tile(input_vector, [2, 1, 1])
    signals = np.concatenate([input_vector, -input_vector, np.ones((1, 2, 2))], axis=0)
    signals = np.expand_dims(np.expand_dims(signals, -2), -1)
    diss = np.ones((4, 3))

    inputs = [Input(signals.shape), Input(diss.shape)]

    # test local
    caps = Capsule(prototype_distribution=(2, 2))
    A = LinearTransformation(output_dim=3,
                             linear_map_initializer=initializers.Constant(2), axis=4)
    caps.add(A)
    outputs = caps(list_to_dict(inputs))

    # two channels, two capsules
    matrices = np.array([[[[1, 2, 3], [3, 4, 5]],
                          [[1, 1, 1], [1, 1, 1]]],
                         [[[0, 0, 0], [-1, -1, -1]],
                          [[0, 1, 0], [1, 0, 1]]],
                         [[[1, 1, 1], [1, 1, 1]],
                          [[1, 1, 1], [1, 1, 1]]]])
    A.set_weights([matrices])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    np_signal = np_linear_transformation(signals, matrices, 3, 4)  # dim with batch no -1 support

    assert np.all(y[1] == diss)
    assert np.all(y[0] == np_signal)

    # ---------------- local, test with signal rank 3, axis=3
    axis = 3
    output_dim = 2
    scope = 'local'
    channels = 3
    capsules = 4

    signals = np.random.rand(capsules, channels, 5, 6, 7)
    diss = np.ones((capsules, channels))
    matrices = np.random.rand(capsules, channels, 5, output_dim)

    inputs = [Input(signals.shape), Input(diss.shape)]

    # test local
    caps = Capsule(prototype_distribution=(2, 2))
    A = LinearTransformation(output_dim=output_dim,
                             linear_map_initializer=initializers.Constant(2), axis=axis, scope=scope)
    caps.add(A)
    outputs = caps(list_to_dict(inputs))
    A.set_weights([matrices])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    matrices = repeat_matrices(matrices, scope, capsules, channels)

    np_signal = np_linear_transformation(signals, matrices, output_dim, axis)

    assert np.all(y[1] == diss)
    assert np.all(np.isclose(np_signal, y[0], atol=np.sqrt(1.e-7)))

    # ---------------- global, test with signal rank 2, axis=3
    axis = 4
    output_dim = 11
    scope = 'global'
    channels = 3
    capsules = 4

    signals = np.random.rand(capsules, channels, 5, 6)
    diss = np.ones((capsules, channels))
    matrices = np.random.rand(6, output_dim)

    inputs = [Input(signals.shape), Input(diss.shape)]

    # test local
    caps = Capsule(prototype_distribution=(2, 2))
    A = LinearTransformation(output_dim=output_dim,
                             linear_map_initializer=initializers.Constant(2), axis=axis, scope=scope)
    caps.add(A)
    outputs = caps(list_to_dict(inputs))
    A.set_weights([matrices])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    matrices = repeat_matrices(matrices, scope, capsules, channels)

    np_signal = np_linear_transformation(signals, matrices, output_dim, axis)

    assert np.all(y[1] == diss)
    assert np.all(np.isclose(np_signal, y[0], atol=np.sqrt(1.e-7)))

    # ---------------- channel_wise, test with signal rank 2, axis=3
    axis = 4
    output_dim = 5
    scope = 'channel_wise'
    channels = 3
    capsules = 4

    signals = np.random.rand(capsules, channels, 5, 6)
    diss = np.ones((capsules, channels))
    matrices = np.random.rand(channels, 6, output_dim)

    inputs = [Input(signals.shape), Input(diss.shape)]

    # test local
    caps = Capsule(prototype_distribution=(2, 2))
    A = LinearTransformation(output_dim=output_dim,
                             linear_map_initializer=initializers.Constant(2), axis=axis, scope=scope)
    caps.add(A)
    outputs = caps(list_to_dict(inputs))
    A.set_weights([matrices])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    matrices = repeat_matrices(matrices, scope, capsules, channels)

    np_signal = np_linear_transformation(signals, matrices, output_dim, axis)

    assert np.all(y[1] == diss)
    assert np.all(np.isclose(np_signal, y[0], atol=np.sqrt(1.e-7)))

    # ---------------- capsule_wise, test with signal rank 2, axis=3
    axis = 4
    output_dim = 3
    scope = 'capsule_wise'
    channels = 3
    capsules = 4

    signals = np.random.rand(capsules, channels, 5, 6)
    diss = np.ones((capsules, channels))
    matrices = np.random.rand(capsules, 6, output_dim)

    inputs = [Input(signals.shape), Input(diss.shape)]

    # test local
    caps = Capsule(prototype_distribution=(2, 2))
    A = LinearTransformation(output_dim=output_dim,
                             linear_map_initializer=initializers.Constant(2), axis=axis, scope=scope)
    caps.add(A)
    outputs = caps(list_to_dict(inputs))
    A.set_weights([matrices])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    matrices = repeat_matrices(matrices, scope, capsules, channels)

    np_signal = np_linear_transformation(signals, matrices, output_dim, axis)

    assert np.all(y[1] == diss)
    assert np.all(np.isclose(np_signal, y[0], atol=np.sqrt(1.e-7)))

    # ***************** Test Bias
    # ---------------- local, test with signal rank 1, axis=-1
    def np_add_biases(signals, biases, axes, scope):
        def inv_perm(perm):
            inverse = [0] * len(perm)
            for i, p in enumerate(perm):
                inverse[p] = i
            return inverse

        signals_shape = signals.shape
        axes = [axis-1 for axis in axes]

        ndim = len(signals_shape)
        atom_axes = list(range(2, ndim))
        batch_axes = []
        for axis in atom_axes:
            if axis not in axes:
                batch_axes.append(axis)

        perm = batch_axes + axes
        perm = [p-2 for p in perm]

        out_signals = np.zeros(signals_shape)

        for i in range(out_signals.shape[0]):  # caps
                for j in range(out_signals.shape[1]):  # channels
                    signal = signals[i, j, :]
                    bias = biases[i, j, :]

                    signal = np.transpose(signal, perm)

                    out_signals[i, j, :] = np.transpose(signal + bias, inv_perm(perm))

        return out_signals

    def repeat_biases(biases, scope, capsules, channels):
        if scope == 'global':
            biases = np.tile(np.expand_dims(np.expand_dims(biases, 0), 0),
                             [capsules, channels] + list(np.ones(len(biases.shape), dtype=int)))
        elif scope == 'channel_wise':
            biases = np.tile(np.expand_dims(biases, 0),
                             [capsules, 1] + list(np.ones(len(biases.shape)-1, dtype=int)))
        elif scope == 'capsule_wise':
            biases = np.tile(np.expand_dims(biases, 1),
                             [1, channels] + list(np.ones(len(biases.shape)-1, dtype=int)))

        return biases

    axes = -1
    scope = 'local'
    channels = 3
    capsules = 4

    signals = np.random.rand(capsules, channels, 6)
    diss = np.ones((capsules, channels))
    biases = np.random.rand(capsules, channels, 6)

    inputs = [Input(signals.shape), Input(diss.shape)]

    # test local
    caps = Capsule(prototype_distribution=(2, 2))
    A = AddBias(axes=axes, scope=scope)
    caps.add(A)
    outputs = caps(list_to_dict(inputs))
    A.set_weights([biases])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    np_signal = biases + signals

    # biases = repeat_matrices(biases, scope, capsules, channels)
    #
    # np_signal = np_linear_transformation(signals, matrices, output_dim, axis)

    assert np.all(y[1] == diss)
    assert np.all(np.isclose(np_signal, y[0], atol=np.sqrt(1.e-7)))

    # ---------------- local, test with signal rank 3, axis=[4]
    axes = [4]
    scope = 'local'
    channels = 3
    capsules = 4

    signals = np.random.rand(capsules, channels, 6, 4, 3)
    diss = np.ones((capsules, channels))
    biases = np.random.rand(capsules, channels, 4)

    inputs = [Input(signals.shape), Input(diss.shape)]

    # test local
    caps = Capsule(prototype_distribution=(2, 2))
    A = AddBias(axes=axes, scope=scope)
    caps.add(A)
    outputs = caps(list_to_dict(inputs))
    A.set_weights([biases])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    biases = repeat_biases(biases, scope, capsules, channels)

    np_signal = np_add_biases(signals, biases, axes, scope)

    assert np.all(y[1] == diss)
    assert np.all(np.isclose(np_signal, y[0], atol=np.sqrt(1.e-7)))

    # ---------------- local, test with signal rank 4, axis=[4, 6]
    axes = [4, 6]
    scope = 'local'
    channels = 3
    capsules = 4

    signals = np.random.rand(capsules, channels, 6, 4, 3, 2)
    diss = np.ones((capsules, channels))
    biases = np.random.rand(capsules, channels, 4, 2)

    inputs = [Input(signals.shape), Input(diss.shape)]

    # test local
    caps = Capsule(prototype_distribution=(2, 2))
    A = AddBias(axes=axes, scope=scope)
    caps.add(A)
    outputs = caps(list_to_dict(inputs))
    A.set_weights([biases])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    biases = repeat_biases(biases, scope, capsules, channels)

    np_signal = np_add_biases(signals, biases, axes, scope)

    assert np.all(y[1] == diss)
    assert np.all(np.isclose(np_signal, y[0], atol=np.sqrt(1.e-7)))

    # ---------------- local, test with signal rank 4, axis=[3, 6]
    axes = [3, 6]
    scope = 'global'
    channels = 3
    capsules = 4

    signals = np.random.rand(capsules, channels, 6, 4, 3, 2)
    diss = np.ones((capsules, channels))
    biases = np.random.rand(6, 2)

    inputs = [Input(signals.shape), Input(diss.shape)]

    # test local
    caps = Capsule(prototype_distribution=(2, 2))
    A = AddBias(axes=axes, scope=scope)
    caps.add(A)
    outputs = caps(list_to_dict(inputs))
    A.set_weights([biases])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    biases = repeat_biases(biases, scope, capsules, channels)

    np_signal = np_add_biases(signals, biases, axes, scope)

    assert np.all(y[1] == diss)
    assert np.all(np.isclose(np_signal, y[0], atol=np.sqrt(1.e-7)))

    # ---------------- local, test with signal rank 4, axis=[3, 6]
    axes = [3, 6]
    scope = 'capsule_wise'
    channels = 3
    capsules = 4

    signals = np.random.rand(capsules, channels, 6, 4, 3, 2)
    diss = np.ones((capsules, channels))
    biases = np.random.rand(capsules, 6, 2)

    inputs = [Input(signals.shape), Input(diss.shape)]

    # test local
    caps = Capsule(prototype_distribution=(2, 2))
    A = AddBias(axes=axes, scope=scope)
    caps.add(A)
    outputs = caps(list_to_dict(inputs))
    A.set_weights([biases])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    biases = repeat_biases(biases, scope, capsules, channels)

    np_signal = np_add_biases(signals, biases, axes, scope)

    assert np.all(y[1] == diss)
    assert np.all(np.isclose(np_signal, y[0], atol=np.sqrt(1.e-7)))

    # ---------------- local, test with signal rank 4, axis=[3, 6]
    axes = [3, 6]
    scope = 'channel_wise'
    channels = 3
    capsules = 4

    signals = np.random.rand(capsules, channels, 6, 4, 3, 2)
    diss = np.ones((capsules, channels))
    biases = np.random.rand(channels, 6, 2)

    inputs = [Input(signals.shape), Input(diss.shape)]

    # test local
    caps = Capsule(prototype_distribution=(2, 2))
    A = AddBias(axes=axes, scope=scope)
    caps.add(A)
    outputs = caps(list_to_dict(inputs))
    A.set_weights([biases])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    biases = repeat_biases(biases, scope, capsules, channels)

    np_signal = np_add_biases(signals, biases, axes, scope)

    assert np.all(y[1] == diss)
    assert np.all(np.isclose(np_signal, y[0], atol=np.sqrt(1.e-7)))


def test_measuring_module():
    # define input vector with three channels and two capsules
    signals = np.array([[[2, 0],  # first capsule input
                         [0, 0],
                         [-1, 2]],
                        [[1, 0],  # second capsule input
                         [1, 1],
                         [-1, -1]]])

    # we have four protos and three channels
    diss = np.random.rand(4, 3)

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    # # ------------------- test order_p = inf with signal_output signals
    # generate caps: first capsule --> three protos; second --> one proto
    caps = Capsule(prototype_distribution=[3, 1])

    Distance = MinkowskiDistance(order_p=np.inf,
                                 prototype_initializer='zeros',
                                 linear_factor=None)

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    # four protos of dimension 2
    protos = np.array([[0, 0],
                       [0, 1],
                       [-1, 1],
                       [2, 1]])
    Distance.set_weights([protos])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # check if the tiling of the signal is correct
    assert np.all(y[0][:, caps._proto_distrib[0], :] == signals[0, :])
    assert np.all(y[0][:, caps._proto_distrib[1], :] == signals[1, :])

    diss_result = np.array([[2, 0, 2],  # first proto against all channels of resp. capsule
                            [2, 1, 1],  # second ...
                            [3, 1, 1],
                            [1, 1, 3]])
    # check distance values
    assert np.all(np.isclose(diss_result + diss, y[1]))

    # # ------------------- test order_p = 1 with signal_output = protos
    caps = Capsule(prototype_distribution=[1, 3])

    Distance = MinkowskiDistance(order_p=1,
                                 prototype_initializer='zeros',
                                 signal_output='protos',
                                 linear_factor=0.5)

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    # four protos of dimension 2
    protos = np.array([[0, 0],
                       [0, 1],
                       [-1, 1],
                       [2, 1]])
    Distance.set_weights([protos])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # check if the tiling of the signal is correct
    for i in range(caps.proto_number):
        assert np.all(y[0][:, i, :] == protos[i, :])

    diss_result = np.array([[2, 0, 3],  # first proto against all channels of resp. capsule
                            [2, 1, 3],  # second ...
                            [3, 2, 2],
                            [2, 1, 5]])
    # check distance values
    assert np.all(np.isclose((diss_result + diss) / 2, y[1]))

    # # ------------------- test order_p = 2
    caps = Capsule(prototype_distribution=[2, 2])

    Distance = MinkowskiDistance(order_p=2,
                                 prototype_initializer='zeros',
                                 linear_factor=0.75)

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    # four protos of dimension 2
    protos = np.array([[0, 0],
                       [0, 1],
                       [-1, 1],
                       [2, 1]])
    Distance.set_weights([protos])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # check if the tiling of the signal is correct
    assert np.all(y[0][:, caps._proto_distrib[0], :] == signals[0, :])
    assert np.all(y[0][:, caps._proto_distrib[1], :] == signals[1, :])

    # compute distances by numpy
    diss_result = np.zeros_like(diss)
    for i in range(diss_result.shape[0]):  # protos
        for j in range(diss_result.shape[1]):  # channels
            for idx, x in enumerate(caps._proto_distrib):
                if x.count(i) > 0:
                    diss_result[i, j] = np.power(np.sum(
                        np.power(np.abs(signals[idx, j, :] - protos[i, :]), Distance.order_p)), 1 / Distance.order_p)
                    break

    # check distance values
    assert np.all(np.isclose(0.75 * diss_result + 0.25 * diss, y[1], atol=np.power(Distance.epsilon, 1/Distance.order_p)))

    # # ------------------- test order_p = 3
    caps = Capsule(prototype_distribution=[1, 3])

    Distance = MinkowskiDistance(order_p=3,
                                 prototype_initializer='zeros',
                                 signal_output='protos')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    # four protos of dimension 2
    protos = np.array([[0, 0],
                       [0, 1],
                       [-1, 1],
                       [2, 1]])
    Distance.set_weights([protos])

    model = Model(inputs, dict_to_list(outputs))
    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # check if the tiling of the signal is correct
    for i in range(caps.proto_number):
        assert np.all(y[0][:, i, :] == protos[i, :])

    # compute distances by numpy
    diss_result = np.zeros_like(diss)
    for i in range(diss_result.shape[0]):  # protos
        for j in range(diss_result.shape[1]):  # channels
            for idx, x in enumerate(caps._proto_distrib):
                if x.count(i) > 0:
                    diss_result[i, j] = np.power(np.sum(
                        np.power(np.abs(signals[idx, j, :] - protos[i, :]), Distance.order_p)),
                        1 / Distance.order_p)
                    break

    # check distance values
    assert np.all(np.isclose((diss_result + diss) / 2, y[1], atol=np.power(Distance.epsilon, 1 / Distance.order_p)))

    # # ------------------- test order_p = 2.2 with matrix input
    # define input vector with three channels and two capsules
    signals = np.array([[[2, 0],  # first capsule input
                         [0, 0],
                         [-1, 2]],
                        [[1, 0],  # second capsule input
                         [1, 1],
                         [-1, -1]]])

    signals = np.reshape(np.tile(signals, [2, 1, 2]), [4, 3, 2, 2])

    # we have four protos and three channels
    diss = np.zeros((4, 3))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = MinkowskiDistance(order_p=2.2,
                                 prototype_initializer='zeros')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    # # four protos of dimension 2
    protos = np.array([[[0, 0], [0, 0]],
                       [[0, 1], [0, 1]],
                       [[-1, 1], [-1, 1]],
                       [[2, 1], [2, 1]]])
    # protos = np.array([[[0, 0], [0, 0]],
    #                    [[0, 1], [0, 1]]])

    Distance.set_weights([protos])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # check if the tiling of the signal is correct
    assert np.all(y[0][:, caps._proto_distrib[0], :] == signals[0, :])
    assert np.all(y[0][:, caps._proto_distrib[1], :] == signals[1, :])

    # compute distances by numpy
    diss_result = np.zeros_like(diss)
    for i in range(diss_result.shape[0]):  # protos
        for j in range(diss_result.shape[1]):  # channels
            for idx, x in enumerate(caps._proto_distrib):
                if x.count(i) > 0:
                    diss_result[i, j] = np.power(np.sum(
                        np.power(np.abs(signals[idx, j, :] - protos[i, :]), Distance.order_p)), 1 / Distance.order_p)
                    break

    # check distance values
    assert np.all(np.isclose((diss_result + diss) / 2, y[1], atol=np.power(Distance.epsilon, 1 / Distance.order_p)))

    # ------------------- pre-training tests ----------------
    # ------------------- equal signal over capsules; no labels
    # define four clusters
    clusters = np.array([[1, 1],
                         [-1, 1],
                         [1, -1],
                         [-1, -1]]) * 5
    x_train = np.concatenate((np.random.multivariate_normal(clusters[0, :], np.eye(2), 100000),
                              np.random.multivariate_normal(clusters[1, :], np.eye(2), 100000),
                              np.random.multivariate_normal(clusters[2, :], np.eye(2), 100000),
                              np.random.multivariate_normal(clusters[3, :], np.eye(2), 100000)))
    x_train = np.expand_dims(np.expand_dims(x_train, 1), 1)
    x_train = np.tile(x_train, [1, 4, 1, 1])

    inputs = Input((2,))

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = MinkowskiDistance()

    caps.add(InputModule(signal_shape=2, dissimilarity_initializer='zeros'))
    caps.add(Distance)

    outputs = caps(inputs)

    Distance.pre_training(x_train, capsule_inputs_are_equal=True)

    protos = Distance.get_weights()[0]

    for c in clusters:
        idx = np.isclose(protos, c, atol=0.05)
        idx = np.bitwise_and(idx[:, 0], idx[:, 1])
        assert np.any(idx)
        # pop out matching proto
        protos = np.delete(protos, np.argmax(idx), 0)

    # ------ with matrix input
    # define four clusters
    clusters = np.array([[1, 1, 1, 1],
                         [-1, 1, -1, 1],
                         [1, -1, 1, -1],
                         [-1, -1, -1, -1]]) * 5
    x_train = np.concatenate((np.random.multivariate_normal(clusters[0, :], np.eye(4), 100000),
                              np.random.multivariate_normal(clusters[1, :], np.eye(4), 100000),
                              np.random.multivariate_normal(clusters[2, :], np.eye(4), 100000),
                              np.random.multivariate_normal(clusters[3, :], np.eye(4), 100000)))
    x_train = np.reshape(x_train, [-1, 2, 2])
    x_train = np.expand_dims(np.expand_dims(x_train, 1), 1)
    x_train = np.tile(x_train, [1, 4, 1, 1, 1])

    inputs = Input((2, 2))

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = MinkowskiDistance()

    caps.add(InputModule(signal_shape=(-1, 2, 2), dissimilarity_initializer='zeros'))
    caps.add(Distance)

    outputs = caps(inputs)

    Distance.pre_training(x_train, capsule_inputs_are_equal=True)

    protos = Distance.get_weights()[0]

    for c in clusters:
        idx = np.isclose(np.reshape(protos, (-1, 4)), c, atol=0.05)
        idx = np.bitwise_and(idx[:, 0], idx[:, 1])
        assert np.any(idx)
        # pop out matching proto
        protos = np.delete(protos, np.argmax(idx), 0)

    # -------------------equal signal over capsules; with labels
    # define four clusters
    clusters = np.array([[1, 1],
                         [-1, 1],
                         [1, -1],
                         [-1, -1]]) * 5
    x_train = np.concatenate((np.random.multivariate_normal(clusters[0, :], np.eye(2), 100000),
                              np.random.multivariate_normal(clusters[1, :], np.eye(2), 100000),
                              np.random.multivariate_normal(clusters[2, :], np.eye(2), 100000),
                              np.random.multivariate_normal(clusters[3, :], np.eye(2), 100000)))
    y_train = np.concatenate((np.ones((100000,)) * 0,
                              np.ones((100000,)) * 1,
                              np.ones((100000,)) * 2,
                              np.ones((100000,)) * 3))
    # shuffel the date
    idx = np.arange(0, y_train.shape[0])
    np.random.shuffle(idx)

    x_train = x_train[idx, :]
    y_train = y_train[idx]

    x_train = np.expand_dims(np.expand_dims(x_train, 1), 1)
    x_train = np.tile(x_train, [1, 4, 1, 1])

    inputs = Input((2,))

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = MinkowskiDistance()

    caps.add(InputModule(signal_shape=2, dissimilarity_initializer='zeros'))
    caps.add(Distance)

    outputs = caps(inputs)

    Distance.pre_training(x_train, y_train=to_categorical(y_train, 4),  capsule_inputs_are_equal=True)

    protos = Distance.get_weights()[0]

    for c in clusters:
        idx = np.isclose(protos, c, atol=0.05)
        idx = np.bitwise_and(idx[:, 0], idx[:, 1])
        assert np.any(idx)
        # pop out matching proto
        protos = np.delete(protos, np.argmax(idx), 0)

    # ---------------- with to categorical
    # define four clusters
    clusters = np.array([[1, 1],
                         [-1, 1],
                         [1, -1],
                         [-1, -1]]) * 5
    x_train = np.concatenate((np.random.multivariate_normal(clusters[0, :], np.eye(2), 100000),
                              np.random.multivariate_normal(clusters[1, :], np.eye(2), 100000),
                              np.random.multivariate_normal(clusters[2, :], np.eye(2), 100000),
                              np.random.multivariate_normal(clusters[3, :], np.eye(2), 100000)))
    y_train = np.concatenate((np.ones((100000,)) * 0,
                              np.ones((100000,)) * 1,
                              np.ones((100000,)) * 2,
                              np.ones((100000,)) * 3))
    # shuffel the date
    idx = np.arange(0, y_train.shape[0])
    np.random.shuffle(idx)

    x_train = x_train[idx, :]
    y_train = y_train[idx]

    x_train = np.expand_dims(np.expand_dims(x_train, 1), 1)
    x_train = np.tile(x_train, [1, 4, 1, 1])

    inputs = Input((2,))

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = MinkowskiDistance()

    caps.add(InputModule(signal_shape=2, dissimilarity_initializer='zeros'))
    caps.add(Distance)

    outputs = caps(inputs)

    Distance.pre_training(x_train, y_train=to_categorical(y_train, 4), capsule_inputs_are_equal=True)

    protos = Distance.get_weights()[0]

    for c in clusters:
        idx = np.isclose(protos, c, atol=0.05)
        idx = np.bitwise_and(idx[:, 0], idx[:, 1])
        assert np.any(idx)
        # pop out matching proto
        protos = np.delete(protos, np.argmax(idx), 0)

    # ------ with matrix input and non equal proto distrib
    samples = 100000
    # define four clusters
    clusters = [np.array([[1, 1, 1, 1],
                         [-1, 1, -1, 1],
                         [1, -1, 1, -1]]) * 5,
                np.array([[-1, -1, -1, -1]]) * 5,
                np.array([[-1, 1, -1, -1]]) * 5,
                np.array([[-1, -1, 1, -1],
                         [-1, 1, 1, 1]]) * 5]
    x_train = None
    for i, c in enumerate(clusters):
        for cluster in c:
            if x_train is None:
                x_train = np.random.multivariate_normal(cluster, np.eye(4), samples)
                y_train = np.ones((samples,)) * i
            else:
                x_train = np.concatenate((x_train,
                                          np.random.multivariate_normal(cluster, np.eye(4), samples)))
                y_train = np.concatenate((y_train,
                                          np.ones((samples,)) * i))

    # shuffel the date
    idx = np.arange(0, y_train.shape[0])
    np.random.shuffle(idx)

    x_train = x_train[idx, :]
    y_train = y_train[idx]

    x_train = np.reshape(x_train, [-1, 2, 2])
    x_train = np.expand_dims(np.expand_dims(x_train, 1), 1)
    x_train = np.tile(x_train, [1, 4, 1, 1, 1])

    inputs = Input((2, 2))

    caps = Capsule(prototype_distribution=[len(c) for c in clusters])

    Distance = MinkowskiDistance()

    caps.add(InputModule(signal_shape=(-1, 2, 2), dissimilarity_initializer='zeros'))
    caps.add(Distance)

    outputs = caps(inputs)

    Distance.pre_training(x_train, y_train=y_train, capsule_inputs_are_equal=True)

    protos = Distance.get_weights()[0]
    proto_distrib = caps.proto_distrib

    for i, c in enumerate(clusters):
        for cluster in c:
            idx = np.isclose(np.reshape(protos[proto_distrib[i], :], (-1, 4)), cluster, atol=0.05)
            idx = np.bitwise_and(idx[:, 0], idx[:, 1])
            assert np.any(idx)
            # pop out matching proto
            del proto_distrib[i][np.argmax(idx)]

    # ------ non equal capsule inputs and  no labels
    samples = 100000
    channels = 100
    # define four clusters
    clusters = [np.array([[1, 1, 1, 1],
                          [-1, 1, -1, 1],
                          [1, -1, 1, -1]]) * 5,
                np.array([[-1, -1, -1, -1],
                          [-1, -1, -1, -1],
                          [-1, -1, -1, -1]]) * 5,
                np.array([[-1, 1, -1, -1],
                          [-1, 1, -1, -1],
                          [-1, 1, -1, -1]]) * 5,
                np.array([[-1, -1, 1, -1],
                          [-1, 1, 1, 1],
                          [-1, 1, 1, 1]]) * 5]
    x_train = None
    for i, c in enumerate(clusters):
        for cluster in c:
            if x_train is None:
                x_train = np.random.multivariate_normal(cluster, np.eye(4), samples)
            else:
                x_train = np.concatenate((x_train,
                                          np.random.multivariate_normal(cluster, np.eye(4), samples)))

    x_train = np.reshape(x_train, [4, -1, channels, 2, 2])
    x_train = np.transpose(x_train, [1, 0, 2, 3, 4])

    inputs = Input((channels, 2, 2))

    caps = Capsule(prototype_distribution=[3, 1, 1, 2])

    Distance = MinkowskiDistance()

    caps.add(InputModule(signal_shape=(channels, 2, 2), dissimilarity_initializer='zeros'))
    caps.add(Distance)

    outputs = caps(inputs)

    Distance.pre_training(x_train, capsule_inputs_are_equal=False)

    protos = Distance.get_weights()[0]
    proto_distrib = caps.proto_distrib

    for i, p in enumerate(caps.proto_distrib):
        for cluster in clusters[i][list(range(len(p))), :]:
            idx = np.isclose(np.reshape(protos[proto_distrib[i], :], (-1, 4)), cluster, atol=0.05)
            idx = np.bitwise_and(idx[:, 0], idx[:, 1])
            assert np.any(idx)
            # pop out matching proto
            del proto_distrib[i][np.argmax(idx)]

    # ------ non equal capsule inputs and  with labels
    samples = 100000
    channels = 1000
    # define four clusters
    clusters = [np.array([[1, 1, 1, 1],
                          [-1, 1, -1, 1],
                          [1, -1, 1, -1]]) * 5,
                np.array([[-1, -1, -1, -1],
                          [-1, -1, -1, -1],
                          [-1, -1, -1, -1]]) * 5,
                np.array([[-1, 1, -1, -1],
                          [-1, 1, -1, -1],
                          [-1, 1, -1, -1]]) * 5,
                np.array([[-1, -1, 1, -1],
                          [-1, 1, 1, 1],
                          [-1, 1, 1, 1]]) * 5]
    x_train = None
    for i, c in enumerate(clusters):
        for cluster in c:
            if x_train is None:
                x_train = np.random.multivariate_normal(cluster, np.eye(4), samples)
                y_train = np.ones((samples // channels,)) * i
            else:
                x_train = np.concatenate((x_train,
                                          np.random.multivariate_normal(cluster, np.eye(4), samples)))
                y_train = np.concatenate((y_train,
                                          np.ones((samples // channels,)) * i))

    x_train = np.reshape(x_train, [-1, channels, 2, 2])

    # shuffel the data
    # alternating class correspondence
    y_train = np.reshape(y_train, [-1, 4])
    y_train = y_train[:, 0]
    x_train = np.reshape(x_train, [-1, 4, channels, 2, 2])

    inputs = Input((channels, 2, 2))

    caps = Capsule(prototype_distribution=[3, 1, 1, 2])

    Distance = MinkowskiDistance()

    caps.add(InputModule(signal_shape=(channels, 2, 2), dissimilarity_initializer='zeros'))
    caps.add(Distance)

    outputs = caps(inputs)

    Distance.pre_training(x_train, y_train=y_train, capsule_inputs_are_equal=False)

    protos = Distance.get_weights()[0]
    proto_distrib = caps.proto_distrib

    for i, p in enumerate(caps.proto_distrib):
        for cluster in clusters[i][list(range(len(p))), :]:
            idx = np.isclose(np.reshape(protos[proto_distrib[i], :], (-1, 4)), cluster, atol=0.05)
            idx = np.bitwise_and(idx[:, 0], idx[:, 1])
            assert np.any(idx)
            # pop out matching proto
            del proto_distrib[i][np.argmax(idx)]


def test_competition_module():

    # ++++++++++++++ test not equally distributed
    # define input vector with seven protos and three capsules of dim 2
    # (batch x protos x dim)
    signals = np.array([[1, 0],
                        [0, 1],
                        [1, 1],
                        [-1, 2],
                        [2, 1],
                        [-1, -1]], dtype='float32')

    # we have seven protos
    diss = np.array([1, 0.5, 3,
                     2,
                     2.3, 4], dtype='float32')

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    # ----------------- test nearest
    # generate caps:
    caps = Capsule(prototype_distribution=[3, 1, 2])

    caps.add(NearestCompetition())
    outputs = caps(list_to_dict(inputs))

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # check
    assert np.all(y[0][:, 0, :] == signals[1, :])
    assert np.all(y[0][:, 1, :] == signals[3, :])
    assert np.all(y[0][:, 2, :] == signals[4, :])

    assert np.all(y[1] == diss[[1, 3, 4]])

    # ------------- test softmax
    # generate caps:
    caps = Capsule(prototype_distribution=[3, 1, 2])

    caps.add(GibbsCompetition())
    outputs = caps(list_to_dict(inputs))

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # check
    for idx, p in enumerate(caps._proto_distrib):
        c = np.exp(-diss[p]) / np.sum(np.exp(-diss[p]), keepdims=True)
        result_signals = np.dot(c, signals[p, :])
        result_diss = np.dot(c, diss[p])
        assert np.all(np.isclose(result_signals, y[0][:, idx, :]))
        assert np.all(np.isclose(result_diss, y[1][:, idx]))

    # ++++++++++++++ test not equally distributed
    # define input vector with seven protos and three capsules of dim 2
    # (batch x protos x dim)
    signals = np.array([[1, 0],
                        [0, 1],
                        [1, 1],
                        [-1, 2],
                        [2, 1],
                        [-1, -1]], dtype='float32')

    # we have seven protos
    diss = np.array([1, 0.5,
                     3, 2,
                     2.3, 4], dtype='float32')

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    # ----------------- test nearest
    # generate caps:
    caps = Capsule(prototype_distribution=(2, 3))

    caps.add(NearestCompetition())
    outputs = caps(list_to_dict(inputs))

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # check
    assert np.all(y[0][:, 0, :] == signals[1, :])
    assert np.all(y[0][:, 1, :] == signals[3, :])
    assert np.all(y[0][:, 2, :] == signals[4, :])

    assert np.all(y[1] == diss[[1, 3, 4]])

    # ------------- test softmax
    # generate caps:
    caps = Capsule(prototype_distribution=(2, 3))

    caps.add(GibbsCompetition())
    outputs = caps(list_to_dict(inputs))

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # check
    for idx, p in enumerate(caps._proto_distrib):
        c = np.exp(-diss[p]) / np.sum(np.exp(-diss[p]), keepdims=True)
        result_signals = np.dot(c, signals[p, :])
        result_diss = np.dot(c, diss[p])
        assert np.all(np.isclose(result_signals, y[0][:, idx, :]))
        assert np.all(np.isclose(result_diss, y[1][:, idx]))

    # # ++++++++++++++ test: with predefined result
    # define input vector with seven protos and three capsules of dim 2
    # (batch x protos x dim)
    signals = np.array([[2, 1],
                        [0, 4],
                        [1, 1],
                        [-1, 2],
                        [2, 1],
                        [-1, -1]], dtype='float32')

    # we have seven protos
    diss = np.array([0, 0, 0,
                     0,
                     0, 0], dtype='float32')

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    # ----------------- test nearest
    # generate caps:
    caps = Capsule(prototype_distribution=[3, 1, 2])

    caps.add(NearestCompetition())
    outputs = caps(list_to_dict(inputs))

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # check
    assert np.all(y[0][:, 0, :] == signals[0, :])
    assert np.all(y[0][:, 1, :] == signals[3, :])
    assert np.all(y[0][:, 2, :] == signals[4, :])

    assert np.all(y[1] == diss[[1, 3, 4]])

    # ------------- test softmax
    # generate caps:
    caps = Capsule(prototype_distribution=[3, 1, 2])

    caps.add(GibbsCompetition())
    outputs = caps(list_to_dict(inputs))

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # check
    assert np.all(y[1] == [0, 0, 0])
    assert np.all(y[0] == np.array([[1, 2], [-1, 2], [1/2, 0]]))


def test_omega_distance():
    def np_omega_distance(signals, protos, omega, projected_atom_shape):
        shape = signals.shape
        signals = np.reshape(signals, shape[0:3] + (-1,))
        protos = np.reshape(protos, (shape[1], -1))
        signals = np.transpose(signals, [0, 2, 1, 3])
        if omega.ndim == 3:
            signals_ = []
            protos_ = []
            for i in range(signals.shape[2]):
                signals_.append(np.tensordot(signals[:, :, i:i+1, :], omega[i, :], [3, 0]))
                protos_.append(np.dot(protos[i:i+1,:], omega[i,:]))
            signals = np.concatenate(signals_, 2)
            protos = np.concatenate(protos_)
        else:
            signals = np.dot(signals, omega)
            protos = np.dot(protos, omega)

        z = signals - protos
        diss = np.sqrt(np.sum(np.square(z), -1))

        signals = np.transpose(signals, [0, 2, 1, 3])
        diss = np.transpose(diss, [0, 2, 1])

        signals = np.reshape(signals, shape[0:3] + projected_atom_shape)
        protos = np.reshape(protos, (shape[1],) + projected_atom_shape)

        return diss, signals, protos

    # ---------------- local matrix, vector input
    # protos x dim_old x dim_new
    omega = np.random.rand(4, 5, 3)

    signals = np.random.rand(4, 6, 5)
    protos = np.random.rand(4, 5)

    # we have four protos and three channels
    diss = np.zeros((4, 6))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = OmegaDistance(prototype_initializer='zeros', projected_atom_shape=3, linear_factor=None)

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, omega])

    model = Model(inputs, dict_to_list(outputs))

    diss = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)[1]
    diss_np = np_omega_distance(np.expand_dims(signals, 0), protos, omega, projected_atom_shape=(3,))[0]

    print(outputs)

    assert np.allclose(diss, diss_np)

    # ---------------- global matrix, vector input
    # protos x dim_old x dim_new
    omega = np.random.rand(5, 3)

    signals = np.random.rand(4, 6, 5)
    protos = np.random.rand(4, 5)

    # we have four protos and three channels
    diss = np.zeros((4, 6))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = OmegaDistance(prototype_initializer='zeros', projected_atom_shape=3, linear_factor=None,
                             matrix_scope='global')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, omega])

    model = Model(inputs, dict_to_list(outputs))

    diss = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)[1]
    diss_np = np_omega_distance(np.expand_dims(signals, 0), protos, omega, projected_atom_shape=(3,))[0]

    print(outputs)

    assert np.allclose(diss, diss_np)

    # ---------------- local matrix, matrix input
    # protos x dim_old x dim_new
    omega = np.random.rand(4, 9, 4)

    signals = np.random.rand(4, 6, 3, 3)
    protos = np.random.rand(4, 3, 3)

    # we have four protos and three channels
    diss = np.zeros((4, 6))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = OmegaDistance(prototype_initializer='zeros', projected_atom_shape=(2, 2), linear_factor=None,
                             matrix_scope='local')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, omega])

    model = Model(inputs, dict_to_list(outputs))

    diss = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)[1]
    diss_np = np_omega_distance(np.expand_dims(signals, 0), protos, omega, projected_atom_shape=(2, 2))[0]

    print(outputs)

    assert np.allclose(diss, diss_np)

    # ---------------- global matrix, matrix input
    # protos x dim_old x dim_new
    omega = np.random.rand(9, 4)

    signals = np.random.rand(4, 6, 3, 3)
    protos = np.random.rand(4, 3, 3)

    # we have four protos and three channels
    diss = np.zeros((4, 6))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = OmegaDistance(prototype_initializer='zeros', projected_atom_shape=(2, 2), linear_factor=None,
                             matrix_scope='global')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, omega])

    model = Model(inputs, dict_to_list(outputs))

    diss = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)[1]
    diss_np = np_omega_distance(np.expand_dims(signals, 0), protos, omega, projected_atom_shape=(2, 2))[0]

    print(outputs)

    assert np.allclose(diss, diss_np)

    # ---------------- local matrix, vector input, get projections True
    # protos x dim_old x dim_new
    omega = np.random.rand(4, 5, 3)

    signals = np.random.rand(4, 6, 5)
    protos = np.random.rand(4, 5)

    # we have four protos and three channels
    diss = np.zeros((4, 6))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = OmegaDistance(prototype_initializer='zeros', projected_atom_shape=(3,), linear_factor=None,
                             matrix_scope='local', signal_output='projected_signals')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, omega])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)
    diss_np, signals_np, protos_np = np_omega_distance(np.expand_dims(signals, 0), protos, omega, projected_atom_shape=(3,))
    print(outputs)

    assert np.allclose(y[1], diss_np)
    assert np.allclose(y[0], signals_np)

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = OmegaDistance(prototype_initializer='zeros', projected_atom_shape=(3,), linear_factor=None,
                             matrix_scope='local', signal_output='projected_protos')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, omega])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    assert np.allclose(y[1], diss_np)
    assert np.allclose(y[0], np.tile(np.expand_dims(protos_np, 1), [1, signals.shape[1], 1]))

    # ---------------- global matrix, vector input, get projections True
    # protos x dim_old x dim_new
    omega = np.random.rand(5, 3)

    signals = np.random.rand(4, 6, 5)
    protos = np.random.rand(4, 5)

    # we have four protos and three channels
    diss = np.zeros((4, 6))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = OmegaDistance(prototype_initializer='zeros', projected_atom_shape=(3,), linear_factor=None,
                             matrix_scope='global', signal_output='projected_signals')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, omega])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)
    diss_np, signals_np, protos_np = np_omega_distance(np.expand_dims(signals, 0), protos, omega,
                                                       projected_atom_shape=(3,))
    print(outputs)

    assert np.allclose(y[1], diss_np)
    assert np.allclose(y[0], signals_np)

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = OmegaDistance(prototype_initializer='zeros', projected_atom_shape=(3,), linear_factor=None,
                             matrix_scope='global', signal_output='projected_protos')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, omega])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    assert np.allclose(y[1], diss_np)
    assert np.allclose(y[0], np.tile(np.expand_dims(protos_np, 1), [1, signals.shape[1], 1]))

    # ---------------- local matrix, matrix input, get projections True
    # protos x dim_old x dim_new
    omega = np.random.rand(4, 9, 4)

    signals = np.random.rand(4, 6, 3, 3)
    protos = np.random.rand(4, 3, 3)

    # we have four protos and three channels
    diss = np.zeros((4, 6))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = OmegaDistance(prototype_initializer='zeros', projected_atom_shape=(2,2), linear_factor=None,
                             matrix_scope='local', signal_output='projected_signals')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, omega])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)
    diss_np, signals_np, protos_np = np_omega_distance(np.expand_dims(signals, 0), protos, omega,
                                                       projected_atom_shape=(2,2))
    print(outputs)

    assert np.allclose(y[1], diss_np)
    assert np.allclose(y[0], signals_np)

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = OmegaDistance(prototype_initializer='zeros', projected_atom_shape=(2,2), linear_factor=None,
                             matrix_scope='local', signal_output='projected_protos')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, omega])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    assert np.allclose(y[1], diss_np)
    assert np.allclose(y[0], np.tile(np.expand_dims(protos_np, 1), [1, signals.shape[1], 1, 1]))

    # ---------------- global matrix, matrix input, get projections True
    # protos x dim_old x dim_new
    omega = np.random.rand(9, 4)

    signals = np.random.rand(4, 6, 3, 3)
    protos = np.random.rand(4, 3, 3)

    # we have four protos and three channels
    diss = np.zeros((4, 6))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = OmegaDistance(prototype_initializer='zeros', projected_atom_shape=(2, 2), linear_factor=None,
                             matrix_scope='global', signal_output='projected_signals')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, omega])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)
    diss_np, signals_np, protos_np = np_omega_distance(np.expand_dims(signals, 0), protos, omega,
                                                       projected_atom_shape=(2, 2))
    print(outputs)

    assert np.allclose(y[1], diss_np)
    assert np.allclose(y[0], signals_np)

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = OmegaDistance(prototype_initializer='zeros', projected_atom_shape=(2, 2), linear_factor=None,
                             matrix_scope='global', signal_output='projected_protos')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, omega])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    assert np.allclose(y[1], diss_np)
    assert np.allclose(y[0], np.tile(np.expand_dims(protos_np, 1), [1, signals.shape[1], 1, 1]))


def test_tangent_distance():
    def np_tangent_distance(signals, protos, subspaces, proto_distrib):
        # compute distances by numpy
        diss_result = np.zeros((protos.shape[0], signals.shape[1]))
        for i in range(diss_result.shape[0]):  # protos
            for j in range(diss_result.shape[1]):  # channels
                for idx, x in enumerate(proto_distrib):
                    if x.count(i) > 0:
                        diss_result[i, j] = np.sqrt(np.sum(
                            np.square(np.dot((signals[idx, j, :] - protos[i, :]).flatten(),
                                             (np.eye(np.prod(signals.shape[2:])) - np.matmul(np.array(subspaces[i, :, :]),
                                                                                    np.array(subspaces[i, :, :]).transpose()))))))
                        break

        return diss_result

    def np_get_projected_signals(signals, protos, subspaces, proto_distrib):
        # compute distances by numpy
        projected_signals = np.zeros((protos.shape[0],) + signals.shape[1:])
        for i in range(diss_result.shape[0]):  # protos
            for j in range(diss_result.shape[1]):  # channels
                for idx, x in enumerate(proto_distrib):
                    if x.count(i) > 0:
                        projected_signals[i, j, :] = np.reshape(np.dot((signals[idx, j, :] - protos[i, :]).flatten(),
                                                            np.matmul(np.array(subspaces[i, :, :]),
                                                                      np.array(subspaces[i, :, :]).transpose())), signals.shape[2:])\
                                                     + protos[i, :]
                        break

        return projected_signals

    def np_parametrized_signals(signals, protos, subspaces, proto_distrib, projected_atom_shape):
        # compute distances by numpy
        projected_signals = np.zeros((protos.shape[0], signals.shape[1]) + projected_atom_shape)
        for i in range(diss_result.shape[0]):  # protos
            for j in range(diss_result.shape[1]):  # channels
                for idx, x in enumerate(proto_distrib):
                    if x.count(i) > 0:
                        projected_signals[i, j, :] = np.reshape(np.dot((signals[idx, j, :] - protos[i, :]).flatten(),
                                                            np.array(subspaces[i, :, :])), projected_atom_shape)
                        break

        return projected_signals

    # --------------- test TANGENT DISTANCE, local, vector
    signals = np.array([[[2, 0],  # first capsule input
                         [0, 10],
                         [-1, 3]],
                        [[1, 0],  # second capsule input
                         [1, 1],
                         [-1, -1]]])

    signals = np.reshape(np.tile(signals, [2, 1, 1]), [4, 3, 2])

    # we have four protos and three channels
    diss = np.zeros((4, 3))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = TangentDistance(prototype_initializer='zeros', projected_atom_shape=1, linear_factor=None)

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    # # four protos of dimension 2
    protos = np.array([[0, 0],
                       [0, 1],
                       [-1, 1],
                       [2, 1]])

    subspaces = np.array([[[1], [0]],
                          [[1], [0]],
                          [[1], [0]],
                          [[1], [0]]])

    Distance.set_weights([protos, subspaces])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # check if the tiling of the signal is correct
    assert np.all(np.isclose(signals[0, :], y[0][:, caps._proto_distrib[0], :]))
    assert np.all(np.isclose(signals[1, :], y[0][:, caps._proto_distrib[1], :]))

    # compute distances by numpy
    diss_result = np_tangent_distance(signals, protos, subspaces, caps._proto_distrib)

    # check distance values
    assert np.all(np.isclose(diss_result, y[1], atol=np.power(Distance.epsilon, 1 / 2)))

    # --------------- test TANGENT DISTANCE, global, vector
    signals = np.random.rand(4, 3, 10)
    protos = np.random.rand(4, 10)
    subspaces, _, _ = np.linalg.svd(np.random.rand(10, 3), full_matrices=False)

    # we have four protos and three channels
    diss = np.zeros((4, 3))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = TangentDistance(prototype_initializer='zeros', projected_atom_shape=3, linear_factor=None,
                               matrix_scope='global')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, subspaces])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # check if the tiling of the signal is correct
    assert np.all(np.isclose(signals[0, :], y[0][:, caps._proto_distrib[0], :]))
    assert np.all(np.isclose(signals[1, :], y[0][:, caps._proto_distrib[1], :]))

    # compute distances by numpy
    diss_result = np_tangent_distance(signals, protos,
                                      np.tile(np.expand_dims(subspaces, 0), [protos.shape[0], 1, 1]),
                                      caps._proto_distrib)

    # check distance values
    assert np.all(np.isclose(diss_result, y[1], atol=np.power(Distance.epsilon, 1 / 2)))

    # --------------- test TANGENT DISTANCE, local, matrix
    signals = np.random.rand(4, 3, 4, 5)
    protos = np.random.rand(4, 4, 5)
    subspaces, _, _ = np.linalg.svd(np.random.rand(4, 20, 16), full_matrices=False)

    # we have four protos and three channels
    diss = np.zeros((4, 3))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = TangentDistance(prototype_initializer='zeros', projected_atom_shape=(4, 4), linear_factor=None)

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, subspaces])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # check if the tiling of the signal is correct
    assert np.all(np.isclose(signals[0, :], y[0][:, caps._proto_distrib[0], :]))
    assert np.all(np.isclose(signals[1, :], y[0][:, caps._proto_distrib[1], :]))

    # compute distances by numpy
    diss_result = np_tangent_distance(signals, protos, subspaces, caps._proto_distrib)

    # check distance values
    assert np.all(np.isclose(diss_result, y[1], atol=np.power(Distance.epsilon, 1 / 2)))

    # --------------- test TANGENT DISTANCE, global, matrix
    signals = np.random.rand(4, 3, 3, 3)
    protos = np.random.rand(4, 3, 3)
    subspaces, _, _ = np.linalg.svd(np.random.rand(9, 4), full_matrices=False)

    # we have four protos and three channels
    diss = np.zeros((4, 3))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = TangentDistance(prototype_initializer='zeros', projected_atom_shape=(2, 2), linear_factor=None,
                               matrix_scope='global')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, subspaces])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # check if the tiling of the signal is correct
    assert np.all(np.isclose(signals[0, :], y[0][:, caps._proto_distrib[0], :]))
    assert np.all(np.isclose(signals[1, :], y[0][:, caps._proto_distrib[1], :]))

    # compute distances by numpy
    diss_result = np_tangent_distance(signals, protos,
                                      np.tile(np.expand_dims(subspaces, 0), [protos.shape[0], 1, 1]),
                                      caps._proto_distrib)

    # check distance values
    assert np.all(np.isclose(diss_result, y[1], atol=np.power(Distance.epsilon, 1 / 2)))

    # --------------- test TANGENT DISTANCE Output projected, local, matrix
    signals = np.random.rand(4, 3, 4, 5)
    protos = np.random.rand(4, 4, 5)
    subspaces, _, _ = np.linalg.svd(np.random.rand(4, 20, 16), full_matrices=False)

    # we have four protos and three channels
    diss = np.zeros((4, 3))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = TangentDistance(prototype_initializer='zeros', projected_atom_shape=(4, 4), linear_factor=None,
                               signal_output='projected_signals')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, subspaces])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # compute distances by numpy
    diss_result = np_tangent_distance(signals, protos, subspaces, caps._proto_distrib)

    # check distance values
    assert np.all(np.isclose(diss_result, y[1], atol=np.power(Distance.epsilon, 1 / 2)))

    np_signals = np_get_projected_signals(signals, protos, subspaces, caps._proto_distrib)

    # check if the tiling of the signal is correct
    assert np.all(np.isclose(np_signals[0, :], y[0][:, caps._proto_distrib[0], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[1, :], y[0][:, caps._proto_distrib[1], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[2, :], y[0][:, caps._proto_distrib[2], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[3, :], y[0][:, caps._proto_distrib[3], :], atol=np.power(Distance.epsilon, 1 / 2)))

    # --------------- test TANGENT DISTANCE Output projected, global, matrix
    signals = np.random.rand(4, 3, 3, 3)
    protos = np.random.rand(4, 3, 3)
    subspaces, _, _ = np.linalg.svd(np.random.rand(9, 4), full_matrices=False)

    # we have four protos and three channels
    diss = np.zeros((4, 3))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = TangentDistance(prototype_initializer='zeros', projected_atom_shape=(2, 2), linear_factor=None,
                               matrix_scope='global', signal_output='projected_signals')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, subspaces])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # compute distances by numpy
    diss_result = np_tangent_distance(signals, protos,
                                      np.tile(np.expand_dims(subspaces, 0), [protos.shape[0], 1, 1]),
                                      caps._proto_distrib)

    # check distance values
    assert np.all(np.isclose(diss_result, y[1], atol=np.power(Distance.epsilon, 1 / 2)))

    np_signals = np_get_projected_signals(signals, protos,
                                          np.tile(np.expand_dims(subspaces, 0), [protos.shape[0], 1, 1]),
                                          caps._proto_distrib)

    # check if the tiling of the signal is correct
    assert np.all(np.isclose(np_signals[0, :], y[0][:, caps._proto_distrib[0], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[1, :], y[0][:, caps._proto_distrib[1], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[2, :], y[0][:, caps._proto_distrib[2], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[3, :], y[0][:, caps._proto_distrib[3], :], atol=np.power(Distance.epsilon, 1 / 2)))

    # --------------- test TANGENT DISTANCE Output parametrized, local, matrix
    signals = np.random.rand(4, 3, 4, 5)
    protos = np.random.rand(4, 4, 5)
    subspaces, _, _ = np.linalg.svd(np.random.rand(4, 20, 16), full_matrices=False)

    # we have four protos and three channels
    diss = np.zeros((4, 3))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = TangentDistance(prototype_initializer='zeros', projected_atom_shape=(4, 4), linear_factor=None,
                               signal_output='parameterized_signals')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, subspaces])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # compute distances by numpy
    diss_result = np_tangent_distance(signals, protos, subspaces, caps._proto_distrib)

    # check distance values
    assert np.all(np.isclose(diss_result, y[1], atol=np.power(Distance.epsilon, 1 / 2)))

    np_signals = np_parametrized_signals(signals, protos, subspaces, caps._proto_distrib, (4, 4))

    # check if the tiling of the signal is correct
    assert np.all(np.isclose(np_signals[0, :], y[0][:, caps._proto_distrib[0], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[1, :], y[0][:, caps._proto_distrib[1], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[2, :], y[0][:, caps._proto_distrib[2], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[3, :], y[0][:, caps._proto_distrib[3], :], atol=np.power(Distance.epsilon, 1 / 2)))

    # --------------- test TANGENT DISTANCE Output parametrized, global, matrix
    signals = np.random.rand(4, 3, 3, 3)
    protos = np.random.rand(4, 3, 3)
    subspaces, _, _ = np.linalg.svd(np.random.rand(9, 4), full_matrices=False)

    # we have four protos and three channels
    diss = np.zeros((4, 3))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = TangentDistance(prototype_initializer='zeros', projected_atom_shape=(2, 2), linear_factor=None,
                               matrix_scope='global', signal_output='parameterized_signals')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, subspaces])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # compute distances by numpy
    diss_result = np_tangent_distance(signals, protos,
                                      np.tile(np.expand_dims(subspaces, 0), [protos.shape[0], 1, 1]),
                                      caps._proto_distrib)

    # check distance values
    assert np.all(np.isclose(diss_result, y[1], atol=np.power(Distance.epsilon, 1 / 2)))

    np_signals = np_parametrized_signals(signals, protos,
                                          np.tile(np.expand_dims(subspaces, 0), [protos.shape[0], 1, 1]),
                                          caps._proto_distrib, (2, 2))

    # check if the tiling of the signal is correct
    assert np.all(np.isclose(np_signals[0, :], y[0][:, caps._proto_distrib[0], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[1, :], y[0][:, caps._proto_distrib[1], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[2, :], y[0][:, caps._proto_distrib[2], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[3, :], y[0][:, caps._proto_distrib[3], :], atol=np.power(Distance.epsilon, 1 / 2)))


def test_restricted_tangent_distance():
    def np_restricted_tangent_distance(signals, protos, subspaces, bounds, proto_distrib):
        # compute distances by numpy
        diss_result = np.zeros((protos.shape[0], signals.shape[1]))
        for i in range(diss_result.shape[0]):  # protos
            for j in range(diss_result.shape[1]):  # channels
                for idx, x in enumerate(proto_distrib):
                    if x.count(i) > 0:
                        proto = protos[i, :].flatten()
                        signal = signals[idx, j, :].flatten()
                        subspace = subspaces[i, :, :]
                        bound = bounds[i, :]
                        diff = signal - proto

                        matching_proto = np.dot(subspace, np.clip(np.dot(diff, subspace), -bound, bound)) + proto

                        diss_result[i, j] = np.sqrt(np.sum(np.square(signal - matching_proto)))
                        break

        return diss_result

    def np_get_projected_signals(signals, protos, subspaces, bounds, proto_distrib):
        # compute distances by numpy
        projected_signals = np.zeros((protos.shape[0],) + signals.shape[1:])
        for i in range(diss_result.shape[0]):  # protos
            for j in range(diss_result.shape[1]):  # channels
                for idx, x in enumerate(proto_distrib):
                    if x.count(i) > 0:
                        proto = protos[i, :].flatten()
                        signal = signals[idx, j, :].flatten()
                        subspace = subspaces[i, :, :]
                        bound = bounds[i, :]
                        diff = signal - proto

                        matching_proto = np.dot(subspace, np.clip(np.dot(diff, subspace), -bound, bound)) + proto

                        projected_signals[i, j, :] = np.reshape(matching_proto, signals.shape[2:])
                        break

        return projected_signals

    def np_parametrized_signals(signals, protos, subspaces, bounds, proto_distrib, projected_atom_shape):
        # compute distances by numpy
        projected_signals = np.zeros((protos.shape[0], signals.shape[1]) + projected_atom_shape)
        for i in range(diss_result.shape[0]):  # protos
            for j in range(diss_result.shape[1]):  # channels
                for idx, x in enumerate(proto_distrib):
                    if x.count(i) > 0:
                        proto = protos[i, :].flatten()
                        signal = signals[idx, j, :].flatten()
                        subspace = subspaces[i, :, :]
                        bound = bounds[i, :]
                        diff = signal - proto

                        matching_proto = np.clip(np.dot(diff, subspace), -bound, bound)

                        projected_signals[i, j, :] = np.reshape(matching_proto, projected_atom_shape)
                        break

        return projected_signals

    # --------------- test TANGENT DISTANCE, local, vector
    signals = np.array([[[2, 0],  # first capsule input
                         [0, 10],
                         [-1, 3]],
                        [[1, 0],  # second capsule input
                         [1, 1],
                         [-1, -1]]])

    signals = np.reshape(np.tile(signals, [2, 1, 1]), [4, 3, 2])

    # we have four protos and three channels
    diss = np.zeros((4, 3))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = RestrictedTangentDistance(prototype_initializer='zeros', projected_atom_shape=1, linear_factor=None)

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    # # four protos of dimension 2
    protos = np.array([[0, 0],
                       [0, 1],
                       [-1, 1],
                       [2, 1]])

    subspaces = np.array([[[1], [0]],
                          [[1], [0]],
                          [[1], [0]],
                          [[1], [0]]])

    bounds = np.array([[100],
                       [100],
                       [100],
                       [100]])

    Distance.set_weights([protos, subspaces, bounds])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # check if the tiling of the signal is correct
    assert np.all(np.isclose(signals[0, :], y[0][:, caps._proto_distrib[0], :]))
    assert np.all(np.isclose(signals[1, :], y[0][:, caps._proto_distrib[1], :]))

    # compute distances by numpy
    diss_result = np_restricted_tangent_distance(signals, protos, subspaces, bounds, caps._proto_distrib)

    # check distance values
    assert np.all(np.isclose(diss_result, y[1], atol=np.power(Distance.epsilon, 1 / 2)))

    # --------------- test TANGENT DISTANCE, global, vector
    signals = np.random.rand(4, 3, 10)
    protos = np.random.rand(4, 10)
    bounds = np.random.rand(3)
    subspaces, _, _ = np.linalg.svd(np.random.rand(10, 3), full_matrices=False)

    # we have four protos and three channels
    diss = np.zeros((4, 3))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = RestrictedTangentDistance(prototype_initializer='zeros', projected_atom_shape=3, linear_factor=None,
                                         matrix_scope='global')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, subspaces, bounds])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # check if the tiling of the signal is correct
    assert np.all(np.isclose(signals[0, :], y[0][:, caps._proto_distrib[0], :]))
    assert np.all(np.isclose(signals[1, :], y[0][:, caps._proto_distrib[1], :]))

    # compute distances by numpy
    diss_result = np_restricted_tangent_distance(signals, protos,
                                                 np.tile(np.expand_dims(subspaces, 0), [protos.shape[0], 1, 1]),
                                                 np.tile(np.expand_dims(bounds, 0), [protos.shape[0], 1]),
                                                 caps._proto_distrib)

    # check distance values
    assert np.all(np.isclose(diss_result, y[1], atol=np.power(Distance.epsilon, 1 / 2)))

    # --------------- test TANGENT DISTANCE, local, matrix
    signals = np.random.rand(4, 3, 4, 5)
    protos = np.random.rand(4, 4, 5)
    bounds = np.random.rand(4, 16)
    subspaces, _, _ = np.linalg.svd(np.random.rand(4, 20, 16), full_matrices=False)

    # we have four protos and three channels
    diss = np.zeros((4, 3))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = RestrictedTangentDistance(prototype_initializer='zeros', projected_atom_shape=(4, 4), linear_factor=None)

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, subspaces, bounds])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # check if the tiling of the signal is correct
    assert np.all(np.isclose(signals[0, :], y[0][:, caps._proto_distrib[0], :]))
    assert np.all(np.isclose(signals[1, :], y[0][:, caps._proto_distrib[1], :]))

    # compute distances by numpy
    diss_result = np_restricted_tangent_distance(signals, protos, subspaces, bounds, caps._proto_distrib)

    # check distance values
    assert np.all(np.isclose(diss_result, y[1], atol=np.power(Distance.epsilon, 1 / 2)))

    # --------------- test TANGENT DISTANCE, global, matrix
    signals = np.random.rand(4, 3, 3, 3)
    protos = np.random.rand(4, 3, 3)
    bounds = np.random.rand(4)
    subspaces, _, _ = np.linalg.svd(np.random.rand(9, 4), full_matrices=False)

    # we have four protos and three channels
    diss = np.zeros((4, 3))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = RestrictedTangentDistance(prototype_initializer='zeros', projected_atom_shape=(2, 2), linear_factor=None,
                               matrix_scope='global')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, subspaces, bounds])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # check if the tiling of the signal is correct
    assert np.all(np.isclose(signals[0, :], y[0][:, caps._proto_distrib[0], :]))
    assert np.all(np.isclose(signals[1, :], y[0][:, caps._proto_distrib[1], :]))

    # compute distances by numpy
    diss_result = np_restricted_tangent_distance(signals, protos,
                                                 np.tile(np.expand_dims(subspaces, 0), [protos.shape[0], 1, 1]),
                                                 np.tile(np.expand_dims(bounds, 0), [protos.shape[0], 1]),
                                                 caps._proto_distrib)

    # check distance values
    assert np.all(np.isclose(diss_result, y[1], atol=np.power(Distance.epsilon, 1 / 2)))

    # --------------- test TANGENT DISTANCE Output projected, local, matrix
    signals = np.random.rand(4, 3, 4, 5)
    protos = np.random.rand(4, 4, 5)
    bounds = np.random.rand(4, 12)
    subspaces, _, _ = np.linalg.svd(np.random.rand(4, 20, 12), full_matrices=False)

    # we have four protos and three channels
    diss = np.zeros((4, 3))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = RestrictedTangentDistance(prototype_initializer='zeros', projected_atom_shape=(3, 4), linear_factor=None,
                                         signal_output='projected_signals')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, subspaces, bounds])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # compute distances by numpy
    diss_result = np_restricted_tangent_distance(signals, protos, subspaces, bounds, caps._proto_distrib)

    # check distance values
    assert np.all(np.isclose(diss_result, y[1], atol=np.power(Distance.epsilon, 1 / 2)))

    np_signals = np_get_projected_signals(signals, protos, subspaces, bounds, caps._proto_distrib)

    # check if the tiling of the signal is correct
    assert np.all(np.isclose(np_signals[0, :], y[0][:, caps._proto_distrib[0], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[1, :], y[0][:, caps._proto_distrib[1], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[2, :], y[0][:, caps._proto_distrib[2], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[3, :], y[0][:, caps._proto_distrib[3], :], atol=np.power(Distance.epsilon, 1 / 2)))

    # --------------- test TANGENT DISTANCE Output projected, global, matrix
    signals = np.random.rand(4, 3, 3, 3)
    protos = np.random.rand(4, 3, 3)
    bounds = np.random.rand(4)
    subspaces, _, _ = np.linalg.svd(np.random.rand(9, 4), full_matrices=False)

    # we have four protos and three channels
    diss = np.zeros((4, 3))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = RestrictedTangentDistance(prototype_initializer='zeros', projected_atom_shape=(2, 2), linear_factor=None,
                                         matrix_scope='global', signal_output='projected_signals')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, subspaces, bounds])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # compute distances by numpy
    diss_result = np_restricted_tangent_distance(signals, protos,
                                                 np.tile(np.expand_dims(subspaces, 0), [protos.shape[0], 1, 1]),
                                                 np.tile(np.expand_dims(bounds, 0), [protos.shape[0], 1]),
                                                 caps._proto_distrib)

    # check distance values
    assert np.all(np.isclose(diss_result, y[1], atol=np.power(Distance.epsilon, 1 / 2)))

    np_signals = np_get_projected_signals(signals, protos,
                                          np.tile(np.expand_dims(subspaces, 0), [protos.shape[0], 1, 1]),
                                          np.tile(np.expand_dims(bounds, 0), [protos.shape[0], 1]),
                                          caps._proto_distrib)

    # check if the tiling of the signal is correct
    assert np.all(np.isclose(np_signals[0, :], y[0][:, caps._proto_distrib[0], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[1, :], y[0][:, caps._proto_distrib[1], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[2, :], y[0][:, caps._proto_distrib[2], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[3, :], y[0][:, caps._proto_distrib[3], :], atol=np.power(Distance.epsilon, 1 / 2)))

    # --------------- test TANGENT DISTANCE Output parametrized, local, matrix
    signals = np.random.rand(4, 3, 4, 5)
    protos = np.random.rand(4, 4, 5)
    bounds = np.random.rand(4, 16)
    subspaces, _, _ = np.linalg.svd(np.random.rand(4, 20, 16), full_matrices=False)

    # we have four protos and three channels
    diss = np.zeros((4, 3))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = RestrictedTangentDistance(prototype_initializer='zeros', projected_atom_shape=(4, 4), linear_factor=None,
                                         signal_output='parameterized_signals')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, subspaces, bounds])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # compute distances by numpy
    diss_result = np_restricted_tangent_distance(signals, protos, subspaces, bounds, caps._proto_distrib)

    # check distance values
    assert np.all(np.isclose(diss_result, y[1], atol=np.power(Distance.epsilon, 1 / 2)))

    np_signals = np_parametrized_signals(signals, protos, subspaces, bounds, caps._proto_distrib, (4, 4))

    # check if the tiling of the signal is correct
    assert np.all(np.isclose(np_signals[0, :], y[0][:, caps._proto_distrib[0], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[1, :], y[0][:, caps._proto_distrib[1], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[2, :], y[0][:, caps._proto_distrib[2], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[3, :], y[0][:, caps._proto_distrib[3], :], atol=np.power(Distance.epsilon, 1 / 2)))

    # --------------- test TANGENT DISTANCE Output parametrized, global, matrix
    signals = np.random.rand(4, 3, 3, 3)
    protos = np.random.rand(4, 3, 3)
    bounds = np.random.rand(4)
    subspaces, _, _ = np.linalg.svd(np.random.rand(9, 4), full_matrices=False)

    # we have four protos and three channels
    diss = np.zeros((4, 3))

    # build inputs
    inputs = [Input(signals.shape), Input(diss.shape)]

    caps = Capsule(prototype_distribution=(1, 4))

    Distance = RestrictedTangentDistance(prototype_initializer='zeros', projected_atom_shape=(2, 2), linear_factor=None,
                                         matrix_scope='global', signal_output='parameterized_signals')

    caps.add(Distance)
    outputs = caps(list_to_dict(inputs))

    Distance.set_weights([protos, subspaces, bounds])

    model = Model(inputs, dict_to_list(outputs))

    y = model.predict([np.expand_dims(signals, 0), np.expand_dims(diss, 0)], batch_size=1, verbose=0)

    print(outputs)

    # compute distances by numpy
    diss_result = np_restricted_tangent_distance(signals, protos,
                                                 np.tile(np.expand_dims(subspaces, 0), [protos.shape[0], 1, 1]),
                                                 np.tile(np.expand_dims(bounds, 0), [protos.shape[0], 1]),
                                                 caps._proto_distrib)

    # check distance values
    assert np.all(np.isclose(diss_result, y[1], atol=np.power(Distance.epsilon, 1 / 2)))

    np_signals = np_parametrized_signals(signals, protos,
                                         np.tile(np.expand_dims(subspaces, 0), [protos.shape[0], 1, 1]),
                                         np.tile(np.expand_dims(bounds, 0), [protos.shape[0], 1]),
                                         caps._proto_distrib, (2, 2))

    # check if the tiling of the signal is correct
    assert np.all(np.isclose(np_signals[0, :], y[0][:, caps._proto_distrib[0], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[1, :], y[0][:, caps._proto_distrib[1], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[2, :], y[0][:, caps._proto_distrib[2], :], atol=np.power(Distance.epsilon, 1 / 2)))
    assert np.all(np.isclose(np_signals[3, :], y[0][:, caps._proto_distrib[3], :], atol=np.power(Distance.epsilon, 1 / 2)))


if __name__ == "__main__":
    main()

