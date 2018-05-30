# -*- coding: utf-8 -*-
""" It's not the traditional GLVQ, because we replace the GLVQ loss function with negative softmax and cross-entropy.
Final result is around 97%.
The traditional GLVQ stops at around 90%.
"""

from keras import callbacks
from keras.models import Model
from keras.utils import to_categorical
from keras import optimizers
from keras import Input
from keras.callbacks import EarlyStopping
from keras.datasets import mnist

from anysma import Capsule
from anysma.modules import InputModule
from anysma.modules.routing import SqueezeRouting
from anysma.modules.measuring import RestrictedTangentDistance
from anysma.modules.competition import NearestCompetition
from anysma.modules.final import Classification
from anysma.callbacks import TensorBoard

import os

import matplotlib
matplotlib.use('Agg')  # needed to avoid cloud errors
from matplotlib import pyplot as plt

import numpy as np


num_tangents = 12
batch_size = 128
num_classes = 10
epochs = 25
lr = 0.001
lr_decay_factor = 0.98
save_dir = './output'

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train.astype('float32') / 255., -1)
x_test = np.expand_dims(x_test.astype('float32') / 255., -1)
y_train = to_categorical(y_train.astype('float32'))
y_test = to_categorical(y_test.astype('float32'))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

inputs = Input(x_train.shape[1:])

# define rTDLVQ prototypes
diss = RestrictedTangentDistance(projected_atom_shape=num_tangents,
                                 linear_factor=None,
                                 squared_dissimilarity=True,
                                 signal_output='signals')

caps = Capsule(prototype_distribution=(1, num_classes))
caps.add(InputModule(signal_shape=(1,) + x_train.shape[1:4], trainable=False, dissimilarity_initializer='zeros'))
caps.add(diss)
caps.add(SqueezeRouting())
caps.add(NearestCompetition(use_for_loop=False))
caps.add(Classification(probability_transformation='neg_softmax', name='lvq_capsule'))

outputs = caps(inputs)

# pre-train the model over 10000 random digits
idx = np.random.randint(0, len(x_train) - 1, (min(10000, len(x_train)),))
pre_train_model = Model(inputs=inputs, outputs=diss.input[0])
diss_input = pre_train_model.predict(x_train[idx, :], batch_size=batch_size)
diss.pre_training(diss_input, y_train[idx], capsule_inputs_are_equal=True)

model = Model(inputs, outputs[2])
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics={'lvq_capsule': 'accuracy'})

model.summary()

# callbacks
log = callbacks.CSVLogger(os.path.join(save_dir, 'log.csv'))
tb = TensorBoard(log_dir=os.path.join(save_dir, 'tensorboard-logs'),
                 batch_size=batch_size)

es = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto')

model.fit(x_train, y_train,
          validation_data=[x_test, y_test],
          epochs=epochs,
          shuffle=True,
          verbose=1,
          callbacks=[log, tb, es])

model.save_weights(os.path.join(save_dir, 'trained_model.h5'))

plots_per_proto = num_tangents + 1
protos = np.squeeze(diss.get_weights()[0], -1)
tangents = diss.get_weights()[1]
for i in range(num_classes):
    for j in range(plots_per_proto):
        plt.subplot(plots_per_proto, num_classes, j * num_classes + i + 1)
        if j == 0:
            img = plt.imshow(protos[i, :, :])
        else:
            img = plt.imshow(np.reshape(tangents[i, :, j-1], [28, 28]))
        img.set_cmap('gray')
        plt.axis('off')
plt.savefig(os.path.join(save_dir, 'protos.png'), bbox_inches='tight')

